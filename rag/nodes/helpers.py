"""
Helper functions for RAG graph nodes
Shared utilities used across multiple node functions
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import config
from rag.models.intent_analysis import QueryIntent
from prompts.translation_prompts import TRANSLATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def format_context_with_citations(documents: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
    """Format context with citation IDs for source tracking"""
    if not documents:
        return "", {}

    formatted_context = ""
    source_map = {}

    for i, doc in enumerate(documents, 1):
        source_id = f"[{i}]"
        content = doc.get("content", "")
        score = doc.get("score", 0)
        source = doc.get("source", doc.get("metadata", {}).get("source", "unknown"))

        source_map[source_id] = {"content": content, "source": source, "score": score}

        formatted_context += f"\n{source_id} (Score: {score:.3f}) {content}\n"

    return formatted_context, source_map


def check_context_quality(
    documents: List[Dict[str, Any]], min_score: float = config.MIN_QUALITY_SCORE
) -> Dict[str, Any]:
    """
    Validate context quality before generation to prevent hallucinations

    Args:
        documents: List of retrieved documents with scores
        min_score: Minimum acceptable relevance score

    Returns:
        Dict with:
            - sufficient (bool): Whether context is good enough
            - reason (str): Explanation if insufficient
            - message (str): Fallback message for user
            - max_score (float): Highest relevance score
            - avg_score (float): Average relevance score
    """
    if not documents or len(documents) == 0:
        return {
            "sufficient": False,
            "reason": "No documents retrieved",
            "message": (
                "I don't have enough information in my knowledge base to answer this question confidently. "
                "Could you provide more context or rephrase your question?"
            ),
            "max_score": 0.0,
            "avg_score": 0.0,
        }

    # Extract scores
    scores = [doc.get("score", 0.0) for doc in documents if "score" in doc]

    if not scores:
        # No scores available - assume sufficient (better than blocking)
        return {
            "sufficient": True,
            "reason": "No scores available, proceeding",
            "message": "",
            "max_score": 1.0,
            "avg_score": 1.0,
        }

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    # Quality checks
    if max_score < min_score:
        return {
            "sufficient": False,
            "reason": f"Max relevance score ({max_score:.3f}) below threshold ({min_score})",
            "message": (
                f"The information I found isn't very relevant to your question "
                f"(confidence: {max_score * 100:.1f}%). I'd rather admit uncertainty than provide unreliable information. "
                "Could you rephrase or provide more details?"
            ),
            "max_score": max_score,
            "avg_score": avg_score,
        }

    if avg_score < config.MIN_AVG_RELEVANCE_SCORE:
        return {
            "sufficient": False,
            "reason": f"Average relevance score ({avg_score:.3f}) too low",
            "message": (
                f"While I found some information, most of it isn't very relevant "
                f"(average confidence: {avg_score * 100:.1f}%). To give you accurate information, "
                "I need better matching sources. Can you clarify your question?"
            ),
            "max_score": max_score,
            "avg_score": avg_score,
        }

    # Context is sufficient
    return {
        "sufficient": True,
        "reason": f"Quality OK (max: {max_score:.3f}, avg: {avg_score:.3f})",
        "message": "",
        "max_score": max_score,
        "avg_score": avg_score,
    }


def enrich_context_with_memory(state, agent) -> Tuple[str, Dict[str, Dict]]:
    """
    Enrich generation context with memory information

    Combines:
    - Core memory (persona + facts)
    - Recent conversation (for continuity)
    - Retrieved context (documents + archival)

    Returns:
        Tuple of (enriched_context_string, source_map_dict)
    """
    from utils.context import format_core_memory

    context_parts = []

    # 1. Core Memory (always included)
    core_memory = format_core_memory(
        state.human_persona, state.agent_persona, state.core_facts
    )
    context_parts.append(f"CORE MEMORY:\n{core_memory}")

    # 2. Recent Conversation Context (if available)
    # For clarification queries, use ALL messages since answer might be from earlier
    if state.retrieved_recall and len(state.retrieved_recall) > 0:
        recent_messages = []
        is_clarification = state.query_intent == QueryIntent.CLARIFICATION
        messages_to_use = (
            state.retrieved_recall if is_clarification else state.retrieved_recall[-5:]
        )

        for msg in messages_to_use:
            recent_messages.append(f"{msg['role']}: {msg['content']}")
        if recent_messages:
            header = (
                "CONVERSATION HISTORY" if is_clarification else "RECENT CONVERSATION"
            )
            context_parts.append(f"\n{header}:\n" + "\n".join(recent_messages))

    # 3. Retrieved Context (documents + archival + KG)
    cited_context, source_map = format_context_with_citations(state.final_context)
    if cited_context:
        context_parts.append(f"\nRETRIEVED CONTEXT WITH CITATIONS:\n{cited_context}")

    # 4. Synthesized Analysis (if available)
    if state.synthesized_context:
        context_parts.append(
            f"\nSYNTHESIZED ANALYSIS (Multi-Document):\n{state.synthesized_context}"
        )

    # 5. Chain-of-Thought Reasoning (if available)
    if state.cot_reasoning:
        context_parts.append(
            f"\nREASONING TRACE (Chain-of-Thought):\n{state.cot_reasoning}"
        )

    return "\n\n".join(context_parts), source_map


def apply_mmr(
    query: str,
    documents: List[Dict[str, Any]],
    lambda_: float,
    k: int,
    embedding_service,
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance for diversity

    MMR balances relevance and diversity to avoid redundant results.
    Formula: MMR = λ * Relevance - (1-λ) * MaxSimilarity

    Args:
        query: Query text
        documents: List of documents with embeddings
        lambda_: Balance between relevance (1.0) and diversity (0.0)
        k: Number of documents to select
        embedding_service: Service to generate embeddings

    Returns:
        List of k diverse documents
    """
    if len(documents) <= k:
        return documents

    try:
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)

        # Get document embeddings (or generate if missing)
        doc_embeddings = []
        for doc in documents:
            if "embedding" in doc and doc["embedding"]:
                doc_embeddings.append(doc["embedding"])
            else:
                # Generate embedding for document
                content = doc.get("content", doc.get("text", ""))
                emb = embedding_service.generate_embedding(content)
                doc["embedding"] = emb
                doc_embeddings.append(emb)

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance: cosine similarity with query
                relevance = cosine_similarity(query_embedding, doc_embeddings[idx])

                # Diversity: max similarity to already selected documents
                if selected_indices:
                    max_sim = max(
                        cosine_similarity(doc_embeddings[idx], doc_embeddings[s])
                        for s in selected_indices
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr = lambda_ * relevance - (1 - lambda_) * max_sim
                mmr_scores.append((idx, mmr))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            logger.debug(f"MMR selected doc {best_idx} with score {best_score:.3f}")

        return [documents[i] for i in selected_indices]

    except Exception as e:
        logger.error(f"MMR calculation failed: {e}", exc_info=True)
        return documents[:k]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity [-1, 1]
    """
    try:
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}", exc_info=True)
        return 0.0


def _is_factual_clarification(query: str, recalled_messages: List[Dict[str, Any]]) -> bool:
    """
    Determine if a clarification query requires document retrieval (factual)
    or can be answered from recall memory (simple clarification).
    
    Professional analysis using multiple signals instead of pattern matching.
    
    Args:
        query: User query (clarification intent)
        recalled_messages: Recent conversation history
        
    Returns:
        True if documents needed (factual request), False if recall sufficient
    """
    query_lower = query.lower()
    
    # Signal 1: Factual interrogatives (requests new information)
    factual_interrogatives = [
        # Portuguese
        "exemplo", "exemplos",          # examples
        "como", "qual", "quais",        # how, what, which
        "por que", "porque",            # why
        "onde", "quando",               # where, when
        "detalhe", "detalhes",          # details
        "especific",                    # specific/específico
        "diferença", "diferenças",      # differences
        "comparação", "compare",        # comparison
        # English
        "example", "examples",
        "how", "what", "which",
        "why", "where", "when",
        "detail", "details",
        "specific",
        "difference", "differences",
        "comparison", "compare",
        "show", "demonstrate",
        "illustrate", "explain"
    ]
    
    has_factual_interrogative = any(term in query_lower for term in factual_interrogatives)
    
    # Signal 2: Previous response was incomplete/insufficient
    # Check if last assistant message indicates lack of information
    insufficient_indicators = [
        "não tenho informações",        # "I don't have information"
        "don't have information",
        "não tenho dados",              # "I don't have data"
        "don't have data",
        "não encontrei",                # "I didn't find"
        "didn't find",
        "couldn't find",
        "forneça mais contexto",        # "provide more context"
        "provide more context",
        "preciso de mais",              # "I need more"
        "need more",
        "não consigo responder",        # "I can't answer"
        "can't answer",
        "unable to answer"
    ]
    
    last_assistant_msg = None
    for msg in reversed(recalled_messages):
        if msg.get("role") == "assistant":
            last_assistant_msg = msg.get("content", "").lower()
            break
    
    previous_was_insufficient = False
    if last_assistant_msg:
        previous_was_insufficient = any(
            indicator in last_assistant_msg 
            for indicator in insufficient_indicators
        )
    
    # Signal 3: Query contains specific nouns/technical terms (not just pronouns)
    # Extract words of 4+ characters (filter out articles, pronouns)
    stop_words = {
        "agora", "isso", "esse", "essa", "aquele", "aquela",  # Portuguese
        "that", "this", "these", "those", "them",              # English
        "você", "voce", "pode", "mais", "sobre",
        "give", "show", "tell", "make", "please"
    }
    
    words = re.findall(r'\b\w{4,}\b', query_lower)
    content_words = [w for w in words if w not in stop_words]
    has_specific_terms = len(content_words) >= 2
    
    # Signal 4: Query length (very short queries likely just need clarification)
    query_length = len(query.split())
    is_substantial_query = query_length >= 4
    
    # Decision logic: Combine signals
    # HIGH confidence factual request:
    if has_factual_interrogative and is_substantial_query:
        logger.debug(
            f"Factual clarification: factual_term=True, length={query_length}, "
            f"specific_terms={has_specific_terms}"
        )
        return True
    
    # MEDIUM-HIGH: Previous response was insufficient + user asks for specifics
    if previous_was_insufficient and has_specific_terms:
        logger.debug(
            f"Factual clarification: prev_insufficient=True, specific_terms={content_words[:3]}"
        )
        return True
    
    # LOW confidence: Short vague query → likely just needs recall clarification
    if query_length <= 3 and not has_factual_interrogative:
        logger.debug(f"Simple clarification: short query, no factual markers")
        return False
    
    # DEFAULT: If in doubt for clarifications, check factual interrogatives
    decision = has_factual_interrogative or has_specific_terms
    logger.debug(
        f"Clarification analysis: factual_interrog={has_factual_interrogative}, "
        f"specific_terms={has_specific_terms}, length={query_length} → {decision}"
    )
    return decision


def should_retrieve_documents(
    query: str, intent, recalled_messages: List[Dict[str, Any]]
) -> bool:
    """
    Decide if document retrieval is needed or if recall memory is sufficient.

    Args:
        query: User query (rewritten)
        intent: Query intent (qa, chitchat, etc)
        recalled_messages: Recent conversation messages

    Returns:
        True if documents needed, False if recall is sufficient
    """
    if not recalled_messages or len(recalled_messages) == 0:
        logger.info("Document retrieval: YES (no recall history)")
        return True

    if intent and intent.value.lower() in ["chitchat", "greeting"]:
        logger.info(f"Document retrieval: NO (intent={intent}, recall sufficient)")
        return False
    
    if intent and intent.value.lower() == "clarification":
        needs_documents = _is_factual_clarification(query, recalled_messages)
        
        if needs_documents:
            logger.info(
                f"Document retrieval: YES (factual clarification detected - "
                f"query='{query[:60]}...')"
            )
            return True
        else:
            logger.info(f"Document retrieval: NO (simple clarification, recall sufficient)")
            return False

    query_lower = query.lower()

    strong_follow_up_patterns = [
        # Portuguese - explicit follow-up requests
        "pode explicar",        # "can you explain" (the previous thing)
        "explique melhor",      # "explain better"
        "mais detalhes",        # "more details"
        "o que você disse",     # "what did you say"
        "você mencionou",       # "you mentioned"
        "você falou",           # "you said"
        "como você disse",      # "as you said"
        "conforme mencionado",  # "as mentioned"
        "me explique",          # "explain to me"
        "como assim",           # "what do you mean"
        "o que quis dizer",     # "what did you mean"
        "não entendi",          # "I didn't understand"
        "pode repetir",         # "can you repeat"
        "elabore",              # "elaborate"
        # English - explicit follow-up requests
        "tell me more",
        "explain that",
        "what did you say",
        "you mentioned",
        "you said",
        "as you said",
        "can you explain",
        "what do you mean",
        "i didn't understand",
        "elaborate on",
        "go on",
        "continue",
    ]

    weak_follow_up_patterns = [
        "sobre isso",           # "about that" - could be new topic
        "about that",
        "more about",
        "mais sobre",
        "e o",                  # "and the" - could be continuing or new
        "and the",
        "what about",
        "e sobre",
    ]

    is_strong_follow_up = any(pattern in query_lower for pattern in strong_follow_up_patterns)

    if is_strong_follow_up:
        recent_content_length = sum(
            len(msg.get("content", "").split())
            for msg in recalled_messages[-3:]
            if msg.get("role") == "assistant"
        )

        if recent_content_length > config.MIN_FOLLOW_UP_WORDS:
            logger.info(
                f"Document retrieval: NO (STRONG follow-up pattern detected, "
                f"recall has {recent_content_length} words)"
            )
            return False

    is_weak_follow_up = any(pattern in query_lower for pattern in weak_follow_up_patterns)

    if is_weak_follow_up:
        recent_content_length = sum(
            len(msg.get("content", "").split())
            for msg in recalled_messages[-3:]
            if msg.get("role") == "assistant"
        )

        common_words = {
            "what", "which", "where", "when", "that", "this", "with", "from",
            "about", "have", "does", "como", "qual", "quais", "onde", "quando",
            "para", "sobre", "mais", "pode", "você", "voce", "the", "and"
        }
        query_terms = set(
            word.lower() for word in re.findall(r'\b\w{4,}\b', query)
            if word.lower() not in common_words
        )

        recall_content = " ".join(
            msg.get("content", "").lower()
            for msg in recalled_messages[-5:]
        )
        terms_in_recall = sum(1 for term in query_terms if term in recall_content)
        topic_overlap = terms_in_recall / len(query_terms) if query_terms else 1.0

        if topic_overlap < 0.3 and len(query_terms) >= 2:
            logger.info(
                f"Document retrieval: YES (weak follow-up pattern but topic is new - "
                f"overlap={topic_overlap:.1%}, terms={list(query_terms)[:5]})"
            )
            return True

        if recent_content_length > config.MIN_FOLLOW_UP_WORDS:
            logger.info(
                f"Document retrieval: NO (weak follow-up with topic match, "
                f"recall has {recent_content_length} words, overlap={topic_overlap:.1%})"
            )
            return False

    factual_intents = ["qa", "question_answering", "factual", "compare", "aggregate"]
    if intent and intent.value.lower() in factual_intents and not (is_strong_follow_up or is_weak_follow_up):
        logger.info(f"Document retrieval: YES (factual intent={intent}, not follow-up)")
        return True

    logger.info("Document retrieval: YES (default fallback)")
    return True


def is_non_english(text: str) -> bool:
    """Check if text is non-English using language detection"""
    try:
        # detect() returns language code (pt, en, es, etc.)
        lang = detect(text)
        return lang != "en"
    except LangDetectException:
        # In case of failure (e.g., text with only numbers), assume False
        return False


def translate_to_english(text: str, openai_client) -> str:
    """
    Translate text to English using LLM

    Returns:
        English translation
    """
    try:
        response = openai_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=200,
        )

        translation = response.choices[0].message.content
        translation = translation.strip() if translation else text
        return translation

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return text  # Fallback to original


def export_metrics_to_json(metrics: Dict, compression_stats: Dict, agent_id: str):
    """Export metrics to JSON file for dashboard (System2 observability)"""
    import json
    import os
    from datetime import datetime

    metrics_dir = "metrics_logs"
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().isoformat()
    metrics_entry = {
        "timestamp": timestamp,
        "agent_id": agent_id,
        "retrieval_metrics": metrics,
        "compression_stats": compression_stats,
        "config": {
            "enable_dynamic_weights": config.ENABLE_DYNAMIC_WEIGHTS,
            "enable_hierarchical_retrieval": config.ENABLE_HIERARCHICAL_RETRIEVAL,
            "enable_context_compression": config.ENABLE_CONTEXT_COMPRESSION,
        },
    }

    # Append to daily log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(metrics_dir, f"metrics_{date_str}.jsonl")

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_entry) + "\n")
        logger.debug(f"Metrics exported to {log_file}")
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}", exc_info=True)
