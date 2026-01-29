"""
Helper functions for RAG graph nodes
Shared utilities used across multiple node functions
"""

import logging
from typing import Any, Dict, List, Tuple

from langdetect import detect

import config

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
    documents: List[Dict[str, Any]], min_score: float = 0.3
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

    if avg_score < 0.2:
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
    if state.retrieved_recall and len(state.retrieved_recall) > 0:
        recent_messages = []

        # FIX: For clarification queries, use ALL messages (answer might be from earlier)
        from rag import QueryIntent

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
        logger.error(f"MMR calculation failed: {e}")
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
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


def should_retrieve_documents(
    query: str, intent, recalled_messages: List[Dict[str, Any]]
) -> bool:
    """
    Decide if document retrieval is needed or if recall memory is sufficient

    OPTIMIZATION: Saves embeddings + retrieval costs when answer is in recent conversation

    Args:
        query: User query (rewritten)
        intent: Query intent (qa, chitchat, etc)
        recalled_messages: Recent conversation messages

    Returns:
        True if documents needed, False if recall is sufficient
    """
    # RULE 1: Always retrieve for first message (no recall context)
    if not recalled_messages or len(recalled_messages) == 0:
        logger.info("Document retrieval: YES (no recall history)")
        return True

    # RULE 2: Chitchat/greeting → likely sufficient in recall
    if intent and intent.value.lower() in ["chitchat", "greeting", "clarification"]:
        logger.info(f"Document retrieval: NO (intent={intent}, recall sufficient)")
        return False

    # RULE 3: Follow-up indicators → check if answer might be in recall
    follow_up_patterns = [
        "o que você disse",
        "você mencionou",
        "você falou",
        "como você disse",
        "conforme mencionado",
        "what did you say",
        "you mentioned",
        "you said",
        "as you said",
        "isso",
        "aquilo",
        "that",
        "this",
        "it",
        "explain that",
        "explique isso",
        "sobre isso",
        "about that",
    ]

    query_lower = query.lower()
    is_follow_up = any(pattern in query_lower for pattern in follow_up_patterns)

    if is_follow_up:
        # Check if recent messages contain substantial content (not just greetings)
        recent_content_length = sum(
            len(msg.get("content", "").split())
            for msg in recalled_messages[-3:]  # Last 3 messages
            if msg.get("role") == "assistant"
        )

        if recent_content_length > 50:  # Substantial previous response
            logger.info(
                f"Document retrieval: NO (follow-up detected, "
                f"recall has {recent_content_length} words)"
            )
            return False

    # RULE 6: New factual query → needs documents
    factual_intents = ["qa", "question_answering", "factual", "compare", "aggregate"]
    if intent and intent.value.lower() in factual_intents and not is_follow_up:
        logger.info(f"Document retrieval: YES (factual intent={intent}, not follow-up)")
        return True

    # DEFAULT: Retrieve documents (safe fallback)
    logger.info("Document retrieval: YES (default fallback)")
    return True


def is_non_english(text: str) -> bool:
    """Check if text is non-English using language detection"""
    try:
        # detect() returns language code (pt, en, es, etc.)
        lang = detect(text)
        return lang != "en"
    except:
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
                {
                    "role": "system",
                    "content": "You are a translator. Translate the user's text to English. Return ONLY the translation, nothing else.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=200,
        )

        translation = response.choices[0].message.content
        translation = translation.strip() if translation else text
        return translation

    except Exception as e:
        logger.error(f"Translation failed: {e}")
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
        logger.error(f"Failed to export metrics: {e}")
