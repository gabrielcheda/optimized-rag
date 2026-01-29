"""
Rewrite Query Node
Optimizes query for better retrieval using unified System 2 approach
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState
from rag.nodes.helpers import is_non_english, translate_to_english

logger = logging.getLogger(__name__)


def rewrite_query_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Rewrite query for better retrieval (Unified System 2 approach).
    Updates state with the optimized query and its variants.
    
    CRITICAL: Translates non-English queries before rewriting for consistency.
    """
    query = state.user_input
    intent = state.query_intent
    
    # CRITICAL: Translate non-English queries to English BEFORE rewriting
    # This ensures rewrites are in English, consistent with retrieval expectations
    original_query = query
    query_language = "original"
    
    if is_non_english(query):
        logger.info(f"Detected non-English query in rewrite phase, translating...")
        query = translate_to_english(query, agent.openai_client)
        query_language = "translated"
        logger.info(f"Translated for rewrite: '{original_query[:50]}...' -> '{query[:50]}...'")
    
    # Call the unified rewriter (makes only 1 LLM call)
    rewrite_result = agent.query_rewriter.rewrite(
        query=query,  # Now using English query
        intent=intent,
        conversation_history=state.messages
    )
    
    # Extract data from UnifiedRewrite object saved in 'metadata'
    metadata = rewrite_result.get("metadata", {})
    
    # Strategic logging to monitor cost and latency savings
    strategies = rewrite_result.get("strategies", [])
    logger.info(
        f"Query optimized ({query_language}): "
        f"'{original_query[:50]}...' -> '{rewrite_result['rewritten'][:50]}...'. "
        f"Strategies: {strategies or 'none'}, "
        f"LLM Calls saved: {rewrite_result.get('operations_saved', 0)}"
    )

    # Prepare query variants for state (used in parallel/multi-vector search)
    # Get transformed queries or keep original if field is null
    query_variants = [
        metadata.get("contextualized_query") or rewrite_result["rewritten"],
        metadata.get("reformulated_query") or rewrite_result["rewritten"],
        metadata.get("simplified_query") or query
    ]

    # Return for automatic State update in LangGraph
    return {
        "rewritten_query": rewrite_result["rewritten"],
        "query_variants": list(set(query_variants)),  # Remove duplicates
        "cot_reasoning": metadata.get("reasoning", state.cot_reasoning),
        "translated_query": query if query_language == "translated" else None  # Store translation
    }
