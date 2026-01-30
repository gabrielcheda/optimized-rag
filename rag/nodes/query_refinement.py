"""
Query Refinement Node
Refines query based on previous results for iterative improvement
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState
from langchain_core.messages import SystemMessage, HumanMessage
from rag.nodes.helpers import is_non_english, translate_to_english
from prompts.query_refinement_prompts import (
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_PROMPT_TEMPLATE
)
logger = logging.getLogger(__name__)


def query_refinement_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Refine query based on previous results (Paper-compliant: iterative refinement)
    
    When initial retrieval is insufficient, this node analyzes the gap between
    the query and retrieved context, then reformulates the query for better results.
    """
    query = state.user_input
    previous_context = state.rag_context
    refinement_count = state.refinement_count
    
    # Early exit if previous refinement didn't improve score
    # Prevents redundant refinements when quality isn't improving
    retrieved_docs = state.retrieved_documents or []
    if refinement_count > 0 and retrieved_docs:
        # Get current top score
        current_top_score = max(d.get('score', 0) for d in retrieved_docs[:5]) if retrieved_docs else 0
        # Get previous score from state (stored during last refinement)
        previous_top_score = getattr(state, 'previous_refinement_score', None)
        
        if previous_top_score is not None and current_top_score <= previous_top_score:
            logger.info(
                f"No improvement detected (current={current_top_score:.3f} <= "
                f"previous={previous_top_score:.3f}), stopping refinement"
            )
            return {
                "refinement_count": refinement_count + 1,
                "previous_refinement_score": current_top_score
            }
    
    logger.info(f"Query refinement attempt {refinement_count + 1}")
    
    try:
        # Store current top score for next iteration comparison
        current_top_score = max(d.get('score', 0) for d in retrieved_docs[:5]) if retrieved_docs else 0
        
        refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            query=query,
            previous_context=previous_context[:500] + "..."
        )
        
        messages = [
            SystemMessage(content=REFINEMENT_SYSTEM_PROMPT),
            HumanMessage(content=refinement_prompt)
        ]
        
        response = agent.llm.invoke(messages)
        refined_query = response.content if hasattr(response, 'content') else str(response)
        
        # CRITICAL: Translate non-English refined queries to English
        if is_non_english(refined_query):
            logger.info(f"Detected non-English refined query, translating...")
            original_refined = refined_query
            refined_query = translate_to_english(refined_query, agent.openai_client)
            logger.info(f"Translated refined query: '{original_refined[:50]}...' -> '{refined_query[:50]}...'")
        
        logger.info(f"Refined query: {refined_query[:100]}...")
        
        return {
            "rewritten_query": refined_query,
            "refinement_count": refinement_count + 1,
            "previous_refinement_score": current_top_score
        }
        
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
        return {
            "refinement_count": refinement_count + 1
        }
