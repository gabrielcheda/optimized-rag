"""
Decision functions for RAG graph conditional edges
These functions determine routing in the LangGraph workflow
"""

import logging

import config
from rag import QueryIntent

logger = logging.getLogger(__name__)


def should_use_cot(state, agent) -> str:
    """
    Decide if Chain-of-Thought reasoning is needed (Paper-compliant)

    CoT is triggered for:
    - Multi-hop questions (comparison, aggregation)
    - Complex queries requiring step-by-step reasoning
    - Queries with COMPARE or AGGREGATE intents

    Returns:
        "cot" or "skip"
    """
    if not config.ENABLE_COT_REASONING:
        return "skip"

    # Check query intent
    intent_enum = state.query_intent
    if intent_enum and intent_enum in [
        QueryIntent.COMPARISON,
        QueryIntent.MULTI_HOP_REASONING,
    ]:
        logger.info(f"CoT triggered by intent: {intent_enum.value}")
        return "cot"

    # Check query complexity (word count, nested questions)
    query = state.user_input
    word_count = len(query.split())
    has_multiple_questions = query.count("?") > 1 or " and " in query.lower()

    if word_count > 20 or has_multiple_questions:
        logger.info(
            f"CoT triggered by complexity: {word_count} words, multiple questions: {has_multiple_questions}"
        )
        return "cot"

    # Check if evaluation suggests low confidence
    quality_eval = state.quality_eval
    confidence = quality_eval.get("confidence", 1.0)
    if confidence < 0.5:
        logger.info(f"CoT triggered by low confidence: {confidence:.2f}")
        return "cot"

    return "skip"


def decide_next_action(state, agent) -> str:
    """
    Unified decision point after generate_response
    Priority: tools > refine > continue

    Returns:
        "tools", "refine", or "continue"
    """
    if state.tool_calls:
        logger.info("Tool calls detected, routing to process_tool_calls")
        return "tools"

    # Check if query should be refined (delegates to should_refine_query)
    refinement_decision = should_refine_query(state, agent)
    if refinement_decision == "refine":
        return "refine"

    # Default: continue to memory update
    return "continue"


def should_refine_query(state, agent) -> str:
    """
    Decide if query should be refined (Paper-compliant: iterative refinement)

    Refinement is triggered when:
    - Low context quality
    - Low confidence in answer
    - Haven't exceeded max refinement attempts

    Returns:
        "refine" or "continue"
    """
    if not config.ENABLE_QUERY_REFINEMENT:
        return "continue"

    refinement_count = state.refinement_count
    if refinement_count >= 2:  # Max 2 refinement attempts
        logger.info("Max refinement attempts reached")
        return "continue"

    # Check context quality
    quality_eval = state.quality_eval
    is_relevant = quality_eval.get("is_relevant", True)
    confidence = quality_eval.get("confidence", 1.0)

    if not is_relevant or confidence < 0.4:
        logger.info(
            f"Query refinement triggered (relevant={is_relevant}, conf={confidence:.2f})"
        )
        return "refine"

    # Check if answer is too short (might indicate insufficient context)
    answer = state.agent_response or ""
    if not isinstance(answer, str):
        answer = str(answer)
    if len(answer.split()) < 20:
        logger.info("Query refinement triggered (short answer)")
        return "refine"

    return "continue"
