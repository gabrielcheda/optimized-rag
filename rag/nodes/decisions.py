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
    Decide if Chain-of-Thought reasoning is needed.

    CoT is triggered for:
    - Multi-hop questions (comparison, aggregation)
    - Complex queries requiring step-by-step reasoning
    - Queries with COMPARE or AGGREGATE intents

    Returns:
        "cot" or "skip"
    """
    if not config.ENABLE_COT_REASONING:
        return "skip"

    intent_enum = state.query_intent
    is_multi_hop_intent = intent_enum and intent_enum in [
        QueryIntent.COMPARISON,
        QueryIntent.MULTI_HOP_REASONING,
    ]

    if is_multi_hop_intent:
        logger.info(f"CoT triggered by intent: {intent_enum.value}")
        return "cot"

    query = state.user_input
    word_count = len(query.split())
    has_multiple_questions = query.count("?") > 1

    if has_multiple_questions and word_count > config.COT_WORD_COUNT_THRESHOLD:
        logger.info(
            f"CoT triggered by complexity: {word_count} words, multiple questions: {has_multiple_questions}"
        )
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

    refinement_decision = should_refine_query(state, agent)
    if refinement_decision == "refine":
        return "refine"

    return "continue"


def should_refine_query(state, agent) -> str:
    """
    Decide if query should be refined.

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
    if refinement_count >= config.MAX_REFINEMENT_ATTEMPTS:
        logger.info("Max refinement attempts reached")
        return "continue"

    quality_eval = state.quality_eval
    is_relevant = quality_eval.get("is_relevant", True)
    confidence = quality_eval.get("confidence", 1.0)

    if not is_relevant or confidence < config.REFINEMENT_CONFIDENCE_THRESHOLD:
        logger.info(
            f"Query refinement triggered (relevant={is_relevant}, conf={confidence:.2f})"
        )
        return "refine"

    answer = state.agent_response or ""
    if len(answer.split()) < config.MIN_ANSWER_WORD_COUNT:
        logger.info("Query refinement triggered (short answer)")
        return "refine"

    return "continue"
