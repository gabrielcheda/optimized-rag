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
    Priority: tools > web_search_fallback > refine > continue

    FASE 6.1: Added web search fallback when factuality is POOR

    Returns:
        "tools", "web_search", "refine", or "continue"
    """
    if state.tool_calls:
        logger.info("Tool calls detected, routing to process_tool_calls")
        return "tools"

    # FASE 6.1: Check if we should try web search fallback
    web_search_decision = should_try_web_search(state, agent)
    if web_search_decision == "web_search":
        return "web_search"

    refinement_decision = should_refine_query(state, agent)
    if refinement_decision == "refine":
        return "refine"

    return "continue"


def should_try_web_search(state, agent) -> str:
    """
    FASE 6.1: Decide if we should try web search as fallback

    Web search is triggered when:
    - Factuality score is POOR (<0.35)
    - Verification failed (support_ratio = 0)
    - Response is a fallback/refusal message
    - We haven't already tried web search this session
    - Web search is available in the agent

    Returns:
        "web_search" or "continue"
    """
    # Check if web search fallback is enabled
    if not getattr(config, 'ENABLE_WEB_SEARCH_FALLBACK', True):
        return "continue"

    # Check if we've already tried web search
    web_search_attempted = getattr(state, 'web_search_attempted', False)
    if web_search_attempted:
        logger.debug("Web search already attempted this session")
        return "continue"

    # Helper function to check agent has web search
    def _has_web_search():
        return (
            hasattr(agent, 'hierarchical_retriever') and
            agent.hierarchical_retriever and
            getattr(agent.hierarchical_retriever, 'web_search', None) is not None
        )

    # TRIGGER 1: Factuality was POOR
    factuality_result = getattr(state, 'factuality_score', None)
    if factuality_result:
        factuality_score = factuality_result.get('factuality_score', 1.0) if isinstance(factuality_result, dict) else 0.5
        quality_level = factuality_result.get('quality_level', 'UNKNOWN') if isinstance(factuality_result, dict) else 'UNKNOWN'

        if quality_level == 'POOR' or factuality_score < 0.35:
            if _has_web_search():
                logger.info(
                    f"FASE 6.1: Triggering web search - "
                    f"factuality={factuality_score:.2f} ({quality_level})"
                )
                return "web_search"

    # TRIGGER 2: Verification completely failed (0 claims supported)
    verification_passed = getattr(state, 'verification_passed', True)
    support_ratio = getattr(state, 'support_ratio', 1.0)

    if not verification_passed and support_ratio == 0.0:
        if _has_web_search():
            logger.info(
                f"FASE 6.1: Triggering web search - "
                f"verification failed (support_ratio=0)"
            )
            return "web_search"

    # TRIGGER 3: Response is a fallback/refusal message (context quality insufficient)
    response = getattr(state, 'agent_response', '') or ''
    fallback_patterns = [
        "i need better matching sources",
        "can you clarify your question",
        "i don't have documents",
        "não tenho documentos",
        "insufficient context",
        "average confidence",
        "isn't very relevant",
    ]

    response_lower = response.lower()
    is_fallback_response = any(pattern in response_lower for pattern in fallback_patterns)

    if is_fallback_response:
        if _has_web_search():
            logger.info(
                f"FASE 6.1: Triggering web search - "
                f"fallback response detected"
            )
            return "web_search"
        else:
            logger.warning(
                f"FASE 6.1: Would trigger web search but not available - "
                f"fallback response detected"
            )

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
