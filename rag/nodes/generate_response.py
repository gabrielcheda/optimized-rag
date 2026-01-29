"""
Generate Response Node
Generates final AI response using enriched context
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

import config
from agent.state import MemGPTState
from prompts.generate_response import CLARIFICATION_INSTRUCTION, FEW_SHOT_EXAMPLES, SYSTEM_PROMPT_TEMPLATE
from rag.nodes.helpers import check_context_quality, enrich_context_with_memory

logger = logging.getLogger(__name__)


def generate_response_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Generate AI response (Paper-compliant: few-shot prompting)"""
    # 🔥 ENRICHED CONTEXT: Combines core memory + conversation + retrieved docs
    enriched_context, source_map = enrich_context_with_memory(state, agent)

    # Check if this is a clarification question (uses recall memory instead of docs)
    from rag import QueryIntent

    is_clarification = state.query_intent == QueryIntent.CLARIFICATION
    has_recall_memory = state.retrieved_recall and len(state.retrieved_recall) > 0

    # BYPASS: For clarification with recall memory, skip document quality check
    if is_clarification and has_recall_memory:
        logger.info(
            f"Clarification intent with {len(state.retrieved_recall)} recall messages - using recall memory"
        )
        context_quality = {
            "sufficient": True,
            "reason": "Using recall memory for clarification",
        }
    else:
        # Check context quality before generating response (CRITICAL: prevents hallucination)
        context_quality = check_context_quality(state.final_context, min_score=0.3)

    if not context_quality["sufficient"]:
        logger.warning(
            f"Insufficient context quality: {context_quality['reason']}, "
            f"max_score={context_quality.get('max_score', 0):.2f}"
        )

        # Return early with honest fallback message
        fallback_message = context_quality["message"]

        return {
            "agent_response": fallback_message,
            "messages": [{"role": "assistant", "content": fallback_message}],
            "faithfulness_score": {
                "score": 0.0,
                "reasoning": "Insufficient context - honest fallback",
            },
            "context_quality": context_quality,
            "source_map": {},
        }

    

    SYSTEM_PROMPT_TEMPLATE.format(few_shot_examples=FEW_SHOT_EXAMPLES, enriched_context=enriched_context)

    
    classification_prompt=""
    if is_clarification:
        classification_prompt=CLARIFICATION_INSTRUCTION

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE + classification_prompt),
        HumanMessage(content=state.user_input),
    ]

    # Generate response
    response = agent.llm_with_tools.invoke(messages)

    # Extract tool calls if present
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = response.tool_calls
        logger.info(f"LLM requested {len(tool_calls)} tool calls")

    answer = response.content if hasattr(response, "content") else str(response)
    # Ensure answer is a string
    if not isinstance(answer, str):
        answer = str(answer)

    # Paper-compliant: Faithfulness Evaluation
    faithfulness = {}
    if agent.evaluator and state.final_context:
        faithfulness = agent.evaluator.faithfulness_score(
            answer=answer, context=state.final_context, llm=agent.llm
        )
        logger.info(
            f"Faithfulness score: {faithfulness.get('score', 0):.2f} - "
            f"{faithfulness.get('reasoning', 'N/A')[:100]}"
        )

    # OPTIMIZATION: Factuality Scorer - Calculate comprehensive quality score
    factuality_result = {}
    auto_refused = False

    # FIX: Skip factuality check for clarification intent (uses recall memory, not documents)
    # Factuality scorer expects documents in final_context, but clarifications use recall memory
    if is_clarification and has_recall_memory:
        logger.info("Skipping factuality check for clarification (recall-based answer)")
        factuality_result = {
            "factuality_score": faithfulness.get("score", 0.7),
            "quality_level": "GOOD",
            "passed": True,
            "method": "clarification_bypass",
            "recommendation": answer,
        }
    elif agent.factuality_scorer:
        try:
            # QUICK WIN: Early exit for high-confidence answers
            faithfulness_score = faithfulness.get("score", 0)
            if faithfulness_score > 0.85 and state.rerank_scores:
                top_rerank_score = (
                    max(state.rerank_scores.values()) if state.rerank_scores else 0
                )
                if top_rerank_score > 0.9:
                    logger.info(
                        f"High-confidence answer detected (faithfulness={faithfulness_score:.2f}, "
                        f"rerank={top_rerank_score:.2f}). Skipping detailed factuality check (early exit)."
                    )
                    factuality_result = {
                        "factuality_score": faithfulness_score,
                        "quality_level": "EXCELLENT",
                        "passed": True,
                        "method": "early_exit",
                        "components": {
                            "support_ratio": 1.0,
                            "citation_coverage": 0.8,
                            "retrieval_confidence": top_rerank_score,
                        },
                        "recommendation": answer,
                    }
                else:
                    factuality_result = (
                        agent.factuality_scorer.calculate_factuality_score(
                            query=state.user_input,
                            answer=answer,
                            retrieved_docs=state.final_context,
                            source_map=source_map,
                        )
                    )
            else:
                factuality_result = agent.factuality_scorer.calculate_factuality_score(
                    query=state.user_input,
                    answer=answer,
                    retrieved_docs=state.final_context,
                    source_map=source_map,
                )

            # Auto-refuse low-quality answers (threshold: 0.25)
            # BUT: Trust faithfulness score as fallback when it's high
            faithfulness_override = faithfulness_score >= 0.7

            if (
                not factuality_result.get("passed", False)
                and agent.factuality_scorer.should_refuse_answer(
                    factuality_result["factuality_score"], threshold=0.25
                )
                and not faithfulness_override
            ):  # Don't refuse if faithfulness is high
                auto_refused = True
                fallback_message = factuality_result["recommendation"]

                logger.warning(
                    f"Auto-refusing answer due to low factuality: "
                    f"{factuality_result['factuality_score']:.2f} "
                    f"({factuality_result['quality_level']})"
                )

                return {
                    "agent_response": fallback_message,
                    "messages": [{"role": "assistant", "content": fallback_message}],
                    "faithfulness_score": {
                        "score": 0.0,
                        "reasoning": "Auto-refused due to low factuality",
                    },
                    "factuality_score": factuality_result,
                    "source_map": {},
                    "tool_calls": [],
                    "auto_refused": True,
                }
            elif faithfulness_override and not factuality_result.get("passed", False):
                logger.info(
                    f"Trusting faithfulness score ({faithfulness_score:.2f}) over factuality "
                    f"({factuality_result['factuality_score']:.2f}) - allowing answer through"
                )

            logger.info(
                f"Factuality score: {factuality_result['factuality_score']:.3f} "
                f"({factuality_result['quality_level']}) - "
                f"support={factuality_result['components']['support_ratio']:.2f}, "
                f"citations={factuality_result['components']['citation_coverage']:.2f}"
            )
        except Exception as e:
            logger.error(f"Factuality scoring failed: {e}")
            factuality_result = {}

    # Track LLM costs if cost tracker enabled
    if agent.cost_tracker and hasattr(response, "usage"):
        try:
            usage = getattr(response, "usage", None)
            if usage:
                agent.cost_tracker.track_llm(
                    model=config.LLM_MODEL,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")

    return {
        "agent_response": answer,
        "messages": [{"role": "assistant", "content": answer}],
        "faithfulness_score": faithfulness,
        "factuality_score": factuality_result,
        "source_map": source_map,
        "tool_calls": tool_calls,
        "auto_refused": auto_refused,
    }
