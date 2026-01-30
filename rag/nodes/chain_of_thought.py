"""
Chain of Thought Node
Executes step-by-step reasoning for complex queries
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import MemGPTState
from prompts.chain_of_thought import COT_REASONING_TEMPLATE, COT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def chain_of_thought_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Apply Chain-of-Thought reasoning for complex multi-hop queries (Paper-compliant)

    Breaks down complex questions into reasoning steps, processes each step,
    and synthesizes final answer. Essential for comparison and aggregation queries.
    """
    query = state.user_input
    rag_context = state.rag_context

    logger.info("Applying Chain-of-Thought reasoning")

    # Format the prompt template
    cot_prompt = COT_REASONING_TEMPLATE.format(query=query, rag_context=rag_context)

    try:
        # Generate CoT reasoning
        messages = [
            SystemMessage(content=COT_SYSTEM_PROMPT),
            HumanMessage(content=cot_prompt),
        ]

        response = agent.llm.invoke(messages)
        cot_reasoning = response.content if hasattr(response, "content") else str(response)

        # Parse reasoning steps
        reasoning_steps = []
        for line in cot_reasoning.split("\n"):
            if line.strip().startswith("Step") or line.strip().startswith("Conclusion"):
                reasoning_steps.append(line.strip())

        logger.info(f"CoT completed: {len(reasoning_steps)} reasoning steps generated")

        return {
            "cot_reasoning": cot_reasoning,
            "reasoning_steps": reasoning_steps,
            "needs_multi_hop": True,
        }

    except Exception as e:
        logger.error(f"CoT reasoning failed: {e}")
        return {"cot_reasoning": "", "reasoning_steps": [], "needs_multi_hop": False}
