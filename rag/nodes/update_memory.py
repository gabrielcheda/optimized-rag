"""
Update Memory Node
Updates memory systems after conversation
"""

import logging
from typing import Any, Dict

from agent.state import MemGPTState
from utils.context import calculate_tokens

logger = logging.getLogger(__name__)


def update_memory_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Update memory systems after conversation

    Stores:
    1. Conversation messages (recall memory)
    2. Important facts (archival memory, if flagged)
    3. Logs operation for tracking
    """
    agent_id = state.agent_id
    conversation_id = state.conversation_id

    # 1. Store conversation in recall memory (via both db_ops and memory_manager)
    try:
        # Store user message
        agent.memory_manager.save_message(
            conversation_id=conversation_id,
            role="user",
            content=state.user_input,
            tokens_used=calculate_tokens(state.user_input),
        )

        # Store assistant response
        if state.agent_response:
            agent.memory_manager.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=state.agent_response,
                tokens_used=calculate_tokens(state.agent_response),
            )

        logger.info(
            f"Saved conversation to recall memory (conversation_id={conversation_id})"
        )
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

    # 2. Store important facts to archival memory (if flagged by state)
    if state.should_save_to_archival and state.pending_archival_inserts:
        try:
            for content in state.pending_archival_inserts:
                agent.memory_manager.archival_memory_insert(
                    content=content,
                    metadata={
                        "conversation_id": conversation_id,
                        "timestamp": "auto",
                        "source": "conversation_extraction",
                    },
                )
            logger.info(
                f"Saved {len(state.pending_archival_inserts)} items to archival memory"
            )
        except Exception as e:
            logger.error(f"Failed to save to archival memory: {e}")

    # 3. FIX: Automatically extract and store important facts to core memory
    # This ensures the agent remembers important personal information
    try:
        _extract_and_store_core_facts(state, agent)
    except Exception as e:
        logger.warning(f"Core memory extraction failed (non-critical): {e}")

    # 4. Log memory operations (for debugging and metrics)
    if state.memory_operations_log:
        for operation in state.memory_operations_log:
            logger.debug(f"Memory operation: {operation}")

    logger.info("Memory update completed")

    return {}


def _extract_and_store_core_facts(state: MemGPTState, agent) -> None:
    """
    Extract important personal facts from conversation and store in core memory

    Uses LLM to identify:
    - User's name, preferences, profession
    - Important dates, locations, relationships
    - User's stated goals or interests
    """
    # Only extract if we have substantial conversation
    if not state.user_input or len(state.user_input.split()) < 5:
        return

    # Skip for simple queries (greetings, clarifications, simple Q&A)
    skip_intents = ["chitchat", "greeting", "clarification"]
    if state.query_intent and state.query_intent.value.lower() in skip_intents:
        return

    # Use LLM to extract personal facts
    try:
        response = agent.llm.invoke(
            [
                {
                    "role": "system",
                    "content": """You are a fact extractor. Analyze the user message and extract ONLY personal facts about the user.
            
Rules:
- Return ONLY new facts about the USER (not general knowledge questions)
- Facts should be concise, one-line statements
- If there are NO personal facts, return EMPTY (no text)
- Examples of facts: "User's name is Gabriel", "User works in AI", "User prefers concise answers"

Format: One fact per line, or empty if no facts.""",
                },
                {"role": "user", "content": f"User message: {state.user_input}"},
            ]
        )

        facts_text = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )

        # Parse and store facts
        if facts_text and facts_text.lower() not in ["empty", "none", ""]:
            facts = [f.strip() for f in facts_text.split("\n") if f.strip()]

            for fact in facts[:3]:  # Max 3 facts per message
                if len(fact) > 10:  # Minimum length
                    success = agent.memory_manager.add_core_fact(fact)
                    if success:
                        logger.info(f"Added core fact: {fact[:50]}...")

    except Exception as e:
        logger.debug(f"Fact extraction skipped: {e}")
