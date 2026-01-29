"""
Context Management Utilities
Token counting and context window management
"""

import logging
from typing import Any, Dict, List, Tuple

import tiktoken

from config import (
    CONTEXT_WARNING_THRESHOLD,
    LLM_MODEL,
    MAX_CONTEXT_TOKENS,
    TOKEN_ALLOCATION,
)

logger = logging.getLogger(__name__)

# Cache for tiktoken encodings (avoid recreating on every call)
_encoding_cache: Dict[str, Any] = {}


def _get_encoding(model: str):
    """Get cached tiktoken encoding for model"""
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache[model]


def calculate_tokens(text: str, model: str = LLM_MODEL) -> int:
    """
    Calculate token count for given text using tiktoken

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer

    Returns:
        Number of tokens
    """
    try:
        # Use cached encoding
        encoding = _get_encoding(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token counting failed, using approximation: {e}")
        # Fallback approximation: ~4 characters per token
        return len(text) // 4


def calculate_messages_tokens(
    messages: List[Dict[str, str]], model: str = LLM_MODEL
) -> int:
    """
    Calculate total tokens for a list of messages

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name for tokenizer

    Returns:
        Total token count including message formatting overhead
    """
    try:
        encoding = _get_encoding(model)
        tokens_per_message = (
            3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = 1  # If there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    except Exception as e:
        logger.warning(f"Message token counting failed, using approximation: {e}")
        # Fallback: sum of individual message tokens
        return sum(calculate_tokens(msg.get("content", ""), model) for msg in messages)


def check_context_overflow(
    system_prompt: str,
    core_memory: str,
    function_definitions: str,
    retrieved_context: str,
    conversation_messages: List[Dict[str, str]],
) -> Tuple[bool, int, Dict[str, int]]:
    """
    Check if context is approaching or exceeding token limit

    Args:
        system_prompt: System prompt text
        core_memory: Core memory text (human + agent persona)
        function_definitions: Function/tool definitions
        retrieved_context: Retrieved archival/recall context
        conversation_messages: List of conversation messages

    Returns:
        Tuple of (is_overflow, total_tokens, token_breakdown)
    """
    # Calculate tokens for each component
    token_breakdown = {
        "system_prompt": calculate_tokens(system_prompt),
        "core_memory": calculate_tokens(core_memory),
        "function_definitions": calculate_tokens(function_definitions),
        "retrieved_context": calculate_tokens(retrieved_context),
        "conversation": calculate_messages_tokens(conversation_messages),
    }

    total_tokens = sum(token_breakdown.values())
    threshold_tokens = int(MAX_CONTEXT_TOKENS * CONTEXT_WARNING_THRESHOLD)

    is_overflow = total_tokens >= threshold_tokens

    if is_overflow:
        logger.warning(
            f"Context approaching limit: {total_tokens}/{MAX_CONTEXT_TOKENS} tokens "
            f"(threshold: {threshold_tokens})"
        )

    return is_overflow, total_tokens, token_breakdown


def format_core_memory(
    human_persona: str, agent_persona: str, facts: List[str] = []
) -> str:
    """
    Format core memory for inclusion in context

    Args:
        human_persona: Human description
        agent_persona: Agent description
        facts: Optional list of important facts

    Returns:
        Formatted core memory string
    """
    # Fix: Handle None default properly
    if facts is None:
        facts = []

    memory_parts = [
        "### Core Memory",
        "",
        "**Human:**",
        human_persona,
        "",
        "**Agent:**",
        agent_persona,
    ]

    if facts:
        memory_parts.extend(["", "**Important Facts:**"])
        for i, fact in enumerate(facts, 1):
            memory_parts.append(f"{i}. {fact}")

    return "\n".join(memory_parts)


def truncate_conversation(
    messages: List[Dict[str, str]], max_tokens: int, keep_recent: int = 5
) -> List[Dict[str, str]]:
    """
    Truncate conversation to fit within token limit while keeping recent messages

    Args:
        messages: List of conversation messages
        max_tokens: Maximum tokens allowed for conversation
        keep_recent: Number of recent messages to always keep

    Returns:
        Truncated list of messages
    """
    if len(messages) <= keep_recent:
        return messages

    # Always keep the most recent messages
    recent_messages = messages[-keep_recent:]
    older_messages = messages[:-keep_recent]

    # Calculate tokens for recent messages
    recent_tokens = calculate_messages_tokens(recent_messages)

    if recent_tokens >= max_tokens:
        logger.warning("Recent messages exceed token limit")
        return recent_messages

    # Add older messages until token limit is reached
    remaining_tokens = max_tokens - recent_tokens
    selected_older = []

    for msg in reversed(older_messages):
        msg_tokens = calculate_messages_tokens([msg])
        if remaining_tokens - msg_tokens >= 0:
            selected_older.insert(0, msg)
            remaining_tokens -= msg_tokens
        else:
            break

    result = selected_older + recent_messages
    logger.info(
        f"Truncated conversation from {len(messages)} to {len(result)} messages"
    )

    return result


def should_trigger_paging(total_tokens: int) -> bool:
    """
    Determine if memory paging should be triggered

    Args:
        total_tokens: Current total token count

    Returns:
        True if paging should be triggered
    """
    threshold = int(MAX_CONTEXT_TOKENS * CONTEXT_WARNING_THRESHOLD)
    return total_tokens >= threshold


def get_available_token_space(
    current_tokens: int, component: str = "conversation"
) -> int:
    """
    Calculate available token space for a specific component

    Args:
        current_tokens: Current total tokens used
        component: Component name from TOKEN_ALLOCATION

    Returns:
        Available tokens for the component
    """
    if component not in TOKEN_ALLOCATION:
        logger.warning(f"Unknown component: {component}")
        return 0

    allocated = TOKEN_ALLOCATION[component]
    remaining_total = MAX_CONTEXT_TOKENS - current_tokens

    return min(allocated, remaining_total)
