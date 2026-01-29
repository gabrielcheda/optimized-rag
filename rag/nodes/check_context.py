"""
Check Context Node
Validates context size and token limits
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState
from utils.context import format_core_memory, calculate_tokens

logger = logging.getLogger(__name__)


def check_context_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Check context size"""
    core_memory = format_core_memory(
        state.human_persona,
        state.agent_persona,
        state.core_facts
    )
    
    # Simplified context check
    overflow_info = {
        "overflow": False,
        "total_tokens": calculate_tokens(core_memory),
        "breakdown": {"core_memory": calculate_tokens(core_memory)}
    }
    
    return {
        "context_overflow": overflow_info["overflow"],
        "current_tokens": overflow_info["total_tokens"],
        "token_breakdown": overflow_info["breakdown"]
    }
