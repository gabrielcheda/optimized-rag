"""
Receive Input Node
Validates and processes initial user input
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def receive_input_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Receive and validate input"""
    logger.info(f"Received input: {state.user_input[:50]}...")
    
    return {
        "iteration_count": state.iteration_count + 1
    }
