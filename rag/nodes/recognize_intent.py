"""
Recognize Intent Node
Identifies the intent of the user's query
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def recognize_intent_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Recognize query intent (Paper-compliant: pre-retrieval)"""
    query = state.user_input
    
    # Get conversation history for context
    conversation_history = state.messages
    
    # Recognize intent
    intent_result = agent.intent_recognizer.recognize(query, conversation_history)
    
    logger.info(
        f"Intent recognized: {intent_result.intent.value} "
        f"(confidence: {intent_result.confidence:.2f})"
    )
    
    return {
        "query_intent": intent_result.intent,  # Keep as enum, not .value!
        "intent_confidence": intent_result.confidence
    }
