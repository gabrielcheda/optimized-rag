"""
Retrieve Memory Node
Retrieves from archival and recall memory systems
"""

import logging
from typing import Any, Dict

import config
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def retrieve_memory_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Retrieve from memory systems (archival + recall)

    CRITICAL: This is the primary memory retrieval node.
    Without this, the agent has no access to:
    - Long-term knowledge (archival)
    - Conversation history (recall)
    - Personalization data
    """
    query = state.rewritten_query if state.rewritten_query else state.user_input
    conversation_id = state.conversation_id

    archival_results = []
    recall_results = []

    # 1. ARCHIVAL MEMORY: Semantic search for relevant long-term knowledge
    try:
        if state.needs_memory_retrieval:
            archival_results = agent.memory_manager.archival_memory_search(
                query=query, top_k=config.ARCHIVAL_SEARCH_RESULTS
            )

            # Format for compatibility with retrieval pipeline
            for result in archival_results:
                result["source"] = "archival_memory"
                result["score"] = result.get("similarity", 0.0)

            logger.info(
                f"Retrieved {len(archival_results)} results from archival memory"
            )
    except Exception as e:
        logger.error(f"Archival memory retrieval failed: {e}")
        archival_results = []

    # 2. RECALL MEMORY: Get recent conversation context
    try:
        recall_results_raw = agent.memory_manager.get_recent_messages(
            conversation_id=conversation_id, limit=config.RECALL_SEARCH_RESULTS
        )

        # FIX: Also do semantic search in conversation history for clarification queries
        # This finds relevant messages even if they're not recent
        if state.query_intent and state.query_intent.value.lower() == "clarification":
            try:
                semantic_results = agent.memory_manager.conversation_search(
                    conversation_id=conversation_id, query=query, limit=10
                )

                # Merge results, avoiding duplicates
                seen_contents = {msg["content"] for msg in recall_results_raw}
                for msg in semantic_results:
                    if msg.get("content") not in seen_contents:
                        recall_results_raw.append(msg)
                        seen_contents.add(msg.get("content"))

                logger.info(
                    f"Semantic search added {len(semantic_results)} relevant messages"
                )
            except Exception as e:
                logger.warning(f"Semantic recall search failed: {e}")

        # Format for state
        for msg in recall_results_raw:
            recall_results.append(
                {
                    "content": msg["content"],
                    "role": msg["role"],
                    "timestamp": msg.get("created_at", ""),
                    "source": "recall_memory",
                }
            )

        logger.info(f"Retrieved {len(recall_results)} messages from recall memory")
    except Exception as e:
        logger.error(f"Recall memory retrieval failed: {e}")
        recall_results = []

    logger.info(
        f"Memory retrieved: {len(archival_results)} archival, "
        f"{len(recall_results)} recall"
    )

    return {"retrieved_archival": archival_results, "retrieved_recall": recall_results}
