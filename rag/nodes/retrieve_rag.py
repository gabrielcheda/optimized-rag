"""
Retrieve RAG Node
Retrieves from RAG sources (documents, knowledge graph)
"""

import logging
from typing import Any, Dict

import config
from agent.state import MemGPTState
from rag import QueryIntent

logger = logging.getLogger(__name__)


def retrieve_rag_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Retrieve from RAG sources (intent-aware)"""
    # OPTIMIZATION: Check if document retrieval is needed
    needs_docs = getattr(state, "needs_document_retrieval", True)

    if not needs_docs:
        logger.info("OPTIMIZATION: Skipping document retrieval (recall sufficient)")

        # FIX: For clarification, use ALL recall messages as context
        # (User messages contain the questions we need to find)
        recall_as_context = []
        if state.retrieved_recall:
            # Use all messages, not just last 3
            for msg in state.retrieved_recall:
                content = msg.get("content", "")
                if len(content.strip()) > 5:  # Skip very short messages
                    recall_as_context.append(
                        {
                            "content": f"[{msg.get('role', 'unknown')}]: {content}",
                            "score": 0.95,  # High confidence from conversation
                            "source": "recall_memory",
                            "metadata": {
                                "timestamp": msg.get("timestamp"),
                                "role": msg.get("role"),
                            },
                        }
                    )

        logger.info(
            f"Using {len(recall_as_context)} recall messages as context (bypassing retrieval)"
        )

        return {
            "retrieved_documents": recall_as_context,
            "translated_query": None,
            "final_context": recall_as_context,  # Set directly
            "rag_context": "\n\n".join([r["content"] for r in recall_as_context]),
            "quality_eval": {
                "is_relevant": True,
                "confidence": 0.9,
                "should_reretrieve": False,
            },
        }

    # Use rewritten query for retrieval
    query = state.rewritten_query if state.rewritten_query else state.user_input
    agent_id = state.agent_id
    intent_enum = state.query_intent  # Already an enum

    retrieval_config = agent.intent_recognizer.get_retrieval_strategy(intent_enum)

    # DW-GRPO: Use hierarchical retrieval if enabled
    if agent.hierarchical_retriever and config.ENABLE_HIERARCHICAL_RETRIEVAL:
        hierarchical_response = agent.hierarchical_retriever.retrieve(
            query=query,
            agent_id=agent_id,
            intent=intent_enum or QueryIntent.QUESTION_ANSWERING,
            top_k=retrieval_config["top_k"],
        )
        rag_results = hierarchical_response["results"]

        # Log cost metrics
        if config.ENABLE_COST_TRACKING:
            cost = hierarchical_response["cost_metrics"]
            logger.info(
                f"DW-GRPO metrics: tier={hierarchical_response['tier_name']}, "
                f"confidence={hierarchical_response['confidence']:.3f}, "
                f"sources={cost['total_sources_queried']}, "
                f"time={hierarchical_response['response_time']:.3f}s"
            )
    else:
        # Fallback to traditional hybrid retrieval
        rag_results = agent.hybrid_retriever.retrieve(
            query=query,
            sources=["documents", "conversation"],
            top_k=retrieval_config["top_k"],
        )

    # Paper-compliant: Knowledge Graph Retrieval (multi-hop)
    kg_results = []
    if agent.kg_retriever and config.ENABLE_KNOWLEDGE_GRAPH:
        try:
            # Clean query for KG (remove refinement metadata)
            clean_kg_query = (
                query.replace("Refined Query:", "").replace('"', "").strip()
            )
            kg_entities = agent.kg_retriever.query_knowledge_graph(
                agent_id=state.agent_id, query=clean_kg_query
            )
            # Convert KG results to standard format
            for kg_item in kg_entities[:5]:
                kg_results.append(
                    {
                        "content": f"{kg_item['path']} (confidence: {kg_item['confidence']:.2f})",
                        "score": kg_item["confidence"],
                        "source": "knowledge_graph",
                        "metadata": kg_item,
                    }
                )
            logger.info(f"KG retrieved {len(kg_results)} related entities")
        except Exception as e:
            logger.warning(f"KG retrieval failed: {e}")

    logger.info(
        f"RAG retrieved: {len(rag_results)} docs + {len(kg_results)} KG entities "
        f"(intent: {intent_enum.value if intent_enum else 'unknown'}, top_k: {retrieval_config['top_k']})"
    )

    # CRITICAL: Return translated_query so it persists in state for reranking
    return {
        "retrieved_documents": rag_results + kg_results,
        "translated_query": query,  # Persist English translation for CrossEncoder
    }
