"""
Route Query Node
Routes query to appropriate data sources based on intent
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState
from rag.nodes.helpers import should_retrieve_documents

logger = logging.getLogger(__name__)


def route_query_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Route query to appropriate data sources (Paper-compliant: routing)"""
    # Use rewritten query for routing
    query = state.rewritten_query if state.rewritten_query else state.user_input
    intent = state.query_intent
    
    # OPTIMIZATION: Smart routing decision - check if recall is sufficient before hitting documents
    # This saves embeddings + retrieval costs when answer is already in recent conversation
    needs_document_retrieval = should_retrieve_documents(
        query=query,
        intent=intent,
        recalled_messages=state.retrieved_recall or []
    )
    
    # Route query with intent awareness
    sources = agent.query_router.route(query)
    sources_str = str(sources) if not isinstance(sources, str) else sources
    
    logger.info(
        f"Query routed to: {sources_str}, intent: {intent}, "
        f"document_retrieval_needed: {needs_document_retrieval}"
    )
    
    return {
        "needs_memory_retrieval": True,
        "needs_document_retrieval": needs_document_retrieval
    }
