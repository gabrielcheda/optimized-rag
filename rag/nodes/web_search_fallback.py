"""
FASE 6.1: Web Search Fallback Node
Triggers web search when document-based retrieval fails to produce
high-quality answers (low factuality scores).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def web_search_fallback_node(state, agent) -> Dict[str, Any]:
    """
    FASE 6.1: Web search fallback when factuality is POOR

    This node is triggered when:
    - Document retrieval succeeded but verification failed
    - Factuality score was POOR (<0.35)
    - Web search hasn't been attempted yet

    It uses the hierarchical retriever's TIER 3 capability to
    perform agentic web search and augment the context.

    Args:
        state: Current MemGPT state
        agent: RAG agent instance

    Returns:
        Updated state with web search results added to context
    """
    query = state.user_input
    logger.info(f"FASE 6.1: Web search fallback triggered for: '{query[:50]}...'")

    # Mark that we've attempted web search
    web_search_attempted = True

    # Get existing context
    existing_context = list(state.final_context) if state.final_context else []

    try:
        # Check if hierarchical retriever is available
        if not hasattr(agent, 'hierarchical_retriever') or not agent.hierarchical_retriever:
            logger.warning("Web search fallback: No hierarchical retriever available")
            return {
                "web_search_attempted": web_search_attempted,
                "web_search_success": False,
                "web_search_reason": "No hierarchical retriever available"
            }

        # Check if web search is available
        web_search = getattr(agent.hierarchical_retriever, 'web_search', None)
        if not web_search:
            logger.warning("Web search fallback: Web search not configured")
            return {
                "web_search_attempted": web_search_attempted,
                "web_search_success": False,
                "web_search_reason": "Web search not configured"
            }

        # Trigger TIER 3 agentic retrieval
        logger.info("FASE 6.1: Triggering hierarchical retriever TIER 3 (web search)")

        tier_3_results = agent.hierarchical_retriever.trigger_tier_3(
            query=query,
            existing_context=existing_context
        )

        if tier_3_results:
            # Filter web search results
            web_results = [
                r for r in tier_3_results
                if r.get('source', '').startswith('web_search')
            ]

            if web_results:
                logger.info(
                    f"FASE 6.1: Web search returned {len(web_results)} results"
                )

                # Merge with existing context (web results first for priority)
                merged_context = web_results + existing_context

                return {
                    "final_context": merged_context,
                    "web_search_attempted": web_search_attempted,
                    "web_search_success": True,
                    "web_search_results_count": len(web_results),
                    "web_search_reason": "Web search augmented context successfully"
                }
            else:
                logger.info("FASE 6.1: TIER 3 executed but no web search was needed (LLM decided)")
                return {
                    "web_search_attempted": web_search_attempted,
                    "web_search_success": False,
                    "web_search_reason": "LLM decided web search not needed"
                }
        else:
            logger.warning("FASE 6.1: TIER 3 returned no results")
            return {
                "web_search_attempted": web_search_attempted,
                "web_search_success": False,
                "web_search_reason": "TIER 3 returned no results"
            }

    except Exception as e:
        logger.error(f"FASE 6.1: Web search fallback error: {e}", exc_info=True)

        # Try direct web search as last resort
        try:
            if hasattr(agent.hierarchical_retriever, 'web_search'):
                web_search = agent.hierarchical_retriever.web_search
                if web_search:
                    logger.info("FASE 6.1: Attempting direct web search fallback")
                    direct_results = web_search.search(query, max_results=3)

                    if direct_results:
                        # Format results
                        formatted_results = []
                        for item in direct_results:
                            formatted_results.append({
                                'content': f"[Web Search - Direct] {item.get('content', item.get('snippet', ''))}",
                                'score': 0.7,
                                'source': 'web_search_direct_fallback',
                                'metadata': {
                                    'url': item.get('url', ''),
                                    'title': item.get('title', ''),
                                    'fallback': True
                                }
                            })

                        merged_context = formatted_results + existing_context

                        logger.info(
                            f"FASE 6.1: Direct web search returned {len(formatted_results)} results"
                        )

                        return {
                            "final_context": merged_context,
                            "web_search_attempted": web_search_attempted,
                            "web_search_success": True,
                            "web_search_results_count": len(formatted_results),
                            "web_search_reason": "Direct web search fallback succeeded"
                        }
        except Exception as e2:
            logger.error(f"FASE 6.1: Direct web search also failed: {e2}")

        return {
            "web_search_attempted": web_search_attempted,
            "web_search_success": False,
            "web_search_reason": f"Error: {str(e)}"
        }
