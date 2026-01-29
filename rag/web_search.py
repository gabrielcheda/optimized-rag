"""
Web Search Tool
Integrates external search APIs (Tavily, DuckDuckGo)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """External web search"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        """
        Initialize web search tool
        
        Args:
            tavily_api_key: Optional Tavily API key
        """
        self.tavily_api_key = tavily_api_key
        self.tavily_client = None
        
        # Try Tavily first
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                logger.info("Tavily search enabled")
            except ImportError:
                logger.warning("tavily-python not installed, using fallback")
            except Exception as e:
                logger.warning(f"Tavily init failed: {e}")
        
        # DuckDuckGo as fallback
        self.ddg_available = False
        try:
            from duckduckgo_search import DDGS
            self.ddg = DDGS()
            self.ddg_available = True
            logger.info("DuckDuckGo search available")
        except ImportError:
            logger.warning("duckduckgo-search not installed")
        except Exception as e:
            logger.warning(f"DuckDuckGo init failed: {e}")
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        include_raw_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search the web
        
        Args:
            query: Search query
            max_results: Maximum number of results
            include_raw_content: Whether to include full content
        
        Returns:
            List of search results
        """
        # Try Tavily first
        if self.tavily_client:
            try:
                return self._search_tavily(query, max_results, include_raw_content)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}, trying fallback")
        
        # Fallback to DuckDuckGo
        if self.ddg_available:
            try:
                return self._search_ddg(query, max_results)
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")
        
        logger.error("No search backend available")
        return []
    
    def search_with_context(
        self,
        query: str,
        context: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search with additional context
        
        Args:
            query: Search query
            context: Additional context to refine search
            max_results: Maximum results
        
        Returns:
            Search results
        """
        # Enhance query with context
        enhanced_query = f"{query} {context[:100]}"
        
        return self.search(enhanced_query, max_results, include_raw_content=True)
    
    def _search_tavily(
        self,
        query: str,
        max_results: int,
        include_raw_content: bool
    ) -> List[Dict[str, Any]]:
        """Search using Tavily"""
        if self.tavily_client is None:
            raise RuntimeError("Tavily client not initialized")
        
        response = self.tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content
        )
        
        results = []
        for result in response.get('results', []):
            results.append({
                "title": result.get('title', ''),
                "url": result.get('url', ''),
                "content": result.get('content', ''),
                "raw_content": result.get('raw_content', '') if include_raw_content else '',
                "score": result.get('score', 0.0),
                "source": "tavily"
            })
        
        logger.info(f"Tavily returned {len(results)} results")
        return results
    
    def _search_ddg(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        search_results = self.ddg.text(query, max_results=max_results)
        
        results = []
        for result in search_results:
            results.append({
                "title": result.get('title', ''),
                "url": result.get('href', ''),
                "content": result.get('body', ''),
                "raw_content": '',
                "score": 0.0,
                "source": "duckduckgo"
            })
        
        logger.info(f"DuckDuckGo returned {len(results)} results")
        return results
