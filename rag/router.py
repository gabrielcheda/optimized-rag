"""
Query Router
Routes queries to appropriate sources and decomposes complex queries
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""
    ARCHIVAL_MEMORY = "archival_memory"
    DOCUMENTS = "documents"
    WEB_SEARCH = "web_search"
    CONVERSATION_HISTORY = "conversation_history"
    HYBRID = "hybrid"


class QueryRouter:
    """Routes queries and decomposes complex questions"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def route(self, query: str, available_sources: Optional[List[DataSource]] = None) -> Dict[str, Any]:
        """Route query to best data source(s)"""
        if available_sources is None:
            available_sources = list(DataSource)
        
        # DETERMINISTIC ROUTING: Always try DOCUMENTS first
        # This ensures we search local knowledge base before external sources
        # The hierarchical retriever will escalate to web search if needed
        
        sources = [DataSource.DOCUMENTS]
        reasoning = "Always search documents first (local knowledge base priority)"
        confidence = 1.0
        
        # Add ARCHIVAL_MEMORY for personalization queries
        if any(word in query.lower() for word in ["me", "my", "i ", "remember", "you told"]):
            sources.append(DataSource.ARCHIVAL_MEMORY)
            reasoning = "Documents + archival memory (personalization detected)"
        
        # Add CONVERSATION_HISTORY for follow-up questions
        if any(word in query.lower() for word in ["that", "it", "this", "previous", "earlier"]):
            sources.append(DataSource.CONVERSATION_HISTORY)
            reasoning = "Documents + conversation history (follow-up detected)"
        
        decision = {
            "sources": sources,
            "reasoning": reasoning,
            "confidence": confidence
        }
        
        logger.info(f"Routed to: {[s.value for s in decision['sources']]}")
        
        return decision
    
    def _build_routing_prompt(self, query: str, available_sources: List[DataSource]) -> str:
        source_descriptions = {
            DataSource.ARCHIVAL_MEMORY: "Long-term memory from past interactions",
            DataSource.DOCUMENTS: "Recently uploaded documents (PDFs, texts)",
            DataSource.WEB_SEARCH: "Real-time web search for current info",
            DataSource.CONVERSATION_HISTORY: "Recent conversation context",
            DataSource.HYBRID: "Multiple sources combined"
        }
        
        sources_list = "\n".join([
            f"- {source.value}: {source_descriptions[source]}"
            for source in available_sources
        ])
        
        return f"""Route the query to the best data source.

Available sources:
{sources_list}

Guidelines:
- DOCUMENTS: PRIORITY - always try first for uploaded files, knowledge base content
- ARCHIVAL_MEMORY: personal info, learned facts, past conversations
- CONVERSATION_HISTORY: context from current conversation
- HYBRID: when multiple sources beneficial
- WEB_SEARCH: LAST RESORT - only for current events, real-time data not in documents

IMPORTANT: Always prefer DOCUMENTS over WEB_SEARCH unless query explicitly asks for current/recent information.
Respond:
SOURCE: [source name]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]"""
    
    def _parse_routing_decision(self, response: str, available_sources: List[DataSource]) -> Dict[str, Any]:
        lines = response.strip().split('\n')
        
        source_str = None
        confidence = 0.7
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("SOURCE:"):
                source_str = line.replace("SOURCE:", "").strip().lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except:
                    confidence = 0.7
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        sources = []
        
        if source_str:
            if "hybrid" in source_str or "multiple" in source_str:
                sources = [DataSource.DOCUMENTS, DataSource.ARCHIVAL_MEMORY]
            else:
                for ds in available_sources:
                    if ds.value in source_str or ds.name.lower() in source_str:
                        sources.append(ds)
        
        if not sources:
            sources = [DataSource.DOCUMENTS]
        
        return {
            "sources": sources,
            "confidence": confidence,
            "reasoning": reasoning or "Auto routing"
        }
    
