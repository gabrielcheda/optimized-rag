"""
Hierarchical Retriever for DW-GRPO

Implements tiered retrieval strategy to minimize cost by progressively escalating
retrieval scope only when necessary. Avoids expensive operations (KG, web search)
for queries that can be answered with local context.
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

import config
from rag.models.intent_analysis import QueryIntent

logger = logging.getLogger(__name__)


class RetrievalTier(Enum):
    """Hierarchical retrieval tiers"""
    TIER_1 = 1  # Core memory only (cheapest)
    TIER_2 = 2  # + Document store
    TIER_3 = 3  # + Knowledge Graph + Web (most expensive)


class ConfidenceEvaluator:
    """Evaluate confidence of retrieval results to determine tier escalation"""
    
    @staticmethod
    def evaluate_confidence(
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent
    ) -> float:
        """
        Evaluate confidence of retrieval results
        
        Args:
            results: Retrieved results
            query: Original query
            intent: Query intent
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if not results:
            return 0.0
        
        # Factor 1: Score distribution (higher scores = more confidence)
        scores = [r.get('score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        # Factor 2: Result count (more results = more confidence)
        count_factor = min(len(results) / 5.0, 1.0)  # Normalize to 5 results
        
        # Factor 3: Score variance (consistent scores = more confidence)
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency_factor = max(0, 1.0 - variance)
        else:
            consistency_factor = 0.5
        
        # Factor 4: Top score quality (high top score = strong match)
        top_score_factor = max_score
        
        # Statistical confidence (original calculation)
        statistical_confidence = (
            0.35 * avg_score +
            0.25 * count_factor +
            0.20 * consistency_factor +
            0.20 * top_score_factor
        )
        
        # OPTIMIZATION: Blend with Self-RAG semantic confidence (60/40 statistical/semantic)
        # Prevents false confidence (e.g., hierarchical=1.0 but Self-RAG=0.1)
        # Extract semantic confidence from results if available
        semantic_confidence = None
        for r in results:
            if 'semantic_confidence' in r:
                semantic_confidence = r.get('semantic_confidence', None)
                break
        
        if semantic_confidence is not None:
            # Blend: configurable statistical/semantic weights
            confidence = (
                config.HIERARCHICAL_CONFIDENCE_BLEND_WEIGHT * statistical_confidence + 
                config.HIERARCHICAL_SEMANTIC_BLEND_WEIGHT * semantic_confidence
            )
            logger.debug(
                f"Blended confidence: {confidence:.3f} "
                f"(statistical={statistical_confidence:.3f}, semantic={semantic_confidence:.3f})"
            )
        else:
            confidence = statistical_confidence
        
        # Intent-specific adjustments
        if intent in ['qa', 'search'] and max_score > config.HIERARCHICAL_BOOST_THRESHOLD:
            # Simple queries with high-scoring match get boosted confidence
            confidence = min(confidence * config.HIERARCHICAL_BOOST_MULTIPLIER, 1.0)
        elif intent == 'multi_hop' and len(results) < 3:
            # Complex queries need more context
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    @staticmethod
    def should_escalate(
        confidence: float,
        threshold: float,
        current_tier: RetrievalTier,
        intent: QueryIntent
    ) -> bool:
        """
        Determine if retrieval should escalate to next tier
        
        Args:
            confidence: Current confidence score
            threshold: Confidence threshold for satisfaction
            current_tier: Current retrieval tier
            intent: Query intent
            
        Returns:
            True if should escalate to next tier
        """
        # Already at max tier
        if current_tier == RetrievalTier.TIER_3:
            return False
        
        # Low confidence triggers escalation
        if confidence < threshold:
            logger.info(
                f"Escalation triggered: confidence={confidence:.3f} < threshold={threshold:.3f}"
            )
            return True
        
        # Intent-specific escalation rules
        if intent == 'multi_hop' and current_tier == RetrievalTier.TIER_1:
            # Multi-hop queries typically need more than core memory
            logger.info("Escalation triggered: multi_hop intent requires broader context")
            return True
        
        if intent == 'recent' and current_tier == RetrievalTier.TIER_1:
            # Recent queries need document store for temporal information
            logger.info("Escalation triggered: recent intent requires temporal context")
            return True
        
        return False


class HierarchicalRetriever:
    """
    Hierarchical retrieval with progressive tier escalation.
    
    Tier 1: Core memory only (instant, no API calls)
    Tier 2: + Document store (moderate cost, embedding search)
    Tier 3: + Knowledge graph + Web search (highest cost)
    
    Escalates only when confidence is insufficient.
    """
    
    def __init__(
        self,
        memory_manager,
        document_store,
        hybrid_retriever,
        llm,
        kg_retriever=None,
        web_search=None,
        confidence_threshold: float = 0.55,
        enable_tier_3: bool = True
    ):
        """
        Args:
            memory_manager: MemoryManager instance
            document_store: DocumentStore instance
            hybrid_retriever: HybridRetriever instance
            llm: Language model for agentic Tier 3 decisions
            kg_retriever: Knowledge graph retriever (optional)
            web_search: Web search instance (optional)
            confidence_threshold: Minimum confidence to avoid escalation (reduced from 0.7 to 0.55)
            enable_tier_3: Whether to enable Tier 3 (expensive operations)
        """
        self.memory_manager = memory_manager
        self.document_store = document_store
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm
        self.kg_retriever = kg_retriever
        self.web_search = web_search
        self.confidence_threshold = confidence_threshold
        self.enable_tier_3 = enable_tier_3
        
        # Create LLM with tools for agentic Tier 3
        from agent.rag_tools import create_rag_tools
        if web_search:
            tools = create_rag_tools(document_store, web_search)
            # Filter only web_search tool
            self.tier_3_tools = [t for t in tools if 'web_search' in t.name.lower()]
            self.llm_with_tools = llm.bind_tools(self.tier_3_tools) if self.tier_3_tools else llm
        else:
            self.llm_with_tools = llm
        
        self.evaluator = ConfidenceEvaluator()
        
        # Statistics
        self.stats = {
            'tier_1_queries': 0,
            'tier_2_queries': 0,
            'tier_3_queries': 0,
            'escalations': 0,
            'avg_confidence_tier_1': 0.0,
            'avg_confidence_tier_2': 0.0,
            'avg_confidence_tier_3': 0.0
        }
        
        logger.info(
            f"HierarchicalRetriever initialized: threshold={confidence_threshold}, "
            f"tier_3_enabled={enable_tier_3}"
        )
    
    def retrieve(
        self,
        query: str,
        agent_id: str,
        intent: QueryIntent,
        top_k: int = 10,
        force_tier: Optional[RetrievalTier] = None
    ) -> Dict[str, Any]:
        """
        Hierarchical retrieval with progressive escalation.
        
        Args:
            query: Query text
            agent_id: Agent identifier
            intent: Query intent
            top_k: Number of results to return
            force_tier: Force specific tier (for testing/override)
            
        Returns:
            Dict with results, confidence, tier_used, and cost_metrics
        """
        start_time = datetime.now()
        
        # Track costs
        cost_metrics = {
            'embedding_calls': 0,
            'llm_calls': 0,
            'kg_queries': 0,
            'web_searches': 0,
            'total_sources_queried': 0
        }
        
        all_results = []
        current_tier = RetrievalTier.TIER_1
        confidence = 0.0
        
        # Tier 1: Core memory only
        if force_tier is None or force_tier == RetrievalTier.TIER_1:
            logger.info(f"TIER 1: Querying core memory for '{query[:50]}...'")
            
            tier_1_results = self._retrieve_tier_1(agent_id, query)
            all_results.extend(tier_1_results)
            cost_metrics['total_sources_queried'] = 1
            
            confidence = self.evaluator.evaluate_confidence(
                all_results, query, intent
            )
            
            self.stats['tier_1_queries'] += 1
            self._update_avg_confidence('tier_1', confidence)
            
            logger.info(
                f"TIER 1 complete: {len(tier_1_results)} results, confidence={confidence:.3f}"
            )
            
            # Check if escalation needed
            if not self.evaluator.should_escalate(
                confidence, self.confidence_threshold, current_tier, intent
            ):
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Query satisfied at TIER 1 (confidence={confidence:.3f} >= {self.confidence_threshold:.3f})"
                )
                return self._format_response(
                    all_results[:top_k], confidence, current_tier,
                    cost_metrics, response_time
                )
        
        # Tier 2: + Document store
        if force_tier is None or force_tier == RetrievalTier.TIER_2:
            current_tier = RetrievalTier.TIER_2
            self.stats['escalations'] += 1
            
            logger.info(f"TIER 2: Escalating to document store")
            
            tier_2_results = self._retrieve_tier_2(agent_id, query, top_k)
            all_results.extend(tier_2_results)
            cost_metrics['embedding_calls'] += 1  # Semantic search
            cost_metrics['total_sources_queried'] = 2
            
            confidence = self.evaluator.evaluate_confidence(
                all_results, query, intent
            )
            
            self.stats['tier_2_queries'] += 1
            self._update_avg_confidence('tier_2', confidence)
            
            logger.info(
                f"TIER 2 complete: {len(tier_2_results)} new results, confidence={confidence:.3f}"
            )
            
            # Check if escalation needed
            if not self.evaluator.should_escalate(
                confidence, self.confidence_threshold, current_tier, intent
            ) or not self.enable_tier_3:
                response_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Query satisfied at TIER 2 (confidence={confidence:.3f})"
                )
                return self._format_response(
                    all_results[:top_k], confidence, current_tier,
                    cost_metrics, response_time
                )
        
        # Tier 3: + Knowledge Graph + Web (if enabled)
        if self.enable_tier_3 and (force_tier is None or force_tier == RetrievalTier.TIER_3):
            current_tier = RetrievalTier.TIER_3
            self.stats['escalations'] += 1
            
            logger.info(f"TIER 3: Escalating to AGENTIC retrieval (LLM with tools)")
            
            # Pass Tier 1+2 context to LLM for informed decision
            tier_3_results = self._retrieve_tier_3(agent_id, query, top_k, all_results)
            all_results.extend(tier_3_results)
            
            # Track costs (LLM always called, but tools only if LLM decides)
            cost_metrics['llm_calls'] += 1  # Agentic decision
            
            # Check if web search was actually used (LLM decided)
            web_search_used = any(r.get('source') == 'web_search_agentic' for r in tier_3_results)
            if web_search_used:
                cost_metrics['web_searches'] += 1
                logger.info("TIER 3: Web search was used (LLM decided)")
            else:
                logger.info("TIER 3: No web search needed (LLM decided local context sufficient)")
            
            # KG would be similar (if implemented as tool)
            cost_metrics['total_sources_queried'] = 2 + (1 if web_search_used else 0)  # memory + docs + web (if used)
            
            confidence = self.evaluator.evaluate_confidence(
                all_results, query, intent
            )
            
            self.stats['tier_3_queries'] += 1
            self._update_avg_confidence('tier_3', confidence)
            
            logger.info(
                f"TIER 3 complete: {len(tier_3_results)} new results, confidence={confidence:.3f}"
            )
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        return self._format_response(
            all_results[:top_k], confidence, current_tier,
            cost_metrics, response_time
        )
    
    def _retrieve_tier_1(self, agent_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Tier 1: Core memory only
        
        Returns:
            List of results from core memory
        """
        try:
            # Query core memory (cheapest, no API calls)
            core_memory = self.memory_manager.get_core_memory()
            
            results = []
            
            # Simple keyword matching in core memory
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            # Check human persona
            human_text = core_memory.get('human', '')
            if human_text:
                human_terms = set(human_text.lower().split())
                overlap = len(query_terms & human_terms)
                if overlap > 0:
                    score = overlap / len(query_terms)
                    results.append({
                        'content': f"[Human Context] {human_text}",
                        'score': score,
                        'source': 'core_memory_human',
                        'tier': 1
                    })
            
            # Check agent persona
            agent_text = core_memory.get('agent', '')
            if agent_text:
                agent_terms = set(agent_text.lower().split())
                overlap = len(query_terms & agent_terms)
                if overlap > 0:
                    score = overlap / len(query_terms)
                    results.append({
                        'content': f"[Agent Context] {agent_text}",
                        'score': score,
                        'source': 'core_memory_agent',
                        'tier': 1
                    })
            
            # Check facts
            facts = core_memory.get('facts', [])
            for fact in facts:
                if isinstance(fact, dict):
                    fact_text = fact.get('text', '')
                else:
                    fact_text = str(fact)
                
                if fact_text:
                    fact_terms = set(fact_text.lower().split())
                    overlap = len(query_terms & fact_terms)
                    if overlap > 0:
                        score = overlap / len(query_terms)
                        results.append({
                            'content': f"[Fact] {fact_text}",
                            'score': score,
                            'source': 'core_memory_facts',
                            'tier': 1
                        })
            
            return sorted(results, key=lambda x: x['score'], reverse=True)
        
        except Exception as e:
            logger.error(f"Tier 1 retrieval error: {e}")
            return []
    
    def _retrieve_tier_2(
        self,
        agent_id: str,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Tier 2: Document store retrieval
        
        Returns:
            List of results from document store
        """
        try:
            # Use hybrid retriever for document store
            results = self.hybrid_retriever.retrieve(
                query=query,
                sources=["documents"],
                top_k=top_k
            )
            
            # Tag with tier
            for r in results:
                r['tier'] = 2
            
            return results
        
        except Exception as e:
            logger.error(f"Tier 2 retrieval error: {e}")
            return []
    
    def _retrieve_tier_3(
        self,
        agent_id: str,
        query: str,
        top_k: int,
        tier_1_2_context: List[Dict[str, Any]] = [{}]
    ) -> List[Dict[str, Any]]:
        """
        Tier 3: AGENTIC retrieval with LLM-driven tool usage
        
        LLM analyzes Tier 1+2 context and DECIDES if it needs:
        - Web search (for current/recent info)
        - Knowledge graph (for entity relationships)
        
        Only calls expensive APIs if LLM determines it's necessary.
        
        Returns:
            List of results from agentic tool usage
        """
        results = []
        
        # Format Tier 1+2 context for LLM
        context_summary = "No local context available."
        if tier_1_2_context:
            top_results = tier_1_2_context[:3]
            context_summary = "\n".join([
                f"[{i+1}] (score: {r.get('score', 0):.2f}) {r.get('content', '')[:200]}..."
                for i, r in enumerate(top_results)
            ])
        
        # Agentic Tier 3 prompt
        agentic_prompt = f"""You are a research assistant with access to external tools.

User Query: {query}

Local Context (from Tier 1+2):
{context_summary}

Your task:
1. Analyze if local context is SUFFICIENT to answer the query
2. If local context is OLD, INCOMPLETE, or MISSING:
   - Use web_search tool for current/recent information
   - Reformulate the search query if needed for better results
3. If local context is SUFFICIENT:
   - Simply confirm "Local context is sufficient"

DECISION CRITERIA:
- Query mentions "latest", "recent", "current", "2025", "2026" → likely needs web_search
- Local context has low scores (<0.5) → likely needs web_search
- Query is well-covered by local context → NO tools needed

Be CONSERVATIVE: Only use web_search when truly necessary to save costs."""
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content="You are a research assistant. Use tools ONLY when local context is insufficient."),
                HumanMessage(content=agentic_prompt)
            ]
            
            # LLM decides if it needs tools
            response = self.llm_with_tools.invoke(messages)
            
            # Check if LLM called tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"Tier 3: LLM decided to use {len(response.tool_calls)} tool(s)")
                
                for tool_call in response.tool_calls:
                    if 'web_search' in tool_call['name'].lower():
                        # LLM decided web search is necessary
                        search_query = tool_call['args'].get('query', query)
                        max_results = tool_call['args'].get('max_results', 3)
                        
                        logger.info(f"Tier 3: LLM triggering web_search for: '{search_query}'")
                        
                        if self.web_search:
                            web_results = self.web_search.search(search_query, max_results=max_results)
                            
                            for web_item in web_results:
                                results.append({
                                    'content': f"[Web Search] {web_item.get('content', web_item.get('snippet', ''))}",
                                    'score': 0.8,  # High relevance (LLM decided it was needed)
                                    'source': 'web_search_agentic',
                                    'tier': 3,
                                    'metadata': {
                                        'url': web_item.get('url', ''),
                                        'title': web_item.get('title', ''),
                                        'llm_decided': True
                                    }
                                })
                            
                            logger.info(f"Tier 3: Retrieved {len(web_results)} web results")
                    
                    # Future: Add KG tool support here
            else:
                logger.info(f"Tier 3: LLM decided local context is SUFFICIENT (no tools needed)")
                # LLM didn't call any tools = local context is enough
                # Return empty to signal no additional retrieval needed
        
        except Exception as e:
            logger.error(f"Tier 3 agentic retrieval error: {e}")
            # Fallback: try web search directly if available
            if self.web_search:
                try:
                    logger.warning(f"Tier 3: Falling back to direct web search due to error")
                    web_results = self.web_search.search(query, max_results=3)
                    for web_item in web_results:
                        results.append({
                            'content': f"[Web Search - Fallback] {web_item.get('content', web_item.get('snippet', ''))}",
                            'score': 0.7,
                            'source': 'web_search_fallback',
                            'tier': 3,
                            'metadata': {'url': web_item.get('url', ''), 'fallback': True}
                        })
                except:
                    pass
        
        return results
    
    def _format_response(
        self,
        results: List[Dict[str, Any]],
        confidence: float,
        tier: RetrievalTier,
        cost_metrics: Dict[str, int],
        response_time: float
    ) -> Dict[str, Any]:
        """Format retrieval response"""
        return {
            'results': results,
            'confidence': confidence,
            'tier_used': tier.value,
            'tier_name': tier.name,
            'cost_metrics': cost_metrics,
            'response_time': response_time,
            'count': len(results)
        }
    
    def _update_avg_confidence(self, tier: str, confidence: float):
        """Update running average confidence for tier"""
        key = f'avg_confidence_{tier}'
        count_key = f'{tier}_queries'
        
        current_avg = self.stats[key]
        count = self.stats[count_key]
        
        # Running average
        new_avg = ((current_avg * (count - 1)) + confidence) / count
        self.stats[key] = new_avg
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        total_queries = (
            self.stats['tier_1_queries'] +
            self.stats['tier_2_queries'] +
            self.stats['tier_3_queries']
        )
        
        stats = self.stats.copy()
        
        if total_queries > 0:
            stats['tier_1_percentage'] = self.stats['tier_1_queries'] / total_queries * 100
            stats['tier_2_percentage'] = self.stats['tier_2_queries'] / total_queries * 100
            stats['tier_3_percentage'] = self.stats['tier_3_queries'] / total_queries * 100
            stats['escalation_rate'] = self.stats['escalations'] / total_queries * 100
        else:
            stats['tier_1_percentage'] = 0.0
            stats['tier_2_percentage'] = 0.0
            stats['tier_3_percentage'] = 0.0
            stats['escalation_rate'] = 0.0
        
        stats['total_queries'] = total_queries
        stats['confidence_threshold'] = self.confidence_threshold
        stats['tier_3_enabled'] = self.enable_tier_3
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'tier_1_queries': 0,
            'tier_2_queries': 0,
            'tier_3_queries': 0,
            'escalations': 0,
            'avg_confidence_tier_1': 0.0,
            'avg_confidence_tier_2': 0.0,
            'avg_confidence_tier_3': 0.0
        }
        logger.info("Statistics reset")
