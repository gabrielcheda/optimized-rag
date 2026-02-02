"""
Selective Reranking
Applies reranking only when necessary to reduce costs and latency
"""

from typing import List, Dict, Any, Optional
import logging

from rag.models.intent_analysis import QueryIntent

logger = logging.getLogger(__name__)


class SelectiveReranker:
    """
    FASE 6: Reranking for precision-critical applications

    In FASE 6 mode, reranking is ALWAYS applied for maximum precision.
    Selective mode is disabled by default to prioritize precision over cost.
    """

    def __init__(
        self,
        openai_reranker=None,
        cross_encoder_reranker=None,
        enable_selective: bool = False  # FASE 6: Disabled by default (always rerank)
    ):
        """
        Initialize selective reranker

        Args:
            openai_reranker: OpenAI reranker instance
            cross_encoder_reranker: CrossEncoder reranker instance
            enable_selective: Enable selective reranking logic
                              FASE 6: Default is False (always rerank for precision)
        """
        self.openai_reranker = openai_reranker
        self.cross_encoder_reranker = cross_encoder_reranker
        self.enable_selective = enable_selective

        # Statistics
        self.total_queries = 0
        self.reranking_skipped = 0
        self.reranking_applied = 0

        logger.info(
            f"FASE 6 SelectiveReranker initialized: enable_selective={enable_selective}, "
            f"openai={openai_reranker is not None}, cross_encoder={cross_encoder_reranker is not None}"
        )
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        intent: QueryIntent = QueryIntent.QUESTION_ANSWERING,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Selectively apply reranking based on query characteristics
        
        Args:
            query: User query
            results: Retrieved results
            intent: Query intent (qa, chat, search, multi_hop, etc.)
            top_k: Number of results to return
            
        Returns:
            Reranked results (or original if reranking skipped)
        """
        self.total_queries += 1
        
        if not self.enable_selective:
            # Always rerank if selective mode disabled
            return self._apply_reranking(query, results, intent, top_k)
        
        # Check if reranking is necessary
        should_rerank, reason = self._should_rerank(results, intent)
        
        if not should_rerank:
            self.reranking_skipped += 1
            logger.info(f"Skipping reranking: {reason}")
            return results[:top_k]
        
        self.reranking_applied += 1
        logger.info(f"Applying reranking: {reason}")
        return self._apply_reranking(query, results, intent, top_k)
    
    def _should_rerank(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> tuple[bool, str]:
        """
        FASE 6: Determine if reranking is necessary

        In FASE 6, almost all intents trigger reranking for maximum precision.

        Returns:
            Tuple of (should_rerank: bool, reason: str)
        """
        from rag import QueryIntent

        # FASE 6: Expanded list of precision intents (almost everything)
        PRECISION_INTENTS = {
            QueryIntent.QUESTION_ANSWERING,
            QueryIntent.MULTI_HOP_REASONING,
            QueryIntent.COMPARISON,
            QueryIntent.FACT_CHECKING,
            QueryIntent.SUMMARIZATION,
            QueryIntent.SEARCH,
        }
        # FASE 6: Expanded string values for compatibility
        PRECISION_INTENT_VALUES = {
            'qa', 'multi_hop', 'compare', 'factual', 'question_answering',
            'comparison', 'fact_checking', 'summarization', 'search'
        }
        
        # Extrair valor do enum de forma segura
        intent_value = intent.value if hasattr(intent, 'value') else str(intent).lower()
        
        # Verificar tanto o enum quanto o valor string separadamente
        if intent in PRECISION_INTENTS or intent_value in PRECISION_INTENT_VALUES:
            return True, f"Precision intent ({intent_value}) - always rerank"
        
        # Rule 1: Too few results - but FORCE reranking if scores are very low
        # (e.g., cross-language queries where embedding fails but CrossEncoder works)
        if len(results) <= 5:
            scores = [r.get('score', 0) for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score < 0.05:  # Very low embedding scores - need CrossEncoder!
                logger.info(f"Applying reranking: Low embedding scores ({avg_score:.3f}), CrossEncoder needed")
                return True, f"Low embedding scores ({avg_score:.3f}), CrossEncoder needed"
            return False, "Too few results (≤5)"
        
        # Rule 2: Check score variance
        scores = [r.get('score', 0) for r in results[:10]]
        if not scores:
            return True, "No scores available"
        
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score)**2 for s in scores) / len(scores)
        
        # High variance = clear winners, no need to rerank
        if score_variance > 0.1:
            return False, f"High score variance ({score_variance:.3f})"
        
        # Low variance = similar scores, reranking helps
        if score_variance < 0.05:
            return True, f"Low score variance ({score_variance:.3f})"
        
        # Rule 3: Intent-based decision
        # Factual queries benefit more from reranking
        if intent in ['qa', 'multi_hop', 'compare']:
            return True, f"Intent requires precision ({intent})"
        
        # Chat queries are more forgiving
        if intent == 'chat':
            # Only rerank if top score is low
            if scores[0] < 0.7:
                return True, "Low top score for chat"
            return False, "Chat query with good top score"
        
        # Default: rerank
        return True, "Default policy"
    
    def _apply_reranking(
        self,
        query: str,
        results: List[Dict[str, Any]],
        intent: QueryIntent,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Apply appropriate reranking strategy based on intent

        Args:
            query: User query
            results: Results to rerank
            intent: Query intent (enum or string)
            top_k: Number to return

        Returns:
            Reranked results
        """
        # FASE 6.1: Get intent value (handle both enum and string)
        intent_value = intent.value if hasattr(intent, 'value') else str(intent).lower()

        # Factual intent values that benefit from CrossEncoder
        FACTUAL_INTENTS = {
            'qa', 'multi_hop', 'compare', 'question_answering',
            'multi_hop_reasoning', 'comparison', 'fact_checking'
        }

        # Conversational intent values that can use faster reranking
        CONVERSATIONAL_INTENTS = {'chat', 'search', 'conversational', 'clarification'}

        # Choose reranking strategy based on intent
        if intent_value in FACTUAL_INTENTS:
            # Factual queries: use CrossEncoder (more accurate)
            if self.cross_encoder_reranker and self.cross_encoder_reranker.is_available():
                logger.info(f"Using CrossEncoder for factual intent: {intent_value}")
                return self.cross_encoder_reranker.rerank(query, results, top_k)
            elif self.openai_reranker:
                logger.info(f"CrossEncoder unavailable, using OpenAI for: {intent_value}")
                return self.openai_reranker.rerank(query, results, top_k)

        elif intent_value in CONVERSATIONAL_INTENTS:
            # Chat/search: use OpenAI reranker (faster)
            if self.openai_reranker:
                logger.info(f"Using OpenAI reranker for conversational intent: {intent_value}")
                return self.openai_reranker.rerank(query, results, top_k)
            elif self.cross_encoder_reranker and self.cross_encoder_reranker.is_available():
                logger.info(f"OpenAI unavailable, using CrossEncoder for: {intent_value}")
                return self.cross_encoder_reranker.rerank(query, results, top_k)

        # FASE 6.1: Default - try any available reranker (don't give up!)
        if self.cross_encoder_reranker and self.cross_encoder_reranker.is_available():
            logger.info(f"Using CrossEncoder for unmatched intent: {intent_value}")
            return self.cross_encoder_reranker.rerank(query, results, top_k)
        elif self.openai_reranker:
            logger.info(f"Using OpenAI reranker for unmatched intent: {intent_value}")
            return self.openai_reranker.rerank(query, results, top_k)

        # No reranker available at all
        logger.warning(f"No reranker available for intent: {intent_value}, returning original results")
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get reranking statistics
        
        Returns:
            Dict with statistics
        """
        skip_rate = self.reranking_skipped / self.total_queries if self.total_queries > 0 else 0
        
        return {
            'total_queries': self.total_queries,
            'reranking_applied': self.reranking_applied,
            'reranking_skipped': self.reranking_skipped,
            'skip_rate': skip_rate,
            'skip_rate_percent': f"{skip_rate * 100:.1f}%",
            'estimated_cost_savings': f"{skip_rate * 100:.0f}% of reranking costs"
        }
