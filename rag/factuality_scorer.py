"""
Factuality Score Calculator
Calculates comprehensive factuality score for generated answers
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FactualityScorer:
    """Calculate factuality score for answers"""
    
    def __init__(self, self_rag_evaluator):
        """
        Initialize factuality scorer
        
        Args:
            self_rag_evaluator: SelfRAGEvaluator instance
        """
        self.evaluator = self_rag_evaluator
    
    def calculate_factuality_score(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        source_map: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        FASE 6: Calculate comprehensive factuality score (0-1)
        Recalibrated weights for maximum precision

        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents
            source_map: Optional citation source map

        Returns:
            Dict with factuality score and breakdown
        """
        # Component 1: Support ratio (FASE 6: weight 50% - most critical for precision)
        answer_eval = self.evaluator.evaluate_answer(query, answer, retrieved_docs)
        support_ratio = answer_eval.get('support_ratio', 0.0)

        # Component 2: Citation coverage (FASE 6: weight 25% - still important)
        citation_coverage = self._calculate_citation_coverage(answer, source_map)

        # Component 3: Confidence (FASE 6: weight 20% - verification confidence)
        avg_confidence = answer_eval.get('avg_confidence', 0.0)

        # Component 4: Retrieval quality (FASE 6: weight 5% - less important for precision)
        retrieval_quality = self._calculate_retrieval_quality(retrieved_docs)

        # FASE 6: Recalibrated weights for maximum precision
        # Support ratio is now dominant (50%) to prioritize document-backed claims
        factuality_score = (
            support_ratio * 0.50 +      # FASE 6: Increased from 0.40
            citation_coverage * 0.25 +   # FASE 6: Decreased from 0.30
            avg_confidence * 0.20 +      # FASE 6: Same
            retrieval_quality * 0.05     # FASE 6: Decreased from 0.10
        )

        # FASE 6: Apply penalty for zero citations (hard requirement for precision)
        if citation_coverage == 0.0 and len(answer) > 50:
            logger.warning("FASE 6: Zero citations penalty applied")
            factuality_score *= 0.5  # 50% penalty for no citations
        
        # Determine quality level
        quality_level = self._get_quality_level(factuality_score)
        
        result = {
            'factuality_score': factuality_score,
            'quality_level': quality_level,
            'components': {
                'support_ratio': support_ratio,
                'citation_coverage': citation_coverage,
                'avg_confidence': avg_confidence,
                'retrieval_quality': retrieval_quality
            },
            'answer_evaluation': answer_eval,
            'recommendation': self._get_recommendation(factuality_score)
        }
        
        logger.info(
            f"Factuality score: {factuality_score:.3f} ({quality_level}) - "
            f"support={support_ratio:.2f}, citations={citation_coverage:.2f}, "
            f"conf={avg_confidence:.2f}, retrieval={retrieval_quality:.2f}"
        )
        
        return result
    
    def _calculate_citation_coverage(
        self,
        answer: str,
        source_map: Dict[str, Any] = {}
    ) -> float:
        """Calculate citation coverage score"""
        import re
        
        # Extract citations [N]
        citations = re.findall(r'\[(\d+)\]', answer)
        
        # CORREÇÃO: Se não há citações, retornar 0 (não 0.5)
        if not citations:
            logger.info("No citations found in answer")
            return 0.0
        
        # Log citações encontradas
        logger.info(f"Found {len(citations)} citation(s) in answer: {citations}")
        
        if not source_map:
            # If no source map, accept numeric citations as valid if they're present
            logger.info(f"No source_map provided, accepting numeric citations as valid: {citations}")
            return 0.9  # High score for having citations, even without mapping
        
        # Validate citations - accept both source_map keys AND numeric strings
        valid_citations = [c for c in citations if c in source_map or c.isdigit()]
        invalid_citations = [c for c in citations if c not in valid_citations]
        
        if invalid_citations:
            logger.warning(f"Invalid citations found: {invalid_citations}")
        
        # CORREÇÃO: Método simplificado - não depende de claim extraction
        # Calcula coverage baseado na presença de citações na resposta inteira
        if not valid_citations and citations:
            # Still have citations, just not mapped - moderate score
            logger.info(f"Citations present but not in source_map: {citations}. Accepting as moderate quality.")
            return 0.6
        
        # Se temos citações válidas, calcular coverage de duas formas:
        # 1. Coverage direto: quantas sentenças têm citações?
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        sentences_with_citations = sum(1 for s in sentences if re.search(r'\[(\d+)\]', s))
        sentence_coverage = sentences_with_citations / len(sentences) if sentences else 0.0
        
        # 2. Coverage por claims (fallback)
        claims = self.evaluator._extract_claims(answer)
        claim_coverage = 0.0
        
        if claims and claims != [answer]:  # Se extraiu claims válidos
            cited_claims = sum(1 for claim in claims if re.search(r'\[(\d+)\]', claim))
            claim_coverage = cited_claims / len(claims)
            logger.info(f"Claim coverage: {cited_claims}/{len(claims)} claims cited ({claim_coverage:.2f})")
        
        # Usar o maior dos dois métodos (mais generoso)
        citation_coverage = max(sentence_coverage, claim_coverage)
        
        logger.info(
            f"Citation coverage: sentence={sentence_coverage:.2f}, "
            f"claim={claim_coverage:.2f}, final={citation_coverage:.2f}"
        )
        
        # Penalty for invalid citations
        if invalid_citations and citations:
            validity_ratio = len(valid_citations) / len(citations)
            citation_coverage *= validity_ratio
        
        return citation_coverage
    
    def _calculate_retrieval_quality(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """Calculate retrieval quality score"""
        if not retrieved_docs:
            return 0.0
        
        # Average score of top 5 documents
        top_docs = retrieved_docs[:5]
        scores = [doc.get('score', 0.0) for doc in top_docs]
        
        if not scores:
            return 0.5  # Neutral if no scores
        
        avg_score = sum(scores) / len(scores)
        return avg_score
    
    def _get_quality_level(self, score: float) -> str:
        """FASE 6: Stricter quality level thresholds for precision"""
        if score >= 0.85:      # FASE 6: Raised from 0.8
            return "EXCELLENT"
        elif score >= 0.70:    # FASE 6: Raised from 0.6
            return "GOOD"
        elif score >= 0.50:    # FASE 6: Raised from 0.4
            return "FAIR"
        else:
            return "POOR"

    def _get_recommendation(self, score: float) -> str:
        """FASE 6: Stricter recommendations for precision"""
        if score >= 0.80:      # FASE 6: Raised from 0.7
            return "Answer is highly factual and well-supported. Safe to use."
        elif score >= 0.60:    # FASE 6: Raised from 0.5
            return "Answer is moderately factual. Verify critical claims before using."
        elif score >= 0.45:    # FASE 6: Raised from 0.3
            return "Answer has low factuality. Use with extreme caution or refuse."
        else:
            return "Answer is unreliable. REFUSE to answer - re-retrieve or acknowledge lack of information."

    def should_refuse_answer(self, factuality_score: float, threshold: float = 0.50) -> bool:
        """
        FASE 6: Determine if answer should be refused due to low factuality

        Args:
            factuality_score: Calculated factuality score
            threshold: Minimum acceptable score (FASE 6: default raised to 0.50 from 0.4)

        Returns:
            True if answer should be refused
        """
        return factuality_score < threshold
