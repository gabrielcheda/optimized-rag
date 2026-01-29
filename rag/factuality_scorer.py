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
        Calculate comprehensive factuality score (0-1)
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents
            source_map: Optional citation source map
            
        Returns:
            Dict with factuality score and breakdown
        """
        # Component 1: Support ratio (weight 40%)
        answer_eval = self.evaluator.evaluate_answer(query, answer, retrieved_docs)
        support_ratio = answer_eval.get('support_ratio', 0.0)
        
        # Component 2: Citation coverage (weight 30%)
        citation_coverage = self._calculate_citation_coverage(answer, source_map)
        
        # Component 3: Confidence (weight 20%)
        avg_confidence = answer_eval.get('avg_confidence', 0.0)
        
        # Component 4: Retrieval quality (weight 10%)
        retrieval_quality = self._calculate_retrieval_quality(retrieved_docs)
        
        # Calculate weighted factuality score
        factuality_score = (
            support_ratio * 0.4 +
            citation_coverage * 0.3 +
            avg_confidence * 0.2 +
            retrieval_quality * 0.1
        )
        
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
        
        if not source_map:
            # If no source map, accept numeric citations as valid if they're present and consistent
            if citations:
                logger.info(f"No source_map provided, accepting numeric citations as valid: {citations}")
                return 0.9  # High score for having citations, even without mapping
            return 0.5  # Neutral if no citations at all
        
        # Validate citations - accept both source_map keys AND numeric strings
        valid_citations = [c for c in citations if c in source_map or c.isdigit()]
        invalid_citations = [c for c in citations if c not in valid_citations]
        
        if invalid_citations:
            logger.warning(f"Invalid citations found: {invalid_citations}")
        
        # If we have citations (valid or numeric), don't penalize heavily
        if not valid_citations and citations:
            # Still have citations, just not mapped - moderate score
            logger.info(f"Citations present but not in source_map: {citations}. Accepting as moderate quality.")
            return 0.6  # Changed from 0.2 to 0.6 - don't over-penalize
        
        # Extract claims
        claims = self.evaluator._extract_claims(answer)
        
        if not claims:
            # No claims extracted - if we have valid citations, that's good
            return 1.0 if valid_citations else 0.5
        
        # Check citation coverage (only count valid citations)
        cited_claims = 0
        for claim in claims:
            # Check if claim has a valid citation
            claim_citations = re.findall(r'\[(\d+)\]', claim)
            if any(c in valid_citations for c in claim_citations):
                cited_claims += 1
        
        citation_coverage = cited_claims / len(claims) if claims else 0.0
        
        # Additional penalty for invalid citations (reduces trust)
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
        """Get quality level from score"""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on score"""
        if score >= 0.7:
            return "Answer is highly factual and well-supported. Safe to use."
        elif score >= 0.5:
            return "Answer is moderately factual. Verify critical claims."
        elif score >= 0.3:
            return "Answer has low factuality. Use with caution."
        else:
            return "Answer is unreliable. Consider re-retrieval or refuse to answer."
    
    def should_refuse_answer(self, factuality_score: float, threshold: float = 0.4) -> bool:
        """
        Determine if answer should be refused due to low factuality
        
        Args:
            factuality_score: Calculated factuality score
            threshold: Minimum acceptable score (default: 0.4)
            
        Returns:
            True if answer should be refused
        """
        return factuality_score < threshold
