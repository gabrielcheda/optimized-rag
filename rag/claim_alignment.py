"""
FASE 6: Claim Alignment Scorer
Measures how well answer claims align with retrieved document content

This module provides fine-grained alignment scoring between generated
claims and source documents, enabling precise hallucination detection.
"""

import logging
import re
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClaimAlignmentResult:
    """Result of claim-document alignment analysis"""
    claim: str
    best_document_idx: int
    alignment_score: float  # 0-1, higher = better alignment
    alignment_type: str  # 'exact', 'paraphrase', 'partial', 'weak', 'none'
    matched_text: str  # The text in document that matches
    confidence: float  # Confidence in the alignment assessment


class ClaimAlignmentScorer:
    """
    FASE 6: Scores alignment between answer claims and source documents

    Uses multiple techniques:
    1. Exact/substring matching for factual claims
    2. Semantic similarity for paraphrased content
    3. Key entity matching for partial alignment
    """

    # Alignment type thresholds
    EXACT_THRESHOLD = 0.95
    PARAPHRASE_THRESHOLD = 0.80
    PARTIAL_THRESHOLD = 0.60
    WEAK_THRESHOLD = 0.40

    def __init__(
        self,
        embedding_service=None,
        use_semantic: bool = True,
        strict_mode: bool = True  # FASE 6: Strict mode for precision
    ):
        """
        Initialize claim alignment scorer

        Args:
            embedding_service: Optional embedding service for semantic matching
            use_semantic: Enable semantic similarity matching
            strict_mode: FASE 6 - Require higher alignment for 'supported' status
        """
        self.embedding_service = embedding_service
        self.use_semantic = use_semantic and embedding_service is not None
        self.strict_mode = strict_mode

        logger.info(
            f"FASE 6 ClaimAlignmentScorer initialized: "
            f"semantic={self.use_semantic}, strict_mode={strict_mode}"
        )

    def score_claim_alignment(
        self,
        claim: str,
        documents: List[Dict[str, Any]],
        min_alignment: float = 0.5
    ) -> ClaimAlignmentResult:
        """
        Score how well a claim aligns with retrieved documents

        Args:
            claim: The claim to evaluate
            documents: Retrieved documents to check against
            min_alignment: Minimum alignment score to consider valid

        Returns:
            ClaimAlignmentResult with alignment details
        """
        if not documents:
            return ClaimAlignmentResult(
                claim=claim,
                best_document_idx=-1,
                alignment_score=0.0,
                alignment_type='none',
                matched_text='',
                confidence=1.0
            )

        best_result = None
        best_score = 0.0

        for idx, doc in enumerate(documents):
            content = doc.get('content', '')
            if not content:
                continue

            # Calculate alignment score using multiple methods
            alignment_score, matched_text, method = self._calculate_alignment(
                claim, content
            )

            if alignment_score > best_score:
                best_score = alignment_score
                alignment_type = self._get_alignment_type(alignment_score)
                best_result = ClaimAlignmentResult(
                    claim=claim,
                    best_document_idx=idx,
                    alignment_score=alignment_score,
                    alignment_type=alignment_type,
                    matched_text=matched_text,
                    confidence=self._calculate_confidence(alignment_score, method)
                )

        if best_result is None:
            return ClaimAlignmentResult(
                claim=claim,
                best_document_idx=-1,
                alignment_score=0.0,
                alignment_type='none',
                matched_text='',
                confidence=1.0
            )

        return best_result

    def score_all_claims(
        self,
        claims: List[str],
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score alignment for all claims in an answer

        Args:
            claims: List of claims extracted from answer
            documents: Retrieved documents

        Returns:
            Dict with overall alignment score and per-claim details
        """
        if not claims:
            return {
                'overall_alignment': 0.0,
                'claim_results': [],
                'alignment_distribution': {},
                'supported_ratio': 0.0,
                'recommendation': 'No claims to evaluate'
            }

        claim_results = []
        alignment_types = {'exact': 0, 'paraphrase': 0, 'partial': 0, 'weak': 0, 'none': 0}

        for claim in claims:
            result = self.score_claim_alignment(claim, documents)
            claim_results.append(result)
            alignment_types[result.alignment_type] += 1

        # Calculate overall alignment score
        total_score = sum(r.alignment_score for r in claim_results)
        overall_alignment = total_score / len(claim_results) if claim_results else 0.0

        # Calculate supported ratio (claims with good alignment)
        if self.strict_mode:
            # FASE 6: Only 'exact' and 'paraphrase' count as supported
            supported = alignment_types['exact'] + alignment_types['paraphrase']
        else:
            # Less strict: include 'partial'
            supported = alignment_types['exact'] + alignment_types['paraphrase'] + alignment_types['partial']

        supported_ratio = supported / len(claims) if claims else 0.0

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_alignment, supported_ratio, alignment_types
        )

        logger.info(
            f"FASE 6 Claim Alignment: overall={overall_alignment:.2f}, "
            f"supported={supported_ratio:.2%}, "
            f"distribution={alignment_types}"
        )

        return {
            'overall_alignment': overall_alignment,
            'claim_results': claim_results,
            'alignment_distribution': alignment_types,
            'supported_ratio': supported_ratio,
            'recommendation': recommendation,
            'total_claims': len(claims),
            'supported_claims': supported
        }

    def _calculate_alignment(
        self,
        claim: str,
        document_content: str
    ) -> Tuple[float, str, str]:
        """
        Calculate alignment between claim and document content

        Returns:
            Tuple of (score, matched_text, method_used)
        """
        claim_lower = claim.lower().strip()
        content_lower = document_content.lower()

        # Method 1: Exact substring match
        if claim_lower in content_lower:
            # Find the matched portion
            start_idx = content_lower.find(claim_lower)
            matched = document_content[start_idx:start_idx + len(claim)]
            return 1.0, matched, 'exact'

        # Method 2: Key entity/number matching
        entity_score, matched_entities = self._entity_match_score(claim, document_content)
        if entity_score >= self.PARAPHRASE_THRESHOLD:
            return entity_score, matched_entities, 'entity'

        # Method 3: Semantic similarity (if available)
        if self.use_semantic:
            semantic_score, matched_text = self._semantic_similarity(claim, document_content)
            if semantic_score > entity_score:
                return semantic_score, matched_text, 'semantic'

        # Method 4: Word overlap (fallback)
        overlap_score = self._word_overlap_score(claim, document_content)
        return overlap_score, '', 'overlap'

    def _entity_match_score(
        self,
        claim: str,
        document_content: str
    ) -> Tuple[float, str]:
        """
        Score based on matching key entities (names, numbers, dates)
        """
        # Extract entities from claim
        entities = self._extract_entities(claim)

        if not entities:
            return 0.0, ''

        content_lower = document_content.lower()
        matched = []

        for entity in entities:
            if entity.lower() in content_lower:
                matched.append(entity)

        if not entities:
            return 0.0, ''

        score = len(matched) / len(entities)
        return score, ', '.join(matched)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (names, numbers, dates)"""
        entities = []

        # Proper nouns (capitalized words not at sentence start)
        proper_nouns = re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+', text)
        entities.extend(proper_nouns)

        # Numbers (including decimals and percentages)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\b', text)
        entities.extend(numbers)

        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        entities.extend(dates)

        # Years
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        entities.extend(years)

        return list(set(entities))

    def _semantic_similarity(
        self,
        claim: str,
        document_content: str
    ) -> Tuple[float, str]:
        """
        Calculate semantic similarity between claim and document sentences
        """
        try:
            # Split document into sentences
            sentences = re.split(r'[.!?]+', document_content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            if not sentences:
                return 0.0, ''

            # Generate embeddings
            claim_embedding = self.embedding_service.generate_embedding(claim)

            best_score = 0.0
            best_sentence = ''

            for sentence in sentences[:20]:  # Limit to first 20 sentences
                sent_embedding = self.embedding_service.generate_embedding(sentence)
                similarity = self._cosine_similarity(claim_embedding, sent_embedding)

                if similarity > best_score:
                    best_score = similarity
                    best_sentence = sentence

            return best_score, best_sentence[:200]  # Truncate for display

        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0, ''

    def _word_overlap_score(self, claim: str, document_content: str) -> float:
        """Calculate word overlap score (fallback method)"""
        # Tokenize
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        doc_words = set(re.findall(r'\b\w+\b', document_content.lower()))

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once'
        }

        claim_words -= stop_words
        doc_words -= stop_words

        if not claim_words:
            return 0.0

        overlap = len(claim_words & doc_words)
        return overlap / len(claim_words)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _get_alignment_type(self, score: float) -> str:
        """Determine alignment type from score"""
        if score >= self.EXACT_THRESHOLD:
            return 'exact'
        elif score >= self.PARAPHRASE_THRESHOLD:
            return 'paraphrase'
        elif score >= self.PARTIAL_THRESHOLD:
            return 'partial'
        elif score >= self.WEAK_THRESHOLD:
            return 'weak'
        else:
            return 'none'

    def _calculate_confidence(self, alignment_score: float, method: str) -> float:
        """Calculate confidence in the alignment assessment"""
        # Base confidence from score
        base_confidence = alignment_score

        # Method-specific adjustments
        method_confidence = {
            'exact': 1.0,
            'entity': 0.9,
            'semantic': 0.8,
            'overlap': 0.6
        }

        method_factor = method_confidence.get(method, 0.5)

        return min(base_confidence * method_factor * 1.2, 1.0)

    def _generate_recommendation(
        self,
        overall_alignment: float,
        supported_ratio: float,
        alignment_distribution: Dict[str, int]
    ) -> str:
        """Generate recommendation based on alignment analysis"""
        if overall_alignment >= 0.8 and supported_ratio >= 0.9:
            return "Excellent alignment - answer is well-grounded in documents"
        elif overall_alignment >= 0.6 and supported_ratio >= 0.7:
            return "Good alignment - most claims are supported"
        elif overall_alignment >= 0.4 and supported_ratio >= 0.5:
            return "Partial alignment - some claims need verification"
        elif alignment_distribution.get('none', 0) > 0:
            return f"Poor alignment - {alignment_distribution['none']} claims have no document support"
        else:
            return "Weak alignment - answer may contain unsupported claims"
