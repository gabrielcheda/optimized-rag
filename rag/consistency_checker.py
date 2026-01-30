"""
Consistency Checker - Phase 2 Anti-Hallucination
Detects contradictions and inconsistencies across multiple documents
"""

import logging
from typing import Any, Dict, List, Tuple
import re

from memory.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Checks for contradictions and inconsistencies across retrieved documents
    
    Phase 2: High Priority - Prevents hallucinations from contradictory sources
    """
    
    def __init__(self, embedding_service: EmbeddingService, similarity_threshold: float = 0.85):
        """
        Initialize consistency checker
        
        Args:
            embedding_service: Service for computing embeddings
            similarity_threshold: Threshold for detecting similar claims (0-1)
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
    
    def check_consistency(
        self, 
        documents: List[Dict[str, Any]], 
        query: str
    ) -> Dict[str, Any]:
        """
        Check for contradictions across documents
        
        Args:
            documents: List of retrieved documents with metadata
            query: User query for context
            
        Returns:
            Dict with consistency results and detected contradictions
        """
        if len(documents) < 2:
            return {
                "consistent": True,
                "contradictions": [],
                "confidence": 1.0,
                "warning": None
            }
        
        try:
            # Extract claims from all documents
            all_claims = []
            for idx, doc in enumerate(documents):
                content = doc.get("content", "")
                claims = self._extract_claims(content)
                for claim in claims:
                    all_claims.append({
                        "text": claim,
                        "doc_idx": idx,
                        "source": doc.get("source", f"doc_{idx}")
                    })
            
            if len(all_claims) < 2:
                return {
                    "consistent": True,
                    "contradictions": [],
                    "confidence": 1.0,
                    "warning": "Too few claims to check consistency"
                }
            
            # Find contradictions using embeddings
            contradictions = self._find_contradictions(all_claims)
            
            # Calculate consistency score
            total_pairs = len(all_claims) * (len(all_claims) - 1) / 2
            contradiction_ratio = len(contradictions) / max(total_pairs, 1)
            consistency_score = 1.0 - min(contradiction_ratio, 1.0)
            
            # Determine if documents are consistent enough
            is_consistent = len(contradictions) == 0 or consistency_score >= 0.8
            
            result = {
                "consistent": is_consistent,
                "contradictions": contradictions[:5],  # Return top 5
                "contradiction_count": len(contradictions),
                "confidence": consistency_score,
                "total_claims": len(all_claims),
                "warning": self._generate_warning(contradictions) if contradictions else None
            }
            
            if contradictions:
                logger.warning(
                    f"Consistency check found {len(contradictions)} contradictions "
                    f"(score: {consistency_score:.2f})"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {
                "consistent": True,  # Fail open
                "contradictions": [],
                "confidence": 0.5,
                "warning": f"Consistency check error: {str(e)}"
            }
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text
        
        Args:
            text: Document content
            
        Returns:
            List of claim strings
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sent in sentences:
            sent = sent.strip()
            
            # Filter short sentences
            if len(sent) < 20:
                continue
            
            # Skip meta-statements
            meta_patterns = [
                r'^(this|that|these|those|it|they)\s+(is|are|was|were)',
                r'^(here|there)\s+(is|are)',
                r'^(in conclusion|in summary|overall|finally)',
            ]
            if any(re.match(pattern, sent.lower()) for pattern in meta_patterns):
                continue
            
            claims.append(sent)
        
        return claims
    
    def _find_contradictions(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find contradictory claims using semantic similarity + negation detection
        
        Args:
            claims: List of claim dicts with text, doc_idx, source
            
        Returns:
            List of contradiction pairs
        """
        contradictions = []
        
        # Compute embeddings for all claims
        claim_texts = [c["text"] for c in claims]
        try:
            embeddings = self.embedding_service.generate_embeddings_batch(claim_texts)
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return []
        
        # Compare all pairs
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                # Skip claims from same document
                if claims[i]["doc_idx"] == claims[j]["doc_idx"]:
                    continue
                
                # Check semantic similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                
                # Claims are similar enough to potentially contradict
                if similarity >= self.similarity_threshold:
                    # Check for negation/contradiction markers
                    if self._is_contradiction(claims[i]["text"], claims[j]["text"]):
                        contradictions.append({
                            "claim_1": claims[i]["text"][:200],
                            "claim_2": claims[j]["text"][:200],
                            "source_1": claims[i]["source"],
                            "source_2": claims[j]["source"],
                            "similarity": round(similarity, 3),
                            "type": "semantic_contradiction"
                        })
        
        return contradictions
    
    def _is_contradiction(self, text1: str, text2: str) -> bool:
        """
        Detect if two similar claims contradict each other
        
        Args:
            text1: First claim
            text2: Second claim
            
        Returns:
            True if contradiction detected
        """
        # Normalize texts
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        
        # Check for negation patterns
        negation_pairs = [
            ("is not", "is"),
            ("are not", "are"),
            ("was not", "was"),
            ("were not", "were"),
            ("does not", "does"),
            ("do not", "do"),
            ("did not", "did"),
            ("cannot", "can"),
            ("will not", "will"),
            ("should not", "should"),
            ("no", "yes"),
            ("false", "true"),
            ("incorrect", "correct"),
            ("never", "always"),
        ]
        
        for neg, pos in negation_pairs:
            if (neg in t1_lower and pos in t2_lower) or (pos in t1_lower and neg in t2_lower):
                return True
        
        # Check for contradictory quantities (rough heuristic)
        numbers_1 = re.findall(r'\b\d+\.?\d*\b', text1)
        numbers_2 = re.findall(r'\b\d+\.?\d*\b', text2)
        
        if numbers_1 and numbers_2:
            # If claims have different numbers, might be contradictory
            if set(numbers_1) != set(numbers_2):
                return True
        
        return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _generate_warning(self, contradictions: List[Dict[str, Any]]) -> str:
        """
        Generate user-facing warning message
        
        Args:
            contradictions: List of detected contradictions
            
        Returns:
            Warning message
        """
        count = len(contradictions)
        
        if count == 1:
            return "Warning: Found 1 potential contradiction in sources. Response may be unreliable."
        elif count <= 3:
            return f"Warning: Found {count} contradictions in sources. Please verify information."
        else:
            return f"Warning: Found {count} contradictions in sources. High uncertainty in response."
