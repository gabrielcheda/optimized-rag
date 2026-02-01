"""
Ensemble Verification System
Combines multiple verification strategies for robust claim verification
"""

import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


_embedding_cache: Dict[str, List[float]] = {}
_cache_hits = 0
_cache_misses = 0


def _get_content_hash(content: str) -> str:
    """Generate hash for content to use as cache key"""
    return hashlib.md5(content.encode()).hexdigest()


def get_cache_stats() -> Dict[str, int]:
    """Get embedding cache statistics"""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate": hit_rate,
        "cache_size": len(_embedding_cache)
    }


class EnsembleVerifier:
    """Verifies claims using multiple strategies and combines results"""

    def __init__(
        self,
        llm,
        embedding_service,
        keyword_threshold: float = 0.3,
        embedding_threshold: float = 0.65,
        ensemble_agreement: int = 1
    ):
        """
        Initialize ensemble verifier

        Args:
            llm: Language model for LLM-based verification
            embedding_service: Embedding service for similarity-based verification
            keyword_threshold: Minimum Jaccard similarity for keyword matching (0-1)
            embedding_threshold: Minimum cosine similarity for embedding matching (0-1)
            ensemble_agreement: Minimum number of methods that must agree (1-3)
        """
        self.llm = llm
        self.embedding_service = embedding_service
        self.keyword_threshold = keyword_threshold
        self.embedding_threshold = embedding_threshold
        self.ensemble_agreement = ensemble_agreement

        self.max_cache_size = 500

        logger.info(
            f"EnsembleVerifier initialized: keyword_threshold={keyword_threshold}, "
            f"embedding_threshold={embedding_threshold}, ensemble_agreement={ensemble_agreement}, "
            f"embedding_cache_enabled=True"
        )

    def verify_claim(
        self, claim: str, documents: List[Dict[str, Any]], max_chars_per_doc: int = 2000
    ) -> Dict[str, Any]:
        """
        Verify claim using ensemble of methods

        Args:
            claim: Factual claim to verify
            documents: Retrieved documents
            max_chars_per_doc: Max characters to use per document

        Returns:
            Dict with verification results
        """
        llm_result = self._llm_verification(claim, documents, max_chars_per_doc)
        keyword_result = self._keyword_verification(claim, documents)
        embedding_result = self._embedding_verification(claim, documents)
        final_supported, final_confidence = self._combine_results(
            llm_result, keyword_result, embedding_result
        )

        return {
            "supported": final_supported,
            "confidence": final_confidence,
            "methods": {
                "llm": llm_result,
                "keyword": keyword_result,
                "embedding": embedding_result,
            },
            "claim": claim,
        }

    def _llm_verification(
        self, claim: str, documents: List[Dict[str, Any]], max_chars_per_doc: int
    ) -> Dict[str, Any]:
        """LLM-based verification (highest quality)"""
        docs_content = "\n\n".join(
            [
                f"[Doc {i + 1}] {doc.get('content', '')[:max_chars_per_doc]}"
                for i, doc in enumerate(documents[:5])
            ]
        )

        prompt = f"""Does this claim have supporting evidence in the documents?

Claim: {claim}

Documents:
{docs_content}

Respond:
SUPPORTED: [yes/no]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [quote from document, or 'none']

Evaluation:"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(
                    content="You verify if claims are supported by documents."
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content.lower()

            supported = "supported: yes" in content

            confidence = 0.5
            for line in response.content.split("\n"):
                if "confidence:" in line.lower():
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        confidence = 0.5 if supported else 0.3
                    break

            logger.info(
                f"LLM verification: supported={supported}, confidence={confidence:.2f}"
            )
            return {"supported": supported, "confidence": confidence, "method": "llm"}
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return {"supported": False, "confidence": 0.0, "method": "llm"}

    def _keyword_verification(
        self, claim: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Keyword-based verification (fastest)"""
        claim_words = set(re.findall(r"\b\w+\b", claim.lower()))
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "that",
            "this",
        }
        claim_keywords = claim_words - stop_words

        if not claim_keywords:
            return {"supported": False, "confidence": 0.0, "method": "keyword"}

        best_match_score = 0.0

        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words = set(re.findall(r"\b\w+\b", content))

            intersection = claim_keywords & doc_words
            union = claim_keywords | doc_words
            score = len(intersection) / len(union) if union else 0

            if claim.lower() in content:
                score += 0.3

            best_match_score = max(best_match_score, score)

        supported = best_match_score > self.keyword_threshold
        confidence = min(best_match_score, 1.0)

        return {"supported": supported, "confidence": confidence, "method": "keyword"}

    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or generate and cache it"""
        global _embedding_cache, _cache_hits, _cache_misses

        content_hash = _get_content_hash(text)

        if content_hash in _embedding_cache:
            _cache_hits += 1
            return _embedding_cache[content_hash]

        _cache_misses += 1

        embedding = self.embedding_service.generate_embedding(text)

        if len(_embedding_cache) >= self.max_cache_size:
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]

        _embedding_cache[content_hash] = embedding
        return embedding

    def _embedding_verification(
        self, claim: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Embedding similarity-based verification (balanced)"""
        try:
            claim_embedding = self._get_cached_embedding(claim)

            best_similarity = 0.0

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                content_truncated = content[:2000]
                doc_embedding = self._get_cached_embedding(content_truncated)

                similarity = self._cosine_similarity(claim_embedding, doc_embedding)
                best_similarity = max(best_similarity, similarity)

            supported = best_similarity > self.embedding_threshold
            confidence = best_similarity

            stats = get_cache_stats()
            if stats["total"] % 50 == 0 and stats["total"] > 0:
                logger.info(
                    f"Embedding cache stats: {stats['hits']}/{stats['total']} hits "
                    f"({stats['hit_rate']:.1f}%), size={stats['cache_size']}"
                )

            return {
                "supported": supported,
                "confidence": confidence,
                "method": "embedding",
            }
        except Exception as e:
            logger.error(f"Embedding verification failed: {e}")
            return {"supported": False, "confidence": 0.0, "method": "embedding"}

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _combine_results(
        self, llm_result: Dict, keyword_result: Dict, embedding_result: Dict
    ) -> Tuple[bool, float]:
        """
        Combine results from multiple methods using weighted voting

        Returns:
            Tuple of (supported: bool, confidence: float)
        """
        weights = {
            "llm": 0.5,
            "keyword": 0.3,
            "embedding": 0.2,
        }

        weighted_confidence = (
            llm_result["confidence"] * weights["llm"]
            + keyword_result["confidence"] * weights["keyword"]
            + embedding_result["confidence"] * weights["embedding"]
        )

        votes = [
            llm_result["supported"],
            keyword_result["supported"],
            embedding_result["supported"],
        ]

        supported = sum(votes) >= self.ensemble_agreement

        if sum(votes) == 3:
            weighted_confidence = min(weighted_confidence * 1.2, 1.0)
        elif sum(votes) == 1:
            weighted_confidence = weighted_confidence * 0.7

        logger.info(
            f"Ensemble: LLM={llm_result['supported']}, "
            f"Keyword={keyword_result['supported']}, "
            f"Embedding={embedding_result['supported']} → "
            f"Final={supported} (conf={weighted_confidence:.2f})"
        )

        return supported, weighted_confidence
