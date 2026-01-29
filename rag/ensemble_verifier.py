"""
Ensemble Verification System
Combines multiple verification strategies for robust claim verification
"""

import logging
import math
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class EnsembleVerifier:
    """Verifies claims using multiple strategies and combines results"""

    def __init__(self, llm, embedding_service):
        """
        Initialize ensemble verifier

        Args:
            llm: Language model for LLM-based verification
            embedding_service: Embedding service for similarity-based verification
        """
        self.llm = llm
        self.embedding_service = embedding_service

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
        # Strategy 1: LLM verification (most accurate but expensive)
        llm_result = self._llm_verification(claim, documents, max_chars_per_doc)

        # Strategy 2: Keyword matching (fast and cheap)
        keyword_result = self._keyword_verification(claim, documents)

        # Strategy 3: Embedding similarity (balanced)
        embedding_result = self._embedding_verification(claim, documents)

        # Combine results using weighted voting
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

            # Extract confidence
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
        # Extract keywords from claim
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

            # Jaccard similarity
            intersection = claim_keywords & doc_words
            union = claim_keywords | doc_words
            score = len(intersection) / len(union) if union else 0

            # Bonus for exact phrase match
            if claim.lower() in content:
                score += 0.3

            best_match_score = max(best_match_score, score)

        supported = best_match_score > 0.4
        confidence = min(best_match_score, 1.0)

        return {"supported": supported, "confidence": confidence, "method": "keyword"}

    def _embedding_verification(
        self, claim: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Embedding similarity-based verification (balanced)"""
        try:
            claim_embedding = self.embedding_service.generate_embedding(claim)

            best_similarity = 0.0

            for doc in documents:
                # Generate embedding for document content
                content = doc.get("content", "")
                if not content:
                    continue

                # Truncate if too long
                content_truncated = content[:2000]
                doc_embedding = self.embedding_service.generate_embedding(
                    content_truncated
                )

                similarity = self._cosine_similarity(claim_embedding, doc_embedding)
                best_similarity = max(best_similarity, similarity)

            supported = best_similarity > 0.60  # Relaxed from 0.75
            confidence = best_similarity

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
        # Weights for each method
        weights = {
            "llm": 0.5,  # Highest weight for LLM
            "keyword": 0.3,  # Medium weight for keywords
            "embedding": 0.2,  # Lower weight for embeddings
        }

        # Weighted confidence score
        weighted_confidence = (
            llm_result["confidence"] * weights["llm"]
            + keyword_result["confidence"] * weights["keyword"]
            + embedding_result["confidence"] * weights["embedding"]
        )

        # Voting: at least 2 out of 3 methods must agree
        votes = [
            llm_result["supported"],
            keyword_result["supported"],
            embedding_result["supported"],
        ]

        supported = sum(votes) >= 2

        # Adjust confidence based on agreement
        if sum(votes) == 3:
            # All agree - high confidence
            weighted_confidence = min(weighted_confidence * 1.2, 1.0)
        elif sum(votes) == 1:
            # Only one agrees - low confidence
            weighted_confidence = weighted_confidence * 0.7

        logger.info(
            f"Ensemble: LLM={llm_result['supported']}, "
            f"Keyword={keyword_result['supported']}, "
            f"Embedding={embedding_result['supported']} → "
            f"Final={supported} (conf={weighted_confidence:.2f})"
        )

        return supported, weighted_confidence
