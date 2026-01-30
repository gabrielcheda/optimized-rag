"""
Context Compressor
Compresses retrieved context to essential information, reducing tokens while maintaining relevance.

Paper recommendation: Context compression is CRITICAL for:
- Reducing API costs (less tokens)
- Improving response quality (less noise)
- Increasing relevance (focus on essentials)
- Allowing more documents in context
"""

from typing import List, Dict, Any, Tuple
import logging
import re

import config
from rag.models.intent_analysis import QueryIntent
from memory.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Compresses context using relevance-based sentence selection"""
    
    def __init__(
        self, 
        max_tokens: int = 2000, 
        sentences_per_doc: int = 5,
        embedding_service: EmbeddingService | None = None
    ):
        """
        Initialize context compressor
        
        Args:
            max_tokens: Maximum tokens for compressed context
            sentences_per_doc: Number of top sentences to keep per document
            embedding_service: Optional embedding service for semantic scoring (Phase 2)
        """
        self.max_tokens = max_tokens
        self.sentences_per_doc = sentences_per_doc
        self.embedding_service = embedding_service
        
        # Phase 2: Hybrid scoring (70% semantic + 30% lexical)
        self.use_semantic_scoring = embedding_service is not None
        self.semantic_weight = 0.7
        self.lexical_weight = 0.3
        
        logger.info(
            f"ContextCompressor initialized: max_tokens={max_tokens}, "
            f"sentences_per_doc={sentences_per_doc}, "
            f"semantic_scoring={'enabled' if self.use_semantic_scoring else 'disabled'}"
        )
    
    def compress(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_tokens: int | None = None,
        query_intent: QueryIntent = QueryIntent.QUESTION_ANSWERING
    ) -> List[Dict[str, Any]]:
        """
        Compress context by selecting most relevant sentences
        
        Paper recommendation: Extract only query-relevant sentences,
        removing redundancy and noise.
        
        Args:
            query: User query
            documents: Retrieved documents to compress
            max_tokens: Override default max_tokens
            query_intent: Query intent for dynamic threshold (qa, chat, search, etc.)
            
        Returns:
            List of compressed documents with metadata
        """
        if not documents:
            return []
        
        # OPTIMIZATION: Quality pre-filtering - skip compression on low-relevance docs
        # Dynamic threshold based on query intent (INCREASED for anti-hallucination)
        base_threshold = config.COMPRESSION_INTENT_THRESHOLDS.get(
            query_intent.value if hasattr(query_intent, 'value') else str(query_intent),
            0.45  # Default
        )
        
        # CRITICAL FIX: If very few documents, lower threshold dramatically
        # (cross-language queries have low embedding scores but valid content)
        if len(documents) <= 5:
            max_doc_score = max((d.get('score', 0) for d in documents), default=0)
            if max_doc_score < 0.1:  # All docs have terrible embedding scores
                relevance_threshold = config.COMPRESSION_MIN_THRESHOLD  # Accept anything above 0.5%
                logger.warning(f"Very low document scores (max={max_doc_score:.3f}), using minimal threshold {relevance_threshold}")
            else:
                relevance_threshold = base_threshold
        else:
            relevance_threshold = base_threshold
        
        # Filter out low-quality documents before compression
        filtered_docs = []
        for doc in documents:
            relevance = doc.get('score', 1.0)  # CrossEncoder score from reranking
            if relevance >= relevance_threshold:
                filtered_docs.append(doc)
            else:
                logger.info(
                    f"Skipping compression for low-relevance doc "
                    f"(score={relevance:.3f} < threshold={relevance_threshold})"
                )
        
        if not filtered_docs:
            logger.warning(f"All documents below relevance threshold ({relevance_threshold}), returning empty context")
            return []
        
        documents = filtered_docs
        
        max_tokens = max_tokens or self.max_tokens
        compressed = []
        total_original_length = 0
        total_compressed_length = 0
        
        for doc in documents:
            content = doc.get('content', '')
            total_original_length += len(content)
            
            # Split into sentences
            sentences = self._split_sentences(content)
            
            if not sentences:
                continue
            
            # Phase 2: Score each sentence using hybrid approach
            if self.use_semantic_scoring:
                scored_sentences = self._score_sentences_hybrid(query, sentences)
            else:
                # Fallback to lexical only
                scored_sentences = []
                for sent in sentences:
                    score = self._score_sentence_lexical(query, sent)
                    scored_sentences.append((sent, score))
            
            # Select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:self.sentences_per_doc]]
            
            # Reconstruct in original order (preserve flow)
            top_sentences_set = set(top_sentences)
            ordered_sentences = [s for s in sentences if s in top_sentences_set]
            
            compressed_content = ' '.join(ordered_sentences)
            total_compressed_length += len(compressed_content)
            
            # Create compressed document
            compressed_doc = {
                **doc,
                'content': compressed_content,
                'original_content': content,
                'compressed': True,
                'original_length': len(content),
                'compressed_length': len(compressed_content),
                'compression_ratio': len(compressed_content) / len(content) if len(content) > 0 else 0,
                'sentences_kept': len(ordered_sentences),
                'sentences_total': len(sentences)
            }
            
            compressed.append(compressed_doc)
        
        # Calculate stats
        tokens_saved = total_original_length - total_compressed_length
        compression_ratio = total_compressed_length / total_original_length if total_original_length > 0 else 0
        
        logger.info(
            f"Compressed {len(documents)} documents: "
            f"{total_original_length} → {total_compressed_length} chars "
            f"({compression_ratio:.1%} ratio, ~{tokens_saved} chars saved)"
        )
        
        return compressed
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Uses simple regex splitting on punctuation.
        Can be enhanced with NLTK or spaCy for better accuracy.
        """
        if not text:
            return []
        
        # Split on sentence-ending punctuation followed by space
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter out very short sentences (likely fragments)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def _score_sentences_hybrid(self, query: str, sentences: List[str]) -> List[Tuple[str, float]]:
        """
        Phase 2: Score sentences using hybrid semantic + lexical approach
        
        Args:
            query: User query
            sentences: List of sentences to score
            
        Returns:
            List of (sentence, score) tuples
        """
        try:
            # Type guard: ensure embedding_service is available
            if not self.embedding_service:
                raise ValueError("Embedding service not available")
            
            # Compute query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Compute sentence embeddings in batch
            sentence_embeddings = self.embedding_service.generate_embeddings_batch(sentences)
            
            scored = []
            for sent, sent_emb in zip(sentences, sentence_embeddings):
                # Semantic score (cosine similarity)
                semantic_score = self._cosine_similarity(query_embedding, sent_emb)
                
                # Lexical score (keyword overlap)
                lexical_score = self._score_sentence_lexical(query, sent)
                
                # Hybrid score (70% semantic + 30% lexical)
                hybrid_score = (
                    self.semantic_weight * semantic_score + 
                    self.lexical_weight * lexical_score
                )
                
                scored.append((sent, hybrid_score))
            
            return scored
            
        except Exception as e:
            logger.error(f"Semantic scoring failed, falling back to lexical: {e}")
            # Fallback to lexical scoring
            return [(sent, self._score_sentence_lexical(query, sent)) for sent in sentences]
    
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
    
    def _score_sentence_lexical(self, query: str, sentence: str) -> float:
        """
        Score sentence relevance to query
        
        Simple keyword overlap method. Can be enhanced with:
        - Embedding-based similarity
        - TF-IDF weighting
        - Named entity matching
        
        Args:
            query: User query
            sentence: Sentence to score
            
        Returns:
            Relevance score (0.0-1.0)
        """
        # Normalize
        query_lower = query.lower()
        sentence_lower = sentence.lower()
        
        # Extract words (remove punctuation)
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        sent_words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        # Remove common stop words (simple list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
        query_words -= stop_words
        sent_words -= stop_words
        
        if not query_words:
            return 0.0
        
        # Calculate keyword overlap
        overlap = len(query_words & sent_words)
        score = overlap / len(query_words)
        
        # Bonus for exact phrase match
        if query_lower in sentence_lower:
            score += 0.2
        
        # Normalize to 0-1
        return min(score, 1.0)
    
    def get_compression_stats(self, compressed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get compression statistics
        
        Args:
            compressed_docs: List of compressed documents
            
        Returns:
            Dictionary with compression statistics
        """
        if not compressed_docs:
            return {
                'total_documents': 0,
                'total_original_length': 0,
                'total_compressed_length': 0,
                'tokens_saved': 0,
                'compression_ratio': 0,
                'avg_sentences_kept': 0
            }
        
        total_original = sum(d.get('original_length', 0) for d in compressed_docs)
        total_compressed = sum(d.get('compressed_length', 0) for d in compressed_docs)
        avg_sentences = sum(d.get('sentences_kept', 0) for d in compressed_docs) / len(compressed_docs)
        
        return {
            'total_documents': len(compressed_docs),
            'total_original_length': total_original,
            'total_compressed_length': total_compressed,
            'tokens_saved': total_original - total_compressed,
            'compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'avg_sentences_kept': avg_sentences
        }
