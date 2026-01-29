"""
OpenAI Reranking
Advanced reranking using OpenAI embeddings with MMR and RRF
"""

from typing import List, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)


class OpenAIReranker:
    """Rerank using OpenAI embeddings for improved relevance"""
    
    def __init__(self, openai_client, model: str = "text-embedding-3-large"):
        """
        Initialize OpenAI reranker
        
        Args:
            openai_client: OpenAI client instance
            model: Embedding model (text-embedding-3-large for best quality)
        """
        self.client = openai_client
        self.model = model
        logger.info(f"Initialized OpenAI reranker with {model}")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using OpenAI embeddings
        
        Args:
            query: User query
            results: Retrieved results
            top_k: Number to return
        
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        try:
            # OPTIMIZATION: Batch embed query + all documents in 1-2 API calls
            # Reduces 10-15 individual embedding calls to 1-2 batched calls
            # Saves ~$0.0013/query and ~1s processing time
            
            # Prepare all texts for batch embedding
            contents = [query] + [r.get('content', '')[:8000] for r in results]
            
            # Batch embed (OpenAI supports up to 2048 texts per call)
            batch_response = self.client.embeddings.create(
                input=contents,
                model=self.model
            )
            
            # Extract embeddings
            query_emb = batch_response.data[0].embedding
            content_embeddings = [batch_response.data[i+1].embedding for i in range(len(results))]
            
            # Calculate refined scores
            for i, result in enumerate(results):
                content_emb = content_embeddings[i]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_emb, content_emb)
                
                # Combine with original score if exists
                original_score = result.get('similarity', 0) or result.get('score', 0)
                
                # Weighted combination: 70% new embedding, 30% original
                result['rerank_score'] = 0.7 * similarity + 0.3 * original_score
                result['embedding'] = content_emb
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"OpenAI batch-reranked {len(results)} results (1 API call vs {len(results)+1} individual)")
            
            return reranked[:top_k]
        
        except Exception as e:
            logger.error(f"OpenAI reranking failed: {e}")
            # Fallback: return original order
            return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class MMRDiversifier:
    """Maximal Marginal Relevance for diversity"""
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR diversifier
        
        Args:
            lambda_param: Balance between relevance (high) and diversity (low)
        """
        self.lambda_param = lambda_param
    
    def diversify(
        self,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply MMR to diversify results
        
        Args:
            query_embedding: Query embedding vector
            results: Results with 'embedding' field
            top_k: Number to return
        
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        # Edge case: filter out invalid embeddings (empty, wrong type, or containing NaN/Inf)
        import math
        valid_results = [
            r for r in results
            if r.get('embedding') and
               isinstance(r['embedding'], list) and
               len(r['embedding']) > 0 and
               all(isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v) for v in r['embedding'])
        ]
        
        if not valid_results:
            logger.warning("MMR: No valid embeddings found, returning original results")
            return results[:top_k]
        
        if len(valid_results) < len(results):
            logger.warning(f"MMR: Filtered {len(results) - len(valid_results)} results with invalid embeddings")
        
        selected = []
        remaining = valid_results.copy()
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for doc in remaining:
                doc_emb = doc.get('embedding')
                if not doc_emb:
                    # If no embedding, use rerank_score or 0
                    mmr_scores.append((doc.get('rerank_score', 0), doc))
                    continue
                
                # Relevance to query
                relevance = self._cosine_similarity(query_embedding, doc_emb)
                
                # Diversity from selected documents
                if selected:
                    max_similarity = max([
                        self._cosine_similarity(doc_emb, s.get('embedding', []))
                        for s in selected
                        if s.get('embedding')
                    ])
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # MMR score
                mmr_score = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                mmr_scores.append((mmr_score, doc))
            
            if not mmr_scores:
                break
            
            # Select best MMR score
            best_score, best_doc = max(mmr_scores, key=lambda x: x[0])
            best_doc['mmr_score'] = best_score
            selected.append(best_doc)
            remaining.remove(best_doc)
        
        logger.info(f"MMR diversified to {len(selected)} results")
        
        return selected
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class ReciprocalRankFusion:
    """Combine rankings from multiple sources using RRF"""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF
        
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        result_lists: List[List[Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion
        
        Args:
            result_lists: List of result lists from different sources
            top_k: Number to return
        
        Returns:
            Fused results
        """
        # Track RRF scores by content (using content as key)
        rrf_scores = {}
        doc_map = {}
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                content = doc.get('content', '')
                
                # RRF formula: score = sum(1 / (k + rank))
                rrf_score = 1 / (self.k + rank)
                
                if content in rrf_scores:
                    rrf_scores[content] += rrf_score
                else:
                    rrf_scores[content] = rrf_score
                    doc_map[content] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_map.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True
        )
        
        # Add RRF score to results
        fused_results = []
        for content, doc in sorted_docs[:top_k]:
            doc['rrf_score'] = rrf_scores[content]
            fused_results.append(doc)
        
        logger.info(f"RRF fused {len(result_lists)} lists into {len(fused_results)} results")
        
        return fused_results


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker (Paper-compliant: superior accuracy)
    
    Uses sentence-transformers cross-encoder models for accurate relevance scoring.
    Cross-encoders jointly encode query-document pairs for direct relevance prediction,
    achieving higher accuracy than bi-encoder (embedding) approaches.
    
    Key advantages:
    - Direct query-document interaction (not just vector similarity)
    - State-of-the-art reranking accuracy
    - Typically used as final reranking stage after initial retrieval
    
    References: Paper Section on Advanced Reranking Techniques
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512
    ):
        """
        Initialize Cross-Encoder reranker
        
        Args:
            model_name: HuggingFace model name (default: MS-MARCO trained model)
            max_length: Maximum sequence length for encoding
        
        Popular models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
        - cross-encoder/ms-marco-TinyBERT-L-2-v2 (very fast, lower quality)
        - cross-encoder/ms-marco-electra-base (slower, best quality)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            self.model = CrossEncoder(model_name, max_length=max_length)
            logger.info(f"Initialized CrossEncoder: {model_name}")
        except ImportError as e:
            logger.warning(f"sentence-transformers not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}", exc_info=True)
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using Cross-Encoder
        
        Args:
            query: User query
            results: Retrieved results with 'content' field
            top_k: Number of top results to return
        
        Returns:
            Reranked results with 'cross_encoder_score' field
        """
        if not results:
            return []
        
        if self.model is None:
            logger.warning("CrossEncoder not available, returning original results")
            return results[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                content = result.get('content', '')
                # Truncate content if too long
                if len(content) > 2000:
                    content = content[:2000]
                pairs.append([query, content])
            
            # Predict relevance scores (range: typically -10 to 10, higher = more relevant)
            scores = self.model.predict(pairs)
            
            # Normalize scores to 0-1 range using sigmoid
            import math
            normalized_scores = [1 / (1 + math.exp(-score)) for score in scores]
            
            # Add scores to results
            for result, score, norm_score in zip(results, scores, normalized_scores):
                # CRITICAL: Preserve original embedding score before overwriting
                if 'score' in result and 'embedding_score' not in result:
                    result['embedding_score'] = result['score']
                
                # Update main score with CrossEncoder (higher quality)
                result['score'] = float(norm_score)
                result['cross_encoder_score'] = float(norm_score)
                result['cross_encoder_raw_score'] = float(score)
            
            # Sort by cross-encoder score
            reranked = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
            
            logger.info(
                f"CrossEncoder reranked {len(results)} docs, "
                f"top score: {reranked[0]['cross_encoder_score']:.3f}"
            )
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
            return results[:top_k]
    
    def is_available(self) -> bool:
        """Check if CrossEncoder is available"""
        return self.model is not None
