"""
Hybrid Retrieval
Combines vector search, keyword search (BM25), and multi-source retrieval
"""

from typing import List, Dict, Any, Optional
import logging
import math

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    FASE 6: Hybrid retrieval with adaptive weights by intent

    Different query intents benefit from different balances between
    semantic and keyword search. FASE 6 optimizes weights for precision.
    """

    # FASE 6: Intent-specific weight configurations for precision
    INTENT_WEIGHTS = {
        # Factual queries: Higher semantic weight for meaning, keyword for exact terms
        'question_answering': {'alpha': 0.55, 'beta': 0.40, 'gamma': 0.05},
        'fact_checking': {'alpha': 0.50, 'beta': 0.45, 'gamma': 0.05},

        # Multi-hop: Need diverse results, semantic important
        'multi_hop_reasoning': {'alpha': 0.60, 'beta': 0.30, 'gamma': 0.10},

        # Comparison: Need to find both entities
        'comparison': {'alpha': 0.50, 'beta': 0.45, 'gamma': 0.05},

        # Summarization: Need broad coverage
        'summarization': {'alpha': 0.65, 'beta': 0.25, 'gamma': 0.10},

        # Search: Keyword more important
        'search': {'alpha': 0.45, 'beta': 0.50, 'gamma': 0.05},

        # Clarification: Use conversation context (handled separately)
        'clarification': {'alpha': 0.70, 'beta': 0.20, 'gamma': 0.10},

        # Conversational: Semantic meaning more important
        'conversational': {'alpha': 0.70, 'beta': 0.20, 'gamma': 0.10},

        # Default/fallback
        'default': {'alpha': 0.55, 'beta': 0.35, 'gamma': 0.10},
    }

    def __init__(
        self,
        memory_manager,
        document_store,
        agent_id: str,
        alpha: float = 0.55,     # FASE 6: Adjusted default
        beta: float = 0.35,      # FASE 6: Adjusted default
        gamma: float = 0.10,     # FASE 6: Adjusted default
        weight_manager=None,
        use_adaptive_weights: bool = True  # FASE 6: Enable adaptive weights
    ):
        """
        Initialize hybrid retriever

        Args:
            memory_manager: MemoryManager instance
            document_store: DocumentStore instance
            agent_id: Agent ID for multi-tenant isolation
            alpha: Weight for semantic search (used if no adaptive weights)
            beta: Weight for keyword search (used if no adaptive weights)
            gamma: Weight for freshness/recency (used if no adaptive weights)
            weight_manager: DynamicWeightManager for adaptive weights (optional)
            use_adaptive_weights: FASE 6 - Use intent-specific weights
        """
        self.memory_manager = memory_manager
        self.document_store = document_store
        self.agent_id = agent_id
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_manager = weight_manager
        self.use_adaptive_weights = use_adaptive_weights

        # Try to load BM25
        self.bm25_available = self._check_bm25()

        logger.info(
            f"FASE 6 HybridRetriever initialized: adaptive_weights={use_adaptive_weights}, "
            f"default_weights=({alpha:.2f}, {beta:.2f}, {gamma:.2f})"
        )

    def get_weights_for_intent(self, intent: str) -> tuple:
        """
        FASE 6: Get optimal weights based on query intent

        Args:
            intent: Query intent string (e.g., 'question_answering', 'search')

        Returns:
            Tuple of (alpha, beta, gamma) weights
        """
        # Normalize intent string
        intent_key = intent.lower().replace(' ', '_') if intent else 'default'

        # Get intent-specific weights or fallback to default
        weights = self.INTENT_WEIGHTS.get(intent_key, self.INTENT_WEIGHTS['default'])

        logger.debug(
            f"FASE 6 weights for intent '{intent_key}': "
            f"alpha={weights['alpha']}, beta={weights['beta']}, gamma={weights['gamma']}"
        )

        return weights['alpha'], weights['beta'], weights['gamma']
    
    def _check_bm25(self) -> bool:
        """Check if BM25 is available"""
        try:
            from rank_bm25 import BM25Okapi
            return True
        except ImportError:
            logger.warning("rank-bm25 not available. Install with: pip install rank-bm25")
            return False
    
    def retrieve(
        self,
        query: str,
        sources: List[str],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from multiple sources
        
        Args:
            query: Search query
            sources: List of source names ('archival', 'documents', 'conversation')
            top_k: Number of results per source
        
        Returns:
            Combined results from all sources
        """
        all_results = []
        
        # Retrieve from each source
        if 'archival' in sources or 'archival_memory' in sources:
            archival_results = self._retrieve_archival(query, top_k)
            all_results.extend(archival_results)
        
        if 'documents' in sources:
            doc_results = self._retrieve_documents(query, top_k)
            all_results.extend(doc_results)
        
        if 'conversation' in sources or 'conversation_history' in sources:
            conv_results = self._retrieve_conversation(query, top_k)
            all_results.extend(conv_results)
        
        logger.info(f"Retrieved {len(all_results)} total results from {len(sources)} sources")
        
        return all_results
    
    def _retrieve_archival(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from archival memory"""
        try:
            results = self.memory_manager.archival_memory_search(query, top_k=top_k)
            
            # Add source metadata
            for result in results:
                result['source'] = 'archival_memory'
            
            return results
        except Exception as e:
            logger.error(f"Archival retrieval failed: {e}")
            return []
    
    def _retrieve_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from document store"""
        try:
            results = self.document_store.search(
                agent_id=self.agent_id,
                query=query,
                top_k=top_k
            )
            
            # Add source metadata
            for result in results:
                result['source'] = 'documents'
            
            return results
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _retrieve_conversation(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve from conversation history"""
        try:
            conversation_id = self.memory_manager.agent_id
            results = self.memory_manager.conversation_search(conversation_id, query, limit=top_k)
            
            # Format as standard results
            formatted = []
            for msg in results:
                formatted.append({
                    'content': msg['content'],
                    'source': 'conversation_history',
                    'metadata': {
                        'role': msg['role'],
                        'timestamp': msg.get('created_at', '')
                    },
                    'similarity': 0.5  # Default score
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Conversation retrieval failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        corpus: List[str],
        embeddings: List[List[float]],
        query_embedding: List[float],
        top_k: int = 10,
        documents_metadata: Optional[List[Dict[str, Any]]] = None,
        query_intent: Optional[str] = None  # FASE 6: Intent for adaptive weights
    ) -> List[Dict[str, Any]]:
        """
        FASE 6: Hybrid search with adaptive weights based on query intent

        Args:
            query: Search query
            corpus: List of documents
            embeddings: Document embeddings
            query_embedding: Query embedding
            top_k: Number of results
            documents_metadata: Optional metadata with timestamps for temporal boost
            query_intent: FASE 6 - Query intent for adaptive weight selection

        Returns:
            Hybrid ranked results with temporal awareness
        """
        from datetime import datetime, timedelta
        import config

        # FASE 6: Get adaptive weights based on intent
        if self.use_adaptive_weights and query_intent:
            alpha, beta, gamma = self.get_weights_for_intent(query_intent)
            logger.info(
                f"FASE 6: Using adaptive weights for '{query_intent}': "
                f"semantic={alpha:.2f}, keyword={beta:.2f}, temporal={gamma:.2f}"
            )
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        # Semantic scores
        semantic_scores = []
        for doc_emb in embeddings:
            similarity = self._cosine_similarity(query_embedding, doc_emb)
            semantic_scores.append(similarity)
        
        # Keyword scores (BM25 if available)
        if self.bm25_available:
            keyword_scores = self._bm25_scores(query, corpus)
        else:
            # Fallback: simple keyword overlap
            keyword_scores = self._simple_keyword_scores(query, corpus)
        
        # Paper-compliant: Temporal Awareness (recency boost)
        temporal_scores = []
        if documents_metadata and config.ENABLE_TEMPORAL_BOOST:
            current_time = datetime.now()
            for metadata in documents_metadata:
                # Extract timestamp from metadata
                timestamp = metadata.get('created_at') or metadata.get('uploaded_at')
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            timestamp = None
                    
                    if timestamp:
                        # Calculate time decay using exponential decay
                        days_old = (current_time - timestamp).total_seconds() / 86400
                        half_life = config.RECENCY_HALF_LIFE_DAYS
                        decay_factor = 0.5 ** (days_old / half_life)
                        temporal_score = config.RECENCY_WEIGHT * decay_factor
                    else:
                        temporal_score = 0.0
                else:
                    temporal_score = 0.0
                
                temporal_scores.append(temporal_score)
        else:
            temporal_scores = [0.0] * len(corpus)
        
        # FASE 6: Combine scores with adaptive weights
        hybrid_scores = []
        for i in range(len(corpus)):
            semantic = semantic_scores[i]
            keyword = keyword_scores[i]
            temporal = temporal_scores[i]

            # FASE 6: Use adaptive weights (alpha, beta, gamma from intent)
            hybrid_score = alpha * semantic + beta * keyword + gamma * temporal
            
            result = {
                'content': corpus[i],
                'hybrid_score': hybrid_score,
                'semantic_score': semantic,
                'keyword_score': keyword,
                'temporal_score': temporal,
                'embedding': embeddings[i]
            }
            
            # Preserve metadata
            if documents_metadata and i < len(documents_metadata):
                result['metadata'] = documents_metadata[i]
            
            hybrid_scores.append(result)
        
        # Sort and return top_k
        ranked = sorted(hybrid_scores, key=lambda x: x['hybrid_score'], reverse=True)
        
        return ranked[:top_k]
    
    def _bm25_scores(self, query: str, corpus: List[str]) -> List[float]:
        """Calculate BM25 scores"""
        from rank_bm25 import BM25Okapi
        
        # Edge case: empty corpus or all whitespace documents
        if not corpus or all(len(doc.split()) == 0 for doc in corpus):
            logger.warning("BM25: Empty or whitespace-only corpus, returning zeros")
            return [0.0] * len(corpus)
        
        # Tokenize
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        tokenized_query = query.lower().split()
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # Normalize to 0-1
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        normalized = [float(s / max_score) for s in scores]
        
        return normalized
    
    def _simple_keyword_scores(self, query: str, corpus: List[str]) -> List[float]:
        """Simple keyword overlap scoring (fallback)"""
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in corpus:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / len(query_terms) if query_terms else 0.0
            scores.append(score)
        
        return scores
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
