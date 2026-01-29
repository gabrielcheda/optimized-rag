"""
Embedding Service
Generates embeddings using OpenAI API
"""

from typing import List, Optional
import logging
from openai import OpenAI
from cachetools import LRUCache
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError

from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Retry decorator for embedding calls
retry_embedding_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((
        APIError, APIConnectionError, RateLimitError, APITimeoutError,
        ConnectionError, TimeoutError
    )),
    reraise=True
)


class EmbeddingService:
    """Handles embedding generation using OpenAI API"""
    
    def __init__(self, dimensions: Optional[int] = None, cost_tracker=None):
        """Initialize OpenAI client with manual cache
        
        Args:
            dimensions: Optional embedding dimensions (512, 768, 1024, 1536).
                       Lower dimensions = faster, smaller, cheaper.
                       Recommended: 512 for 66% storage savings with minimal quality loss.
            cost_tracker: Optional CostTracker instance for tracking API costs
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = EMBEDDING_MODEL
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.dimensions = dimensions  # None = use model default (1536)
        self.cost_tracker = cost_tracker  # Track API costs
        
        # Manual cache with explicit control for hit/miss tracking
        self._cache = LRUCache(maxsize=1000)
        self._cache_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Log embedding dimension for monitoring
        embedding_dim = self.get_embedding_dimension()
        logger.info(
            f"Initialized EmbeddingService with model: {self.model} "
            f"(dimension: {embedding_dim})"
        )
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            use_cache: Whether to use LRU cache (default: True)
        
        Returns:
            List of floats representing the embedding vector
        
        Raises:
            ValueError: If text is empty or None
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        
        # OPTIMIZATION: Use cache for query embeddings (30-50% cost savings on repeated queries)
        if use_cache:
            with self._cache_lock:
                if text in self._cache:
                    self._cache_hits += 1
                    logger.debug(f"Cache hit for text of length {len(text)}")
                    return list(self._cache[text])  # Convert tuple back to list
                else:
                    self._cache_misses += 1
        
        # Generate new embedding
        embedding = self._generate_embedding_uncached(text)
        
        # Track costs if cost tracker enabled
        if self.cost_tracker:
            try:
                num_tokens = len(text.split())  # Approximate token count
                self.cost_tracker.track_embedding(
                    model=self.model,
                    num_tokens=num_tokens,
                    num_calls=1
                )
            except Exception as e:
                logger.debug(f"Cost tracking failed: {e}")
        
        # Store in cache
        if use_cache:
            with self._cache_lock:
                self._cache[text] = tuple(embedding)  # Store as tuple for memory efficiency
        
        return embedding
    
    @retry_embedding_call
    def _generate_embedding_uncached(self, text: str) -> List[float]:
        """Generate embedding without caching (with automatic retry on failure)"""
        try:
            # Call API with appropriate parameters based on model and dimensions
            if self.dimensions and "text-embedding-3" in self.model:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    dimensions=self.dimensions
                )
            else:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
            
            embedding = response.data[0].embedding
            
            # Track cost if tracker is available
            if self.cost_tracker:
                num_tokens = len(text.split())  # Rough estimate
                self.cost_tracker.track_embedding(
                    model=self.model,
                    num_tokens=num_tokens,
                    num_calls=1
                )
            
            dim_info = f" (dims={len(embedding)})" if self.dimensions else ""
            logger.debug(f"Generated embedding for text of length {len(text)}{dim_info}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding after retries: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches with optional caching
        
        OPTIMIZATION: Checks cache for each text before batching API calls.
        Saves ~30% on duplicate chunks (e.g., re-uploads, similar documents).
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use LRU cache (default True)
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if not use_cache:
            # Bypass cache for time-sensitive embeddings
            return self._generate_batch_uncached(texts)
        
        # Pre-allocate results list with proper typing
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        uncached_texts = []
        uncached_indices = []
        empty_count = 0
        
        # Check cache for each text
        for i, text in enumerate(texts):
            # Validate input
            if not text or not text.strip():
                logger.warning(f"Skipping empty text at index {i}")
                embeddings[i] = []
                empty_count += 1
                continue
            
            # Explicit cache check
            with self._cache_lock:
                if text in self._cache:
                    self._cache_hits += 1
                    embeddings[i] = list(self._cache[text])  # Convert tuple back to list
                    logger.debug(f"Cache hit for text {i+1}/{len(texts)}")
                else:
                    self._cache_misses += 1
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        
        # Batch process uncached texts
        if uncached_texts:
            valid_texts = len(texts) - empty_count
            cache_hits = valid_texts - len(uncached_texts)
            logger.info(
                f"Cache: {cache_hits}/{valid_texts} hits "
                f"({cache_hits/valid_texts*100:.1f}%). "
                f"Generating {len(uncached_texts)} new embeddings."
            )
            
            new_embeddings = self._generate_batch_uncached(uncached_texts)
            
            # Track costs for uncached embeddings
            if self.cost_tracker:
                try:
                    total_tokens = sum(len(t.split()) for t in uncached_texts)
                    self.cost_tracker.track_embedding(
                        model=self.model,
                        num_tokens=total_tokens,
                        num_calls=len(uncached_texts)
                    )
                except Exception as e:
                    logger.debug(f"Cost tracking failed: {e}")
            
            # Insert new embeddings and populate cache
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                with self._cache_lock:
                    self._cache[texts[idx]] = tuple(emb)  # Store as tuple
        else:
            logger.info(f"Cache: 100% hit rate - zero API calls!")
        
        # Log cache statistics periodically (every 100 operations)
        total_ops = self._cache_hits + self._cache_misses
        if total_ops > 0 and total_ops % 100 == 0:
            stats = self.get_cache_stats()
            logger.info(
                f"Cache Stats: {stats['hit_rate_percent']} hit rate, "
                f"{stats['current_size']}/{stats['max_size']} entries"
            )
        
        return embeddings
    
    @retry_embedding_call
    def _generate_batch_uncached(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings without cache (with automatic retry on failure)"""
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Call API with appropriate parameters based on model and dimensions
                if self.dimensions and "text-embedding-3" in self.model:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                        dimensions=self.dimensions
                    )
                else:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model
                    )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Track cost if tracker is available
                if self.cost_tracker:
                    total_tokens = sum(len(text.split()) for text in batch)  # Rough estimate
                    self.cost_tracker.track_embedding(
                        model=self.model,
                        num_tokens=total_tokens,
                        num_calls=len(batch)
                    )
                
                dim_info = f" (dims={len(batch_embeddings[0])})" if batch_embeddings and self.dimensions else ""
                logger.debug(f"Generated {len(batch_embeddings)} embeddings{dim_info} (batch {i//self.batch_size + 1})")
            
            logger.info(f"Generated {len(all_embeddings)} embeddings total")
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for monitoring performance
        
        Returns:
            Dict with hits, misses, hit_rate, size, and maxsize
        """
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0
            current_size = len(self._cache)
            max_size = self._cache.maxsize
            
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "hit_rate_percent": f"{hit_rate * 100:.1f}%",
                "current_size": current_size,
                "max_size": max_size,
                "cache_full": current_size >= max_size
            }
    
    def clear_cache(self) -> None:
        """
        Clear the embedding cache
        
        Useful for:
        - Freeing memory in long-running processes
        - Testing without cache interference
        - Resetting after model changes
        """
        # Log stats before clearing
        stats = self.get_cache_stats()
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        logger.info(
            f"Embedding cache cleared. Previous stats: {stats['hit_rate_percent']} hit rate, "
            f"{stats['current_size']} entries"
        )
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension (custom if set, otherwise model default)
        """
        # Return custom dimensions if configured
        if self.dimensions:
            return self.dimensions
        
        # OpenAI embedding models default dimensions
        if "ada-002" in self.model or "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-3-large" in self.model:
            return 3072
        else:
            # For other models, generate a test embedding
            logger.warning(f"Unknown embedding dimension for model {self.model}, testing...")
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
