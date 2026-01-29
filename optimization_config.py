"""
Configuration for Optimizations
Centralized settings for new optimization features
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class OptimizationSettings(BaseSettings):
    """Settings for optimization features"""
    
    # Embedding Optimizations
    embedding_dimensions: Optional[int] = Field(
        default=None,
        description="Embedding dimensions (512, 768, 1024, 1536). None = model default. "
                    "Recommended: 512 for 66% storage savings with minimal quality loss."
    )
    embedding_cache_size_mb: int = Field(
        default=50,
        description="Embedding cache size in MB (default: 50MB = ~8000 embeddings)"
    )
    enable_persistent_cache: bool = Field(
        default=False,
        description="Enable persistent disk cache for embeddings (survives restarts)"
    )
    persistent_cache_dir: str = Field(
        default=".cache/embeddings",
        description="Directory for persistent embedding cache"
    )
    
    # Reranking Optimizations
    enable_selective_reranking: bool = Field(
        default=True,
        description="Enable selective reranking (skip when not needed)"
    )
    reranking_score_variance_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Score variance threshold for reranking decision"
    )
    
    # Anti-Hallucination Optimizations
    enable_ensemble_verification: bool = Field(
        default=True,
        description="Enable ensemble verification (LLM + keywords + embeddings)"
    )
    ensemble_llm_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for LLM verification in ensemble"
    )
    ensemble_keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword verification in ensemble"
    )
    ensemble_embedding_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for embedding verification in ensemble"
    )
    claim_verification_context_chars: int = Field(
        default=2000,
        description="Max characters per document for claim verification (increased from 1000)"
    )
    
    # Cost Tracking
    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable detailed cost tracking"
    )
    max_daily_cost: float = Field(
        default=10.0,
        description="Maximum daily cost in USD (safety limit)"
    )
    
    # Performance Optimizations
    enable_parallel_retrieval: bool = Field(
        default=False,
        description="Enable parallel retrieval from multiple sources (requires async)"
    )
    enable_streaming_response: bool = Field(
        default=False,
        description="Enable streaming response generation (requires async)"
    )


# Create global optimization settings instance
optimization_settings = OptimizationSettings()

# Export for backward compatibility
EMBEDDING_DIMENSIONS = optimization_settings.embedding_dimensions
EMBEDDING_CACHE_SIZE_MB = optimization_settings.embedding_cache_size_mb
ENABLE_PERSISTENT_CACHE = optimization_settings.enable_persistent_cache
PERSISTENT_CACHE_DIR = optimization_settings.persistent_cache_dir
ENABLE_SELECTIVE_RERANKING = optimization_settings.enable_selective_reranking
ENABLE_ENSEMBLE_VERIFICATION = optimization_settings.enable_ensemble_verification
CLAIM_VERIFICATION_CONTEXT_CHARS = optimization_settings.claim_verification_context_chars
ENABLE_COST_TRACKING = optimization_settings.enable_cost_tracking
MAX_DAILY_COST = optimization_settings.max_daily_cost
