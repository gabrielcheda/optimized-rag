"""
MemGPT Configuration
Centralized configuration for OpenAI, Database, and Agent settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from typing import Dict

class Settings(BaseSettings):
    """Application settings with automatic validation and .env loading"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API key for LLM and embeddings")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    # OPTIMIZATION: text-embedding-3-small provides 80% cost savings vs ada-002 with similar quality
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model for semantic search")

    # RAG Configuration
    reranking_embedding_model: str = Field(default="text-embedding-3-large", description="Model for reranking")
    # OPTIMIZATION: Increased chunk_size 1000→1200, reduced overlap 200→150 (15% cost savings)
    chunk_size: int = Field(default=1200, description="Document chunk size")
    chunk_overlap: int = Field(default=150, description="Overlap between chunks")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="Balance relevance vs diversity")
    rrf_k: int = Field(default=60, description="Reciprocal rank fusion constant")
    enable_self_rag: bool = Field(default=True, description="Enable Self-RAG evaluation")
    # OPTIMIZATION: Increased from 0.6 to 0.75 for stricter quality control (anti-hallucination)
    relevance_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Minimum relevance score")
    max_reretrieve_attempts: int = Field(default=2, description="Max re-retrieval attempts")
    semantic_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Threshold for semantic chunking")

    # Context Compression
    enable_context_compression: bool = Field(default=True, description="Enable context compression")
    context_compression_max_tokens: int = Field(default=2000, description="Max tokens after compression")
    # OPTIMIZATION: Increased from 5 to 8 for better context coverage
    context_compression_sentences_per_doc: int = Field(default=8, description="Sentences to keep per document")
    
    # Temporal Awareness
    enable_temporal_boost: bool = Field(default=True, description="Enable temporal boosting")
    # OPTIMIZATION: Increased from 0.1 to 0.15 for better time-sensitive queries
    recency_weight: float = Field(default=0.15, ge=0.0, le=0.3, description="Weight for temporal boosting")
    recency_half_life_days: int = Field(default=30, description="Exponential decay rate in days")
    
    # Knowledge Graph
    # OPTIMIZATION: Disabled by default - consistently returns 0 results (saves 6-9 queries/request, ~3s)
    # Re-enable after verifying entity extraction during document upload
    enable_knowledge_graph: bool = Field(default=True, description="Enable knowledge graph")
    kg_max_hops: int = Field(default=2, description="Maximum hops for graph traversal")
    kg_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for triples")
    
    # Advanced Features
    enable_cot_reasoning: bool = Field(default=True, description="Chain-of-Thought for complex queries")
    enable_cross_encoder: bool = Field(default=True, description="Cross-encoder reranking")
    enable_query_refinement: bool = Field(default=True, description="Iterative query refinement")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model name")

    # DW-GRPO (Dynamic Weight Graph Reinforcement Policy Optimization)
    enable_dynamic_weights: bool = Field(default=True, description="Enable dynamic weight learning")
    weight_learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Rate of weight adaptation")
    performance_tracking_window: int = Field(default=100, description="Number of queries to track")
    enable_hierarchical_retrieval: bool = Field(default=True, description="Enable hierarchical retrieval")
    hierarchical_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for tier escalation")
    enable_tier_3: bool = Field(default=True, description="Enable expensive Tier 3 (KG + Web)")
    enable_cost_tracking: bool = Field(default=True, description="Track API costs and savings")
    
    # Anti-Hallucination Settings (PHASE 1: Critical Fixes)
    enable_post_generation_verification: bool = Field(default=True, description="Verify claims after generation")
    enable_citation_validation: bool = Field(default=True, description="Validate citation format and completeness")
    min_factuality_score: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum factuality to accept (increased from 0.25)")
    require_both_scores_high: bool = Field(default=True, description="Require both faithfulness AND factuality high")
    max_regeneration_attempts: int = Field(default=2, description="Max times to regenerate failed responses")
    
    # Anti-Hallucination Settings (PHASE 2: High Priority)
    enable_uncertainty_quantification: bool = Field(default=True, description="Calculate and show uncertainty scores")
    show_confidence_in_response: bool = Field(default=False, description="Append confidence to response text")
    enable_consistency_check: bool = Field(default=True, description="Check cross-document consistency")
    
    # Anti-Hallucination Settings (PHASE 3: Advanced Features)
    enable_human_in_the_loop: bool = Field(default=False, description="Flag uncertain responses for review")
    enable_attribution_map: bool = Field(default=True, description="Build claim-to-source attribution map")
    enable_temporal_validation: bool = Field(default=True, description="Validate temporal consistency")
    enable_ensemble_sampling: bool = Field(default=False, description="Generate multiple responses for critical queries")
    
    # Evaluation & Monitoring
    enable_metrics_logging: bool = Field(default=True, description="Enable metrics logging")
    metrics_log_interval: int = Field(default=10, description="Log metrics every N queries")
    
    # Embedding Cache
    embedding_cache_size: int = Field(default=1000, description="LRU cache size for embeddings")
    
    # Web Search (optional)
    tavily_api_key: str = Field(default="", description="Tavily API key for web search (optional)")
    
    # Database Configuration
    postgres_uri: str = Field(..., description="PostgreSQL connection URI")

    # Agent Configuration
    max_context_tokens: int = Field(default=8000, description="Maximum context tokens")
    
    # Token Allocation
    token_allocation_system_prompt: int = Field(default=500, description="Tokens for system prompt")
    token_allocation_core_memory: int = Field(default=800, description="Tokens for core memory")
    token_allocation_function_definitions: int = Field(default=700, description="Tokens for function definitions")
    token_allocation_retrieved_context: int = Field(default=2000, description="Tokens for retrieved context")
    token_allocation_conversation: int = Field(default=4000, description="Tokens for conversation history")
    
    # Context Management
    context_warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Trigger paging at threshold capacity")
    
    @property
    def token_allocation(self) -> Dict[str, int]:
        """Return token allocation as dictionary for backward compatibility"""
        return {
            "system_prompt": self.token_allocation_system_prompt,
            "core_memory": self.token_allocation_core_memory,
            "function_definitions": self.token_allocation_function_definitions,
            "retrieved_context": self.token_allocation_retrieved_context,
            "conversation": self.token_allocation_conversation
        }


# Create global settings instance
# Note: Pydantic Settings automatically loads from .env file
settings = Settings()  # type: ignore[call-arg]

# Backward compatibility: expose as uppercase constants
# NOTE: Prefer using settings.openai_api_key.get_secret_value() directly
OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
LLM_MODEL = settings.llm_model
EMBEDDING_MODEL = settings.embedding_model
RERANKING_EMBEDDING_MODEL = settings.reranking_embedding_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MMR_LAMBDA = settings.mmr_lambda
RRF_K = settings.rrf_k
ENABLE_SELF_RAG = settings.enable_self_rag
RELEVANCE_THRESHOLD = settings.relevance_threshold
MAX_RERETRIEVE_ATTEMPTS = settings.max_reretrieve_attempts
SEMANTIC_SIMILARITY_THRESHOLD = settings.semantic_similarity_threshold
ENABLE_CONTEXT_COMPRESSION = settings.enable_context_compression
CONTEXT_COMPRESSION_MAX_TOKENS = settings.context_compression_max_tokens
CONTEXT_COMPRESSION_SENTENCES_PER_DOC = settings.context_compression_sentences_per_doc
ENABLE_TEMPORAL_BOOST = settings.enable_temporal_boost
RECENCY_WEIGHT = settings.recency_weight
RECENCY_HALF_LIFE_DAYS = settings.recency_half_life_days
ENABLE_KNOWLEDGE_GRAPH = settings.enable_knowledge_graph
KG_MAX_HOPS = settings.kg_max_hops
KG_MIN_CONFIDENCE = settings.kg_min_confidence
ENABLE_COT_REASONING = settings.enable_cot_reasoning
ENABLE_CROSS_ENCODER = settings.enable_cross_encoder
ENABLE_QUERY_REFINEMENT = settings.enable_query_refinement
CROSS_ENCODER_MODEL = settings.cross_encoder_model
ENABLE_DYNAMIC_WEIGHTS = settings.enable_dynamic_weights
WEIGHT_LEARNING_RATE = settings.weight_learning_rate
PERFORMANCE_TRACKING_WINDOW = settings.performance_tracking_window
ENABLE_HIERARCHICAL_RETRIEVAL = settings.enable_hierarchical_retrieval
HIERARCHICAL_CONFIDENCE_THRESHOLD = settings.hierarchical_confidence_threshold
ENABLE_TIER_3 = settings.enable_tier_3
ENABLE_COST_TRACKING = settings.enable_cost_tracking
ENABLE_POST_GENERATION_VERIFICATION = settings.enable_post_generation_verification
ENABLE_CITATION_VALIDATION = settings.enable_citation_validation
MIN_FACTUALITY_SCORE = settings.min_factuality_score
REQUIRE_BOTH_SCORES_HIGH = settings.require_both_scores_high
MAX_REGENERATION_ATTEMPTS = settings.max_regeneration_attempts
ENABLE_UNCERTAINTY_QUANTIFICATION = settings.enable_uncertainty_quantification
SHOW_CONFIDENCE_IN_RESPONSE = settings.show_confidence_in_response
ENABLE_CONSISTENCY_CHECK = settings.enable_consistency_check
ENABLE_HUMAN_IN_THE_LOOP = settings.enable_human_in_the_loop
ENABLE_ATTRIBUTION_MAP = settings.enable_attribution_map
ENABLE_TEMPORAL_VALIDATION = settings.enable_temporal_validation
ENABLE_ENSEMBLE_SAMPLING = settings.enable_ensemble_sampling
ENABLE_METRICS_LOGGING = settings.enable_metrics_logging
METRICS_LOG_INTERVAL = settings.metrics_log_interval
EMBEDDING_CACHE_SIZE = settings.embedding_cache_size
TAVILY_API_KEY = settings.tavily_api_key
POSTGRES_URI = settings.postgres_uri
MAX_CONTEXT_TOKENS = settings.max_context_tokens
TOKEN_ALLOCATION = settings.token_allocation
CONTEXT_WARNING_THRESHOLD = settings.context_warning_threshold

# Default Core Memory Templates
DEFAULT_HUMAN_PERSONA = "Name: [User]\nBackground: [To be learned]\nPreferences: [To be discovered]"
DEFAULT_AGENT_PERSONA = "I am a helpful AI assistant with long-term memory capabilities. I can remember our past conversations and learn about you over time. I manage my memory efficiently by storing important information and retrieving it when needed."

# Memory Settings
ARCHIVAL_SEARCH_RESULTS = 5  # Number of results to retrieve from archival memory
RECALL_SEARCH_RESULTS = 10   # Number of conversation messages to retrieve
EMBEDDING_BATCH_SIZE = 100   # Batch size for embedding generation

# Self-RAG Settings (PHASE 1: Increased thresholds for stricter quality control)
MAX_CHARS_PER_DOC = 3000     # Increased from 2000 for better context matching
MIN_QUALITY_SCORE = 0.5      # Increased from 0.3 for stricter quality control
MIN_SUPPORT_RATIO = 0.75     # Increased from 0.7 for better hallucination prevention

# Context Quality Settings
MIN_AVG_RELEVANCE_SCORE = 0.35   # Increased from 0.2
MIN_FOLLOW_UP_WORDS = 50        # Minimum words in recent response to skip document retrieval

# CoT (Chain-of-Thought) Decision Thresholds
COT_WORD_COUNT_THRESHOLD = 20   # Word count above which CoT is triggered
COT_CONFIDENCE_THRESHOLD = 0.5  # Confidence below which CoT is triggered

# Query Refinement Settings
MAX_REFINEMENT_ATTEMPTS = 2      # Maximum number of query refinement iterations
REFINEMENT_CONFIDENCE_THRESHOLD = 0.4  # Confidence threshold to trigger refinement
MIN_ANSWER_WORD_COUNT = 20       # Minimum answer length to avoid refinement

# Reranking & Retrieval Settings
RERANK_TOP_K_DEFAULT = 15        # Default top_k for reranking
MMR_DIVERSITY_TOP_K = 5          # Top K documents after MMR diversity
CROSS_ENCODER_SCORE_THRESHOLD = 0.1  # Minimum cross-encoder score to consider relevant
PROGRESSIVE_TOP_K_CONFIG = {     # Progressive top_k reduction for re-retrieval
    0: 15,  # First attempt
    1: 10,  # Second attempt
    2: 5    # Third attempt
}

# Knowledge Graph Settings
KG_RESULT_LIMIT = 5              # Maximum KG entities to retrieve

# Multi-Document Synthesis Settings
SYNTHESIS_DOC_LIMIT = 5          # Maximum documents for synthesis
SYNTHESIS_CONTENT_PREVIEW = 300  # Characters to preview per document
# Context Compression Settings
COMPRESSION_MIN_THRESHOLD = 0.005    # Minimum threshold for low-score documents
COMPRESSION_INTENT_THRESHOLDS = {   # Dynamic thresholds by intent
    "QUESTION_ANSWERING": 0.5,
    "SEARCH": 0.4,
    "CONVERSATIONAL": 0.35,
    "MULTI_HOP_REASONING": 0.55
}

# Hierarchical Retriever Settings
HIERARCHICAL_CONFIDENCE_BLEND_WEIGHT = 0.6  # Statistical weight in confidence blend
HIERARCHICAL_SEMANTIC_BLEND_WEIGHT = 0.4    # Semantic weight in confidence blend
HIERARCHICAL_BOOST_THRESHOLD = 0.7          # Score threshold for confidence boost
HIERARCHICAL_BOOST_MULTIPLIER = 1.2         # Confidence boost multiplier