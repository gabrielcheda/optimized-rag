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
    
    openai_api_key: SecretStr = Field(..., description="OpenAI API key for LLM and embeddings")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model for semantic search")

    reranking_embedding_model: str = Field(default="text-embedding-3-large", description="Model for reranking")
    chunk_size: int = Field(default=1200, description="Document chunk size")
    chunk_overlap: int = Field(default=150, description="Overlap between chunks")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="Balance relevance vs diversity")
    rrf_k: int = Field(default=60, description="Reciprocal rank fusion constant")
    enable_self_rag: bool = Field(default=True, description="Enable Self-RAG evaluation")
    relevance_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Minimum relevance score (was 0.75)")  # FASE 1: Increased
    max_reretrieve_attempts: int = Field(default=2, description="Max re-retrieval attempts")
    semantic_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Threshold for semantic chunking")

    enable_context_compression: bool = Field(default=True, description="Enable context compression")
    context_compression_max_tokens: int = Field(default=2000, description="Max tokens after compression")
    context_compression_sentences_per_doc: int = Field(default=12, description="Sentences to keep per document (was 8)")  # FASE 1: Increased

    enable_temporal_boost: bool = Field(default=True, description="Enable temporal boosting")
    recency_weight: float = Field(default=0.15, ge=0.0, le=0.3, description="Weight for temporal boosting")
    recency_half_life_days: int = Field(default=30, description="Exponential decay rate in days")

    enable_knowledge_graph: bool = Field(default=True, description="Enable knowledge graph")
    kg_max_hops: int = Field(default=2, description="Maximum hops for graph traversal")
    kg_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for triples")

    enable_cot_reasoning: bool = Field(default=True, description="Chain-of-Thought for complex queries")
    enable_cross_encoder: bool = Field(default=True, description="Cross-encoder reranking")
    enable_query_refinement: bool = Field(default=True, description="Iterative query refinement")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model name")

    enable_dynamic_weights: bool = Field(default=True, description="Enable dynamic weight learning")
    weight_learning_rate: float = Field(default=0.01, ge=0.0, le=1.0, description="Rate of weight adaptation")
    performance_tracking_window: int = Field(default=100, description="Number of queries to track")
    enable_hierarchical_retrieval: bool = Field(default=True, description="Enable hierarchical retrieval")
    hierarchical_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for tier escalation")
    enable_tier_3: bool = Field(default=True, description="Enable expensive Tier 3 (KG + Web)")
    enable_cost_tracking: bool = Field(default=True, description="Track API costs and savings")

    enable_post_generation_verification: bool = Field(default=True, description="Verify claims after generation")
    enable_citation_validation: bool = Field(default=True, description="Validate citation format and completeness")
    min_factuality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum factuality to accept (was 0.4)")  # FASE 1: Increased
    require_both_scores_high: bool = Field(default=True, description="Require both faithfulness AND factuality high")
    max_regeneration_attempts: int = Field(default=1, description="Max times to regenerate failed responses (was 2)")  # CORRIGIDO

    enable_uncertainty_quantification: bool = Field(default=True, description="Calculate and show uncertainty scores")
    show_confidence_in_response: bool = Field(default=False, description="Append confidence to response text")
    enable_consistency_check: bool = Field(default=True, description="Check cross-document consistency")

    enable_human_in_the_loop: bool = Field(default=False, description="Flag uncertain responses for review")
    enable_attribution_map: bool = Field(default=True, description="Build claim-to-source attribution map")
    enable_temporal_validation: bool = Field(default=True, description="Validate temporal consistency")
    enable_ensemble_sampling: bool = Field(default=False, description="Generate multiple responses for critical queries")

    enable_metrics_logging: bool = Field(default=True, description="Enable metrics logging")
    metrics_log_interval: int = Field(default=10, description="Log metrics every N queries")

    embedding_cache_size: int = Field(default=1000, description="LRU cache size for embeddings")
    
    # FASE 1: Anti-hallucination thresholds
    min_support_ratio: float = Field(default=0.70, ge=0.0, le=1.0, description="Minimum ratio of supported claims (adjusted from 0.90)")  # CORRIGIDO
    cross_encoder_score_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Threshold to filter weak results (was 0.1)")  # FASE 1: Increased
    
    # FASE 1: Performance and timeout settings
    max_verification_time_ms: int = Field(default=5000, description="Timeout for verification in ms")
    enable_async_verification: bool = Field(default=True, description="Enable async verification when possible")
    verification_cache_size: int = Field(default=100, description="Size of verification cache")

    tavily_api_key: str = Field(default="", description="Tavily API key for web search (optional)")

    postgres_uri: str = Field(..., description="PostgreSQL connection URI")

    max_context_tokens: int = Field(default=8000, description="Maximum context tokens")

    token_allocation_system_prompt: int = Field(default=500, description="Tokens for system prompt")
    token_allocation_core_memory: int = Field(default=800, description="Tokens for core memory")
    token_allocation_function_definitions: int = Field(default=700, description="Tokens for function definitions")
    token_allocation_retrieved_context: int = Field(default=2000, description="Tokens for retrieved context")
    token_allocation_conversation: int = Field(default=4000, description="Tokens for conversation history")

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


settings = Settings()  # type: ignore[call-arg]

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

# FASE 6.1: Web Search Fallback - Triggers web search when factuality is POOR
ENABLE_WEB_SEARCH_FALLBACK = getattr(settings, 'enable_web_search_fallback', True)
WEB_SEARCH_FALLBACK_THRESHOLD = getattr(settings, 'web_search_fallback_threshold', 0.35)

# FASE 1: New constants
MIN_SUPPORT_RATIO_SETTING = settings.min_support_ratio
CROSS_ENCODER_SCORE_THRESHOLD_SETTING = settings.cross_encoder_score_threshold
MAX_VERIFICATION_TIME_MS = settings.max_verification_time_ms
ENABLE_ASYNC_VERIFICATION = settings.enable_async_verification
VERIFICATION_CACHE_SIZE = settings.verification_cache_size

TAVILY_API_KEY = settings.tavily_api_key
POSTGRES_URI = settings.postgres_uri
MAX_CONTEXT_TOKENS = settings.max_context_tokens
TOKEN_ALLOCATION = settings.token_allocation
CONTEXT_WARNING_THRESHOLD = settings.context_warning_threshold

DEFAULT_HUMAN_PERSONA = "Name: [User]\nBackground: [To be learned]\nPreferences: [To be discovered]"
DEFAULT_AGENT_PERSONA = "I am a helpful AI assistant with long-term memory capabilities. I can remember our past conversations and learn about you over time. I manage my memory efficiently by storing important information and retrieving it when needed."

ARCHIVAL_SEARCH_RESULTS = 5
RECALL_SEARCH_RESULTS = 10
EMBEDDING_BATCH_SIZE = 100

MAX_CHARS_PER_DOC = 3000
MIN_QUALITY_SCORE = 0.5
MIN_SUPPORT_RATIO = 0.70  # CORRIGIDO: 0.90 era muito rigoroso, reduzido para 70%

MIN_AVG_RELEVANCE_SCORE = 0.35
MIN_FOLLOW_UP_WORDS = 50

COT_WORD_COUNT_THRESHOLD = 20
COT_CONFIDENCE_THRESHOLD = 0.5

MAX_REFINEMENT_ATTEMPTS = 2
REFINEMENT_CONFIDENCE_THRESHOLD = 0.4
MIN_ANSWER_WORD_COUNT = 20

RERANK_TOP_K_DEFAULT = 15
MMR_DIVERSITY_TOP_K = 5
CROSS_ENCODER_SCORE_THRESHOLD = 0.15  # FASE 1: Was 0.1 → Filtrar resultados fracos
PROGRESSIVE_TOP_K_CONFIG = {
    0: 15,
    1: 10,
    2: 5
}

KG_RESULT_LIMIT = 5

SYNTHESIS_DOC_LIMIT = 5
SYNTHESIS_CONTENT_PREVIEW = 300
COMPRESSION_MIN_THRESHOLD = 0.005
COMPRESSION_INTENT_THRESHOLDS = {
    "QUESTION_ANSWERING": 0.25,
    "SEARCH": 0.2,
    "CONVERSATIONAL": 0.15,
    "MULTI_HOP_REASONING": 0.3
}

HIERARCHICAL_CONFIDENCE_BLEND_WEIGHT = 0.6
HIERARCHICAL_SEMANTIC_BLEND_WEIGHT = 0.4
HIERARCHICAL_BOOST_THRESHOLD = 0.7
HIERARCHICAL_BOOST_MULTIPLIER = 1.2