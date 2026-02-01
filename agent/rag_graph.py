"""
MemGPT RAG Agent
LangGraph workflow with advanced agentic RAG
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import SecretStr

import config
from agent.rag_tools import create_rag_tools
from agent.state import MemGPTState, ChatResponse
from agent.tools import create_memory_tools
from database.operations import DatabaseOperations
from memory.embeddings import EmbeddingService
from memory.manager import MemoryManager

# RAG Components
from rag import (
    ContextCompressor,
    CrossEncoderReranker,
    DocumentStore,
    HybridRetriever,
    IntentRecognizer,
    KnowledgeGraphExtractor,
    KnowledgeGraphRetriever,
    MMRDiversifier,
    OpenAIReranker,
    QueryIntent,
    QueryRewriter,
    QueryRouter,
    RAGASEvaluator,
    RAGEvaluator,
    ReciprocalRankFusion,
    SelfRAGEvaluator,
    SemanticChunker,
    WebSearchTool,
)

from rag.nodes import (
    chain_of_thought_node,
    check_context_node,
    decide_next_action,
    generate_response_node,
    process_tool_calls_node,
    query_refinement_node,
    receive_input_node,
    recognize_intent_node,
    rerank_and_eval_node,
    retrieve_memory_node,
    retrieve_rag_node,
    rewrite_query_node,
    route_query_node,
    should_use_cot,
    synthesize_multi_doc_node,
    verify_response_node,
    should_regenerate,
)

from rag.nodes.update_memory import update_memory_node

logger = logging.getLogger(__name__)


class MemGPTRAGAgent:
    """MemGPT Agent with Advanced Agentic RAG"""

    def __init__(self, agent_id: str, memory_manager: MemoryManager | None = None):
        self.agent_id = agent_id

        self.db_ops = DatabaseOperations()

        if memory_manager:
            self.memory_manager = memory_manager
        else:
            self.memory_manager = MemoryManager(agent_id=agent_id)

        # Initialize LLM FIRST (needed by RAG components)
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,
            api_key=SecretStr(config.OPENAI_API_KEY),
        )

        # OpenAI client for direct API calls (translation, etc.)
        from openai import OpenAI

        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Initialize RAG components (needs self.llm)
        self._initialize_rag()

        memory_tools = create_memory_tools(self.memory_manager)
        rag_tools = create_rag_tools(self.document_store, self.web_search)
        self.all_tools = memory_tools + rag_tools

        self.llm_with_tools = self.llm.bind_tools(self.all_tools)

        self.graph = self._build_graph()

        logger.info(f"MemGPT RAG Agent initialized: {agent_id}")

    def _initialize_rag(self):
        """Initialize RAG components"""
        # Load optimization settings
        try:
            from optimization_config import optimization_settings

            logger.info("Optimization settings loaded")
        except ImportError:
            logger.warning("optimization_config not found, using defaults")
            optimization_settings = None

        if optimization_settings and optimization_settings.enable_cost_tracking:
            from utils.cost_tracker import get_cost_tracker

            self.cost_tracker = get_cost_tracker()
            logger.info("Cost tracking enabled")
        else:
            self.cost_tracker = None

        # Embedding service with dimensionality reduction
        embedding_dimensions = (
            optimization_settings.embedding_dimensions
            if optimization_settings
            else None
        )
        self.embedding_service = EmbeddingService(
            dimensions=embedding_dimensions, cost_tracker=self.cost_tracker
        )

        if embedding_dimensions:
            logger.info(
                f"Embedding service initialized with {embedding_dimensions} dimensions (optimized)"
            )
        else:
            logger.info("Embedding service initialized with default dimensions")

        # Intent recognizer (Paper-compliant: pre-retrieval component)
        self.intent_recognizer = IntentRecognizer(self.llm)

        # Query rewriter (Paper-compliant: pre-retrieval component)
        self.query_rewriter = QueryRewriter(self.llm)

        # Query router
        self.query_router = QueryRouter(self.llm)

        # Knowledge Graph (Paper-compliant: multi-hop reasoning) - MUST be before DocumentStore
        if config.ENABLE_KNOWLEDGE_GRAPH:
            self.kg_extractor = KnowledgeGraphExtractor(
                llm=self.llm, min_confidence=config.KG_MIN_CONFIDENCE
            )
            self.kg_retriever = KnowledgeGraphRetriever(
                max_hops=config.KG_MAX_HOPS, min_confidence=config.KG_MIN_CONFIDENCE
            )
            logger.info("Knowledge Graph enabled")
        else:
            self.kg_extractor = None
            self.kg_retriever = None

        from rag.data_wrangler import DataWrangler

        data_wrangler = DataWrangler(enable_dedup=True, min_quality_score=0.3)

        self.document_store = DocumentStore(
            database_ops=self.db_ops,
            embedding_service=self.embedding_service,
            chunking_strategy=SemanticChunker(
                embedding_service=self.embedding_service,
                similarity_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
            ),
            data_wrangler=data_wrangler,
            kg_extractor=self.kg_extractor,
        )
        from openai import OpenAI

        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.reranker = OpenAIReranker(
            openai_client=openai_client, model=config.RERANKING_EMBEDDING_MODEL
        )

        # Cross-Encoder Reranker (Paper-compliant: superior accuracy)
        if config.ENABLE_CROSS_ENCODER:
            self.cross_encoder = CrossEncoderReranker(
                model_name=config.CROSS_ENCODER_MODEL
            )
            if self.cross_encoder.is_available():
                logger.info("Cross-Encoder reranker enabled")
            else:
                logger.warning("Cross-Encoder requested but not available")
                self.cross_encoder = None
        else:
            self.cross_encoder = None

        # Selective Reranking (OPTIMIZATION: 20-40% cost reduction)
        if optimization_settings and optimization_settings.enable_selective_reranking:
            from rag.selective_reranker import SelectiveReranker

            self.selective_reranker = SelectiveReranker(
                openai_reranker=self.reranker,
                cross_encoder_reranker=self.cross_encoder,
                enable_selective=True,
            )
            logger.info("Selective reranking enabled (optimized)")
        else:
            self.selective_reranker = None

        # DW-GRPO: Dynamic Weight Manager (with database persistence)
        if config.ENABLE_DYNAMIC_WEIGHTS:
            from rag.adaptive_weights import DynamicWeightManager

            self.weight_manager = DynamicWeightManager(
                learning_rate=config.WEIGHT_LEARNING_RATE,
                tracking_window=config.PERFORMANCE_TRACKING_WINDOW,
                enable_learning=True,
                agent_id=self.agent_id,
                enable_db_persistence=True,
            )
            logger.info(
                f"Dynamic Weight Manager enabled (α={config.WEIGHT_LEARNING_RATE}, persistent learning)"
            )
        else:
            self.weight_manager = None

        self.mmr = MMRDiversifier(lambda_param=config.MMR_LAMBDA)

        self.rrf = ReciprocalRankFusion(k=config.RRF_K)

        # Self-RAG evaluator with ensemble verification
        use_ensemble = (
            optimization_settings.enable_ensemble_verification
            if optimization_settings
            else True
        )
        self.self_rag = SelfRAGEvaluator(
            llm=self.llm,
            embedding_service=self.embedding_service,
            use_ensemble=use_ensemble,
        )

        if use_ensemble:
            logger.info("Self-RAG initialized with ensemble verification (optimized)")

        # Hybrid retriever (Paper-compliant: multiple retrieval methods)
        self.hybrid_retriever = HybridRetriever(
            agent_id=self.agent_id,
            memory_manager=self.memory_manager,
            document_store=self.document_store,
            weight_manager=self.weight_manager
            if config.ENABLE_DYNAMIC_WEIGHTS
            else None,
        )

        # Web search (MUST be initialized before HierarchicalRetriever)
        self.web_search = WebSearchTool(tavily_api_key=config.TAVILY_API_KEY)

        # DW-GRPO: Hierarchical Retriever
        if config.ENABLE_HIERARCHICAL_RETRIEVAL:
            from rag.hierarchical_retriever import HierarchicalRetriever

            self.hierarchical_retriever = HierarchicalRetriever(
                memory_manager=self.memory_manager,
                document_store=self.document_store,
                hybrid_retriever=self.hybrid_retriever,
                llm=self.llm,  # CRITICAL: Pass LLM for agentic Tier 3
                kg_retriever=self.kg_retriever
                if config.ENABLE_KNOWLEDGE_GRAPH
                else None,
                web_search=self.web_search if hasattr(self, "web_search") else None,
                confidence_threshold=config.HIERARCHICAL_CONFIDENCE_THRESHOLD,
                enable_tier_3=config.ENABLE_TIER_3,
            )
            logger.info(
                f"Hierarchical Retriever enabled (threshold={config.HIERARCHICAL_CONFIDENCE_THRESHOLD}, agentic Tier 3)"
            )
        else:
            self.hierarchical_retriever = None

        # Context compressor (Paper-compliant: post-retrieval)
        if config.ENABLE_CONTEXT_COMPRESSION:
            self.context_compressor = ContextCompressor(
                max_tokens=config.CONTEXT_COMPRESSION_MAX_TOKENS,
                sentences_per_doc=config.CONTEXT_COMPRESSION_SENTENCES_PER_DOC,
                embedding_service=self.embedding_service,  # Phase 2: Enable semantic scoring
            )
        else:
            self.context_compressor = None

        # RAG evaluator (Paper-compliant: metrics)
        if config.ENABLE_METRICS_LOGGING:
            self.evaluator = RAGEvaluator()
            self.metrics_counter = 0
        else:
            self.evaluator = None

        # RAGAS Evaluator (Paper-compliant: comprehensive evaluation)
        self.ragas_evaluator = RAGASEvaluator(llm=self.llm)
        if self.ragas_evaluator.is_available():
            logger.info("RAGAS evaluator enabled")

        # Factuality Scorer (OPTIMIZATION: Auto-refuse low-quality answers)
        if optimization_settings and optimization_settings.enable_ensemble_verification:
            from rag.factuality_scorer import FactualityScorer

            self.factuality_scorer = FactualityScorer(self.self_rag)
            logger.info("Factuality scorer enabled (optimized)")
        else:
            self.factuality_scorer = None

        logger.info("RAG components initialized")

    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(MemGPTState)

        # Add nodes using imported modular functions
        workflow.add_node("receive_input", lambda s: receive_input_node(s, self))  # type: ignore
        workflow.add_node("recognize_intent", lambda s: recognize_intent_node(s, self))  # type: ignore
        workflow.add_node("rewrite_query", lambda s: rewrite_query_node(s, self))  # type: ignore
        workflow.add_node("route_query", lambda s: route_query_node(s, self))  # type: ignore
        workflow.add_node("check_context", lambda s: check_context_node(s, self))  # type: ignore
        workflow.add_node("retrieve_memory", lambda s: retrieve_memory_node(s, self))  # type: ignore
        workflow.add_node("retrieve_rag", lambda s: retrieve_rag_node(s, self))  # type: ignore
        workflow.add_node("rerank_and_eval", lambda s: rerank_and_eval_node(s, self))  # type: ignore
        workflow.add_node(
            "chain_of_thought",
            lambda s: chain_of_thought_node(s, self),  # type: ignore
        )  # System2: CoT reasoning
        workflow.add_node(
            "synthesize_multi_doc",
            lambda s: synthesize_multi_doc_node(s, self),  # type: ignore
        )  # System2: multi-doc synthesis
        workflow.add_node(
            "query_refinement",
            lambda s: query_refinement_node(s, self),  # type: ignore
        )  # Paper-compliant: iterative refinement
        workflow.add_node(
            "generate_response",
            lambda s: generate_response_node(s, self),  # type: ignore
        )
        workflow.add_node(
            "verify_response",
            lambda s: verify_response_node(s, self),  # type: ignore
        )  # Phase 1: Post-generation verification
        workflow.add_node(
            "process_tool_calls",
            lambda s: process_tool_calls_node(s, self),  # type: ignore
        )
        workflow.add_node("update_memory", lambda s: update_memory_node(s, self))  # type: ignore

        workflow.set_entry_point("receive_input")

        # Add edges - Paper-compliant workflow: Intent → Rewrite → Memory → Route → Retrieve → CoT → Generate
        # Memory retrieval happens before routing so recalled messages are available for routing decisions
        workflow.add_edge("receive_input", "recognize_intent")
        workflow.add_edge("recognize_intent", "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve_memory")  # Memory FIRST!
        workflow.add_edge(
            "retrieve_memory", "route_query"
        )  # Then route with recall available
        workflow.add_edge("route_query", "check_context")
        workflow.add_edge("check_context", "retrieve_rag")
        workflow.add_edge("retrieve_rag", "rerank_and_eval")

        # Conditional: Chain-of-Thought for complex reasoning (System2)
        workflow.add_conditional_edges(
            "rerank_and_eval",
            lambda s: should_use_cot(s, self),
            {"cot": "chain_of_thought", "skip": "synthesize_multi_doc"},
        )

        # After CoT, go to synthesis (always run, decides internally)
        workflow.add_edge("chain_of_thought", "synthesize_multi_doc")

        # Synthesis always runs but may skip internally, then go to generate
        workflow.add_edge("synthesize_multi_doc", "generate_response")

        # Phase 1: Post-generation verification (conditional: regenerate if verification fails)
        workflow.add_conditional_edges(
            "generate_response",
            lambda s: should_regenerate(s, self),
            {
                "regenerate": "generate_response",  # Loop back for regeneration
                "accept": "verify_response",  # Proceed to verification
            },
        )

        # Paper-compliant: Query Refinement Conditional Loop with tool usage
        workflow.add_conditional_edges(
            "verify_response",
            lambda s: decide_next_action(s, self),
            {
                "refine": "query_refinement",
                "tools": "process_tool_calls",
                "continue": "update_memory",
            },
        )

        workflow.add_edge("query_refinement", "retrieve_rag")  # Loop back to retrieval

        workflow.add_edge("process_tool_calls", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def chat(self, user_input: str, conversation_id: str = "default") -> ChatResponse:
        """
        Process user input and generate response

        Args:
            user_input: User message
            conversation_id: Conversation identifier

        Returns:
            ChatResponse with agent response and metadata
        """
        # Get actual core memory from database (not just templates)
        try:
            core_memory_data = self.memory_manager.get_core_memory()
            human_persona = core_memory_data.get(
                "human_persona", config.DEFAULT_HUMAN_PERSONA
            )
            agent_persona = core_memory_data.get(
                "agent_persona", config.DEFAULT_AGENT_PERSONA
            )
            core_facts = core_memory_data.get("facts", [])
        except Exception as e:
            logger.warning(f"Failed to load core memory, using defaults: {e}")
            human_persona = config.DEFAULT_HUMAN_PERSONA
            agent_persona = config.DEFAULT_AGENT_PERSONA
            core_facts = []

        # Initialize state using Pydantic model
        initial_state = MemGPTState(
            needs_document_retrieval=True,
            agent_id=self.agent_id,
            conversation_id=conversation_id,
            user_input=user_input,
            agent_response=None,
            messages=[],
            human_persona=human_persona,
            agent_persona=agent_persona,
            core_facts=core_facts,
            retrieved_archival=[],
            retrieved_recall=[],
            retrieved_documents=[],
            retrieved_web=[],
            rerank_scores={},
            quality_eval={},
            rag_context="",
            query_intent=None,
            intent_confidence=0.0,
            rewritten_query=None,
            query_variants=[],
            current_tokens=0,
            token_breakdown={},
            context_overflow=False,
            pending_archival_inserts=[],
            memory_operations_log=[],
            iteration_count=0,
            max_iterations=5,
            needs_memory_retrieval=True,  # Enable archival memory search by default
            should_save_to_archival=False,
            tool_calls=[],
            tool_results=[],
            refinement_count=0,
            reretrieve_count=0,
            final_context=[],
            compression_stats={},
            cot_reasoning="",
            reasoning_steps=[],
            needs_multi_hop=False,
            synthesized_context=None,
            synthesis_metadata={},
            faithfulness_score={},
            retrieval_metrics={},
            ground_truth=None,
            verification_passed=True,
            support_ratio=1.0,
            regeneration_count=0,
            total_regeneration_count=0,
            citation_validation={},
            consistency_result={},
            uncertainty_info={},
            temporal_validation={},
            requires_human_review=False,
            hitl_reason=None,
            attribution_map={},
        )

        # Run graph
        final_state = self.graph.invoke(initial_state)

        # Access response from Pydantic model (could be dict or model depending on LangGraph version)
        if isinstance(final_state, dict):
            response = final_state.get("agent_response", "No response generated")
            return {
                "agent_response": response,
                "intent": final_state.get("query_intent"),
                "retrieved_docs": len(final_state.get("final_context", [])),
                "refinement_count": final_state.get("refinement_count", 0),
                "quality_score": final_state.get("quality_eval", {}).get(
                    "confidence", 0.0
                ),
            }
        else:
            response = final_state.agent_response or "No response generated"
            return {
                "agent_response": response,
                "intent": final_state.query_intent,
                "retrieved_docs": len(final_state.final_context),
                "refinement_count": final_state.refinement_count,
                "quality_score": final_state.quality_eval.get("confidence", 0.0)
                if final_state.quality_eval
                else 0.0,
            }
