from typing import List, Dict, Any, Annotated, Optional, TypedDict
from operator import add
from pydantic import BaseModel, Field, ConfigDict

from rag.models.intent_analysis import QueryIntent


class ChatResponse(TypedDict):
    """
    Typed response from MemGPTAgent.chat() method.

    Attributes:
        agent_response: The generated response text
        intent: Detected query intent (e.g., QUESTION_ANSWERING, SEARCH)
        retrieved_docs: Number of documents used for context
        refinement_count: Number of query refinement iterations
        quality_score: Confidence score of the response (0.0-1.0)
    """
    agent_response: str
    intent: Optional[QueryIntent]
    retrieved_docs: int
    refinement_count: int
    quality_score: float


class MemGPTState(BaseModel):
    """
    MemGPT Agent State adapted for Pydantic (LangGraph v0.2+)
    Focused on System 1 and System 2 reasoning.
    """
    # Configuration to allow arbitrary types if needed
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Identidade e Sessão ---
    agent_id: str = Field(..., description="ID único do agente")
    conversation_id: str = Field(..., description="ID da sessão de chat")
    
    # --- Interação e Mensagens ---
    user_input: str = Field("", description="Input bruto do usuário")
    agent_response: Optional[str] = None
    
    # O Annotated com 'add' é crucial para o LangGraph acumular o histórico
    messages: Annotated[List[Dict[str, Any]], add] = Field(
        default_factory=list, 
        description="Histórico de mensagens acumulado"
    )

    # --- Memória Estática (Core) ---
    human_persona: str = Field("User", description="Perfil do humano")
    agent_persona: str = Field("Assistant", description="Perfil do agente")
    core_facts: List[str] = Field(default_factory=list)

    # --- Recuperação e Contexto RAG ---
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_archival: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_recall: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_web: List[Dict[str, Any]] = Field(default_factory=list)
    rag_context: str = Field("", description="Contexto final injetado no prompt")
    final_context: List[Dict[str, Any]] = Field(default_factory=list)
    rerank_scores: Dict[str, float] = Field(default_factory=dict)
    reretrieve_count: int = Field(0, description="Contador de re-retrieval (Self-RAG)")
    
    # --- Query Processing (System 1/2) ---
    query_intent: Optional[QueryIntent] = None
    intent_confidence: float = Field(0.0, ge=0.0, le=1.0)
    rewritten_query: Optional[str] = None
    translated_query: Optional[str] = None  # English translation for cross-language retrieval
    query_variants: List[str] = Field(default_factory=list)
    refinement_count: int = Field(0, description="Contador de iterações de busca")

    # --- Raciocínio Deliberativo (System 2 / Agentic) ---
    needs_multi_hop: bool = Field(False, description="Flag para disparar raciocínio complexo")
    cot_reasoning: str = Field("", description="Cadeia de pensamento (Chain-of-Thought)")
    reasoning_steps: List[str] = Field(default_factory=list)
    synthesized_context: Optional[str] = None
    synthesis_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados da síntese multi-documento")

    # --- Métricas e Avaliação (Self-RAG) ---
    quality_eval: Dict[str, Any] = Field(default_factory=dict)
    faithfulness_score: Dict[str, Any] = Field(default_factory=dict, description="Score de fidelidade (alucinação)")
    retrieval_metrics: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: Optional[str] = Field(None, description="Ground truth para avaliação (opcional)")

    # --- Controle de Fluxo e Ferramentas ---
    iteration_count: int = Field(0)
    max_iterations: int = Field(5)
    needs_memory_retrieval: bool = Field(False)
    needs_document_retrieval: bool = Field(True, description="Whether to retrieve from documents (optimization)")
    should_save_to_archival: bool = Field(False)
    pending_archival_inserts: List[str] = Field(default_factory=list)
    memory_operations_log: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- Gestão de Contexto e Tokens ---
    current_tokens: int = Field(0)
    token_breakdown: Dict[str, int] = Field(default_factory=dict)
    context_overflow: bool = Field(False)
    compression_stats: Dict[str, Any] = Field(default_factory=dict)
    
    # --- Anti-Hallucination Verification (Phase 1) ---
    verification_passed: bool = Field(True, description="Whether post-generation verification succeeded")
    support_ratio: float = Field(1.0, description="Ratio of verified claims to total claims")
    regeneration_count: int = Field(0, description="Number of times response was regenerated")
    citation_validation: Dict[str, Any] = Field(default_factory=dict, description="Citation validation results")
    
    # --- Anti-Hallucination Enhancement (Phase 2) ---
    consistency_result: Dict[str, Any] = Field(default_factory=dict, description="Cross-document consistency check")
    uncertainty_info: Dict[str, Any] = Field(default_factory=dict, description="Uncertainty quantification metrics")
    
    # --- Anti-Hallucination Advanced (Phase 3) ---
    temporal_validation: Dict[str, Any] = Field(default_factory=dict, description="Temporal consistency validation")
    requires_human_review: bool = Field(False, description="Whether response needs human review (HITL)")
    hitl_reason: Optional[str] = Field(None, description="Reason for human review requirement")
    attribution_map: Dict[str, Any] = Field(default_factory=dict, description="Claim-to-source attribution mapping")