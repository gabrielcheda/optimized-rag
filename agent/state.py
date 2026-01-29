from typing import List, Dict, Any, Annotated, Optional
from operator import add
from pydantic import BaseModel, Field, ConfigDict

from rag.models.intent_analysis import QueryIntent


class MemGPTState(BaseModel):
    """
    Estado do Agente MemGPT adaptado para Pydantic (LangGraph v0.2+)
    Focado em Raciocínio de Sistema 1 e Sistema 2.
    """
    # Configuração para permitir tipos arbitrários se necessário
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