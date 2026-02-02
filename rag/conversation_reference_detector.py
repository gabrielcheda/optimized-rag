"""
Conversation Reference Detector
Detecta se uma query referencia a conversa atual (nao documentos externos)

Tecnicas utilizadas:
1. Semantic Similarity - Query vs Recall Memory via embeddings
2. Coreference Detection - Detectar referencias anaforicas
3. LLM-based Analysis - LLM analisa se query referencia a conversa
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class ConversationReferenceResult:
    """Resultado da deteccao de referencia a conversa"""
    is_conversation_reference: bool
    confidence: float
    method: str  # 'semantic', 'llm', 'coreference', 'hybrid'
    reasoning: str
    referenced_message_index: Optional[int] = None  # Qual mensagem esta sendo referenciada


class ConversationReferenceDetector:
    """
    Detecta se uma query faz referencia a conversa atual
    usando multiplas tecnicas (nao patterns hardcoded)
    """

    def __init__(
        self,
        llm,
        embedding_service,
        semantic_threshold: float = 0.75,
        enable_llm_fallback: bool = True
    ):
        self.llm = llm
        self.embedding_service = embedding_service
        self.semantic_threshold = semantic_threshold
        self.enable_llm_fallback = enable_llm_fallback

        # Cache de embeddings do historico
        self._history_embeddings_cache = {}

    def detect(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]] = None,
        recall_messages: List[Dict[str, Any]] = None
    ) -> ConversationReferenceResult:
        """
        Detecta se query referencia a conversa

        Args:
            query: Query do usuario
            conversation_history: Historico de mensagens [{role, content}, ...]
            recall_messages: Mensagens recuperadas do recall memory

        Returns:
            ConversationReferenceResult com detalhes da deteccao
        """
        # Usar recall_messages se disponivel, senao conversation_history
        messages = recall_messages or conversation_history or []

        if not messages:
            return ConversationReferenceResult(
                is_conversation_reference=False,
                confidence=0.0,
                method='no_history',
                reasoning='No conversation history available'
            )

        # TECNICA 1: Semantic Similarity com historico
        semantic_result = self._detect_semantic_reference(query, messages)

        if semantic_result.is_conversation_reference and semantic_result.confidence > 0.85:
            logger.info(f"Semantic detection: HIGH confidence ({semantic_result.confidence:.2f})")
            return semantic_result

        # TECNICA 2: Coreference Detection (referencias anaforicas)
        coref_result = self._detect_coreference(query, messages)

        if coref_result.is_conversation_reference and coref_result.confidence > 0.8:
            logger.info(f"Coreference detection: HIGH confidence ({coref_result.confidence:.2f})")
            return coref_result

        # TECNICA 3: LLM-based Analysis (fallback para casos ambiguos)
        if self.enable_llm_fallback:
            # So usar LLM se os metodos anteriores tiveram confianca media (0.4-0.85)
            if semantic_result.confidence > 0.4 or coref_result.confidence > 0.4:
                llm_result = self._detect_via_llm(query, messages)

                # Combinar resultados (ensemble)
                return self._combine_results(semantic_result, coref_result, llm_result)

        # Se nenhuma tecnica detectou referencia com confianca
        return ConversationReferenceResult(
            is_conversation_reference=False,
            confidence=max(semantic_result.confidence, coref_result.confidence),
            method='hybrid',
            reasoning='No strong conversation reference detected'
        )

    def _detect_semantic_reference(
        self,
        query: str,
        messages: List[Dict]
    ) -> ConversationReferenceResult:
        """
        TECNICA 1: Detecta referencia via similaridade semantica

        Se a query e muito similar a uma mensagem anterior (em conceito/topico),
        provavelmente esta referenciando essa mensagem.
        """
        try:
            # Gerar embedding da query
            query_embedding = self.embedding_service.generate_embedding(query)

            # Gerar/recuperar embeddings do historico
            history_embeddings = []
            for i, msg in enumerate(messages):
                content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                if not content:
                    continue

                # Cache key
                cache_key = hash(content[:100])

                if cache_key in self._history_embeddings_cache:
                    emb = self._history_embeddings_cache[cache_key]
                else:
                    emb = self.embedding_service.generate_embedding(content[:500])
                    self._history_embeddings_cache[cache_key] = emb

                history_embeddings.append((i, content, emb))

            if not history_embeddings:
                return ConversationReferenceResult(
                    is_conversation_reference=False,
                    confidence=0.0,
                    method='semantic',
                    reasoning='No valid history messages'
                )

            # Calcular similaridade com cada mensagem
            similarities = []
            for idx, content, emb in history_embeddings:
                sim = self._cosine_similarity(query_embedding, emb)
                similarities.append((idx, content, sim))

            # Ordenar por similaridade
            similarities.sort(key=lambda x: x[2], reverse=True)

            best_idx, best_content, best_sim = similarities[0]

            # Se query e curta e generica, verificar se parece referencia
            query_word_count = len(query.split())

            # Queries curtas com alta similaridade = provavelmente referencia
            if query_word_count <= 10 and best_sim > self.semantic_threshold:
                return ConversationReferenceResult(
                    is_conversation_reference=True,
                    confidence=best_sim,
                    method='semantic',
                    reasoning=f'Query semantically similar to message #{best_idx} (sim={best_sim:.2f})',
                    referenced_message_index=best_idx
                )

            # Queries mais longas precisam de similaridade maior
            elif best_sim > 0.85:
                return ConversationReferenceResult(
                    is_conversation_reference=True,
                    confidence=best_sim,
                    method='semantic',
                    reasoning=f'Strong semantic match with message #{best_idx}',
                    referenced_message_index=best_idx
                )

            return ConversationReferenceResult(
                is_conversation_reference=False,
                confidence=best_sim,
                method='semantic',
                reasoning=f'Similarity ({best_sim:.2f}) below threshold'
            )

        except Exception as e:
            logger.error(f"Semantic detection failed: {e}")
            return ConversationReferenceResult(
                is_conversation_reference=False,
                confidence=0.0,
                method='semantic',
                reasoning=f'Error: {e}'
            )

    def _detect_coreference(
        self,
        query: str,
        messages: List[Dict]
    ) -> ConversationReferenceResult:
        """
        TECNICA 2: Detecta referencias anaforicas

        Identifica pronomes e referencias que apontam para o historico:
        - Ordinais: "primeira", "segunda", "ultima"
        - Demonstrativos: "isso", "aquilo", "esse"
        - Pronomes interrogativos: "qual foi", "o que era"
        - Referencias temporais: "antes", "agora ha pouco", "voce disse"
        """
        query_lower = query.lower()

        # Analisar estrutura da query
        has_ordinal = self._has_ordinal_reference(query_lower)
        has_demonstrative = self._has_demonstrative_reference(query_lower)
        has_temporal = self._has_temporal_reference(query_lower)
        has_meta_reference = self._has_meta_conversation_reference(query_lower)

        # Calcular score baseado nas referencias encontradas
        reference_score = 0.0
        reasons = []

        if has_ordinal:
            reference_score += 0.4
            reasons.append("ordinal reference (primeiro/ultimo/etc)")

        if has_demonstrative:
            reference_score += 0.25
            reasons.append("demonstrative pronoun (isso/aquilo/esse)")

        if has_temporal:
            reference_score += 0.2
            reasons.append("temporal reference (antes/agora/voce disse)")

        if has_meta_reference:
            reference_score += 0.35
            reasons.append("meta-conversation reference (pergunta/resposta/conversa)")

        # Verificar se query e auto-contida (pode ser respondida sem contexto)
        is_self_contained = self._is_self_contained_query(query)

        if is_self_contained:
            reference_score *= 0.5  # Reduzir score se parece auto-contida

        # Se encontrou multiplas referencias, aumentar confianca
        if len(reasons) >= 2:
            reference_score = min(reference_score * 1.2, 1.0)

        is_reference = reference_score > 0.5

        # Tentar identificar qual mensagem esta sendo referenciada
        referenced_idx = None
        if has_ordinal and messages:
            referenced_idx = self._resolve_ordinal_reference(query_lower, len(messages))

        return ConversationReferenceResult(
            is_conversation_reference=is_reference,
            confidence=reference_score,
            method='coreference',
            reasoning='; '.join(reasons) if reasons else 'No coreference detected',
            referenced_message_index=referenced_idx
        )

    def _has_ordinal_reference(self, query: str) -> bool:
        """Detecta referencias ordinais (primeiro, ultimo, etc.)"""
        # Usar analise semantica em vez de lista fixa
        ordinal_concepts = [
            # PT-BR
            "primeir", "segund", "terceir", "quart", "quint",
            "ultim", "penultim", "anterior",
            # EN
            "first", "second", "third", "fourth", "fifth",
            "last", "previous", "prior", "preceding"
        ]
        return any(concept in query for concept in ordinal_concepts)

    def _has_demonstrative_reference(self, query: str) -> bool:
        """Detecta pronomes demonstrativos que referenciam algo anterior"""
        # Demonstrativos que tipicamente referenciam algo da conversa
        demonstratives = [
            # PT-BR
            "isso", "isto", "aquilo", "esse", "este", "aquele",
            "dessa", "desta", "daquela", "nessa", "nesta",
            # EN
            "this", "that", "these", "those"
        ]

        # Verificar se demonstrativo esta em contexto de referencia
        for dem in demonstratives:
            if dem in query:
                # Verificar se nao e parte de uma frase completa
                # Ex: "o que e isso" vs "isso e importante para X"
                words = query.split()
                if len(words) <= 6:  # Query curta com demonstrativo = provavel referencia
                    return True
        return False

    def _has_temporal_reference(self, query: str) -> bool:
        """Detecta referencias temporais a conversa"""
        temporal_markers = [
            # PT-BR
            "antes", "agora ha pouco", "voce disse", "eu disse", "falamos",
            "mencionou", "perguntei", "respondeu", "conversamos",
            # EN
            "earlier", "just now", "you said", "i said", "we discussed",
            "mentioned", "asked", "answered", "talked about"
        ]
        return any(marker in query for marker in temporal_markers)

    def _has_meta_conversation_reference(self, query: str) -> bool:
        """Detecta referencias meta a conversa (pergunta, resposta, topico)"""
        meta_terms = [
            # PT-BR
            "pergunta", "resposta", "conversa", "topico", "assunto",
            "discussao", "dialogo", "questao",
            # EN
            "question", "answer", "conversation", "topic", "subject",
            "discussion", "dialogue"
        ]
        return any(term in query for term in meta_terms)

    def _is_self_contained_query(self, query: str) -> bool:
        """
        Verifica se query pode ser respondida sem contexto da conversa

        Ex: "O que e machine learning?" e auto-contida
        Ex: "O que foi a primeira pergunta?" NAO e auto-contida
        """
        query_lower = query.lower()

        # Indicadores de NAO ser auto-contida
        not_self_contained_indicators = [
            "qual foi", "o que foi", "what was",  # Passado indefinido
            "voce falou", "eu disse", "we talked",  # Referencia a conversa
            "sobre isso", "about that", "about this",  # Referencia anaforica
            "a pergunta", "the question",  # Referencia a pergunta anterior
            "antes", "earlier", "previously",  # Temporal
        ]

        for indicator in not_self_contained_indicators:
            if indicator in query_lower:
                return False

        # Indicadores de ser auto-contida
        self_contained_indicators = [
            "o que e", "what is",  # Definicao
            "como funciona", "how does", "how do",  # Explicacao
            "por que", "why",  # Razao
            "quem e", "who is",  # Identificacao
            "onde fica", "where is",  # Localizacao
        ]

        for indicator in self_contained_indicators:
            if indicator in query_lower:
                return True

        return False

    def _resolve_ordinal_reference(self, query: str, history_length: int) -> Optional[int]:
        """Resolve qual mensagem um ordinal referencia"""
        # Mapear ordinais para indices (0-based)
        # "primeira pergunta" = indice 0 (primeira user message)
        # "ultima pergunta" = indice -1 (ultima user message)

        if "primeir" in query or "first" in query:
            return 0
        elif "segund" in query or "second" in query:
            return 1
        elif "terceir" in query or "third" in query:
            return 2
        elif "ultim" in query or "last" in query:
            return history_length - 1
        elif "penultim" in query or "second to last" in query:
            return history_length - 2 if history_length >= 2 else 0
        elif "anterior" in query or "previous" in query:
            return history_length - 1  # Geralmente a anterior a atual

        return None

    def _detect_via_llm(
        self,
        query: str,
        messages: List[Dict]
    ) -> ConversationReferenceResult:
        """
        TECNICA 3: LLM analisa se query referencia a conversa

        Usado como fallback para casos ambiguos.
        """
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            # Formatar historico resumido
            history_summary = []
            for i, msg in enumerate(messages[-5:]):  # Ultimas 5 mensagens
                content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                role = msg.get('role', 'unknown') if isinstance(msg, dict) else 'message'
                preview = content[:100] + "..." if len(content) > 100 else content
                history_summary.append(f"[{i+1}] {role}: {preview}")

            history_text = "\n".join(history_summary)

            prompt = f"""Analyze if this query is asking about the conversation itself (meta-question) or is a new independent question.

CONVERSATION HISTORY:
{history_text}

CURRENT QUERY: "{query}"

Is this query:
A) Asking about something FROM THE CONVERSATION (e.g., "what was the first question?", "can you repeat that?", "what did I ask about X?")
B) A NEW INDEPENDENT QUESTION that could be asked without any prior context

Respond with ONLY:
CONVERSATION_REFERENCE: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""

            response = self.llm.invoke([
                SystemMessage(content="You analyze queries to determine if they reference conversation history."),
                HumanMessage(content=prompt)
            ])

            content = response.content.lower()

            is_reference = "conversation_reference: yes" in content

            # Extrair confianca
            confidence = 0.5
            if "confidence:" in content:
                try:
                    conf_line = [l for l in content.split('\n') if 'confidence:' in l][0]
                    confidence = float(conf_line.split(':')[1].strip())
                except:
                    confidence = 0.7 if is_reference else 0.3

            # Extrair reasoning
            reasoning = "LLM analysis"
            if "reasoning:" in content:
                try:
                    reason_line = [l for l in response.content.split('\n') if 'REASONING:' in l][0]
                    reasoning = reason_line.split(':')[1].strip()
                except:
                    pass

            return ConversationReferenceResult(
                is_conversation_reference=is_reference,
                confidence=confidence,
                method='llm',
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return ConversationReferenceResult(
                is_conversation_reference=False,
                confidence=0.0,
                method='llm',
                reasoning=f'Error: {e}'
            )

    def _combine_results(
        self,
        semantic: ConversationReferenceResult,
        coref: ConversationReferenceResult,
        llm: ConversationReferenceResult
    ) -> ConversationReferenceResult:
        """
        Combina resultados das 3 tecnicas (ensemble voting)
        """
        # Pesos para cada metodo
        weights = {
            'semantic': 0.35,
            'coreference': 0.30,
            'llm': 0.35
        }

        results = [
            (semantic, weights['semantic']),
            (coref, weights['coreference']),
            (llm, weights['llm'])
        ]

        # Calcular score ponderado
        weighted_score = sum(
            r.confidence * w * (1.0 if r.is_conversation_reference else 0.0)
            for r, w in results
        )

        # Normalizar
        total_weight = sum(w for r, w in results if r.confidence > 0)
        if total_weight > 0:
            weighted_score /= total_weight

        # Votacao: quantos metodos concordam?
        votes = sum(1 for r, _ in results if r.is_conversation_reference and r.confidence > 0.5)

        is_reference = weighted_score > 0.5 or votes >= 2

        # Encontrar melhor referenced_message_index
        ref_idx = None
        for r, _ in results:
            if r.referenced_message_index is not None:
                ref_idx = r.referenced_message_index
                break

        # Combinar reasonings
        reasons = [f"{r.method}: {r.reasoning}" for r, _ in results if r.confidence > 0.3]

        return ConversationReferenceResult(
            is_conversation_reference=is_reference,
            confidence=weighted_score,
            method='hybrid',
            reasoning='; '.join(reasons),
            referenced_message_index=ref_idx
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similaridade de cosseno"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
