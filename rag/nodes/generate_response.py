"""
Generate Response Node
Generates final AI response using enriched context
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

import config
from agent.state import CitedResponse, MemGPTState
from prompts.generate_response import CLARIFICATION_INSTRUCTION, FEW_SHOT_EXAMPLES, SYSTEM_PROMPT_TEMPLATE
from rag.nodes.helpers import check_context_quality, enrich_context_with_memory
from rag.citation_validator import CitationValidator

logger = logging.getLogger(__name__)

# FASE 6: Maximum Precision Prompt (Bilíngue PT-BR/EN)
# Objective: 100% precision, <1% hallucination
STRUCTURED_OUTPUT_PROMPT = """You are a PRECISION-CRITICAL AI assistant. Your ONLY job is to extract and cite information from documents.
Você é um assistente de IA de PRECISÃO CRÍTICA. Seu ÚNICO trabalho é extrair e citar informações dos documentos.

## 🚨 ABSOLUTE RULES (VIOLATION = TOTAL FAILURE) / REGRAS ABSOLUTAS (VIOLAÇÃO = FALHA TOTAL):

### RULE 1: CITE OR DIE
- EVERY factual statement MUST have [N] citation / TODA afirmação factual DEVE ter citação [N]
- No citation = NO statement. Do NOT write anything you cannot cite.
- Sem citação = NÃO escreva. NÃO escreva NADA que você não possa citar.

### RULE 2: DOCUMENTS ARE THE ONLY TRUTH
- Your training knowledge is FORBIDDEN. Pretend you know NOTHING.
- Seu conhecimento de treinamento é PROIBIDO. Finja que você não sabe NADA.
- If it's not in the documents → IT DOES NOT EXIST for you.
- Se não está nos documentos → NÃO EXISTE para você.

### RULE 3: HONEST UNCERTAINTY
- When information is incomplete → say "Based on document [N], I can only confirm X. I cannot find information about Y."
- Quando informação incompleta → diga "Com base no documento [N], só posso confirmar X. Não encontro informação sobre Y."
- When no relevant documents → say "I don't have documents that answer this question."
- Quando sem documentos relevantes → diga "Não tenho documentos que respondam a esta pergunta."

### RULE 4: NO EXTRAPOLATION
- Do NOT infer, guess, assume, or extend beyond explicit document content.
- NÃO infira, adivinhe, assuma, ou extrapole além do conteúdo explícito dos documentos.
- If a document says "A leads to B", you CANNOT say "A always leads to B" unless the document says "always".
- Se documento diz "A leva a B", você NÃO PODE dizer "A sempre leva a B" a menos que o documento diga "sempre".

## CITATION FORMAT / FORMATO DE CITAÇÃO:
- [1] immediately after the claim it supports / [1] imediatamente após a afirmação que suporta
- Example: "The capital is Paris [1]." NOT "The capital is Paris. [1]"
- Multiple sources: "This is supported by both studies [1][2]."

## SELF-CHECK BEFORE RESPONDING / AUTO-VERIFICAÇÃO ANTES DE RESPONDER:
Ask yourself for EACH sentence:
1. ✅ Is there a [N] citation? If NO → Add citation OR delete sentence.
2. ✅ Can I point to the EXACT text in document [N]? If NO → Delete sentence.
3. ✅ Am I adding ANY information not in [N]? If YES → Delete the added info.

DOCUMENTS / DOCUMENTOS:
{documents}

Answer the question with MANDATORY citations / Responda à pergunta com citações OBRIGATÓRIAS:"""


def generate_response_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Generate AI response (Paper-compliant: few-shot prompting)"""
    # Enriched context: Combines core memory + conversation + retrieved docs
    enriched_context, source_map = enrich_context_with_memory(state, agent)

    # Check if this is a clarification question (uses recall memory instead of docs)
    from rag import QueryIntent

    is_clarification = state.query_intent == QueryIntent.CLARIFICATION
    has_recall_memory = state.retrieved_recall and len(state.retrieved_recall) > 0

    # BYPASS: For clarification with recall memory, skip document quality check
    if is_clarification and has_recall_memory:
        logger.info(
            f"Clarification intent with {len(state.retrieved_recall)} recall messages - using recall memory"
        )
        context_quality = {
            "sufficient": True,
            "reason": "Using recall memory for clarification",
        }
    else:
        # Check context quality before generating response (CRITICAL: prevents hallucination)
        context_quality = check_context_quality(state.final_context, min_score=0.3)

    if not context_quality["sufficient"]:
        logger.warning(
            f"Insufficient context quality: {context_quality['reason']}, "
            f"max_score={context_quality.get('max_score', 0):.2f}"
        )

        # Return early with honest fallback message
        fallback_message = context_quality["message"]

        return {
            "agent_response": fallback_message,
            "messages": [{"role": "assistant", "content": fallback_message}],
            "faithfulness_score": {
                "score": 0.0,
                "reasoning": "Insufficient context - honest fallback",
            },
            "context_quality": context_quality,
            "source_map": {},
        }

    # CRITICAL: Additional check for empty final_context (prevents hallucination)
    if not state.final_context or len(state.final_context) == 0:
        logger.error("❌ CRITICAL: final_context is empty despite passing context_quality check!")
        fallback_message = "I cannot answer this question as I don't have relevant documents in my knowledge base. Please try rephrasing your question or check if documents were uploaded correctly."
        return {
            "agent_response": fallback_message,
            "messages": [{"role": "assistant", "content": fallback_message}],
            "verification_passed": False,
            "no_context_available": True,
            "source_map": {},
        }

    # Build explicit context summary to force LLM awareness
    context_summary = "\n📚 DOCUMENTS PROVIDED (USE ONLY THESE):\n\n"
    for i, doc in enumerate(state.final_context[:5]):
        source = doc.get('source', 'Unknown')
        score = doc.get('score', 0.0)
        content_preview = doc.get('content', '')[:500]
        context_summary += f"[{i+1}] {source} (relevance: {score:.2f})\n"
        context_summary += f"Content: {content_preview}...\n\n"
    
    context_summary += "🚨 IMPORTANT: Answer using ONLY the documents [1]-[{0}] above. Do NOT use training knowledge.\n".format(len(state.final_context[:5]))
    
    # Prepend context summary to enriched_context
    enriched_context_with_summary = context_summary + "\n" + enriched_context

    # Build system prompt with few-shot examples and context
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        few_shot_examples=FEW_SHOT_EXAMPLES, 
        enriched_context=enriched_context_with_summary
    )
    
    # Add clarification instruction if needed
    if is_clarification:
        system_prompt += CLARIFICATION_INSTRUCTION

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.user_input),
    ]

    # FIX 2.3: Log prompt size for debugging truncation issues
    prompt_chars = len(system_prompt) + len(state.user_input)
    prompt_tokens_estimate = prompt_chars // 4  # Rough estimate
    logger.info(f"Prompt size: {prompt_chars} chars (~{prompt_tokens_estimate} tokens)")
    if prompt_tokens_estimate > 12000:
        logger.warning(f"Prompt may be too long ({prompt_tokens_estimate} tokens) - risk of truncation")

    # FIX 2.2: Use structured output for factual queries to FORCE citations
    tool_calls = []
    response = None  # Initialize to avoid UnboundLocalError in cost tracking
    use_structured_output = (
        not is_clarification
        and state.final_context
        and len(state.final_context) > 0
        and hasattr(agent, 'llm')  # Check if we have access to base LLM
    )

    if use_structured_output:
        try:
            # Build simplified prompt for structured output
            docs_text = "\n\n".join([
                f"[{i+1}] {doc.get('source', 'Unknown')}: {doc.get('content', '')[:1500]}"
                for i, doc in enumerate(state.final_context[:5])
            ])
            structured_prompt = STRUCTURED_OUTPUT_PROMPT.format(documents=docs_text)

            # Use structured output to force citation format
            structured_llm = agent.llm.with_structured_output(CitedResponse)
            structured_messages = [
                SystemMessage(content=structured_prompt),
                HumanMessage(content=state.user_input),
            ]

            cited_response = structured_llm.invoke(structured_messages)

            # Extract answer with forced citations
            answer = cited_response.answer
            citations_used = cited_response.citations_used

            logger.info(
                f"Structured output: {len(citations_used)} citations used, "
                f"confidence={cited_response.confidence:.2f}"
            )

            # Validate citations are present
            if not citations_used:
                logger.warning("Structured output returned empty citations - falling back to standard generation")
                # Fall through to standard generation
                raise ValueError("Empty citations from structured output")

        except Exception as e:
            logger.warning(f"Structured output failed: {e} - falling back to standard generation")
            # Fall back to standard generation
            response = agent.llm_with_tools.invoke(messages)
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                logger.info(f"LLM requested {len(tool_calls)} tool calls")
            answer = response.content if hasattr(response, "content") else str(response)
    else:
        # Standard generation (for clarification or when structured output not applicable)
        response = agent.llm_with_tools.invoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = response.tool_calls
            logger.info(f"LLM requested {len(tool_calls)} tool calls")
        answer = response.content if hasattr(response, "content") else str(response)

    # Phase 1: Citation validation (if enabled)
    citation_validation = {}
    if config.ENABLE_CITATION_VALIDATION and source_map:
        validator = CitationValidator(strict_mode=True)
        citation_validation = validator.validate_citations(answer, source_map)
        if not citation_validation["valid"]:
            logger.warning(
                f"Citation validation failed: {citation_validation.get('error', 'Unknown error')} "
                f"(cited: {citation_validation.get('cited_sources', [])}, "
                f"uncited: {citation_validation.get('uncited_count', 0)})"
            )

    # Paper-compliant: Faithfulness Evaluation
    faithfulness = {}
    if agent.evaluator and state.final_context:
        faithfulness = agent.evaluator.faithfulness_score(
            answer=answer, context=state.final_context, llm=agent.llm
        )
        logger.info(
            f"Faithfulness score: {faithfulness.get('score', 0):.2f} - "
            f"{faithfulness.get('reasoning', 'N/A')[:100]}"
        )

    # Calculate comprehensive quality score using factuality scorer
    factuality_result = {}
    auto_refused = False

    # FIX: Skip factuality check for clarification intent (uses recall memory, not documents)
    # Factuality scorer expects documents in final_context, but clarifications use recall memory
    if is_clarification and has_recall_memory:
        logger.info("Skipping factuality check for clarification (recall-based answer)")
        factuality_result = {
            "factuality_score": faithfulness.get("score", 0.7),
            "quality_level": "GOOD",
            "passed": True,
            "method": "clarification_bypass",
            "recommendation": answer,
        }
    elif agent.factuality_scorer:
        try:
            # Phase 2: All answers must pass factuality check (early exit removed)
            factuality_result = agent.factuality_scorer.calculate_factuality_score(
                query=state.user_input,
                answer=answer,
                retrieved_docs=state.final_context,
                source_map=source_map,
            )

            # Phase 1: Stricter quality control - require BOTH scores if enabled
            factuality_score = factuality_result["factuality_score"]
            factuality_passed = factuality_result.get("passed", False)
            
            # Check if we should refuse (increased threshold from 0.25 to config.MIN_FACTUALITY_SCORE)
            should_refuse = agent.factuality_scorer.should_refuse_answer(
                factuality_score, threshold=config.MIN_FACTUALITY_SCORE
            )
            
            # If REQUIRE_BOTH_SCORES_HIGH is enabled, require BOTH faithfulness AND factuality
            if config.REQUIRE_BOTH_SCORES_HIGH:
                # Extract faithfulness score (with safe fallback)
                faithfulness_score = faithfulness.get("score", 0.7) if faithfulness else 0.7
                both_low = (faithfulness_score < 0.7) and (factuality_score < 0.5)
                if both_low or (should_refuse and not factuality_passed):
                    auto_refused = True
                    fallback_message = factuality_result["recommendation"]
                    
                    logger.warning(
                        f"Auto-refusing answer - faithfulness: {faithfulness_score:.2f}, "
                        f"factuality: {factuality_score:.2f} ({factuality_result['quality_level']})"
                    )
                    
                    # FIX 1.1: Increment regeneration counter to prevent infinite loop
                    current_regen_count = getattr(state, 'total_regeneration_count', 0)
                    return {
                        "agent_response": fallback_message,
                        "messages": [{"role": "assistant", "content": fallback_message}],
                        "faithfulness_score": {
                            "score": 0.0,
                            "reasoning": "Auto-refused due to low quality scores",
                        },
                        "factuality_score": factuality_result,
                        "source_map": {},
                        "tool_calls": [],
                        "auto_refused": True,
                        "citation_validation": citation_validation,
                        "total_regeneration_count": current_regen_count + 1,
                        "verification_passed": False,
                    }
            else:
                # Original behavior: refuse only if factuality is very low
                if should_refuse and not factuality_passed:
                    auto_refused = True
                    fallback_message = factuality_result["recommendation"]
                    
                    logger.warning(
                        f"Auto-refusing answer due to low factuality: "
                        f"{factuality_score:.2f} ({factuality_result['quality_level']})"
                    )
                    
                    # FIX 1.1: Increment regeneration counter to prevent infinite loop
                    current_regen_count = getattr(state, 'total_regeneration_count', 0)
                    return {
                        "agent_response": fallback_message,
                        "messages": [{"role": "assistant", "content": fallback_message}],
                        "faithfulness_score": {
                            "score": 0.0,
                            "reasoning": "Auto-refused due to low factuality",
                        },
                        "factuality_score": factuality_result,
                        "source_map": {},
                        "tool_calls": [],
                        "auto_refused": True,
                        "citation_validation": citation_validation,
                        "total_regeneration_count": current_regen_count + 1,
                        "verification_passed": False,
                    }

            logger.info(
                f"Factuality score: {factuality_result['factuality_score']:.3f} "
                f"({factuality_result['quality_level']}) - "
                f"support={factuality_result['components']['support_ratio']:.2f}, "
                f"citations={factuality_result['components']['citation_coverage']:.2f}"
            )
        except Exception as e:
            logger.error(f"Factuality scoring failed: {e}")
            factuality_result = {}
    
    # Phase 2: Uncertainty quantification
    uncertainty_info = {}
    if config.ENABLE_UNCERTAINTY_QUANTIFICATION:
        # Extract faithfulness score with safe fallback
        faithfulness_score = faithfulness.get("score", 0.0) if faithfulness else 0.0
        uncertainty_info = _quantify_uncertainty(
            answer=answer,
            faithfulness_score=faithfulness_score,
            factuality_result=factuality_result,
            citation_validation=citation_validation,
            context_quality=context_quality
        )
        
        # Log uncertainty level
        if uncertainty_info.get("high_uncertainty", False):
            logger.warning(
                f"High uncertainty detected: {uncertainty_info['uncertainty_score']:.2f} - "
                f"Reasons: {', '.join(uncertainty_info.get('reasons', []))}"
            )
        
        # Optionally show confidence in response
        if config.SHOW_CONFIDENCE_IN_RESPONSE and uncertainty_info.get("confidence_score"):
            confidence_pct = int(uncertainty_info["confidence_score"] * 100)
            answer = f"{answer}\n\n[Confidence: {confidence_pct}%]"
    
    # Phase 3: Temporal validation (if enabled)
    temporal_validation = {}
    if config.ENABLE_TEMPORAL_VALIDATION:
        try:
            from rag.temporal_validator import TemporalValidator
            
            temporal_validator = TemporalValidator()
            temporal_validation = temporal_validator.validate_temporal_consistency(
                answer=answer,
                documents=state.final_context
            )
            
            if not temporal_validation.get("valid", True):
                logger.warning(
                    f"Temporal validation failed: {temporal_validation.get('inconsistency_count', 0)} issues - "
                    f"{temporal_validation.get('warning', 'Unknown')}"
                )
        except Exception as e:
            logger.error(f"Temporal validation failed: {e}")
            temporal_validation = {"valid": True, "error": str(e)}
    
    # Phase 3: Human-in-the-Loop detection (if enabled)
    requires_human_review = False
    hitl_reason = None
    if config.ENABLE_HUMAN_IN_THE_LOOP and not auto_refused:
        factuality_score = factuality_result.get("factuality_score", 1.0)
        # Extract faithfulness score with safe fallback
        faithfulness_score = faithfulness.get("score", 1.0) if faithfulness else 1.0
        
        # Gray zone detection: 0.4 <= score < 0.6
        in_gray_zone = (
            (0.4 <= factuality_score < 0.6) or 
            (0.4 <= faithfulness_score < 0.6)
        )
        
        # High uncertainty detection
        high_uncertainty = uncertainty_info.get("high_uncertainty", False) if uncertainty_info else False
        
        # Temporal issues
        temporal_issues = not temporal_validation.get("valid", True) if temporal_validation else False
        
        if in_gray_zone or high_uncertainty or temporal_issues:
            requires_human_review = True
            reasons = []
            if in_gray_zone:
                reasons.append(f"ambiguous quality (factuality: {factuality_score:.2f}, faithfulness: {faithfulness_score:.2f})")
            if high_uncertainty:
                reasons.append(f"high uncertainty ({uncertainty_info.get('uncertainty_score', 0):.2f})")
            if temporal_issues:
                reasons.append(f"temporal inconsistencies ({temporal_validation.get('inconsistency_count', 0)})")
            
            hitl_reason = "; ".join(reasons)
            
            logger.warning(
                f"HUMAN-IN-THE-LOOP triggered: {hitl_reason}"
            )

            # FIX 1.2: Store warning in metadata instead of appending to answer
            # This prevents Self-RAG from extracting meta-warnings as claims
            # The warning should be displayed to user AFTER verification, not before

    # Track LLM costs if cost tracker enabled
    if agent.cost_tracker and response is not None and hasattr(response, "usage"):
        try:
            usage = getattr(response, "usage", None)
            if usage:
                agent.cost_tracker.track_llm(
                    model=config.LLM_MODEL,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")

    # FIX 1.2: Build HITL warning message separately (to append AFTER verification)
    hitl_warning_message = None
    if requires_human_review and hitl_reason:
        hitl_warning_message = (
            f"\n\n⚠️ **Note:** This response requires human review due to: {hitl_reason}. "
            f"Please verify the information before using it."
        )

    return {
        "agent_response": answer,
        "messages": [{"role": "assistant", "content": answer}],
        "faithfulness_score": faithfulness,
        "factuality_score": factuality_result,
        "source_map": source_map,
        "tool_calls": tool_calls,
        "auto_refused": auto_refused,
        "citation_validation": citation_validation,
        "uncertainty_info": uncertainty_info if 'uncertainty_info' in locals() else {},
        "temporal_validation": temporal_validation if 'temporal_validation' in locals() else {},
        "requires_human_review": requires_human_review if 'requires_human_review' in locals() else False,
        "hitl_reason": hitl_reason if 'hitl_reason' in locals() else None,
        "hitl_warning_message": hitl_warning_message,  # FIX 1.2: Append to response AFTER verification
    }


def _quantify_uncertainty(
    answer: str,
    faithfulness_score: float,
    factuality_result: Dict[str, Any],
    citation_validation: Dict[str, Any],
    context_quality: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Phase 2: Quantify uncertainty in generated response
    
    Args:
        answer: Generated response
        faithfulness_score: Faithfulness evaluation score
        factuality_result: Factuality scoring results
        citation_validation: Citation validation results
        context_quality: Context quality assessment
        
    Returns:
        Dict with uncertainty metrics and confidence score
    """
    uncertainty_reasons = []
    uncertainty_score = 0.0
    
    # Factor 1: Low faithfulness (weight: 0.3)
    if faithfulness_score < 0.7:
        uncertainty_reasons.append(f"Low faithfulness ({faithfulness_score:.2f})")
        uncertainty_score += 0.3 * (1.0 - faithfulness_score)
    
    # Factor 2: Low factuality (weight: 0.3)
    factuality_score = factuality_result.get("factuality_score", 0.5)
    if factuality_score < 0.5:
        uncertainty_reasons.append(f"Low factuality ({factuality_score:.2f})")
        uncertainty_score += 0.3 * (1.0 - factuality_score)
    
    # Factor 3: Poor citation coverage (weight: 0.2)
    if not citation_validation.get("valid", True):
        citation_count = citation_validation.get("citation_count", 0)
        uncertainty_reasons.append(f"Poor citations ({citation_count} citations)")
        uncertainty_score += 0.2
    
    # Factor 4: Weak context quality (weight: 0.2)
    if not context_quality.get("sufficient", True):
        max_score = context_quality.get("max_score", 0)
        uncertainty_reasons.append(f"Weak context (max={max_score:.2f})")
        uncertainty_score += 0.2 * (1.0 - max_score)
    
    # Factor 5: Hedging language detection
    hedging_patterns = [
        "might", "maybe", "possibly", "probably", "likely", "perhaps",
        "it seems", "appears to", "could be", "may be", "uncertain",
        "not sure", "unclear"
    ]
    hedging_count = sum(1 for pattern in hedging_patterns if pattern in answer.lower())
    if hedging_count >= 3:
        uncertainty_reasons.append(f"Hedging language ({hedging_count} instances)")
        uncertainty_score += min(0.1 * hedging_count, 0.3)
    
    # Normalize uncertainty score (0-1)
    uncertainty_score = min(uncertainty_score, 1.0)
    
    # Calculate confidence (inverse of uncertainty)
    confidence_score = 1.0 - uncertainty_score
    
    # Determine if uncertainty is high (threshold: 0.4)
    high_uncertainty = uncertainty_score >= 0.4
    
    return {
        "uncertainty_score": round(uncertainty_score, 3),
        "confidence_score": round(confidence_score, 3),
        "high_uncertainty": high_uncertainty,
        "reasons": uncertainty_reasons,
        "hedging_count": hedging_count
    }
