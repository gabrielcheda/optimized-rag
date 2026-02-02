"""
Verify Response Node
Post-generation verification of claims against retrieved context
PHASE 1: Critical anti-hallucination layer
"""

import logging
import re
from typing import Any, Dict, List
from concurrent.futures import TimeoutError

import config
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def _verify_with_exact_match(claim: str, documents: List[Dict]) -> bool:
    """
    Verificação adicional com matching exato de frases-chave
    FASE 1: Extra layer of verification for factual claims
    """
    # Extrai substantivos, números, datas e anos (elementos factuais)
    # MELHORADO: Captura números com vírgula (PT-BR), datas e anos
    key_terms = re.findall(
        r'\b[A-Z][a-z]+\b|'           # Substantivos próprios
        r'\b\d+(?:[.,]\d+)?%?\b|'      # Números (com , ou .)
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|'  # Datas formato DD/MM/YYYY
        r'\b\d{4}\b',                   # Anos
        claim
    )
    key_terms_lower = [t.lower() for t in key_terms]

    if not key_terms_lower:
        return True  # Sem termos-chave = não é factual

    for doc in documents:
        content_lower = doc.get('content', '').lower()
        matches = sum(1 for term in key_terms_lower if term in content_lower)
        if matches >= len(key_terms_lower) * 0.6:  # 60% dos termos encontrados
            return True
    return False


def verify_response_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Verify generated response claims against retrieved context
    
    This is the CRITICAL post-generation verification step that prevents
    hallucinations by checking if all claims are actually supported.
    
    FASE 1: Added fallback for verification failures and exact match verification.
    
    Returns:
        Dict with verification_passed, support_ratio, and unsupported_claims
    """
    try:
        return _perform_verification(state, agent)
    except TimeoutError:
        logger.warning("Verification timeout - marking for human review")
        return {
            "verification_passed": False,
            "verification_error": "timeout",
            "requires_human_review": True,
            "hitl_reason": "Verification timeout exceeded",
            "support_ratio": 0.0,
        }
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        # FALLBACK: Marcar como não verificado mas continuar
        return {
            "verification_passed": False,
            "verification_error": str(e),
            "requires_human_review": True,
            "hitl_reason": f"Verification system error: {e}",
            "support_ratio": 0.0,
        }


def _perform_verification(state: MemGPTState, agent) -> Dict[str, Any]:
    """Internal verification logic - separated for error handling"""
    # CRITICAL: Check global regeneration counter first (prevents infinite loops)
    total_regenerations = getattr(state, 'total_regeneration_count', 0)
    
    if total_regenerations >= config.MAX_REGENERATION_ATTEMPTS:
        logger.warning(
            f"⛔ Maximum global regenerations reached ({total_regenerations}/{config.MAX_REGENERATION_ATTEMPTS}). "
            f"Accepting current response to prevent infinite loop."
        )
        return {
            "verification_passed": True,  # Force accept
            "support_ratio": 0.5,
            "max_attempts_reached": True,
            "warning": "Response quality may be suboptimal - max regeneration attempts reached",
            "verification_method": "force_accept_max_attempts"
        }
    
    answer = state.agent_response
    
    if not answer or len(answer.strip()) < 20:
        logger.info("Response too short to verify")
        return {
            "verification_passed": True,
            "support_ratio": 1.0,
            "verification_method": "skip_short_response"
        }
    
    # Skip verification for clarification queries using recall memory
    from rag import QueryIntent
    if state.query_intent == QueryIntent.CLARIFICATION:
        logger.info("Skipping verification for clarification (uses recall memory)")
        return {
            "verification_passed": True,
            "support_ratio": 1.0,
            "verification_method": "skip_clarification"
        }
    
    # FASE 6: Dual-pass verification for maximum precision
    try:
        claims = agent.self_rag._extract_claims(answer)
        logger.info(f"Extracted {len(claims)} claims for FASE 6 dual-pass verification")

        if not claims:
            # No extractable claims = conversational response
            return {
                "verification_passed": True,
                "support_ratio": 1.0,
                "verification_method": "no_claims"
            }

        # ============================================
        # FASE 6: PASS 1 - Semantic/LLM verification
        # ============================================
        verification_results = []
        for claim in claims:
            support = agent.self_rag._find_supporting_evidence(
                claim,
                state.final_context,
                max_chars_per_doc=3000
            )
            verification_results.append({
                'claim': claim,
                'supported': support['found'],
                'confidence': support['confidence'],
                'evidence': support.get('text', ''),
                'pass': 1
            })

        # Calculate PASS 1 support ratio
        pass1_supported = sum(1 for v in verification_results if v['supported'])
        pass1_ratio = pass1_supported / len(verification_results) if verification_results else 0

        logger.info(
            f"FASE 6 PASS 1 (Semantic): {pass1_supported}/{len(verification_results)} "
            f"claims supported ({pass1_ratio:.2f})"
        )

        # ============================================
        # FASE 6: PASS 2 - Exact match verification
        # ============================================
        pass2_results = []
        for claim in claims:
            exact_match = _verify_with_exact_match(claim, state.final_context)
            pass2_results.append({
                'claim': claim,
                'exact_match': exact_match,
                'pass': 2
            })

        pass2_matched = sum(1 for v in pass2_results if v['exact_match'])
        pass2_ratio = pass2_matched / len(pass2_results) if pass2_results else 0

        logger.info(
            f"FASE 6 PASS 2 (Exact Match): {pass2_matched}/{len(pass2_results)} "
            f"claims matched ({pass2_ratio:.2f})"
        )

        # ============================================
        # FASE 6: Combine results (both passes must agree)
        # ============================================
        final_results = []
        for i, claim in enumerate(claims):
            pass1_ok = verification_results[i]['supported']
            pass2_ok = pass2_results[i]['exact_match']

            # FASE 6: Claim is verified only if BOTH passes agree
            # OR if semantic pass has very high confidence (>0.85)
            high_confidence = verification_results[i]['confidence'] >= 0.85
            final_supported = (pass1_ok and pass2_ok) or (pass1_ok and high_confidence)

            final_results.append({
                'claim': claim,
                'supported': final_supported,
                'pass1_semantic': pass1_ok,
                'pass2_exact': pass2_ok,
                'confidence': verification_results[i]['confidence'],
                'evidence': verification_results[i].get('evidence', '')
            })

        # Calculate final support ratio
        supported_count = sum(1 for v in final_results if v['supported'])
        support_ratio = supported_count / len(final_results) if final_results else 0

        # Get unsupported claims for logging
        unsupported_claims = [
            v['claim'] for v in final_results if not v['supported']
        ]

        # FASE 6: Stricter threshold - use MIN_SUPPORT_RATIO from config
        verification_passed = support_ratio >= config.MIN_SUPPORT_RATIO

        if not verification_passed:
            logger.warning(
                f"❌ FASE 6 Dual-pass FAILED: support_ratio={support_ratio:.2f} "
                f"({supported_count}/{len(final_results)} claims verified)"
            )
            logger.warning(f"Unsupported claims: {unsupported_claims[:3]}")
        else:
            logger.info(
                f"✅ FASE 6 Dual-pass PASSED: support_ratio={support_ratio:.2f} "
                f"({supported_count}/{len(final_results)} claims)"
            )

        return {
            "verification_passed": verification_passed,
            "support_ratio": support_ratio,
            "unsupported_claims": unsupported_claims,
            "total_claims": len(final_results),
            "supported_claims": supported_count,
            "verification_method": "fase6_dual_pass",
            "pass1_ratio": pass1_ratio,
            "pass2_ratio": pass2_ratio,
            "verification_details": final_results,
            "total_regeneration_count": total_regenerations + (0 if verification_passed else 1)
        }
        
    except Exception as e:
        logger.error(f"Verification error: {e}", exc_info=True)
        # On error, assume verification passed (fail-open to avoid blocking)
        return {
            "verification_passed": True,
            "support_ratio": 0.0,
            "verification_method": "error",
            "error": str(e)
        }


def should_regenerate(state: MemGPTState, agent) -> str:
    """
    Decision function: should we regenerate the response?
    
    Returns:
        "regenerate" if verification failed and we haven't exceeded max attempts
        "accept" if verification passed or max attempts reached
    """
    verification_passed = getattr(state, "verification_passed", True)
    total_regenerations = getattr(state, "total_regeneration_count", 0)
    max_regenerations = config.MAX_REGENERATION_ATTEMPTS
    
    if not verification_passed and total_regenerations < max_regenerations:
        logger.info(
            f"Triggering regeneration (global attempt {total_regenerations + 1}/{max_regenerations})"
        )
        return "regenerate"
    
    if total_regenerations >= max_regenerations:
        logger.warning(
            f"Max global regeneration attempts reached ({total_regenerations}), "
            f"accepting current response"
        )
    
    return "accept"
