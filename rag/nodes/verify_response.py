"""
Verify Response Node
Post-generation verification of claims against retrieved context
PHASE 1: Critical anti-hallucination layer
"""

import logging
from typing import Any, Dict

import config
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def verify_response_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Verify generated response claims against retrieved context
    
    This is the CRITICAL post-generation verification step that prevents
    hallucinations by checking if all claims are actually supported.
    
    Returns:
        Dict with verification_passed, support_ratio, and unsupported_claims
    """
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
    
    # Use Self-RAG evaluator to extract and verify claims
    try:
        claims = agent.self_rag._extract_claims(answer)
        logger.info(f"Extracted {len(claims)} claims for verification")
        
        if not claims:
            # No extractable claims = conversational response
            return {
                "verification_passed": True,
                "support_ratio": 1.0,
                "verification_method": "no_claims"
            }
        
        # Verify each claim against final context
        verification_results = []
        for claim in claims:
            support = agent.self_rag._find_supporting_evidence(
                claim,
                state.final_context,
                max_chars_per_doc=3000  # Increased from 2500
            )
            verification_results.append({
                'claim': claim,
                'supported': support['found'],
                'confidence': support['confidence'],
                'evidence': support.get('text', '')
            })
        
        # Calculate support ratio
        supported_count = sum(1 for v in verification_results if v['supported'])
        support_ratio = supported_count / len(verification_results) if verification_results else 0
        
        # Get unsupported claims for logging
        unsupported_claims = [
            v['claim'] for v in verification_results if not v['supported']
        ]
        
        # Decision: Pass if support ratio meets threshold
        verification_passed = support_ratio >= config.MIN_SUPPORT_RATIO
        
        if not verification_passed:
            logger.warning(
                f"❌ Verification FAILED: support_ratio={support_ratio:.2f} "
                f"({supported_count}/{len(verification_results)} claims supported)"
            )
            logger.warning(f"Unsupported claims: {unsupported_claims[:3]}")  # Log first 3
        else:
            logger.info(
                f"✅ Verification PASSED: support_ratio={support_ratio:.2f} "
                f"({supported_count}/{len(verification_results)} claims)"
            )
        
        return {
            "verification_passed": verification_passed,
            "support_ratio": support_ratio,
            "unsupported_claims": unsupported_claims,
            "total_claims": len(verification_results),
            "supported_claims": supported_count,
            "verification_method": "claim_level",
            "verification_details": verification_results,
            "total_regeneration_count": total_regenerations + (0 if verification_passed else 1)  # Increment GLOBAL counter on failure
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
