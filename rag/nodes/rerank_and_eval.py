"""
Rerank and Evaluate Node
Reranks retrieved documents and evaluates retrieval quality
"""

import logging
from typing import Any, Dict

import config
from agent.state import MemGPTState
from rag import QueryIntent
from rag.nodes.helpers import apply_mmr

logger = logging.getLogger(__name__)


def rerank_and_eval_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Rerank and evaluate retrieval quality (Paper-compliant: post-retrieval)"""
    # Skip reranking if context already prepared (recall-only mode)
    if not getattr(state, "needs_document_retrieval", True) and state.final_context:
        logger.info("Skipping rerank (using recall context directly)")
        return {
            "retrieved_documents": state.retrieved_documents,
            "final_context": state.final_context,
            "rerank_scores": {},
            "quality_eval": state.quality_eval
            or {"is_relevant": True, "confidence": 0.9},
            "rag_context": state.rag_context,
            "compression_stats": {},
            "retrieval_metrics": {},
            "reretrieve_count": 0,
        }

    # CRITICAL: Use translated query if available (for cross-language retrieval)
    translated = getattr(state, "translated_query", None)
    query = translated if translated else state.user_input
    intent_enum = state.query_intent or QueryIntent.QUESTION_ANSWERING

    # Combine ALL sources including archival and recall
    all_results = []

    # Add archival memory results (long-term knowledge)
    if state.retrieved_archival:
        all_results.extend(state.retrieved_archival)
        logger.info(f"Added {len(state.retrieved_archival)} archival memory results")

    # Add RAG document results
    if state.retrieved_documents:
        all_results.extend(state.retrieved_documents)
        logger.info(f"Added {len(state.retrieved_documents)} document results")

    if not all_results:
        logger.warning("No results from any source (archival, documents, recall)")
        return {
            "rag_context": "",
            "final_context": [],
            "quality_eval": {"is_relevant": False, "should_reretrieve": False},
        }

    # Selective Reranking: use cheap BM25 first, then expensive cross-encoder if needed
    if agent.selective_reranker:
        reranked = agent.selective_reranker.rerank(
            query=query, results=all_results, intent=intent_enum, top_k=config.RERANK_TOP_K_DEFAULT
        )

        # Log statistics
        stats = agent.selective_reranker.get_statistics()
        if stats["total_queries"] % 10 == 0:  # Log every 10 queries
            logger.info(
                f"Selective reranking stats: {stats['skip_rate_percent']} skip rate"
            )
    else:
        # CORREÇÃO 6: Fallback para reranking tradicional quando selective_reranker não disponível
        logger.info("Using traditional reranking (selective_reranker not available)")
        
        # Primeiro OpenAI reranker
        if hasattr(agent, 'reranker') and agent.reranker:
            reranked = agent.reranker.rerank(query, all_results)
        else:
            reranked = all_results

        # Depois Cross-Encoder se disponível
        if hasattr(agent, 'cross_encoder') and agent.cross_encoder and agent.cross_encoder.is_available():
            logger.info("Applying Cross-Encoder reranking (System2)")
            reranked = agent.cross_encoder.rerank(query, reranked, top_k=config.RERANK_TOP_K_DEFAULT)
            logger.info(f"Cross-Encoder reranked to top {config.RERANK_TOP_K_DEFAULT} results")

    # Apply MMR diversity with embeddings
    if len(reranked) > config.MMR_DIVERSITY_TOP_K:
        try:
            diverse_results = apply_mmr(
                query=query,
                documents=reranked,
                lambda_=config.MMR_LAMBDA,
                k=config.MMR_DIVERSITY_TOP_K,
                embedding_service=agent.embedding_service,
            )
            logger.info(
                f"MMR diversity applied: {len(reranked)} -> {len(diverse_results)} documents"
            )
        except Exception as e:
            logger.warning(f"MMR failed, using top results: {e}")
            diverse_results = reranked[:config.MMR_DIVERSITY_TOP_K]
    else:
        diverse_results = reranked[:config.MMR_DIVERSITY_TOP_K]

    # Self-RAG evaluation
    if config.ENABLE_SELF_RAG:
        retrieval_eval = agent.self_rag.evaluate_retrieval(query, diverse_results)
    else:
        retrieval_eval = {
            "is_relevant": True,
            "confidence": 1.0,
            "should_reretrieve": False,
        }
    
    # FIX 3.7: Self-RAG → TIER 3 Escalation
    # If Self-RAG detects irrelevance and we're still at TIER 1/2, trigger web search
    is_relevant = retrieval_eval.get("is_relevant", True)
    relevance_confidence = retrieval_eval.get("confidence", 1.0)
    current_tier = getattr(state, "retrieval_tier", "TIER_2")
    
    if (
        not is_relevant 
        and relevance_confidence < 0.3 
        and current_tier in ["TIER_1", "TIER_2"]
        and hasattr(agent, "hierarchical_retriever")
    ):
        logger.warning(
            f"⚠️ Self-RAG detected low relevance ({relevance_confidence:.2f}) "
            f"after {current_tier} - escalating to TIER 3 web search"
        )
        
        try:
            # Trigger agentic web search
            tier_3_results = agent.hierarchical_retriever.tier_3_agentic_search(
                query=query,
                existing_context=[doc.get("content", "") for doc in diverse_results[:3]]
            )
            
            if tier_3_results and len(tier_3_results) > 0:
                logger.info(f"✅ TIER 3 web search returned {len(tier_3_results)} results")
                
                # Merge web results with existing context
                diverse_results.extend(tier_3_results)
                
                # Re-evaluate with new web context
                if config.ENABLE_SELF_RAG:
                    retrieval_eval = agent.self_rag.evaluate_retrieval(query, diverse_results)
                    logger.info(
                        f"Re-evaluation after TIER 3: relevant={retrieval_eval.get('is_relevant')}, "
                        f"confidence={retrieval_eval.get('confidence', 0):.2f}"
                    )
            else:
                logger.warning("TIER 3 web search returned no additional results")
                
        except Exception as e:
            logger.error(f"TIER 3 escalation failed: {e}", exc_info=True)
    
    # Phase 2: Consistency checking (detect contradictions)
    consistency_result = {}
    if config.ENABLE_CONSISTENCY_CHECK and len(diverse_results) >= 2:
        try:
            from rag.consistency_checker import ConsistencyChecker
            
            consistency_checker = ConsistencyChecker(
                embedding_service=agent.embedding_service,
                similarity_threshold=0.85
            )
            consistency_result = consistency_checker.check_consistency(
                documents=diverse_results,
                query=query
            )
            
            # If contradictions detected, adjust confidence
            if not consistency_result.get("consistent", True):
                contradiction_count = consistency_result.get("contradiction_count", 0)
                logger.warning(
                    f"Consistency check detected {contradiction_count} contradictions - "
                    f"lowering confidence"
                )
                # Reduce retrieval confidence based on contradictions
                original_confidence = retrieval_eval.get("confidence", 1.0)
                consistency_penalty = min(contradiction_count * 0.15, 0.5)
                retrieval_eval["confidence"] = max(original_confidence - consistency_penalty, 0.3)
                retrieval_eval["consistency_warning"] = consistency_result.get("warning")
            
            logger.info(
                f"Consistency check: {'PASSED' if consistency_result.get('consistent') else 'FAILED'} "
                f"(score: {consistency_result.get('confidence', 0):.2f})"
            )
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            consistency_result = {"consistent": True, "error": str(e)}

    # Embed semantic confidence into results
    semantic_confidence = retrieval_eval.get("confidence", 1.0)
    for result in diverse_results:
        result["semantic_confidence"] = semantic_confidence

    # Early exit for zero-relevance
    if reranked:
        top_cross_encoder_score = max(r.get("score", 0) for r in reranked[:config.MMR_DIVERSITY_TOP_K])
        if top_cross_encoder_score < config.CROSS_ENCODER_SCORE_THRESHOLD:
            logger.warning(
                f"Zero relevance detected (CrossEncoder={top_cross_encoder_score:.3f}), "
                "skipping re-retrieval to prevent wasteful API calls"
            )
            retrieval_eval["should_reretrieve"] = False
            retrieval_eval["is_relevant"] = False
            retrieval_eval["confidence"] = 0.0

    # Paper-compliant: Self-RAG Re-retrieval Loop with Progressive top_k
    reretrieve_count = state.reretrieve_count
    if (
        retrieval_eval.get("should_reretrieve")
        and reretrieve_count < config.MAX_RERETRIEVE_ATTEMPTS
    ):
        logger.warning(
            f"Self-RAG triggered re-retrieval (attempt {reretrieve_count + 1})"
        )

        # OPTIMIZATION: Progressive top_k reduction
        progressive_top_k = config.PROGRESSIVE_TOP_K_CONFIG.get(reretrieve_count, config.MMR_DIVERSITY_TOP_K)

        logger.info(f"COST OPTIMIZATION: Using top_k={progressive_top_k}")

        # Refine query
        refined_query = agent.query_rewriter.reformulate(
            state.user_input, intent=intent_enum
        )

        # New retrieval with progressive top_k
        new_results = agent.hybrid_retriever.retrieve(
            query=refined_query,
            sources=["documents", "archival"],
            top_k=progressive_top_k,
        )

        # Merge with previous results using RRF
        if new_results:
            diverse_results = agent.rrf.fuse([diverse_results, new_results])[:5]
            reretrieve_count += 1
            logger.info(f"Re-retrieval completed: {len(new_results)} new docs")

    # Context Compression (if enabled)
    compression_stats = {}
    if agent.context_compressor and config.ENABLE_CONTEXT_COMPRESSION:
        logger.info("Applying Context Compression (System2)")
        compressed_results = agent.context_compressor.compress(
            query=query, documents=diverse_results, query_intent=intent_enum
        )
        compression_stats = agent.context_compressor.get_compression_stats(
            compressed_results
        )
        logger.info(
            f"Context compressed: {compression_stats.get('tokens_saved', 0)} tokens saved "
            f"({compression_stats.get('compression_ratio', 0):.1%} compression), "
            f"{len(compressed_results)} docs retained"
        )
    else:
        compressed_results = diverse_results

    # Format RAG context string
    rag_context = "\n\n".join(
        [
            f"[{i + 1}] (score: {doc.get('score', 0):.3f}): {doc.get('content', '')}"
            for i, doc in enumerate(compressed_results)
        ]
    )

    return {
        "retrieved_documents": reranked,
        "final_context": compressed_results,
        "rerank_scores": {str(i): r.get("score", 0) for i, r in enumerate(reranked)},
        "quality_eval": retrieval_eval,
        "rag_context": rag_context,
        "compression_stats": compression_stats,
        "reretrieve_count": reretrieve_count,
        "consistency_result": consistency_result if 'consistency_result' in locals() else {},
    }
