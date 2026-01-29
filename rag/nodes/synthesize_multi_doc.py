"""
Synthesize Multi-Document Node
Synthesizes evidence from multiple sources
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState
from langchain_core.messages import HumanMessage
from rag import QueryIntent

logger = logging.getLogger(__name__)


def synthesize_multi_doc_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """Synthesize evidence from multiple documents (System2 feature)"""
    query = state.user_input
    docs = state.final_context
    intent_enum = state.query_intent
    
    # Check if synthesis is needed (complex intents + multiple sources)
    complex_intents = [QueryIntent.MULTI_HOP_REASONING, QueryIntent.COMPARISON, QueryIntent.SUMMARIZATION]
    should_synthesize = intent_enum and intent_enum in complex_intents and len(docs) > 2
    
    if not should_synthesize:
        logger.info(f"Skipping multi-doc synthesis (intent={intent_enum}, docs={len(docs)})")
        return {}  # Skip synthesis
    
    logger.info(f"Multi-doc synthesis triggered: intent={intent_enum}, docs={len(docs)}")
    
    # Extract key evidence from each document
    evidence_list = []
    for i, doc in enumerate(docs[:5]):  # Limit to top 5
        content = doc.get('content', '')
        source = doc.get('source', 'unknown')
        score = doc.get('score', 0.0)
        evidence_list.append(f"[Source {i+1} ({source}, score={score:.2f})]: {content[:300]}...")
    
    # Create synthesis prompt
    synthesis_prompt = f"""You are synthesizing information from multiple sources to answer a complex question.

Question: {query}

Evidence from multiple sources:
{chr(10).join(evidence_list)}

Your task:
1. Identify common themes and contradictions across sources
2. Synthesize a coherent answer that integrates evidence from all relevant sources
3. Note any conflicting information and explain the most likely answer
4. Cite which sources support each claim (e.g., "According to Sources 1 and 3...")

Provide a well-synthesized answer:"""
    
    try:
        synthesis_response = agent.llm.invoke([HumanMessage(content=synthesis_prompt)])
        synthesized_answer = synthesis_response.content
        
        logger.info(f"Multi-document synthesis completed for {len(docs)} sources")
        return {
            "synthesized_context": synthesized_answer,
            "synthesis_metadata": {
                "sources_used": len(evidence_list),
                "synthesis_applied": True
            }
        }
    except Exception as e:
        logger.error(f"Multi-document synthesis failed: {e}")
        return {"synthesis_metadata": {"synthesis_applied": False, "error": str(e)}}
