"""
Multi-Document Synthesis Prompts
Prompts for synthesizing information from multiple sources
"""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing information from multiple sources to provide comprehensive answers."""

SYNTHESIS_PROMPT_TEMPLATE = """You are synthesizing information from multiple sources to answer a complex question.

Question: {query}

Evidence from multiple sources:
{evidence_list}

Your task:
1. Identify common themes and contradictions across sources
2. Synthesize a coherent answer that integrates evidence from all relevant sources
3. Note any conflicting information and explain the most likely answer
4. Cite which sources support each claim (e.g., "According to Sources 1 and 3...")

Provide a well-synthesized answer:"""
