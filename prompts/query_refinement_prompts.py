"""
Query Refinement Prompts
Prompts for iterative query improvement
"""

REFINEMENT_SYSTEM_PROMPT = """You are a query refinement system. Reformulate queries to improve retrieval."""

REFINEMENT_PROMPT_TEMPLATE = """The previous query didn't retrieve sufficient information.

Original Query: {query}

Previous Context Retrieved:
{previous_context}

Task: Reformulate the query to retrieve missing information. Consider:
1. Are there alternative phrasings or synonyms?
2. Are there specific aspects not covered?
3. Should we break down the query into sub-questions?

Provide a refined query that addresses the gaps:"""
