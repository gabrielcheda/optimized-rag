"""
Evaluation Prompts
Prompts for RAG evaluation metrics
"""

FAITHFULNESS_EVALUATION_PROMPT = """Rate the faithfulness of the answer to the provided context on a scale of 0.0 to 1.0.

Context:
{context_text}

Answer:
{answer}

Faithfulness means: Are all claims in the answer supported by the context?
- 1.0 = All claims directly supported
- 0.7-0.9 = Most claims supported
- 0.4-0.6 = Some claims supported
- 0.0-0.3 = Few/no claims supported

Provide your evaluation in this format:
SCORE: [0.0-1.0]
REASONING: [brief explanation]
"""
