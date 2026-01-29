# Chain of Thought Prompts

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides clear step-by-step reasoning."
)

COT_REASONING_TEMPLATE = """You are answering a complex question that requires step-by-step reasoning.

Question: {query}

Context Information:
{rag_context}

Please break down your reasoning into clear steps:
1. Identify the key sub-questions or components
2. Address each component systematically
3. Synthesize the final answer

Provide your reasoning in the format:
Step 1: [analysis]
Step 2: [analysis]
...
Conclusion: [final answer]
"""
