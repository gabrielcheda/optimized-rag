"""
Self-RAG Evaluation Prompts
Prompts for retrieval evaluation, claim extraction, and evidence verification
"""

RETRIEVAL_EVALUATION_PROMPT = """Evaluate if retrieved documents are relevant to the query.

Query: {query}

Retrieved Documents:
{docs_summary}

Answer with:
RELEVANT: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]

Evaluation:"""

RETRIEVAL_EVALUATION_SYSTEM = "You evaluate retrieval quality."

CLAIM_EXTRACTION_PROMPT = """Extract individual factual claims from this answer. List each claim separately.

Answer: {answer}

Extract only factual claims (not opinions or questions). Format as:
1. [claim]
2. [claim]
...

Claims:"""

CLAIM_EXTRACTION_SYSTEM = "You extract factual claims from text."

EVIDENCE_VERIFICATION_PROMPT = """Does this claim have supporting evidence in the documents?

Claim: {claim}

Documents:
{docs_content}

Respond:
SUPPORTED: [yes/no]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [quote from document that supports claim, or 'none']
DOCUMENT: [which document number, or 'none']

Evaluation:"""

EVIDENCE_VERIFICATION_SYSTEM = "You verify if claims are supported by documents."

ANSWER_EVALUATION_PROMPT = """Evaluate if this answer is supported by the documents and free of hallucinations.

Query: {query}

Answer: {answer}

Supporting Documents:
{docs_content}

Evaluate:
SUPPORTED: [yes/no - is answer grounded in documents?]
CONFIDENCE: [0.0-1.0]
HALLUCINATION: [yes/no - does answer contain unsupported claims?]
COMPLETENESS: [full/partial/none - how complete is the answer?]
REASONING: [brief explanation]

Evaluation:"""

ANSWER_EVALUATION_SYSTEM = "You evaluate answer quality and detect hallucinations."
