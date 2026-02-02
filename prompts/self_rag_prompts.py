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

Extract only factual claims (not opinions or questions). 
IMPORTANT: Preserve any citation markers [N] that appear in the original answer.

Format as:
1. [claim with citations if present]
2. [claim with citations if present]
...

Claims:"""

# FASE 1: Prompt mais preciso para extração de claims
CLAIM_EXTRACTION_SYSTEM = """You extract ONLY verifiable factual claims from text.

## INCLUDE:
- Statements with specific facts, numbers, dates, names
- Declarative statements that can be true or false
- Cause-effect relationships stated as fact

## EXCLUDE:
- Opinions ("I think", "seems like", "probably")
- Questions or hypotheticals
- Meta-statements about the response itself
- Vague generalizations without specifics
- Emotional expressions or greetings

Return each claim on a new line, numbered 1, 2, 3...
If no factual claims exist, return "NO_FACTUAL_CLAIMS"."""

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

# CORREÇÃO 8: Prompt melhorado para aceitar infereências razoáveis
ANSWER_EVALUATION_PROMPT = """Evaluate if this answer is supported by the documents and free of hallucinations.

Query: {query}

Answer: {answer}

Supporting Documents:
{docs_content}

Evaluate:
SUPPORTED: [yes/no - is answer grounded in documents? Accept reasonable inferences from document content]
CONFIDENCE: [0.0-1.0]
HALLUCINATION: [yes/no - does answer contain claims that contradict or are completely absent from documents?]
COMPLETENESS: [full/partial/none - how complete is the answer?]
REASONING: [brief explanation]

NOTE: An answer can be SUPPORTED even if it paraphrases, summarizes, or makes reasonable inferences from the documents.
Only mark as hallucination if claims directly contradict or have zero basis in the provided documents.

Evaluation:"""

ANSWER_EVALUATION_SYSTEM = "You evaluate answer quality and detect hallucinations."
