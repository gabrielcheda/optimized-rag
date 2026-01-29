from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict


class QueryIntent(Enum):
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    FACT_CHECKING = "fact_checking"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    CLARIFICATION = "clarification"
    CONVERSATIONAL = "conversational"
    INSTRUCTION = "instruction"
    SEARCH = "search"

class IntentAnalysis(BaseModel):
    intent: QueryIntent
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    requires_multi_source: bool
    requires_reasoning: bool
    requires_factual_answer: bool
    requires_context: bool


