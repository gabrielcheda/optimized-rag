from pydantic import BaseModel, Field
from typing import Optional

class UnifiedRewrite(BaseModel):
    simplified_query: Optional[str] = Field(None, description="A shorter, more direct version of the query.")
    contextualized_query: Optional[str] = Field(None, description="The query with pronouns (it, that, etc.) resolved based on history.")
    reformulated_query: Optional[str] = Field(None, description="The query optimized for vector search keywords.")
    corrected_query: Optional[str] = Field(None, description="The query with spelling/grammar fixed.")
    reasoning: str = Field(..., description="Short explanation of why these transformations were applied.")