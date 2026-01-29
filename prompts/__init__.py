# Prompts Module
# Centralized location for all LLM prompts

from prompts.chain_of_thought import COT_REASONING_TEMPLATE, COT_SYSTEM_PROMPT
from prompts.generate_response import (
    CLARIFICATION_INSTRUCTION,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT_TEMPLATE,
)

__all__ = [
    "COT_SYSTEM_PROMPT",
    "COT_REASONING_TEMPLATE",
    "FEW_SHOT_EXAMPLES",
    "SYSTEM_PROMPT_TEMPLATE",
    "CLARIFICATION_INSTRUCTION",
]
