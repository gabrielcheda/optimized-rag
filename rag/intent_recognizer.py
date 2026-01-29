"""
Intent Recognizer
Classifies user queries into intents for specialized handling
Based on RAG paper recommendations for online RAG workflow
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging

from prompts.intent_recognizer_prompts import INTENT_PROMPT
from rag.models.intent_analysis import IntentAnalysis, QueryIntent

logger = logging.getLogger(__name__)





class IntentRecognizer:
    """Recognizes query intent using LLM"""
    
    def __init__(self, llm):
        """
        Initialize intent recognizer
        
        Args:
            llm: Language model for classification
        """
        self.llm = llm
        self.intent_descriptions = {
            QueryIntent.QUESTION_ANSWERING: "Direct factual questions requiring specific answers (What is...? How does...?)",
            QueryIntent.SUMMARIZATION: "Requests for summarizing content (Summarize..., Give me an overview...)",
            QueryIntent.COMPARISON: "Comparing two or more entities (Compare X and Y, What's the difference between...)",
            QueryIntent.FACT_CHECKING: "Verifying truth of claims (Is it true that...? Verify that...)",
            QueryIntent.MULTI_HOP_REASONING: "Complex questions requiring multiple reasoning steps",
            QueryIntent.CLARIFICATION: "Asking for more details about previous answer (Tell me more, What about...)",
            QueryIntent.CONVERSATIONAL: "General chat or small talk",
            QueryIntent.INSTRUCTION: "Task instructions (Upload..., Search for..., Create...)",
            QueryIntent.SEARCH: "General information search (Find information about...)"
        }
    
    def recognize(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> IntentAnalysis:
        """
        Recognize query intent
        
        Args:
            query: User query
            conversation_history: Recent conversation for context
        
        Returns:
            IntentAnalysis with intent, confidence, and characteristics
        """
        try:
            prompt = self._build_intent_prompt(conversation_history)
            
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Query: {query}")
            ]
            
            structured_output = self.llm.with_structured_output(IntentAnalysis)
            result = structured_output.invoke(messages)
            
            logger.info(
                f"Intent recognized: {result.intent.value} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            return IntentAnalysis(
                intent=QueryIntent.QUESTION_ANSWERING,
                confidence=0.0,
                reasoning="Defaulted due to error",
                requires_multi_source=False,
                requires_reasoning=False,
                requires_context=False,
                requires_factual_answer=False
            )
    
    def _build_intent_prompt(
        self,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build prompt for intent classification"""
        
        intents_list = "\n".join([
            f"- {intent.value}: {desc}"
            for intent, desc in self.intent_descriptions.items()
        ])
        
        context_info = ""
        if conversation_history:
            recent = conversation_history[-3:]
            context_info = "\nRecent conversation:\n" + "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..."
                for msg in recent
            ])
        
        return INTENT_PROMPT.format(
            intents_list=intents_list,
            context_info=context_info
        )
    

    def _map_intent_string(self, intent_str: Optional[str]) -> QueryIntent:
        """Map string to QueryIntent enum"""
        if not intent_str:
            return QueryIntent.QUESTION_ANSWERING
        
        intent_str = intent_str.lower()
        
        # Direct mapping
        for intent in QueryIntent:
            if intent.value in intent_str:
                return intent
        
        # Keyword mapping
        if any(word in intent_str for word in ["question", "what", "how", "why", "qa"]):
            return QueryIntent.QUESTION_ANSWERING
        
        if any(word in intent_str for word in ["summarize", "summary", "overview"]):
            return QueryIntent.SUMMARIZATION
        
        if any(word in intent_str for word in ["compare", "comparison", "difference", "versus"]):
            return QueryIntent.COMPARISON
        
        if any(word in intent_str for word in ["verify", "fact", "check", "true"]):
            return QueryIntent.FACT_CHECKING
        
        if any(word in intent_str for word in ["multi", "complex", "reasoning", "step"]):
            return QueryIntent.MULTI_HOP_REASONING
        
        if any(word in intent_str for word in ["clarify", "explain", "more", "detail"]):
            return QueryIntent.CLARIFICATION
        
        if any(word in intent_str for word in ["chat", "talk", "hello", "hi"]):
            return QueryIntent.CONVERSATIONAL
        
        if any(word in intent_str for word in ["upload", "create", "delete", "run"]):
            return QueryIntent.INSTRUCTION
        
        if any(word in intent_str for word in ["search", "find", "look"]):
            return QueryIntent.SEARCH
        
        # Default
        return QueryIntent.QUESTION_ANSWERING
    
    def get_retrieval_strategy(self, intent: QueryIntent | None) -> Dict[str, Any]:
        """
        Get recommended retrieval strategy for intent
        
        Args:
            intent: Detected intent
        
        Returns:
            Retrieval strategy configuration
        """
        strategies = {
            QueryIntent.QUESTION_ANSWERING: {
                "top_k": 5,
                "use_hybrid": True,
                "use_reranking": True,
                "diversity_weight": 0.3
            },
            QueryIntent.SUMMARIZATION: {
                "top_k": 10,
                "use_hybrid": False,
                "use_reranking": True,
                "diversity_weight": 0.5
            },
            QueryIntent.COMPARISON: {
                "top_k": 8,
                "use_hybrid": True,
                "use_reranking": True,
                "diversity_weight": 0.7
            },
            QueryIntent.FACT_CHECKING: {
                "top_k": 5,
                "use_hybrid": True,
                "use_reranking": True,
                "diversity_weight": 0.2
            },
            QueryIntent.MULTI_HOP_REASONING: {
                "top_k": 12,
                "use_hybrid": True,
                "use_reranking": True,
                "diversity_weight": 0.6
            },
            QueryIntent.CLARIFICATION: {
                "top_k": 3,
                "use_hybrid": False,
                "use_reranking": False,
                "diversity_weight": 0.1
            },
            QueryIntent.CONVERSATIONAL: {
                "top_k": 2,
                "use_hybrid": False,
                "use_reranking": False,
                "diversity_weight": 0.0
            },
            QueryIntent.INSTRUCTION: {
                "top_k": 3,
                "use_hybrid": False,
                "use_reranking": False,
                "diversity_weight": 0.0
            },
            QueryIntent.SEARCH: {
                "top_k": 7,
                "use_hybrid": True,
                "use_reranking": True,
                "diversity_weight": 0.5
            }
        }
        
        # Handle None case
        if intent is None:
            return strategies[QueryIntent.QUESTION_ANSWERING]
        
        return strategies.get(intent, strategies[QueryIntent.QUESTION_ANSWERING])
