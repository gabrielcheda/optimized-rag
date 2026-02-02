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

    def __init__(self, llm, embedding_service=None):
        """
        Initialize intent recognizer

        Args:
            llm: Language model for classification
            embedding_service: Optional embedding service for conversation reference detection
        """
        self.llm = llm
        self.embedding_service = embedding_service

        # Initialize conversation reference detector (advanced detection)
        self.conv_ref_detector = None
        if embedding_service:
            try:
                from rag.conversation_reference_detector import ConversationReferenceDetector
                self.conv_ref_detector = ConversationReferenceDetector(
                    llm=llm,
                    embedding_service=embedding_service,
                    semantic_threshold=0.75,
                    enable_llm_fallback=True
                )
                logger.info("ConversationReferenceDetector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ConversationReferenceDetector: {e}")

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
        conversation_history: Optional[List[Dict[str, str]]] = None,
        recall_messages: Optional[List[Dict[str, Any]]] = None
    ) -> IntentAnalysis:
        """
        Recognize query intent

        Args:
            query: User query
            conversation_history: Recent conversation for context
            recall_messages: Messages from recall memory

        Returns:
            IntentAnalysis with intent, confidence, and characteristics
        """
        try:
            # STEP 1: Check if query references conversation (PRIORITY)
            if self.conv_ref_detector and (conversation_history or recall_messages):
                conv_ref_result = self.conv_ref_detector.detect(
                    query=query,
                    conversation_history=conversation_history or [],
                    recall_messages=recall_messages
                )

                if conv_ref_result.is_conversation_reference and conv_ref_result.confidence > 0.6:
                    logger.info(
                        f"Detected CLARIFICATION via {conv_ref_result.method} "
                        f"(confidence={conv_ref_result.confidence:.2f}): {conv_ref_result.reasoning}"
                    )
                    return IntentAnalysis(
                        intent=QueryIntent.CLARIFICATION,
                        confidence=conv_ref_result.confidence,
                        reasoning=conv_ref_result.reasoning,
                        requires_multi_source=False,
                        requires_reasoning=False,
                        requires_context=True,  # Needs conversation context
                        requires_factual_answer=False
                    )

            # STEP 2: Normal LLM-based intent classification
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
