"""
RAGAS Evaluation Integration
Comprehensive RAG evaluation using RAGAS framework
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS Framework Integration (Paper-compliant: comprehensive evaluation)
    
    RAGAS (RAG Assessment) provides production-grade metrics:
    - Faithfulness: Answer grounded in context
    - Answer Relevancy: Answer addresses query
    - Context Precision: Relevant chunks ranked high
    - Context Recall: All needed info retrieved
    
    References: Paper Section on RAG Evaluation Frameworks
    """
    
    def __init__(self, llm=None):
        """
        Initialize RAGAS evaluator
        
        Args:
            llm: LangChain LLM for LLM-based metrics
        """
        self.llm = llm
        self.ragas_available = False
        
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            self.evaluate_fn = evaluate
            self.metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
            self.ragas_available = True
            logger.info("RAGAS framework initialized successfully")
        except ImportError as e:
            logger.warning(f"RAGAS not installed: {e}")
        except Exception as e:
            logger.error(f"RAGAS initialization failed: {e}", exc_info=True)
    
    def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG response using RAGAS metrics
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer (for context_recall)
        
        Returns:
            Dict of metric scores (0-1 range)
        """
        if not self.ragas_available:
            logger.warning("RAGAS not available, returning empty metrics")
            return {}
        
        try:
            from datasets import Dataset
            
            # Prepare data in RAGAS format
            data = {
                'question': [query],
                'answer': [answer],
                'contexts': [contexts]
            }
            
            if ground_truth:
                data['ground_truth'] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics
            metrics_to_use = [
                self.metrics['faithfulness'],
                self.metrics['answer_relevancy']
            ]
            
            if ground_truth:
                metrics_to_use.extend([
                    self.metrics['context_precision'],
                    self.metrics['context_recall']
                ])
            
            # Evaluate
            result = self.evaluate_fn(dataset, metrics=metrics_to_use)
            
            scores = {}
            # Convert result to dict - RAGAS returns EvaluationResult/DataFrame
            result_dict = result.to_pandas().mean().to_dict() if hasattr(result, 'to_pandas') else dict(result)  # type: ignore[union-attr, call-arg]
            for metric_name, metric_value in result_dict.items():
                if isinstance(metric_value, (int, float)):
                    scores[metric_name] = float(metric_value)
            
            logger.info(f"RAGAS evaluation: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Batch evaluation for efficiency
        
        Args:
            queries: List of queries
            answers: List of answers
            contexts_list: List of context lists
            ground_truths: Optional ground truth answers
        
        Returns:
            Average scores across batch
        """
        if not self.ragas_available:
            return {}
        
        try:
            from datasets import Dataset
            
            data = {
                'question': queries,
                'answer': answers,
                'contexts': contexts_list
            }
            
            if ground_truths:
                data['ground_truth'] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            metrics_to_use = [
                self.metrics['faithfulness'],
                self.metrics['answer_relevancy']
            ]
            
            if ground_truths:
                metrics_to_use.extend([
                    self.metrics['context_precision'],
                    self.metrics['context_recall']
                ])
            
            result = self.evaluate_fn(dataset, metrics=metrics_to_use)
            
            scores = {}
            # Convert result to dict - RAGAS returns EvaluationResult/DataFrame
            result_dict = result.to_pandas().mean().to_dict() if hasattr(result, 'to_pandas') else dict(result)  # type: ignore[union-attr, call-arg]
            for metric_name, metric_value in result_dict.items():
                if isinstance(metric_value, (int, float)):
                    scores[metric_name] = float(metric_value)
            
            logger.info(f"RAGAS batch evaluation ({len(queries)} samples): {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS batch evaluation failed: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Check if RAGAS is available"""
        return self.ragas_available
    
    def get_quality_assessment(self, scores: Dict[str, float]) -> str:
        """
        Get qualitative assessment from scores
        
        Args:
            scores: RAGAS metric scores
        
        Returns:
            Quality assessment string
        """
        if not scores:
            return "No evaluation available"
        
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 0.8:
            quality = "Excellent"
        elif avg_score >= 0.6:
            quality = "Good"
        elif avg_score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Identify weaknesses
        weaknesses = [name for name, score in scores.items() if score < 0.5]
        
        assessment = f"Quality: {quality} (avg: {avg_score:.2f})"
        if weaknesses:
            assessment += f" - Weaknesses: {', '.join(weaknesses)}"
        
        return assessment
