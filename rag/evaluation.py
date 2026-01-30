"""
RAG Evaluation Metrics
Implements standard retrieval and generation evaluation metrics.

Paper recommendation: Continuous monitoring with quantitative metrics is ESSENTIAL for:
- Detecting quality degradation
- A/B testing improvements
- Benchmarking strategies
- Data-driven decisions
"""

from typing import List, Dict, Any, Set
import math
import logging

from prompts.evaluation_prompts import FAITHFULNESS_EVALUATION_PROMPT

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system with standard metrics"""
    
    def __init__(self):
        logger.info("RAGEvaluator initialized")
    
    @staticmethod
    def precision_at_k(
        retrieved: List[Dict[str, Any]],
        relevant_ids: Set[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K
        
        Paper definition: Precision@K = (# relevant in top-K) / K
        
        Args:
            retrieved: List of retrieved documents
            relevant_ids: Set of ground truth relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision score (0.0-1.0)
        """
        if not retrieved or k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_count = sum(1 for doc in top_k if doc.get('id') in relevant_ids)
        
        precision = relevant_count / k
        return precision
    
    @staticmethod
    def recall_at_k(
        retrieved: List[Dict[str, Any]],
        relevant_ids: Set[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K
        
        Paper definition: Recall@K = (# relevant in top-K) / (total relevant)
        
        Args:
            retrieved: List of retrieved documents
            relevant_ids: Set of ground truth relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall score (0.0-1.0)
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_count = sum(1 for doc in top_k if doc.get('id') in relevant_ids)
        
        recall = relevant_count / len(relevant_ids)
        return recall
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved: List[Dict[str, Any]],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank)
        
        Paper definition: MRR = 1 / (rank of first relevant doc)
        
        Args:
            retrieved: List of retrieved documents
            relevant_ids: Set of ground truth relevant document IDs
            
        Returns:
            MRR score (0.0-1.0)
        """
        if not retrieved or not relevant_ids:
            return 0.0
        
        for i, doc in enumerate(retrieved, 1):
            if doc.get('id') in relevant_ids:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[Dict[str, Any]],
        relevance_scores: Dict[str, float],
        k: int = 5
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Paper definition: NDCG@K = DCG@K / IDCG@K
        where DCG@K = Σ (rel_i / log2(i+1)) for i=1..K
        
        Args:
            retrieved: List of retrieved documents
            relevance_scores: Dict mapping doc_id to relevance score
            k: Number of top documents to consider
            
        Returns:
            NDCG score (0.0-1.0)
        """
        if not retrieved or not relevance_scores:
            return 0.0
        
        def dcg(docs, k):
            score = 0.0
            for i, doc in enumerate(docs[:k], 1):
                doc_id = doc.get('id')
                rel = relevance_scores.get(doc_id, 0.0)
                score += rel / math.log2(i + 1)
            return score
        
        # Calculate DCG for retrieved docs
        dcg_score = dcg(retrieved, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_docs = sorted(
            [{'id': doc_id, 'rel': rel} for doc_id, rel in relevance_scores.items()],
            key=lambda x: x['rel'],
            reverse=True
        )
        idcg_score = dcg(ideal_docs, k)
        
        if idcg_score == 0:
            return 0.0
        
        ndcg = dcg_score / idcg_score
        return ndcg
    
    def faithfulness_score(
        self,
        answer: str,
        context: List[Dict[str, Any]],
        llm
    ) -> Dict[str, Any]:
        """
        Evaluate answer faithfulness to context using LLM
        
        Paper recommendation: Check if all claims in answer
        are supported by retrieved context.
        
        Args:
            answer: Generated answer
            context: Retrieved context documents
            llm: Language model for evaluation
            
        Returns:
            Dict with score and reasoning
        """
        if not answer or not context:
            return {'score': 0.0, 'reasoning': 'Empty answer or context'}
        
        # Combine context
        context_text = "\n\n".join([
            f"[Doc {i+1}]: {doc.get('content', '')[:500]}"
            for i, doc in enumerate(context[:5])
        ])
        
        prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
            context_text=context_text[:2000],
            answer=answer[:1000]
        )
        
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response
            content = response.content
            score = 0.5  # default
            reasoning = ""
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('SCORE:'):
                    try:
                        score_str = line.split(':', 1)[1].strip()
                        score = float(score_str)
                        score = max(0.0, min(1.0, score))  # Clamp to [0,1]
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return {
                'score': score,
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Faithfulness scoring failed: {e}")
            return {'score': 0.5, 'reasoning': f'Evaluation failed: {e}'}
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive retrieval evaluation
        
        Args:
            query: User query
            retrieved: Retrieved documents
            ground_truth: Ground truth with relevant_ids and relevance_scores
            
        Returns:
            Dict with all metrics
        """
        relevant_ids = set(ground_truth.get('relevant_ids', []))
        relevance_scores = ground_truth.get('relevance_scores', {})
        
        metrics = {
            'query': query,
            'num_retrieved': len(retrieved),
            'num_relevant': len(relevant_ids)
        }
        
        # Calculate metrics only if ground truth provided
        if relevant_ids:
            metrics.update({
                'precision@5': self.precision_at_k(retrieved, relevant_ids, k=5),
                'precision@10': self.precision_at_k(retrieved, relevant_ids, k=10),
                'recall@5': self.recall_at_k(retrieved, relevant_ids, k=5),
                'recall@10': self.recall_at_k(retrieved, relevant_ids, k=10),
                'mrr': self.mean_reciprocal_rank(retrieved, relevant_ids)
            })
        
        if relevance_scores:
            metrics.update({
                'ndcg@5': self.ndcg_at_k(retrieved, relevance_scores, k=5),
                'ndcg@10': self.ndcg_at_k(retrieved, relevance_scores, k=10)
            })
        
        return metrics
    
    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for logging
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted string
        """
        parts = []
        
        if 'precision@5' in metrics:
            parts.append(f"P@5={metrics['precision@5']:.3f}")
        if 'recall@5' in metrics:
            parts.append(f"R@5={metrics['recall@5']:.3f}")
        if 'mrr' in metrics:
            parts.append(f"MRR={metrics['mrr']:.3f}")
        if 'ndcg@5' in metrics:
            parts.append(f"NDCG@5={metrics['ndcg@5']:.3f}")
        
        return " | ".join(parts) if parts else "No metrics available"
