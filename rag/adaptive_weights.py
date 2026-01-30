"""
Dynamic Weight Manager for DW-GRPO (Dynamic Weight Graph Reinforcement Policy Optimization)

This module implements adaptive weight adjustment based on historical performance,
replacing fixed weights (α, β, γ) with learned, query-specific weights.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Single fact lookup
    MODERATE = "moderate"      # Multiple facts, some reasoning
    COMPLEX = "complex"        # Multi-hop reasoning, synthesis


class RetrievalSource(Enum):
    """Retrieval source types"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    TEMPORAL = "temporal"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class PerformanceTracker:
    """Track retrieval performance per query type and source"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent queries to track
        """
        self.window_size = window_size
        
        # Track success rates: {(intent, source): [success_scores]}
        self.performance_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        # Track weight performance: {(intent, complexity): {'semantic': score, 'keyword': score, ...}}
        self.weight_performance: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Track query features for learning
        self.query_features: List[Dict] = []
        
        logger.info(f"PerformanceTracker initialized with window={window_size}")
    
    def record_query(
        self,
        query: str,
        intent: str,
        complexity: QueryComplexity,
        weights: Dict[str, float],
        confidence: float,
        success: bool,
        response_time: float
    ):
        """
        Record query performance for learning
        
        Args:
            query: Original query
            intent: Query intent
            complexity: Query complexity level
            weights: Weights used for retrieval
            confidence: Confidence of the result
            success: Whether result was satisfactory
            response_time: Time taken for retrieval
        """
        # Calculate success score (0.0-1.0)
        success_score = confidence if success else confidence * 0.5
        
        # Record per source
        for source, weight in weights.items():
            key = (intent, source)
            self.performance_history[key].append(success_score * weight)
            
            # Keep only recent history
            if len(self.performance_history[key]) > self.window_size:
                self.performance_history[key].pop(0)
        
        # Record per complexity
        complexity_key = (intent, complexity.value)
        for source, weight in weights.items():
            self.weight_performance[complexity_key][source].append(success_score)
            
            # Keep window
            if len(self.weight_performance[complexity_key][source]) > self.window_size:
                self.weight_performance[complexity_key][source].pop(0)
        
        # Record features
        feature = {
            'query': query,
            'intent': intent,
            'complexity': complexity.value,
            'weights': weights.copy(),
            'confidence': confidence,
            'success': success,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        self.query_features.append(feature)
        
        # Keep window
        if len(self.query_features) > self.window_size:
            self.query_features.pop(0)
        
        logger.debug(
            f"Recorded query performance: intent={intent}, complexity={complexity.value}, "
            f"confidence={confidence:.3f}, success={success}"
        )
    
    def get_source_performance(self, intent: str, source: str) -> float:
        """
        Get average performance for intent-source combination
        
        Returns:
            Average success score (0.0-1.0)
        """
        key = (intent, source)
        history = self.performance_history.get(key, [])
        
        if not history:
            return 0.5  # Neutral default
        
        return sum(history) / len(history)
    
    def get_optimal_weights_for_complexity(
        self,
        intent: str,
        complexity: QueryComplexity
    ) -> Optional[Dict[str, float]]:
        """
        Get optimal weights based on historical performance
        
        Returns:
            Dict of source weights or None if insufficient data
        """
        key = (intent, complexity.value)
        performance = self.weight_performance.get(key)
        
        if not performance or not any(performance.values()):
            return None
        
        # Calculate average performance per source
        source_scores = {}
        for source, scores in performance.items():
            if scores:
                source_scores[source] = sum(scores) / len(scores)
        
        if not source_scores:
            return None
        
        # Normalize to sum to 1.0
        total = sum(source_scores.values())
        if total == 0:
            return None
        
        optimal_weights = {
            source: score / total
            for source, score in source_scores.items()
        }
        
        return optimal_weights
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'total_queries': len(self.query_features),
            'window_size': self.window_size,
            'tracked_combinations': len(self.performance_history),
            'average_confidence': 0.0,
            'success_rate': 0.0
        }
        
        if self.query_features:
            stats['average_confidence'] = sum(
                q['confidence'] for q in self.query_features
            ) / len(self.query_features)
            
            stats['success_rate'] = sum(
                1 for q in self.query_features if q['success']
            ) / len(self.query_features)
        
        return stats


class QueryFeatureExtractor:
    """Extract features from queries for weight optimization"""
    
    @staticmethod
    def extract_complexity(query: str, intent: str) -> QueryComplexity:
        """
        Determine query complexity
        
        Args:
            query: Query text
            intent: Query intent
            
        Returns:
            QueryComplexity enum
        """
        # Simple heuristics (can be improved with ML)
        query_lower = query.lower()
        
        # Complex indicators
        complex_keywords = [
            'compare', 'difference between', 'relationship',
            'why', 'how does', 'explain', 'analyze',
            'multiple', 'all', 'every', 'comprehensive'
        ]
        
        # Simple indicators
        simple_keywords = [
            'what is', 'who is', 'when', 'where',
            'define', 'name', 'list'
        ]
        
        word_count = len(query.split())
        
        # Check complexity
        if any(kw in query_lower for kw in complex_keywords):
            return QueryComplexity.COMPLEX
        
        if intent in ['multi_hop', 'reasoning', 'synthesis']:
            return QueryComplexity.COMPLEX
        
        if any(kw in query_lower for kw in simple_keywords) and word_count < 10:
            return QueryComplexity.SIMPLE
        
        # Default to moderate
        return QueryComplexity.MODERATE
    
    @staticmethod
    def extract_domain_indicators(query: str) -> Dict[str, float]:
        """
        Extract domain-specific indicators
        
        Returns:
            Dict of domain scores
        """
        query_lower = query.lower()
        
        domains = {
            'temporal': ['recent', 'latest', 'yesterday', 'today', 'last week', 'ago', 'when'],
            'technical': ['code', 'function', 'class', 'algorithm', 'implementation', 'technical'],
            'factual': ['what is', 'define', 'meaning', 'definition'],
            'conversational': ['how are', 'tell me about', 'can you', 'please', 'help me']
        }
        
        scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[domain] = min(score / 3.0, 1.0)  # Normalize to 0-1
        
        return scores


class DynamicWeightManager:
    """
    Manage dynamic weight adjustment based on query characteristics and historical performance.
    
    Implements the core DW-GRPO algorithm for cost optimization.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tracking_window: int = 100,
        enable_learning: bool = True,
        agent_id: str = "default",
        enable_db_persistence: bool = True
    ):
        """
        Args:
            learning_rate: Rate of weight adaptation (0.0-1.0)
            tracking_window: Number of recent queries to track
            enable_learning: Whether to enable online learning
            agent_id: Agent identifier for database persistence
            enable_db_persistence: Whether to persist learning to database
        """
        self.learning_rate = learning_rate
        self.enable_learning = enable_learning
        self.agent_id = agent_id
        self.enable_db_persistence = enable_db_persistence
        
        self.tracker = PerformanceTracker(window_size=tracking_window)
        self.feature_extractor = QueryFeatureExtractor()
        
        # Initialize database persistence
        self.db = None
        if enable_db_persistence:
            try:
                from database.dw_grpo_persistence import DWGRPODatabase
                self.db = DWGRPODatabase()
                logger.info(f"Database persistence enabled for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Database persistence failed to initialize: {e}")
                self.db = None
        
        # Default fallback weights (original fixed weights)
        self.default_weights = {
            'semantic': 0.6,
            'keyword': 0.3,
            'temporal': 0.1,
            'knowledge_graph': 0.0
        }
        
        # Intent-based weight templates (starting point for learning)
        self.intent_templates = {
            'qa': {'semantic': 0.7, 'keyword': 0.2, 'temporal': 0.05, 'knowledge_graph': 0.05},
            'search': {'semantic': 0.5, 'keyword': 0.4, 'temporal': 0.05, 'knowledge_graph': 0.05},
            'multi_hop': {'semantic': 0.4, 'keyword': 0.2, 'temporal': 0.05, 'knowledge_graph': 0.35},
            'recent': {'semantic': 0.4, 'keyword': 0.2, 'temporal': 0.35, 'knowledge_graph': 0.05},
            'conversational': {'semantic': 0.6, 'keyword': 0.25, 'temporal': 0.1, 'knowledge_graph': 0.05}
        }
        
        logger.info(
            f"DynamicWeightManager initialized: learning_rate={learning_rate}, "
            f"window={tracking_window}, learning_enabled={enable_learning}"
        )
    
    def calculate_optimal_weights(
        self,
        query: str,
        intent: str,
        conversation_history: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal weights for the given query.
        
        This is the core DW-GRPO algorithm:
        1. Extract query features (complexity, domain)
        2. Retrieve historical performance for similar queries
        3. Adjust template weights based on learned performance
        4. Apply learning rate for gradual adaptation
        
        Args:
            query: Query text
            intent: Query intent
            conversation_history: Recent conversation for context
            
        Returns:
            Dict of optimized weights for each retrieval source
        """
        # Extract features
        complexity = self.feature_extractor.extract_complexity(query, intent)
        domain_scores = self.feature_extractor.extract_domain_indicators(query)
        
        # Get template weights for intent
        base_weights = self.intent_templates.get(intent, self.default_weights).copy()
        
        # If learning disabled, return template
        if not self.enable_learning:
            logger.debug(f"Learning disabled, using template weights for intent={intent}")
            return base_weights
        
        # Try to load learned weights from database (persistent learning)
        learned_weights = None
        if self.db:
            learned_weights = self.db.load_learned_weights(
                agent_id=self.agent_id,
                intent=intent,
                complexity=complexity.value
            )
        
        # Fallback to in-memory learned weights
        if not learned_weights:
            learned_weights = self.tracker.get_optimal_weights_for_complexity(intent, complexity)
        
        if learned_weights:
            # Blend template and learned weights using learning rate
            optimized_weights = {}
            for source in base_weights.keys():
                base = base_weights[source]
                learned = learned_weights.get(source, base)
                
                # Gradual adaptation: w_new = (1-α) * w_template + α * w_learned
                optimized_weights[source] = (
                    (1 - self.learning_rate) * base + self.learning_rate * learned
                )
            
            logger.debug(
                f"Applied learning: complexity={complexity.value}, "
                f"learned adjustment applied with α={self.learning_rate}"
            )
        else:
            optimized_weights = base_weights.copy()
            logger.debug(
                f"Insufficient learning data for {intent}+{complexity.value}, "
                f"using template weights"
            )
        
        # Apply domain-specific adjustments
        if domain_scores.get('temporal', 0) > 0.5:
            # Boost temporal weight for time-sensitive queries
            boost = 0.15 * domain_scores['temporal']
            optimized_weights['temporal'] += boost
            optimized_weights['semantic'] -= boost * 0.7
            optimized_weights['keyword'] -= boost * 0.3
        
        if domain_scores.get('technical', 0) > 0.5:
            # Boost keyword weight for technical queries
            boost = 0.1 * domain_scores['technical']
            optimized_weights['keyword'] += boost
            optimized_weights['semantic'] -= boost
        
        # Normalize to sum to 1.0 (excluding knowledge_graph which is added separately)
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {
                source: weight / total_weight
                for source, weight in optimized_weights.items()
            }
        
        logger.info(
            f"Optimized weights for query (intent={intent}, complexity={complexity.value}): "
            f"semantic={optimized_weights['semantic']:.3f}, "
            f"keyword={optimized_weights['keyword']:.3f}, "
            f"temporal={optimized_weights['temporal']:.3f}, "
            f"kg={optimized_weights.get('knowledge_graph', 0):.3f}"
        )
        
        return optimized_weights
    
    def record_feedback(
        self,
        query: str,
        intent: str,
        weights: Dict[str, float],
        confidence: float,
        success: bool,
        response_time: float,
        tier_reached: int = 2,
        cost_estimate: float = 0.0
    ):
        """
        Record feedback for online learning (System1-System2 persistent feedback)
        
        Args:
            query: Original query
            intent: Query intent
            weights: Weights that were used
            confidence: Result confidence
            success: Whether result was satisfactory
            response_time: Response time in seconds
            tier_reached: Which tier satisfied the query (1, 2, or 3)
            cost_estimate: Estimated API cost
        """
        if not self.enable_learning:
            return
        
        complexity = self.feature_extractor.extract_complexity(query, intent)
        
        # Record to in-memory tracker
        self.tracker.record_query(
            query=query,
            intent=intent,
            complexity=complexity,
            weights=weights,
            confidence=confidence,
            success=success,
            response_time=response_time
        )
        
        # Save to database for persistent learning
        if self.db:
            self.db.save_performance(
                agent_id=self.agent_id,
                query_text=query,
                intent=intent,
                complexity=complexity.value,
                weights=weights,
                confidence=confidence,
                success=success,
                response_time=response_time,
                tier_reached=tier_reached,
                cost_estimate=cost_estimate
            )
            
            # Periodically update learned weights in database
            stats = self.tracker.get_statistics()
            if stats['total_queries'] % 10 == 0:  # Every 10 queries
                optimal = self.tracker.get_optimal_weights_for_complexity(intent, complexity)
                if optimal:
                    self.db.save_learned_weights(
                        agent_id=self.agent_id,
                        intent=intent,
                        complexity=complexity.value,
                        weights=optimal,
                        sample_count=stats['total_queries'],
                        avg_confidence=stats['average_confidence'],
                        success_rate=stats['success_rate']
                    )
        
        logger.debug(f"Recorded feedback: confidence={confidence:.3f}, success={success}, tier={tier_reached}")
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = self.tracker.get_statistics()
        stats['learning_enabled'] = self.enable_learning
        stats['learning_rate'] = self.learning_rate
        return stats
    
    def reset_learning(self):
        """Reset all learning history"""
        self.tracker = PerformanceTracker(window_size=self.tracker.window_size)
        logger.info("Learning history reset")
