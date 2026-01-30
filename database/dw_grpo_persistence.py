"""
DW-GRPO Database Persistence Module
Handles storage and retrieval of performance metrics and learned weights in PostgreSQL
System1-System2 Paper Compliance: Persistent Feedback Loop
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional
import hashlib
import json

import config

logger = logging.getLogger(__name__)


class DWGRPODatabase:
    """Database operations for DW-GRPO persistence (System1-System2 feedback loop)"""
    
    def __init__(self, postgres_uri: str = ''):
        """
        Initialize database connection
        
        Args:
            postgres_uri: PostgreSQL connection string (defaults to config.POSTGRES_URI)
        """
        self.postgres_uri = postgres_uri or config.POSTGRES_URI
        self.conn = None
        self._ensure_connection()
        logger.info("DW-GRPO Database initialized")
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(self.postgres_uri)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def _query_hash(self, query: str) -> str:
        """Generate SHA256 hash of query for deduplication"""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()
    
    def save_performance(
        self,
        agent_id: str,
        query_text: str,
        intent: str,
        complexity: str,
        weights: Dict[str, float],
        confidence: float,
        success: bool,
        response_time: float,
        tier_reached: int,
        cost_estimate: float = 0.0,
        metadata: Dict = {}
    ) -> bool:
        """
        Save query performance to database
        
        Args:
            agent_id: Agent identifier
            query_text: Original query
            intent: Query intent (qa, search, multi_hop, etc.)
            complexity: Query complexity (simple, moderate, complex)
            weights: Weights used {semantic, keyword, temporal, knowledge_graph}
            confidence: Confidence score (0.0-1.0)
            success: Whether query was successful
            response_time: Time taken in seconds
            tier_reached: Which tier satisfied the query (1, 2, or 3)
            cost_estimate: Estimated API cost
            metadata: Additional metadata
        
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connection():
            return False
        
        try:
            if self.conn is None:
                logger.error("Database connection not available")
                return False
            
            query_hash = self._query_hash(query_text)
            metadata_json = json.dumps(metadata or {})
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dw_grpo_performance (
                        agent_id, query_text, query_hash, intent, complexity,
                        weight_semantic, weight_keyword, weight_temporal, weight_kg,
                        confidence, success, response_time, tier_reached,
                        cost_estimate, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    agent_id, query_text, query_hash, intent, complexity,
                    weights.get('semantic', 0.0),
                    weights.get('keyword', 0.0),
                    weights.get('temporal', 0.0),
                    weights.get('knowledge_graph', 0.0),
                    confidence, success, response_time, tier_reached,
                    cost_estimate, metadata_json
                ))
            
            if self.conn is not None:
                self.conn.commit()
            logger.debug(f"Saved performance: intent={intent}, confidence={confidence:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save performance: {e}")
            if self.conn is not None:
                self.conn.rollback()
            return False
    
    def load_learned_weights(
        self,
        agent_id: str,
        intent: str,
        complexity: str
    ) -> Optional[Dict[str, float]]:
        """
        Load learned optimal weights for intent/complexity
        
        Args:
            agent_id: Agent identifier
            intent: Query intent
            complexity: Query complexity
        
        Returns:
            Dict of weights or None if not found
        """
        if not self._ensure_connection():
            return None
        
        try:
            if self.conn is None:
                logger.error("Database connection not available")
                return None
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        weight_semantic,
                        weight_keyword,
                        weight_temporal,
                        weight_kg,
                        sample_count,
                        avg_confidence,
                        success_rate,
                        last_updated
                    FROM dw_grpo_learned_weights
                    WHERE agent_id = %s AND intent = %s AND complexity = %s
                """, (agent_id, intent, complexity))
                
                row = cur.fetchone()
                
                if row:
                    weights = {
                        'semantic': row['weight_semantic'],
                        'keyword': row['weight_keyword'],
                        'temporal': row['weight_temporal'],
                        'knowledge_graph': row['weight_kg']
                    }
                    logger.debug(
                        f"Loaded weights: intent={intent}, complexity={complexity}, "
                        f"samples={row['sample_count']}, success={row['success_rate']:.2%}"
                    )
                    return weights
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to load learned weights: {e}")
            return None
    
    def save_learned_weights(
        self,
        agent_id: str,
        intent: str,
        complexity: str,
        weights: Dict[str, float],
        sample_count: int,
        avg_confidence: float,
        success_rate: float
    ) -> bool:
        """
        Save or update learned weights
        
        Args:
            agent_id: Agent identifier
            intent: Query intent
            complexity: Query complexity
            weights: Optimal weights
            sample_count: Number of samples used
            avg_confidence: Average confidence
            success_rate: Success rate (0.0-1.0)
        
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            if self.conn is None:
                logger.error("Database connection not available")
                return False
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dw_grpo_learned_weights (
                        agent_id, intent, complexity,
                        weight_semantic, weight_keyword, weight_temporal, weight_kg,
                        sample_count, avg_confidence, success_rate,
                        last_updated
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (agent_id, intent, complexity)
                    DO UPDATE SET
                        weight_semantic = EXCLUDED.weight_semantic,
                        weight_keyword = EXCLUDED.weight_keyword,
                        weight_temporal = EXCLUDED.weight_temporal,
                        weight_kg = EXCLUDED.weight_kg,
                        sample_count = EXCLUDED.sample_count,
                        avg_confidence = EXCLUDED.avg_confidence,
                        success_rate = EXCLUDED.success_rate,
                        last_updated = NOW()
                """, (
                    agent_id, intent, complexity,
                    weights.get('semantic', 0.0),
                    weights.get('keyword', 0.0),
                    weights.get('temporal', 0.0),
                    weights.get('knowledge_graph', 0.0),
                    sample_count, avg_confidence, success_rate
                ))
            
            if self.conn is not None:
                self.conn.commit()
            logger.debug(f"Saved learned weights: intent={intent}, complexity={complexity}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save learned weights: {e}")
            if self.conn is not None:
                self.conn.rollback()
            return False
    
    def get_performance_history(
        self,
        agent_id: str,
        intent: Optional[str] = None,
        complexity: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get performance history for analysis
        
        Args:
            agent_id: Agent identifier
            intent: Filter by intent (optional)
            complexity: Filter by complexity (optional)
            days: Number of days to look back
            limit: Maximum number of records
        
        Returns:
            List of performance records
        """
        if not self._ensure_connection():
            return []
        
        try:
            query = """
                SELECT 
                    query_text, intent, complexity,
                    weight_semantic, weight_keyword, weight_temporal, weight_kg,
                    confidence, success, response_time, tier_reached,
                    cost_estimate, created_at
                FROM dw_grpo_performance
                WHERE agent_id = %s
                AND created_at >= NOW() - INTERVAL '%s days'
            """
            params = [agent_id, days]
            
            if intent:
                query += " AND intent = %s"
                params.append(intent)
            
            if complexity:
                query += " AND complexity = %s"
                params.append(complexity)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            if self.conn is None:
                logger.error("Database connection not available")
                return []
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                
            logger.debug(f"Retrieved {len(results)} performance records")
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []
    
    def update_daily_metrics(
        self,
        agent_id: str,
        tier_reached: int,
        confidence: float,
        response_time: float,
        cost_estimate: float,
        cost_savings: float,
        success: bool
    ) -> bool:
        """
        Update daily aggregated metrics
        
        Args:
            agent_id: Agent identifier
            tier_reached: Tier that satisfied query
            confidence: Confidence score
            response_time: Response time
            cost_estimate: API cost estimate
            cost_savings: Cost saved vs Tier 3
            success: Whether successful
        
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            if self.conn is None:
                logger.error("Database connection not available")
                return False
            
            with self.conn.cursor() as cur:
                # Update or insert today's metrics
                cur.execute("""
                    INSERT INTO dw_grpo_metrics (
                        agent_id, date,
                        tier1_count, tier2_count, tier3_count,
                        total_queries, avg_confidence, avg_response_time,
                        total_cost_estimate, cost_savings_estimate,
                        success_count, failure_count,
                        updated_at
                    ) VALUES (
                        %s, CURRENT_DATE,
                        %s, %s, %s,
                        1, %s, %s,
                        %s, %s,
                        %s, %s,
                        NOW()
                    )
                    ON CONFLICT (agent_id, date)
                    DO UPDATE SET
                        tier1_count = dw_grpo_metrics.tier1_count + EXCLUDED.tier1_count,
                        tier2_count = dw_grpo_metrics.tier2_count + EXCLUDED.tier2_count,
                        tier3_count = dw_grpo_metrics.tier3_count + EXCLUDED.tier3_count,
                        total_queries = dw_grpo_metrics.total_queries + 1,
                        avg_confidence = (dw_grpo_metrics.avg_confidence * dw_grpo_metrics.total_queries + %s) / (dw_grpo_metrics.total_queries + 1),
                        avg_response_time = (dw_grpo_metrics.avg_response_time * dw_grpo_metrics.total_queries + %s) / (dw_grpo_metrics.total_queries + 1),
                        total_cost_estimate = dw_grpo_metrics.total_cost_estimate + %s,
                        cost_savings_estimate = dw_grpo_metrics.cost_savings_estimate + %s,
                        success_count = dw_grpo_metrics.success_count + %s,
                        failure_count = dw_grpo_metrics.failure_count + %s,
                        updated_at = NOW()
                """, (
                    agent_id,
                    1 if tier_reached == 1 else 0,
                    1 if tier_reached == 2 else 0,
                    1 if tier_reached == 3 else 0,
                    confidence, response_time,
                    cost_estimate, cost_savings,
                    1 if success else 0,
                    0 if success else 1,
                    # Update clause parameters
                    confidence, response_time,
                    cost_estimate, cost_savings,
                    1 if success else 0,
                    0 if success else 1
                ))
            
            if self.conn is not None:
                self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update daily metrics: {e}")
            if self.conn is not None:
                self.conn.rollback()
            return False
    
    def get_cost_savings_report(self, agent_id: str, days: int = 30) -> Dict:
        """
        Generate cost savings report
        
        Args:
            agent_id: Agent identifier
            days: Number of days to analyze
        
        Returns:
            Dict with cost savings statistics
        """
        if not self._ensure_connection():
            return {}
        
        try:
            if self.conn is None:
                logger.error("Database connection not available")
                return {}
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        SUM(total_queries) as total_queries,
                        SUM(tier1_count) as tier1_stops,
                        SUM(tier2_count) as tier2_stops,
                        SUM(tier3_count) as tier3_complete,
                        SUM(total_cost_estimate) as actual_cost,
                        SUM(cost_savings_estimate) as total_savings,
                        AVG(avg_confidence) as overall_confidence,
                        SUM(success_count) as total_success,
                        SUM(failure_count) as total_failures
                    FROM dw_grpo_metrics
                    WHERE agent_id = %s
                    AND date >= CURRENT_DATE - INTERVAL '%s days'
                """, (agent_id, days))
                
                row = cur.fetchone()
                
                if row and row['total_queries']:
                    return {
                        'total_queries': row['total_queries'],
                        'tier1_percentage': (row['tier1_stops'] / row['total_queries']) * 100,
                        'tier2_percentage': (row['tier2_stops'] / row['total_queries']) * 100,
                        'tier3_percentage': (row['tier3_complete'] / row['total_queries']) * 100,
                        'actual_cost': row['actual_cost'],
                        'total_savings': row['total_savings'],
                        'savings_percentage': (row['total_savings'] / (row['actual_cost'] + row['total_savings'])) * 100 if (row['actual_cost'] + row['total_savings']) > 0 else 0,
                        'overall_confidence': row['overall_confidence'],
                        'success_rate': (row['total_success'] / row['total_queries']) * 100
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to generate cost savings report: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("DW-GRPO Database connection closed")
