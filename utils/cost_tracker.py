"""
Cost Tracking System
Tracks API costs for embeddings, LLM calls, and reranking
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CostTracker:
    """Track and report API costs"""
    
    # Pricing (as of 2024, in USD per 1K tokens)
    PRICING = {
        # OpenAI Embeddings
        'text-embedding-3-small': 0.00002,
        'text-embedding-3-large': 0.00013,
        'text-embedding-ada-002': 0.0001,
        
        # OpenAI LLMs
        'gpt-4o': {'input': 0.0025, 'output': 0.01},
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        
        # Tavily Search
        'tavily_search': 0.001,  # per search
    }
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize cost tracker
        
        Args:
            save_path: Optional path to save cost data
        """
        self.save_path = save_path
        self.costs: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_cost': 0.0,
            'calls': 0,
            'tokens': {'input': 0, 'output': 0, 'total': 0}
        })
        self.daily_costs: Dict[Any, float] = defaultdict(float)
        self.current_date = datetime.now().date()
        
        # Load existing data if available
        if self.save_path and Path(self.save_path).exists():
            self._load()
    
    def track_embedding(
        self,
        model: str,
        num_tokens: int,
        num_calls: int = 1
    ) -> float:
        """
        Track embedding API call
        
        Args:
            model: Embedding model name
            num_tokens: Number of tokens processed
            num_calls: Number of API calls
            
        Returns:
            Cost in USD
        """
        price_per_1k = self.PRICING.get(model, 0.0001)  # Default to ada-002 price
        cost = (num_tokens / 1000) * price_per_1k
        
        # Type guard for dict access
        model_data = self.costs[model]
        model_data['total_cost'] = float(model_data.get('total_cost', 0.0)) + cost
        model_data['calls'] = int(model_data.get('calls', 0)) + num_calls
        model_data['tokens']['total'] = int(model_data['tokens'].get('total', 0)) + num_tokens
        
        self._update_daily_cost(cost)
        
        logger.debug(f"Embedding cost: ${cost:.6f} ({model}, {num_tokens} tokens)")
        return cost
    
    def track_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Track LLM API call
        
        Args:
            model: LLM model name
            input_tokens: Input tokens
            output_tokens: Output tokens
            
        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(model, {'input': 0.0005, 'output': 0.0015})
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        # Type guard for dict access
        model_data = self.costs[model]
        model_data['total_cost'] = float(model_data.get('total_cost', 0.0)) + total_cost
        model_data['calls'] = int(model_data.get('calls', 0)) + 1
        model_data['tokens']['input'] = int(model_data['tokens'].get('input', 0)) + input_tokens
        model_data['tokens']['output'] = int(model_data['tokens'].get('output', 0)) + output_tokens
        model_data['tokens']['total'] = int(model_data['tokens'].get('total', 0)) + input_tokens + output_tokens
        
        self._update_daily_cost(total_cost)
        
        logger.debug(
            f"LLM cost: ${total_cost:.6f} ({model}, "
            f"in={input_tokens}, out={output_tokens})"
        )
        return total_cost
    
    def track_search(self, num_searches: int = 1) -> float:
        """
        Track web search API call
        
        Args:
            num_searches: Number of searches
            
        Returns:
            Cost in USD
        """
        cost = num_searches * self.PRICING['tavily_search']
        
        model_data = self.costs['tavily_search']
        model_data['total_cost'] = float(model_data.get('total_cost', 0.0)) + cost
        model_data['calls'] = int(model_data.get('calls', 0)) + num_searches
        
        self._update_daily_cost(cost)
        
        logger.debug(f"Search cost: ${cost:.6f} ({num_searches} searches)")
        return cost
    
    def _update_daily_cost(self, cost: float):
        """Update daily cost tracking"""
        today = datetime.now().date()
        
        # Reset if new day
        if today != self.current_date:
            self.current_date = today
            # Keep only last 30 days
            cutoff_date = today - timedelta(days=30)
            self.daily_costs = {
                d: c for d, c in self.daily_costs.items()
                if d >= cutoff_date
            }
        
        self.daily_costs[today] += cost
    
    def get_total_cost(self) -> float:
        """Get total cost across all services"""
        return sum(float(data.get('total_cost', 0.0)) for data in self.costs.values())
    
    def get_daily_cost(self, target_date: Optional[Any] = None) -> float:
        """Get cost for a specific date (default: today)"""
        if target_date is None:
            target_date = datetime.now().date()
        return self.daily_costs.get(target_date, 0.0)
    
    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        total = self.get_total_cost()
        
        breakdown = {}
        for service, data in self.costs.items():
            total_cost = float(data.get('total_cost', 0.0))
            breakdown[service] = {
                'cost': total_cost,
                'percentage': (total_cost / total * 100) if total > 0 else 0,
                'calls': int(data.get('calls', 0)),
                'tokens': data.get('tokens', {})
            }
        
        return {
            'total_cost': total,
            'daily_cost': self.get_daily_cost(),
            'services': breakdown
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        breakdown = self.get_breakdown()
        
        # Calculate averages
        total_calls = sum(int(data.get('calls', 0)) for data in self.costs.values())
        avg_cost_per_call = breakdown['total_cost'] / total_calls if total_calls > 0 else 0
        
        # Last 7 days trend
        today = datetime.now().date()
        last_7_days = [today - timedelta(days=i) for i in range(7)]
        weekly_costs = [self.daily_costs.get(d, 0.0) for d in last_7_days]
        
        return {
            **breakdown,
            'total_calls': total_calls,
            'avg_cost_per_call': avg_cost_per_call,
            'weekly_costs': weekly_costs,
            'weekly_total': sum(weekly_costs)
        }
    
    def print_summary(self):
        """Print cost summary to console"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("COST TRACKING SUMMARY")
        print("="*60)
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Today's Cost: ${stats['daily_cost']:.4f}")
        print(f"Weekly Cost: ${stats['weekly_total']:.4f}")
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Avg Cost/Call: ${stats['avg_cost_per_call']:.6f}")
        print("\nBreakdown by Service:")
        print("-"*60)
        
        for service, data in stats['services'].items():
            print(f"{service:30} ${data['cost']:8.4f} ({data['percentage']:5.1f}%)")
            tokens = data.get('tokens', {})
            if isinstance(tokens, dict):
                total_tokens = tokens.get('total', 0)
                print(f"  Calls: {data['calls']:6}  Tokens: {total_tokens:10,}")
        
        print("="*60 + "\n")
    
    def _save(self):
        """Save cost data to disk"""
        if not self.save_path:
            return
        
        data = {
            'costs': dict(self.costs),
            'daily_costs': {str(k): v for k, v in self.daily_costs.items()},
            'current_date': str(self.current_date)
        }
        
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Cost data saved to {self.save_path}")
    
    def _load(self):
        """Load cost data from disk"""
        if not self.save_path:
            return
            
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load costs with type safety
            loaded_costs = data.get('costs', {})
            for model, cost_data in loaded_costs.items():
                self.costs[model] = cost_data
            
            # Load daily costs
            for date_str, cost in data.get('daily_costs', {}).items():
                try:
                    date_obj = datetime.fromisoformat(date_str).date()
                    self.daily_costs[date_obj] = cost
                except:
                    pass
            
            # Load current date
            try:
                self.current_date = datetime.fromisoformat(
                    data.get('current_date', str(datetime.now().date()))
                ).date()
            except:
                self.current_date = datetime.now().date()
            
            logger.info(f"Cost data loaded from {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to load cost data: {e}")
    
    def __del__(self):
        """Save data on destruction"""
        if self.save_path:
            try:
                self._save()
            except:
                pass


# Global cost tracker instance
_global_tracker = None


def get_cost_tracker(save_path: str = ".cache/cost_tracking.json") -> CostTracker:
    """Get or create global cost tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker(save_path)
    return _global_tracker
