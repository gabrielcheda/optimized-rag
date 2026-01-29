"""
Query Rewriter
Transforms queries for better retrieval
Based on RAG paper recommendations for online RAG workflow
"""

from typing import Dict, Any, List, Optional
import logging
import re

from sympy import simplify

from prompts.query_rewriter_prompts import QUERY_REWRITER_PROMPT
from rag.models.unified_rewrite import UnifiedRewrite
from .intent_recognizer import QueryIntent

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites queries for improved retrieval with conditional optimization"""
    
    def __init__(self, llm):
        """
        Initialize query rewriter
        
        Args:
            llm: Language model for rewriting
        """
        self.llm = llm
        logger.info("QueryRewriter initialized with conditional optimization")
    
    def _needs_simplification(self, query: str) -> bool:
        """Check if query needs simplification"""
        # Long or complex queries benefit from simplification
        word_count = len(query.split())
        has_complex_structure = any(marker in query.lower() for marker in [
            'however', 'moreover', 'furthermore', 'additionally', 'consequently'
        ])
        has_multiple_clauses = query.count(',') > 2 or query.count(' and ') > 2
        
        return word_count > 25 or has_complex_structure or has_multiple_clauses
    
    def _has_ambiguous_references(self, query: str) -> bool:
        """Check if query has pronouns or references needing context"""
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them', 'their', 'he', 'she']
        query_lower = query.lower()
        
        # Check for pronouns at start or in key positions
        words = query_lower.split()
        if words and words[0] in pronouns:
            return True
        
        # Check for ambiguous references
        ambiguous_patterns = [
            'the same', 'the one', 'the other', 'mentioned', 'previous', 'above', 'earlier'
        ]
        return any(pattern in query_lower for pattern in ambiguous_patterns)
    
    def _needs_reformulation(self, query: str, intent: Optional[QueryIntent] = None) -> bool:
        """Check if query needs intent-based reformulation"""
        if not intent:
            return False
        
        # Only complex intents benefit from reformulation
        complex_intents = [
            'MULTI_HOP_REASONING',
            'COMPARISON',
            'AGGREGATE'
        ]
        
        intent_str = str(intent).upper() if intent else ''
        return any(ci in intent_str for ci in complex_intents)
    
    def _has_obvious_errors(self, query: str) -> bool:
        """Check if query has obvious spelling/grammar errors"""
        # Simple heuristics for obvious errors
        has_repeated_chars = bool(re.search(r'(\w)\1{2,}', query))  # e.g., 'hellooo'
        has_mixed_case = bool(re.search(r'[a-z][A-Z]', query))  # e.g., 'heLLo'
        has_excessive_punctuation = query.count('?') > 1 or query.count('!') > 1
        
        return has_repeated_chars or has_mixed_case or has_excessive_punctuation

    def rewrite(
		self,
		query: str,
		intent: Optional[QueryIntent] = None,
		conversation_history: Optional[List[Dict[str, str]]] = None
	) -> Dict[str, Any]:
		
		# 1. Avalia heurísticas (Sistema 1)
        needs = {
			"simplify": self._needs_simplification(query),
			"contextualize": self._has_ambiguous_references(query) and conversation_history,
			"reformulate": self._needs_reformulation(query, intent),
			"correct": self._has_obvious_errors(query)
		}

		# Se nada é necessário, economizamos 100% do custo de LLM
        if not any(needs.values()):
            return {"original": query, "rewritten": query, "applied_strategies": []}

        # 2. Constrói o Prompt Dinâmico (English for precision)
        if(conversation_history):
            history_text = self._format_history(conversation_history) if needs["contextualize"] else "N/A"
		
        prompt = QUERY_REWRITER_PROMPT.format(simplify=needs["simplify"],
            contextualize=needs["contextualize"],
            reformulate=needs["reformulate"],
            correct=needs["correct"],
            query=query,
            history_text=history_text if conversation_history else "N/A")

		# 3. Chamada Única Estruturada
        structured_llm = self.llm.with_structured_output(UnifiedRewrite)
        result = structured_llm.invoke(prompt)

		# 4. Seleção da Melhor Versão (Heurística de Prioridade)
		# Prioridade: Contextualizada > Reformulada > Simplificada > Corrigida
        best_version = (
			result.contextualized_query or 
			result.reformulated_query or 
			result.simplified_query or 
			result.corrected_query or 
			query
		)

        return {
			"original": query,
			"rewritten": best_version,
			"metadata": result.model_dump(),
			"strategies": [k for k, v in needs.items() if v]
		} 
   
    def _format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Formata o histórico recente para o LLM.
        Foca nas últimas mensagens para economizar tokens e manter o foco.
        """
        if not conversation_history:
            return "No previous history available."

        # Pegamos as últimas 3 a 5 mensagens (janela ideal para contextualização)
        recent_context = conversation_history[-5:]

        formatted_messages = []
        for msg in recent_context:
            role = msg.get("role", "user").upper()
            # Limitamos o conteúdo de cada mensagem para evitar 'noise'
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "... [truncated]"

            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def simplify(self, query: str) -> str:
        """
        Simplify complex queries
        Remove unnecessary words, clarify structure
        """
        try:
            # Detect language to maintain consistency
            original_lang = None
            try:
                from langdetect import detect
                original_lang = detect(query)
                lang_instruction = f"IMPORTANT: You MUST maintain the SAME language as the original query ({original_lang}). Do NOT translate."
            except:
                lang_instruction = "IMPORTANT: You MUST maintain the SAME language as the original query. Do NOT translate."
            
            prompt = f"""Simplify this query while preserving its meaning and language.
Remove unnecessary words, clarify ambiguities, make it more direct.
{lang_instruction}

Original: {query}

Simplified:"""
            
            max_retries = 2
            for attempt in range(max_retries):
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                simplified = response.content.strip()
                
                # Clean up
                simplified = simplified.replace("Simplified:", "").strip()
                simplified = simplified.strip('"\'')
                
                # OPTIMIZATION: Validate language consistency to prevent PT→ES drift
                # This prevents CrossEncoder score=0.000 failures
                if original_lang and simplified:
                    try:
                        output_lang = detect(simplified)
                        if output_lang == original_lang:
                            return simplified
                        else:
                            logger.warning(
                                f"Language drift detected: {original_lang}→{output_lang} "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            if attempt < max_retries - 1:
                                # Retry with stronger instruction
                                lang_instruction = f"CRITICAL: Output MUST be in {original_lang}. Answer in {original_lang} ONLY."
                                continue
                            else:
                                # Last attempt failed, return original
                                logger.error(f"Language drift persisted after {max_retries} attempts, returning original query")
                                return query
                    except:
                        # Language detection failed, accept result
                        return simplified if simplified else query
                else:
                    return simplified if simplified else query
            
            return query
        
        except Exception as e:
            logger.error(f"Query simplification failed: {e}")
            return query
    
    def add_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Add conversation context to ambiguous queries
        Example: "What about the other one?" -> "What about the iPhone 15 Pro?"
        """
        if not conversation_history:
            return query
        
        # Check if query is ambiguous (has pronouns or references)
        ambiguous_terms = ["it", "that", "this", "they", "them", "other", "one", "thing"]
        
        has_ambiguity = any(term in query.lower() for term in ambiguous_terms)
        
        if not has_ambiguity:
            return query
        
        try:
            # Detect original language
            original_lang = None
            try:
                from langdetect import detect
                original_lang = detect(query)
            except:
                pass
            
            # Get recent context
            recent = conversation_history[-5:]
            context_text = "\n".join([
                f"{msg['role']}: {msg['content'][:200]}"
                for msg in recent
            ])
            
            prompt = f"""Given this conversation history, resolve ambiguous references in the query.
IMPORTANT: Keep the SAME language as the original query. Do NOT translate.

Conversation:
{context_text}

Query: {query}

Rewrite the query with full context (replace "it", "that", "other one" with actual entities):"""
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            contextualized = response.content.strip()
            
            # Clean up
            contextualized = contextualized.strip('"\'')
            
            # Validate language consistency
            if original_lang and contextualized:
                try:
                    output_lang = detect(contextualized)
                    if output_lang != original_lang:
                        logger.warning(f"add_context() language drift {original_lang}→{output_lang}, returning original")
                        return query
                except:
                    pass
            
            return contextualized if contextualized else query
        
        except Exception as e:
            logger.error(f"Query contextualization failed: {e}")
            return query
    
    def reformulate(
        self,
        query: str,
        intent: Optional[QueryIntent] = None
    ) -> str:
        """
        Reformulate query based on intent
        Adapt phrasing for better retrieval
        """
        if not intent:
            return query
        
        try:
            intent_instructions = {
                QueryIntent.QUESTION_ANSWERING: "Make it a clear, specific question",
                QueryIntent.SUMMARIZATION: "Reformulate as a request for summary",
                QueryIntent.COMPARISON: "Structure as comparison query (X vs Y)",
                QueryIntent.FACT_CHECKING: "Reformulate as verification query",
                QueryIntent.MULTI_HOP_REASONING: "Break into logical sub-questions",
                QueryIntent.SEARCH: "Reformulate as search keywords"
            }
            
            instruction = intent_instructions.get(
                intent,
                "Reformulate for clarity"
            )
            
            # Detect language
            original_lang = None
            try:
                from langdetect import detect
                original_lang = detect(query)
                lang_instruction = f"CRITICAL: Output MUST be in {original_lang}. Do NOT translate to any other language."
            except:
                lang_instruction = "CRITICAL: Keep the EXACT same language as the query. Do NOT translate."
            
            prompt = f"""Reformulate this query for better information retrieval.
Goal: {instruction}
{lang_instruction}

Original: {query}

Reformulated:"""
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            reformulated = response.content.strip()
            
            # Clean up
            reformulated = reformulated.replace("Reformulated:", "").strip()
            reformulated = reformulated.strip('"\'')
            
            # Validate language consistency
            if original_lang and reformulated:
                try:
                    output_lang = detect(reformulated)
                    if output_lang != original_lang:
                        logger.warning(f"reformulate() language drift {original_lang}→{output_lang}, returning original")
                        return query
                except:
                    pass
            
            return reformulated if reformulated else query
        
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            return query
    
    def correct_errors(self, query: str) -> str:
        """
        Correct spelling and grammar errors
        """
        try:
            # Simple heuristic check - if query looks clean, skip
            if len(query.split()) < 3:
                return query
            
            # Detect original language
            original_lang = None
            try:
                from langdetect import detect
                original_lang = detect(query)
            except:
                pass
            
            prompt = f"""Correct any spelling or grammar errors in this query.
If no errors, return the query unchanged.
IMPORTANT: Keep the SAME language. Do NOT translate.

Query: {query}

Corrected:"""
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            corrected = response.content.strip()
            
            # Clean up
            corrected = corrected.replace("Corrected:", "").strip()
            corrected = corrected.strip('"\'')
            
            # Validate language consistency
            if original_lang and corrected:
                try:
                    output_lang = detect(corrected)
                    if output_lang != original_lang:
                        logger.warning(f"correct_errors() language drift {original_lang}→{output_lang}, returning original")
                        return query
                except:
                    pass
            
            # If too different, probably wrong - keep original
            if len(corrected) > len(query) * 1.5 or len(corrected) < len(query) * 0.5:
                return query
            
            return corrected if corrected else query
        
        except Exception as e:
            logger.error(f"Query error correction failed: {e}")
            return query
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries
        For multi-hop reasoning
        """
        try:
            prompt = f"""Break this complex query into 2-4 simpler sub-queries.
Each sub-query should be independently answerable.

Query: {query}

Sub-queries (one per line):"""
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            lines = response.content.strip().split('\n')
            sub_queries = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering
                line = line.lstrip('0123456789.-) ')
                line = line.strip('"\'')
                
                if line and len(line) > 5:
                    sub_queries.append(line)
            
            return sub_queries if sub_queries else [query]
        
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        """
        try:
            prompt = f"""Generate 2-3 alternative phrasings of this query using synonyms.
Keep the same meaning but vary the words.

Query: {query}

Alternatives (one per line):"""
            
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            lines = response.content.strip().split('\n')
            alternatives = [query]  # Include original
            
            for line in lines:
                line = line.strip()
                # Remove numbering
                line = line.lstrip('0123456789.-) ')
                line = line.strip('"\'')
                
                if line and len(line) > 5:
                    alternatives.append(line)
            
            return alternatives[:4]  # Max 4 including original
        
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def _select_best_rewrite(
        self,
        original: str,
        simplified: str,
        contextualized: str,
        reformulated: str,
        corrected: str
    ) -> str:
        """
        Select best rewritten version based on heuristics
        """
        # Priority: contextualized > reformulated > simplified > corrected > original
        
        # Prefer contextualized if significantly different (resolved ambiguity)
        if contextualized != original and len(contextualized) > len(original) * 0.8:
            if len(contextualized) < len(original) * 2:  # Not too long
                return contextualized
        
        # Use reformulated if different
        if reformulated != original and len(reformulated) > 5:
            return reformulated
        
        # Use simplified if different and shorter
        if simplified != original and len(simplified) < len(original):
            return simplified
        
        # Use corrected if different
        if corrected != original:
            return corrected
        
        # Default to original
        return original
    
    def _get_applied_strategies(self, original: str, rewritten: str) -> List[str]:
        """Identify which strategies were applied"""
        strategies = []
        
        if rewritten != original:
            if len(rewritten) < len(original) * 0.8:
                strategies.append("simplification")
            
            if len(rewritten) > len(original) * 1.2:
                strategies.append("contextualization")
            
            # Check for structural changes
            if "vs" in rewritten.lower() and "vs" not in original.lower():
                strategies.append("reformulation")
            
            # Check for corrections (different words)
            orig_words = set(original.lower().split())
            rewr_words = set(rewritten.lower().split())
            
            if len(orig_words - rewr_words) > 0:
                strategies.append("correction")
        
        return strategies if strategies else ["none"]
