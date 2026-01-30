"""
Citation Validator
Validates that generated responses properly cite their sources
PHASE 1: Basic citation validation
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citation format and completeness in generated responses"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize citation validator
        
        Args:
            strict_mode: If True, requires citations for all factual sentences
        """
        self.strict_mode = strict_mode
    
    def validate_citations(
        self, 
        answer: str, 
        source_map: Dict[str, Any],
        allow_no_citations: bool = False
    ) -> Dict[str, Any]:
        """
        Validate citation format and completeness
        
        Args:
            answer: Generated response
            source_map: Mapping of citation numbers to sources
            allow_no_citations: If True, allows responses without citations (for conversational)
        
        Returns:
            Dict with validation results
        """
        # Extract all citations in format [N]
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(citations)
        
        # Check 1: At least one citation exists (unless allowed to skip)
        if not citations:
            if allow_no_citations:
                return {
                    "valid": True,
                    "citation_count": 0,
                    "warning": "No citations found (allowed for conversational response)"
                }
            return {
                "valid": False,
                "error": "No citations found in response",
                "citation_count": 0
            }
        
        # Check 2: All citations map to valid sources
        invalid_citations = []
        for citation in unique_citations:
            # Try both formats: "1" and "[1]"
            citation_key = f"[{citation}]"
            if citation not in source_map and citation_key not in source_map:
                invalid_citations.append(citation)
        
        if invalid_citations:
            return {
                "valid": False,
                "error": f"Invalid citation numbers: {invalid_citations} (not in source_map)",
                "citation_count": len(unique_citations),
                "invalid_citations": invalid_citations
            }
        
        # Check 3: Sources section exists
        has_sources_section = "Sources:" in answer or "sources:" in answer.lower()
        
        # Check 4: Factual claims have citations (in strict mode)
        if self.strict_mode:
            uncited_result = self._check_uncited_claims(answer)
            if not uncited_result["valid"]:
                return uncited_result
        
        return {
            "valid": True,
            "citation_count": len(unique_citations),
            "total_citation_occurrences": len(citations),
            "has_sources_section": has_sources_section,
            "cited_sources": list(unique_citations)
        }
    
    def _check_uncited_claims(self, answer: str) -> Dict[str, Any]:
        """
        Check if factual claims have citations (strict mode)
        
        Returns:
            Dict with validation results
        """
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        # Filter to potential factual sentences
        factual_sentences = []
        conversational_patterns = [
            r'\b(i|you|we|let me|here|this|that|would|could|should)\b',
            r'\b(thank|please|sorry|hope|think|believe)\b',
            r'^(yes|no|sure|ok|okay|well)\b'
        ]
        
        for sent in sentences:
            sent = sent.strip()
            
            # Skip short sentences
            if len(sent) < 30:
                continue
            
            # Skip conversational sentences
            is_conversational = any(
                re.search(pattern, sent.lower()) 
                for pattern in conversational_patterns
            )
            if is_conversational:
                continue
            
            # Skip sources section
            if sent.lower().startswith('source'):
                continue
            
            # Potential factual claim
            factual_sentences.append(sent)
        
        # Check if factual sentences have citations
        uncited_sentences = []
        for sent in factual_sentences:
            if '[' not in sent:  # No citation
                uncited_sentences.append(sent[:80] + "..." if len(sent) > 80 else sent)
        
        # Allow up to 2 uncited factual sentences
        max_uncited_allowed = 2
        
        if len(uncited_sentences) > max_uncited_allowed:
            return {
                "valid": False,
                "error": f"{len(uncited_sentences)} factual sentences without citations (max {max_uncited_allowed} allowed)",
                "uncited_sentences": uncited_sentences[:3],
                "uncited_count": len(uncited_sentences)
            }
        
        return {"valid": True}
    
    def extract_cited_sources(self, answer: str) -> List[str]:
        """Extract all citation numbers from answer"""
        citations = re.findall(r'\[(\d+)\]', answer)
        return list(set(citations))
