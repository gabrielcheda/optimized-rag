"""
Attribution Mapper - Phase 3 Anti-Hallucination
Maps individual claims to their source documents for full auditability
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AttributionMapper:
    """
    Creates detailed claim-to-source attribution map
    
    Phase 3: Advanced feature - provides full auditability and traceability
    """
    
    def __init__(self):
        """Initialize attribution mapper"""
        pass
    
    def create_attribution_map(
        self,
        answer: str,
        source_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create detailed attribution map linking claims to sources
        
        Args:
            answer: Generated response
            source_map: Map of citation IDs to source documents
            
        Returns:
            Dict with attribution map and metadata
        """
        try:
            # Extract claims from answer
            claims = self._extract_claims(answer)
            
            if not claims:
                return {
                    "enabled": True,
                    "claims": [],
                    "total_claims": 0,
                    "attributed_claims": 0,
                    "attribution_rate": 0.0
                }
            
            # Map each claim to its sources
            attributed_claims = []
            for claim in claims:
                sources = self._find_claim_sources(claim, source_map)
                attributed_claims.append({
                    "claim": claim["text"],
                    "sentence_number": claim["sentence_number"],
                    "citations": claim["citations"],
                    "sources": sources,
                    "is_attributed": len(sources) > 0
                })
            
            # Calculate attribution rate
            attributed_count = sum(1 for c in attributed_claims if c["is_attributed"])
            attribution_rate = attributed_count / len(attributed_claims) if attributed_claims else 0.0
            
            result = {
                "enabled": True,
                "claims": attributed_claims,
                "total_claims": len(attributed_claims),
                "attributed_claims": attributed_count,
                "attribution_rate": attribution_rate,
                "fully_traceable": attribution_rate >= 0.95
            }
            
            logger.info(
                f"Attribution map created: {attributed_count}/{len(attributed_claims)} claims attributed "
                f"({attribution_rate:.1%})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Attribution mapping failed: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "claims": [],
                "attribution_rate": 0.0
            }
    
    def _extract_claims(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extract factual claims from answer
        
        Args:
            answer: Generated response
            
        Returns:
            List of claim dicts with text, sentence number, and citations
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        claims = []
        sentence_number = 0
        
        for sent in sentences:
            sent = sent.strip()
            
            # Skip very short sentences
            if len(sent) < 20:
                continue
            
            # Skip meta/conversational sentences
            meta_patterns = [
                r'^(here|this|that|these|those|it)\s+(is|are)',
                r'^(let me|i will|i can|i would)',
                r'^(in (summary|conclusion)|overall|to summarize)',
                r'^\*\*',  # Skip markdown headers
                r'^\[confidence:',  # Skip confidence markers
                r'^⚠️',  # Skip warning markers
            ]
            
            is_meta = any(re.match(pattern, sent.lower()) for pattern in meta_patterns)
            if is_meta:
                continue
            
            # Skip "Sources:" section
            if sent.lower().startswith('source'):
                break
            
            sentence_number += 1
            
            # Extract citations from this sentence
            citations = re.findall(r'\[(\d+)\]', sent)
            
            claims.append({
                "text": sent,
                "sentence_number": sentence_number,
                "citations": citations
            })
        
        return claims
    
    def _find_claim_sources(
        self,
        claim: Dict[str, Any],
        source_map: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Find source documents for a specific claim
        
        Args:
            claim: Claim dict with text and citations
            source_map: Map of citation IDs to source documents
            
        Returns:
            List of source dicts with citation ID and source info
        """
        sources = []
        
        for citation_num in claim["citations"]:
            # Try both formats: "[1]" and "1"
            citation_key = f"[{citation_num}]"
            
            source_info = source_map.get(citation_key) or source_map.get(citation_num)
            
            if source_info:
                sources.append({
                    "citation_id": citation_num,
                    "source": source_info.get("source", "unknown"),
                    "relevance_score": source_info.get("score", 0.0)
                })
        
        return sources
    
    def format_attribution_map(self, attribution_map: Dict[str, Any]) -> str:
        """
        Format attribution map as readable text for inclusion in response
        
        Args:
            attribution_map: Attribution map dict
            
        Returns:
            Formatted string
        """
        if not attribution_map.get("enabled") or attribution_map.get("error"):
            return ""
        
        claims = attribution_map.get("claims", [])
        if not claims:
            return ""
        
        lines = ["\n\n---\n**Attribution Map:**\n"]
        
        for claim in claims:
            sentence_num = claim["sentence_number"]
            claim_text = claim["text"][:100] + "..." if len(claim["text"]) > 100 else claim["text"]
            sources = claim["sources"]
            
            if sources:
                source_list = ", ".join([f"[{s['citation_id']}]" for s in sources])
                lines.append(f"{sentence_num}. \"{claim_text}\" → {source_list}")
            else:
                lines.append(f"{sentence_num}. \"{claim_text}\" → ⚠️ No attribution")
        
        # Add summary
        attribution_rate = attribution_map.get("attribution_rate", 0)
        total = attribution_map.get("total_claims", 0)
        attributed = attribution_map.get("attributed_claims", 0)
        
        lines.append(f"\n**Traceability:** {attributed}/{total} claims ({attribution_rate:.0%})")
        
        return "\n".join(lines)
