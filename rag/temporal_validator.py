"""
Temporal Validator - Phase 3 Anti-Hallucination
Detects temporal inconsistencies and validates date/timeline accuracy
"""

import logging
import re
from typing import Any, Dict, List, Tuple
from datetime import datetime
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class TemporalValidator:
    """
    Validates temporal consistency in generated responses
    
    Phase 3: Advanced feature - prevents date/timeline hallucinations
    """
    
    def __init__(self):
        """Initialize temporal validator"""
        self.current_year = datetime.now().year
        
    def validate_temporal_consistency(
        self, 
        answer: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check for temporal inconsistencies in response
        
        Args:
            answer: Generated response to validate
            documents: Source documents used for generation
            
        Returns:
            Dict with temporal validation results
        """
        try:
            # Extract temporal claims from answer
            answer_dates = self._extract_dates(answer)
            answer_events = self._extract_temporal_events(answer)
            
            if not answer_dates and not answer_events:
                return {
                    "valid": True,
                    "inconsistencies": [],
                    "confidence": 1.0,
                    "warning": None
                }
            
            inconsistencies = []
            
            # Check 1: Internal consistency (dates in answer)
            internal_issues = self._check_internal_consistency(answer_dates, answer_events)
            inconsistencies.extend(internal_issues)
            
            # Check 2: Cross-document consistency
            if documents:
                doc_dates = []
                for doc in documents:
                    content = doc.get("content", "")
                    dates = self._extract_dates(content)
                    doc_dates.extend(dates)
                
                cross_doc_issues = self._check_cross_document_consistency(
                    answer_dates, doc_dates, answer
                )
                inconsistencies.extend(cross_doc_issues)
            
            # Check 3: Future date claims (hallucination red flag)
            future_issues = self._check_future_dates(answer_dates, answer)
            inconsistencies.extend(future_issues)
            
            # Calculate confidence
            confidence = 1.0 - min(len(inconsistencies) * 0.2, 0.8)
            is_valid = len(inconsistencies) == 0
            
            result = {
                "valid": is_valid,
                "inconsistencies": inconsistencies[:5],  # Top 5
                "inconsistency_count": len(inconsistencies),
                "confidence": confidence,
                "temporal_claims": len(answer_dates) + len(answer_events),
                "warning": self._generate_warning(inconsistencies) if inconsistencies else None
            }
            
            if inconsistencies:
                logger.warning(
                    f"Temporal validation found {len(inconsistencies)} inconsistencies "
                    f"(confidence: {confidence:.2f})"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Temporal validation failed: {e}")
            return {
                "valid": True,  # Fail open
                "inconsistencies": [],
                "confidence": 0.5,
                "warning": f"Temporal validation error: {str(e)}"
            }
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dates from text
        
        Args:
            text: Text to extract dates from
            
        Returns:
            List of date dicts with parsed date and original text
        """
        dates = []
        
        # Pattern 1: Explicit years (1900-2099)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        for match in re.finditer(year_pattern, text):
            year = int(match.group(1))
            dates.append({
                "year": year,
                "text": match.group(0),
                "position": match.start()
            })
        
        # Pattern 2: Month + Year (e.g., "January 2020")
        month_year_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19\d{2}|20\d{2})\b'
        for match in re.finditer(month_year_pattern, text, re.IGNORECASE):
            try:
                date_obj = date_parser.parse(match.group(0))
                dates.append({
                    "year": date_obj.year,
                    "month": date_obj.month,
                    "text": match.group(0),
                    "position": match.start()
                })
            except:
                pass
        
        # Pattern 3: Full dates (e.g., "March 15, 2020" or "15/03/2020")
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    date_obj = date_parser.parse(match.group(0))
                    dates.append({
                        "year": date_obj.year,
                        "month": date_obj.month,
                        "day": date_obj.day,
                        "text": match.group(0),
                        "position": match.start()
                    })
                except:
                    pass
        
        return dates
    
    def _extract_temporal_events(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract temporal event descriptions
        
        Args:
            text: Text to extract events from
            
        Returns:
            List of temporal event dicts
        """
        events = []
        
        # Pattern: "before/after X" constructs
        temporal_markers = [
            (r'before\s+(\d{4})', 'before'),
            (r'after\s+(\d{4})', 'after'),
            (r'since\s+(\d{4})', 'since'),
            (r'until\s+(\d{4})', 'until'),
            (r'between\s+(\d{4})\s+and\s+(\d{4})', 'range')
        ]
        
        for pattern, event_type in temporal_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if event_type == 'range':
                    start_year = int(match.group(1))
                    end_year = int(match.group(2))
                    events.append({
                        "type": event_type,
                        "start_year": start_year,
                        "end_year": end_year,
                        "text": match.group(0)
                    })
                else:
                    year = int(match.group(1))
                    events.append({
                        "type": event_type,
                        "year": year,
                        "text": match.group(0)
                    })
        
        return events
    
    def _check_internal_consistency(
        self, 
        dates: List[Dict[str, Any]], 
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check internal temporal consistency
        
        Args:
            dates: Extracted dates
            events: Extracted temporal events
            
        Returns:
            List of inconsistencies found
        """
        inconsistencies = []
        
        # Check event ranges
        for event in events:
            if event["type"] == "range":
                if event["start_year"] > event["end_year"]:
                    inconsistencies.append({
                        "type": "reversed_range",
                        "description": f"Date range reversed: {event['text']}",
                        "severity": "high"
                    })
        
        # Check before/after relationships
        before_years = [e["year"] for e in events if e["type"] == "before"]
        after_years = [e["year"] for e in events if e["type"] == "after"]
        
        for before_year in before_years:
            for after_year in after_years:
                if before_year > after_year:
                    inconsistencies.append({
                        "type": "temporal_contradiction",
                        "description": f"Claims something before {before_year} but after {after_year}",
                        "severity": "high"
                    })
        
        return inconsistencies
    
    def _check_cross_document_consistency(
        self,
        answer_dates: List[Dict[str, Any]],
        doc_dates: List[Dict[str, Any]],
        answer: str
    ) -> List[Dict[str, Any]]:
        """
        Check consistency between answer and source documents
        
        Args:
            answer_dates: Dates extracted from answer
            doc_dates: Dates extracted from documents
            answer: Full answer text
            
        Returns:
            List of inconsistencies
        """
        inconsistencies = []
        
        if not doc_dates:
            return inconsistencies
        
        # Get year ranges from documents
        doc_years = {d["year"] for d in doc_dates if "year" in d}
        answer_years = {d["year"] for d in answer_dates if "year" in d}
        
        # Check for dates in answer not in documents (potential hallucination)
        unsupported_years = answer_years - doc_years
        
        if unsupported_years and len(doc_years) > 0:
            # Only flag if the difference is significant (>5 years from any doc date)
            min_doc_year = min(doc_years)
            max_doc_year = max(doc_years)
            
            for year in unsupported_years:
                if year < min_doc_year - 5 or year > max_doc_year + 5:
                    inconsistencies.append({
                        "type": "unsupported_date",
                        "description": f"Date {year} not found in source documents (doc range: {min_doc_year}-{max_doc_year})",
                        "severity": "medium"
                    })
        
        return inconsistencies
    
    def _check_future_dates(
        self,
        dates: List[Dict[str, Any]],
        answer: str
    ) -> List[Dict[str, Any]]:
        """
        Check for claims about future dates (hallucination indicator)
        
        Args:
            dates: Extracted dates
            answer: Full answer text
            
        Returns:
            List of inconsistencies
        """
        inconsistencies = []
        
        for date_info in dates:
            year = date_info.get("year")
            if not year:
                continue
            
            if year > self.current_year:
                # Check if it's a prediction/forecast (allowed)
                context = answer[max(0, date_info["position"]-50):date_info["position"]+100]
                prediction_markers = ["will", "predict", "forecast", "expect", "plan", "future"]
                
                is_prediction = any(marker in context.lower() for marker in prediction_markers)
                
                if not is_prediction:
                    inconsistencies.append({
                        "type": "future_date_claim",
                        "description": f"Claims fact about future year {year} (current: {self.current_year})",
                        "severity": "high"
                    })
        
        return inconsistencies
    
    def _generate_warning(self, inconsistencies: List[Dict[str, Any]]) -> str:
        """
        Generate user-facing warning message
        
        Args:
            inconsistencies: List of detected inconsistencies
            
        Returns:
            Warning message
        """
        count = len(inconsistencies)
        high_severity = sum(1 for i in inconsistencies if i.get("severity") == "high")
        
        if high_severity > 0:
            return f"Warning: Found {high_severity} high-severity temporal inconsistencies. Dates/timeline may be unreliable."
        elif count <= 2:
            return f"Warning: Found {count} minor temporal inconsistencies. Please verify dates."
        else:
            return f"Warning: Found {count} temporal inconsistencies. Timeline accuracy uncertain."
