"""
FASE 6: Confidence Calibrator
Calibrates and adjusts confidence scores for reliable uncertainty quantification

This module ensures that confidence scores accurately reflect the true probability
of correctness, preventing overconfident hallucinations.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Calibration methods available"""
    PLATT_SCALING = "platt_scaling"
    ISOTONIC = "isotonic"
    TEMPERATURE = "temperature"
    HISTOGRAM = "histogram"
    ENSEMBLE = "ensemble"


@dataclass
class CalibrationResult:
    """Result of confidence calibration"""
    original_confidence: float
    calibrated_confidence: float
    calibration_method: str
    adjustment_factor: float
    reliability_score: float  # How reliable is this calibration
    warning: Optional[str] = None


class ConfidenceCalibrator:
    """
    FASE 6: Confidence Calibrator for precision-critical RAG

    Calibrates confidence scores to ensure they accurately reflect
    the probability of correctness. This prevents overconfident
    hallucinations by adjusting scores based on:

    1. Historical calibration data (if available)
    2. Multi-signal agreement analysis
    3. Uncertainty indicators in the response
    4. Document coverage and alignment scores
    """

    # FASE 6: Conservative calibration parameters
    DEFAULT_TEMPERATURE = 1.5  # Higher = more conservative (lower confidences)
    MIN_CONFIDENCE = 0.05  # Never report 0% confidence
    MAX_CONFIDENCE = 0.95  # Never report 100% confidence (always some uncertainty)

    # Penalty factors for various uncertainty indicators
    UNCERTAINTY_PENALTIES = {
        'hedging_language': 0.15,      # "might", "possibly", "could be"
        'missing_citations': 0.25,      # Claims without document support
        'low_retrieval_scores': 0.20,   # Poor document relevance
        'conflicting_sources': 0.30,    # Documents disagree
        'sparse_coverage': 0.15,        # Few documents cover the topic
        'cross_language': 0.10,         # Query/docs in different languages
    }

    # Confidence boost factors (used sparingly)
    CONFIDENCE_BOOSTS = {
        'exact_match': 0.10,            # Exact text match in documents
        'multiple_sources_agree': 0.08, # 3+ sources say the same thing
        'high_alignment': 0.05,         # High claim-document alignment
    }

    def __init__(
        self,
        temperature: float = 1.5,
        use_ensemble: bool = True,
        strict_mode: bool = True,  # FASE 6: Strict calibration
        calibration_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize confidence calibrator

        Args:
            temperature: Temperature for scaling (higher = more conservative)
            use_ensemble: Use ensemble of calibration methods
            strict_mode: FASE 6 - Apply stricter calibration for precision
            calibration_data: Historical calibration data (optional)
        """
        self.temperature = temperature
        self.use_ensemble = use_ensemble
        self.strict_mode = strict_mode
        self.calibration_data = calibration_data or {}

        # Statistics tracking
        self.calibration_count = 0
        self.total_adjustment = 0.0
        self.warnings_issued = 0

        logger.info(
            f"FASE 6 ConfidenceCalibrator initialized: "
            f"temperature={temperature}, strict_mode={strict_mode}, "
            f"ensemble={use_ensemble}"
        )

    def calibrate(
        self,
        raw_confidence: float,
        signals: Dict[str, Any],
        response_text: Optional[str] = None
    ) -> CalibrationResult:
        """
        Calibrate a confidence score based on multiple signals

        Args:
            raw_confidence: Original confidence score (0-1)
            signals: Dict containing various confidence signals:
                - retrieval_scores: List of document relevance scores
                - claim_alignments: List of claim alignment results
                - citation_coverage: Fraction of claims with citations
                - ensemble_agreement: Agreement between verification methods
                - source_count: Number of supporting sources
            response_text: Generated response for linguistic analysis

        Returns:
            CalibrationResult with calibrated confidence
        """
        self.calibration_count += 1

        # Clamp raw confidence to valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Step 1: Temperature scaling (base calibration)
        temp_scaled = self._temperature_scale(raw_confidence)

        # Step 2: Signal-based adjustments
        signal_adjusted, penalties, boosts = self._apply_signal_adjustments(
            temp_scaled, signals
        )

        # Step 3: Linguistic analysis (if response provided)
        if response_text:
            linguistic_adjusted = self._linguistic_calibration(
                signal_adjusted, response_text
            )
        else:
            linguistic_adjusted = signal_adjusted

        # Step 4: FASE 6 strict mode adjustments
        if self.strict_mode:
            calibrated = self._apply_strict_mode(linguistic_adjusted, signals)
        else:
            calibrated = linguistic_adjusted

        # Clamp to safe range
        calibrated = max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, calibrated))

        # Calculate adjustment factor
        adjustment_factor = calibrated / raw_confidence if raw_confidence > 0 else 1.0
        self.total_adjustment += abs(calibrated - raw_confidence)

        # Determine reliability of this calibration
        reliability = self._calculate_reliability(signals)

        # Generate warning if significant adjustment
        warning = None
        if adjustment_factor < 0.7:
            warning = "Significant confidence reduction applied"
            self.warnings_issued += 1
        elif raw_confidence > 0.9 and calibrated < 0.6:
            warning = "High confidence reduced due to uncertainty indicators"
            self.warnings_issued += 1

        # Determine method used
        method = "ensemble" if self.use_ensemble else "temperature"

        logger.debug(
            f"FASE 6 Calibration: {raw_confidence:.2f} → {calibrated:.2f} "
            f"(factor={adjustment_factor:.2f}, penalties={sum(penalties.values()):.2f})"
        )

        return CalibrationResult(
            original_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_method=method,
            adjustment_factor=adjustment_factor,
            reliability_score=reliability,
            warning=warning
        )

    def calibrate_batch(
        self,
        confidences: List[float],
        signals_list: List[Dict[str, Any]],
        response_texts: Optional[List[str]] = None
    ) -> List[CalibrationResult]:
        """
        Calibrate multiple confidence scores

        Args:
            confidences: List of raw confidence scores
            signals_list: List of signal dictionaries
            response_texts: Optional list of response texts

        Returns:
            List of CalibrationResults
        """
        results = []
        for i, (conf, signals) in enumerate(zip(confidences, signals_list)):
            response = response_texts[i] if response_texts else None
            result = self.calibrate(conf, signals, response)
            results.append(result)

        return results

    def _temperature_scale(self, confidence: float) -> float:
        """
        Apply temperature scaling to confidence

        Higher temperature = more conservative (pushes towards 0.5)
        """
        if confidence <= 0 or confidence >= 1:
            return confidence

        # Convert to logit
        logit = math.log(confidence / (1 - confidence))

        # Apply temperature
        scaled_logit = logit / self.temperature

        # Convert back to probability
        scaled = 1 / (1 + math.exp(-scaled_logit))

        return scaled

    def _apply_signal_adjustments(
        self,
        confidence: float,
        signals: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Adjust confidence based on various signals

        Returns:
            Tuple of (adjusted_confidence, penalties_applied, boosts_applied)
        """
        penalties = {}
        boosts = {}
        adjusted = confidence

        # Penalty: Low retrieval scores
        retrieval_scores = signals.get('retrieval_scores', [])
        if retrieval_scores:
            avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
            if avg_retrieval < 0.5:
                penalty = self.UNCERTAINTY_PENALTIES['low_retrieval_scores']
                penalty *= (0.5 - avg_retrieval) * 2  # Scale by how low
                penalties['low_retrieval'] = penalty
                adjusted -= penalty

        # Penalty: Missing citations
        citation_coverage = signals.get('citation_coverage', 1.0)
        if citation_coverage < 0.8:
            penalty = self.UNCERTAINTY_PENALTIES['missing_citations']
            penalty *= (0.8 - citation_coverage)  # Scale by coverage gap
            penalties['missing_citations'] = penalty
            adjusted -= penalty

        # Penalty: Low ensemble agreement
        ensemble_agreement = signals.get('ensemble_agreement', 1.0)
        if ensemble_agreement < 0.7:
            penalty = self.UNCERTAINTY_PENALTIES['conflicting_sources']
            penalty *= (0.7 - ensemble_agreement)
            penalties['low_ensemble'] = penalty
            adjusted -= penalty

        # Penalty: Sparse coverage
        source_count = signals.get('source_count', 0)
        if source_count < 3:
            penalty = self.UNCERTAINTY_PENALTIES['sparse_coverage']
            penalty *= (3 - source_count) / 3  # Scale by how sparse
            penalties['sparse_coverage'] = penalty
            adjusted -= penalty

        # Penalty: Low claim alignment
        claim_alignments = signals.get('claim_alignments', [])
        if claim_alignments:
            avg_alignment = sum(
                a.alignment_score if hasattr(a, 'alignment_score') else a.get('alignment_score', 0)
                for a in claim_alignments
            ) / len(claim_alignments)
            if avg_alignment < 0.6:
                penalty = 0.20 * (0.6 - avg_alignment)
                penalties['low_alignment'] = penalty
                adjusted -= penalty

        # Boost: Multiple sources agree (only if good alignment)
        if source_count >= 3 and ensemble_agreement >= 0.8:
            boost = self.CONFIDENCE_BOOSTS['multiple_sources_agree']
            boosts['multi_source'] = boost
            adjusted += boost

        # Boost: High alignment scores
        if claim_alignments:
            avg_alignment = sum(
                a.alignment_score if hasattr(a, 'alignment_score') else a.get('alignment_score', 0)
                for a in claim_alignments
            ) / len(claim_alignments)
            if avg_alignment >= 0.85:
                boost = self.CONFIDENCE_BOOSTS['high_alignment']
                boosts['high_alignment'] = boost
                adjusted += boost

        return adjusted, penalties, boosts

    def _linguistic_calibration(
        self,
        confidence: float,
        response_text: str
    ) -> float:
        """
        Adjust confidence based on linguistic uncertainty markers
        """
        text_lower = response_text.lower()

        # Hedging language patterns
        hedging_patterns = [
            'might', 'may', 'could', 'possibly', 'perhaps',
            'it seems', 'appears to', 'likely', 'unlikely',
            'probably', 'not sure', 'uncertain', 'unclear',
            'approximately', 'roughly', 'about', 'around',
            'i think', 'i believe', 'in my opinion'
        ]

        # Count hedging markers
        hedging_count = sum(1 for p in hedging_patterns if p in text_lower)

        if hedging_count >= 3:
            penalty = self.UNCERTAINTY_PENALTIES['hedging_language']
            confidence -= penalty
            logger.debug(f"Applied hedging penalty: {hedging_count} markers found")
        elif hedging_count >= 1:
            penalty = self.UNCERTAINTY_PENALTIES['hedging_language'] * 0.5
            confidence -= penalty

        # Check for explicit uncertainty statements
        explicit_uncertainty = [
            'não tenho certeza', 'não sei', 'não encontrei',
            "i don't know", "i'm not sure", "i couldn't find",
            'the documents do not', 'no information available'
        ]

        for phrase in explicit_uncertainty:
            if phrase in text_lower:
                confidence -= 0.20  # Strong penalty for explicit uncertainty
                break

        return confidence

    def _apply_strict_mode(
        self,
        confidence: float,
        signals: Dict[str, Any]
    ) -> float:
        """
        FASE 6: Apply strict mode calibration for maximum precision

        In strict mode:
        - Any unsupported claim significantly reduces confidence
        - Perfect scores are not possible
        - Conservative rounding is applied
        """
        # FASE 6: Check for any unsupported claims
        unsupported_claims = signals.get('unsupported_claims', 0)
        total_claims = signals.get('total_claims', 0)

        if unsupported_claims > 0 and total_claims > 0:
            # Each unsupported claim reduces confidence
            unsupported_ratio = unsupported_claims / total_claims
            penalty = min(0.30, unsupported_ratio * 0.5)
            confidence -= penalty
            logger.debug(
                f"FASE 6 strict: {unsupported_claims}/{total_claims} "
                f"unsupported claims, penalty={penalty:.2f}"
            )

        # FASE 6: Penalize if verification was not thorough
        verification_depth = signals.get('verification_depth', 'full')
        if verification_depth != 'full':
            confidence *= 0.9  # 10% reduction for incomplete verification

        # FASE 6: Conservative cap
        # Even excellent answers shouldn't be 100% confident
        if confidence > 0.90:
            confidence = 0.85 + (confidence - 0.90) * 0.5
            logger.debug(f"FASE 6 strict: Capped high confidence to {confidence:.2f}")

        return confidence

    def _calculate_reliability(self, signals: Dict[str, Any]) -> float:
        """
        Calculate how reliable this calibration is

        Higher reliability = more signals available, more consistent data
        """
        reliability = 0.5  # Base reliability

        # More signals = more reliable calibration
        signal_count = sum(1 for k in signals if signals.get(k) is not None)
        reliability += min(0.3, signal_count * 0.05)

        # Good retrieval scores = more reliable
        retrieval_scores = signals.get('retrieval_scores', [])
        if retrieval_scores and sum(retrieval_scores) / len(retrieval_scores) > 0.7:
            reliability += 0.1

        # High ensemble agreement = more reliable
        if signals.get('ensemble_agreement', 0) >= 0.8:
            reliability += 0.1

        return min(1.0, reliability)

    def recalibrate_from_feedback(
        self,
        predicted_confidence: float,
        actual_correct: bool
    ) -> None:
        """
        Update calibration parameters based on feedback

        This allows the calibrator to learn from actual outcomes.

        Args:
            predicted_confidence: The confidence we predicted
            actual_correct: Whether the answer was actually correct
        """
        # Simple exponential moving average update
        # If we were overconfident on incorrect answers, increase temperature
        # If we were underconfident on correct answers, decrease temperature

        error = predicted_confidence - (1.0 if actual_correct else 0.0)

        if error > 0.2:  # Overconfident and wrong
            self.temperature = min(3.0, self.temperature * 1.05)
            logger.info(f"Calibrator: Increased temperature to {self.temperature:.2f}")
        elif error < -0.3:  # Underconfident and right
            self.temperature = max(1.0, self.temperature * 0.98)
            logger.info(f"Calibrator: Decreased temperature to {self.temperature:.2f}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get calibration statistics"""
        avg_adjustment = (
            self.total_adjustment / self.calibration_count
            if self.calibration_count > 0 else 0
        )

        return {
            'total_calibrations': self.calibration_count,
            'average_adjustment': avg_adjustment,
            'warnings_issued': self.warnings_issued,
            'current_temperature': self.temperature,
            'strict_mode': self.strict_mode
        }

    def reset_statistics(self) -> None:
        """Reset calibration statistics"""
        self.calibration_count = 0
        self.total_adjustment = 0.0
        self.warnings_issued = 0
