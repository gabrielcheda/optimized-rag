"""
Self-RAG Evaluator
Evaluates retrieval and answer quality, triggers re-retrieval if needed
"""

from typing import List, Dict, Any, Tuple, Optional
import logging

from prompts.self_rag_prompts import (
    RETRIEVAL_EVALUATION_PROMPT,
    RETRIEVAL_EVALUATION_SYSTEM,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_EXTRACTION_SYSTEM,
    EVIDENCE_VERIFICATION_PROMPT,
    EVIDENCE_VERIFICATION_SYSTEM,
)
from config import MAX_CHARS_PER_DOC, MIN_SUPPORT_RATIO

logger = logging.getLogger(__name__)


class SelfRAGEvaluator:
    """Self-evaluation for RAG quality control"""
    
    def __init__(self, llm, embedding_service=None, use_ensemble: bool = True):
        """
        Initialize Self-RAG evaluator
        
        Args:
            llm: Language model for evaluation
            embedding_service: Optional embedding service for ensemble verification
            use_ensemble: Enable ensemble verification (recommended)
        """
        self.llm = llm
        self.embedding_service = embedding_service
        self.use_ensemble = use_ensemble
        
        # Initialize ensemble verifier if enabled
        self.ensemble_verifier = None
        if self.use_ensemble and self.embedding_service:
            try:
                from .ensemble_verifier import EnsembleVerifier
                self.ensemble_verifier = EnsembleVerifier(llm, embedding_service)
                logger.info("Ensemble verification enabled")
            except ImportError:
                logger.warning("Ensemble verifier not available, using standard verification")
                self.use_ensemble = False
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate if retrieved documents are relevant
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
        
        Returns:
            Evaluation results
        """
        if not retrieved_docs:
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "should_reretrieve": True,
                "reasoning": "No documents retrieved"
            }
        
        docs_summary = "\n\n".join([
            f"Doc {i+1}: {doc.get('content', '')[:200]}..."
            for i, doc in enumerate(retrieved_docs[:3])
        ])

        prompt = RETRIEVAL_EVALUATION_PROMPT.format(
            query=query,
            docs_summary=docs_summary
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=RETRIEVAL_EVALUATION_SYSTEM),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            evaluation = self._parse_evaluation(response.content)
            
            logger.info(
                f"Retrieval eval: relevant={evaluation['is_relevant']}, "
                f"confidence={evaluation['confidence']:.2f}"
            )
            
            return evaluation
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "should_reretrieve": False,
                "reasoning": "Evaluation failed"
            }
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract individual factual claims from answer"""
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=CLAIM_EXTRACTION_SYSTEM),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            
            # Parse numbered list
            claims = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    claim = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if claim:
                        claims.append(claim)
            
            return claims if claims else [answer]  # Fallback to full answer
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return [answer]
    
    def _find_supporting_evidence(
        self,
        claim: str,
        documents: List[Dict[str, Any]],
        max_chars_per_doc: int = MAX_CHARS_PER_DOC
    ) -> Dict[str, Any]:
        """Find supporting evidence for a claim in documents"""
        
        # Use ensemble verifier if available (better accuracy)
        if self.ensemble_verifier:
            result = self.ensemble_verifier.verify_claim(claim, documents, max_chars_per_doc)
            return {
                'found': result['supported'],
                'confidence': result['confidence'],
                'text': f"Ensemble verification: {result['methods']}"
            }
        
        # Fallback to standard LLM verification
        docs_content = "\n\n".join([
            f"[Doc {i+1}] {doc.get('content', '')[:max_chars_per_doc]}"
            for i, doc in enumerate(documents[:5])  # Check top 5 docs
        ])
        
        prompt = EVIDENCE_VERIFICATION_PROMPT.format(
            claim=claim,
            docs_content=docs_content
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=EVIDENCE_VERIFICATION_SYSTEM),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content.lower()
            
            supported = 'supported: yes' in content
            
            # Extract confidence
            confidence = 0.0
            for line in response.content.split('\n'):
                if 'confidence:' in line.lower():
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        confidence = 0.5 if supported else 0.0
                    break
            
            # Extract evidence text
            evidence = ""
            for line in response.content.split('\n'):
                if 'evidence:' in line.lower():
                    evidence = line.split(':', 1)[1].strip()
                    break
            
            return {
                'found': supported,
                'confidence': confidence,
                'text': evidence if evidence and evidence.lower() != 'none' else ''
            }
        except Exception as e:
            logger.error(f"Evidence finding failed: {e}")
            return {'found': False, 'confidence': 0.0, 'text': ''}
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate if answer is supported by documents with claim-level verification
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents
        
        Returns:
            Evaluation with support, claims verification, and hallucination detection
        """
        # Extract individual claims from answer
        claims = self._extract_claims(answer)
        logger.info(f"Extracted {len(claims)} claims from answer")

        MAX_CLAIMS_TO_VERIFY = 5
        if len(claims) > MAX_CLAIMS_TO_VERIFY:
            logger.info(
                f"Limiting verification to {MAX_CLAIMS_TO_VERIFY} claims "
                f"(skipping {len(claims) - MAX_CLAIMS_TO_VERIFY})"
            )
            claims = claims[:MAX_CLAIMS_TO_VERIFY]

        claim_verifications = []
        for claim in claims:
            support = self._find_supporting_evidence(
                claim,
                retrieved_docs,
                max_chars_per_doc=2500
            )
            claim_verifications.append({
                'claim': claim,
                'supported': support['found'],
                'confidence': support['confidence'],
                'evidence': support['text']
            })
        
        if claim_verifications:
            supported_count = sum(1 for c in claim_verifications if c['supported'])
            support_ratio = supported_count / len(claim_verifications)
            avg_confidence = sum(c['confidence'] for c in claim_verifications) / len(claim_verifications)
        else:
            support_ratio = 0.0
            avg_confidence = 0.0
        
        is_supported = support_ratio >= MIN_SUPPORT_RATIO
        has_hallucination = support_ratio < 0.5

        docs_content = "\n\n".join([
            doc.get('content', '')[:1000]
            for doc in retrieved_docs[:3]
        ])
        
        prompt = f"""Evaluate answer quality against documents.

Query: {query}

Answer: {answer}

Documents Summary:
{docs_content}

Claims Verification:
{"".join([f"- {c['claim']}: {'SUPPORTED' if c['supported'] else 'UNSUPPORTED'} (conf: {c['confidence']:.2f})\n" for c in claim_verifications])}

Overall Support Ratio: {support_ratio:.1%}

Based on claim-level verification:
1. Is answer grounded in documents? (at least 70% claims supported)
2. Any hallucinations? (>50% claims unsupported)
3. Is answer complete?

Respond:
SUPPORTED: [yes/no]
CONFIDENCE: [0.0-1.0]
HALLUCINATION: [yes/no]
COMPLETENESS: [complete/partial/incomplete]
REASONING: [explanation]

Evaluation:"""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content="You evaluate answer quality against documents."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            evaluation = self._parse_answer_evaluation(response.content)
            
            evaluation['is_supported'] = is_supported
            evaluation['has_hallucination'] = has_hallucination
            evaluation['support_ratio'] = support_ratio
            evaluation['avg_confidence'] = avg_confidence
            evaluation['claims_verified'] = claim_verifications
            
            logger.info(
                f"Answer eval: supported={evaluation['is_supported']}, "
                f"hallucination={evaluation['has_hallucination']}, "
                f"support_ratio={support_ratio:.1%}, "
                f"claims={len(claims)}"
            )
            
            return evaluation
        
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            return {
                "is_supported": True,
                "has_hallucination": False,
                "completeness": "unknown",
                "confidence": 0.5,
                "reasoning": "Evaluation failed"
            }
    
    def should_reretrieve(
        self,
        retrieval_eval: Dict[str, Any],
        answer_eval: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Determine if re-retrieval needed
        
        Args:
            retrieval_eval: Retrieval evaluation
            answer_eval: Optional answer evaluation
        
        Returns:
            Tuple of (should_reretrieve, reason)
        """
        if not retrieval_eval.get('is_relevant', True):
            return True, "Documents not relevant"

        if retrieval_eval.get('confidence', 1.0) < 0.7:
            return True, "Low retrieval confidence"

        if answer_eval:
            if not answer_eval.get('is_supported', True):
                return True, "Answer not supported"
            
            if answer_eval.get('has_hallucination', False):
                return True, "Hallucination detected"
            
            if (answer_eval.get('completeness') == 'incomplete' and
                answer_eval.get('confidence', 1.0) < 0.6):
                return True, "Answer incomplete"
        
        return False, "Quality acceptable"
    
    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse retrieval evaluation"""
        lines = response.strip().split('\n')
        
        is_relevant = True
        confidence = 0.7
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("RELEVANT:"):
                relevant_str = line.replace("RELEVANT:", "").strip().lower()
                is_relevant = relevant_str in ['yes', 'true', '1']
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.7
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        should_reretrieve = (not is_relevant) and (confidence < 0.3)

        return {
            "is_relevant": is_relevant,
            "confidence": confidence,
            "should_reretrieve": should_reretrieve,
            "reasoning": reasoning
        }
    
    def _parse_answer_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse answer evaluation"""
        lines = response.strip().split('\n')
        
        is_supported = True
        has_hallucination = False
        completeness = "unknown"
        confidence = 0.7
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("SUPPORTED:"):
                supported_str = line.replace("SUPPORTED:", "").strip().lower()
                is_supported = supported_str in ['yes', 'true', '1']
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    confidence = 0.7
            elif line.startswith("HALLUCINATION:"):
                halluc_str = line.replace("HALLUCINATION:", "").strip().lower()
                has_hallucination = halluc_str in ['yes', 'true', '1']
            elif line.startswith("COMPLETENESS:"):
                completeness = line.replace("COMPLETENESS:", "").strip().lower()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return {
            "is_supported": is_supported,
            "has_hallucination": has_hallucination,
            "completeness": completeness,
            "confidence": confidence,
            "reasoning": reasoning
        }
