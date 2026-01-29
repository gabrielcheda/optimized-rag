"""
Data Wrangling
Cleans, structures, and deduplicates text data for optimal RAG performance
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and normalizes text"""
    
    @staticmethod
    def remove_noise(text: str) -> str:
        """Remove common noise patterns"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def fix_encoding(text: str) -> str:
        """Fix common encoding issues"""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã©': 'é',
            'Ã¡': 'á',
            'Ã³': 'ó',
        }
        
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters"""
        if keep_punctuation:
            # Keep letters, numbers, basic punctuation, and whitespace
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        else:
            # Keep only letters, numbers, and whitespace
            text = re.sub(r'[^\w\s]+', '', text)
        
        return text
    
    def clean(self, text: str) -> str:
        """Apply all cleaning steps"""
        text = self.fix_encoding(text)
        text = self.remove_noise(text)
        text = self.normalize_whitespace(text)
        return text


class StructureExtractor:
    """Extracts structured information from text"""
    
    @staticmethod
    def extract_tables(text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract tables and convert to JSON"""
        tables = []
        
        # Simple table detection (lines with multiple | or \t)
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line or '\t' in line:
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    tables.append({
                        'type': 'table',
                        'rows': table_lines,
                        'row_count': len(table_lines)
                    })
                    table_lines = []
                in_table = False
        
        # Remove tables from text
        text_without_tables = text
        for table in tables:
            for row in table['rows']:
                text_without_tables = text_without_tables.replace(row, '', 1)
        
        return text_without_tables, tables
    
    @staticmethod
    def extract_lists(text: str) -> List[Dict[str, Any]]:
        """Extract numbered and bulleted lists"""
        lists = []
        
        # Numbered lists
        numbered_pattern = r'^\s*\d+[\.)]\s+(.+)$'
        # Bulleted lists
        bullet_pattern = r'^\s*[•\-\*]\s+(.+)$'
        
        lines = text.split('\n')
        current_list = []
        list_type = None
        
        for line in lines:
            numbered_match = re.match(numbered_pattern, line)
            bullet_match = re.match(bullet_pattern, line)
            
            if numbered_match:
                if list_type != 'numbered':
                    if current_list:
                        lists.append({'type': list_type, 'items': current_list})
                    current_list = []
                    list_type = 'numbered'
                item = numbered_match.group(1)
                current_list.append(item)
            elif bullet_match:
                if list_type != 'bulleted':
                    if current_list:
                        lists.append({'type': list_type, 'items': current_list})
                    current_list = []
                    list_type = 'bulleted'
                item = bullet_match.group(1)
                current_list.append(item)
            else:
                if current_list:
                    lists.append({'type': list_type, 'items': current_list})
                    current_list = []
                    list_type = None
        
        if current_list:
            lists.append({'type': list_type, 'items': current_list})
        
        return lists
    
    @staticmethod
    def extract_code_blocks(text: str) -> Tuple[str, List[str]]:
        """Extract code blocks"""
        code_blocks = []
        
        # Markdown code blocks
        pattern = r'```[\w]*\n(.*?)```'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            code_blocks.append(match.group(1).strip())
        
        # Remove code blocks from text
        text_without_code = re.sub(pattern, '[CODE_BLOCK]', text, flags=re.DOTALL)
        
        return text_without_code, code_blocks
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """Extract metadata from text with hierarchical structure"""
        metadata = {}
        
        # Word count
        metadata['word_count'] = len(text.split())
        
        # Character count
        metadata['char_count'] = len(text)
        
        # Line count
        metadata['line_count'] = len(text.split('\n'))
        
        # Detect language (enhanced with langdetect if available)
        try:
            from langdetect import detect
            metadata['language'] = detect(text[:1000])  # Sample first 1000 chars
        except:
            metadata['language'] = 'en'  # Default fallback
        
        # Hierarchical Structure Detection (Paper-compliant)
        # Detect document structure (sections, subsections, paragraphs)
        lines = text.split('\n')
        
        # Markdown-style headers
        h1_count = sum(1 for line in lines if line.strip().startswith('# '))
        h2_count = sum(1 for line in lines if line.strip().startswith('## '))
        h3_count = sum(1 for line in lines if line.strip().startswith('### '))
        
        metadata['hierarchical_structure'] = {
            'has_structure': h1_count > 0 or h2_count > 0,
            'section_count': h1_count,
            'subsection_count': h2_count,
            'subsubsection_count': h3_count,
            'total_headers': h1_count + h2_count + h3_count
        }
        
        # Extract document outline (titles of sections)
        sections = []
        for line in lines[:100]:  # First 100 lines
            stripped = line.strip()
            if stripped.startswith('# '):
                sections.append({'level': 1, 'title': stripped[2:].strip()})
            elif stripped.startswith('## '):
                sections.append({'level': 2, 'title': stripped[3:].strip()})
            elif stripped.startswith('### '):
                sections.append({'level': 3, 'title': stripped[4:].strip()})
        
        metadata['document_outline'] = sections[:10]  # First 10 sections
        
        # Content type indicators
        metadata['content_type'] = {
            'has_code': '```' in text or 'def ' in text or 'class ' in text,
            'has_tables': '|' in text and '---' in text,  # Markdown tables
            'has_lists': any(line.strip().startswith(('-', '*', '1.')) for line in lines[:50]),
            'has_urls': 'http://' in text or 'https://' in text,
            'has_equations': '$' in text or '\\(' in text  # LaTeX math
        }
        
        return metadata


class Deduplicator:
    """Removes duplicate and near-duplicate content"""
    
    @staticmethod
    def exact_dedup(texts: List[str]) -> List[str]:
        """Remove exact duplicates"""
        seen = set()
        unique_texts = []
        
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)
        
        logger.info(f"Exact dedup: {len(texts)} → {len(unique_texts)}")
        return unique_texts
    
    @staticmethod
    def fuzzy_dedup(texts: List[str], threshold: float = 0.95) -> List[str]:
        """Remove near-duplicates using Levenshtein distance"""
        try:
            from Levenshtein import ratio
        except ImportError:
            logger.warning("python-Levenshtein not available, skipping fuzzy dedup")
            return texts
        
        unique_texts = []
        
        for text in texts:
            is_duplicate = False
            for existing in unique_texts:
                similarity = ratio(text, existing)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
        
        logger.info(f"Fuzzy dedup: {len(texts)} → {len(unique_texts)}")
        return unique_texts
    
    @staticmethod
    def semantic_dedup(
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Remove semantically similar chunks"""
        import math
        
        def cosine_similarity(v1, v2):
            dot = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a * a for a in v1))
            mag2 = math.sqrt(sum(b * b for b in v2))
            return dot / (mag1 * mag2) if mag1 and mag2 else 0
        
        unique_chunks = []
        unique_embeddings = []
        
        for chunk, emb in zip(chunks, embeddings):
            is_duplicate = False
            for existing_emb in unique_embeddings:
                similarity = cosine_similarity(emb, existing_emb)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                unique_embeddings.append(emb)
        
        logger.info(f"Semantic dedup: {len(chunks)} → {len(unique_chunks)}")
        return unique_chunks


class QualityScorer:
    """Scores text quality for filtering low-quality chunks"""
    
    @staticmethod
    def readability_score(text: str) -> float:
        """Simple readability score (0-1)"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        # Sentence count
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Words per sentence
        words_per_sentence = len(words) / max(sentence_count, 1)
        
        # Score based on reasonable ranges
        score = 0.0
        
        # Prefer average word length between 4-8 characters
        if 4 <= avg_word_len <= 8:
            score += 0.3
        
        # Prefer 10-25 words per sentence
        if 10 <= words_per_sentence <= 25:
            score += 0.4
        
        # Has punctuation
        if any(c in text for c in '.,!?;:'):
            score += 0.3
        
        return min(score, 1.0)
    
    @staticmethod
    def information_density(text: str) -> float:
        """Measure information density (0-1)"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 3:
            return 0.0
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Not too much repetition
        word_counts = Counter(words)
        most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
        repetition_penalty = max(0, 1 - (most_common_ratio - 0.1) * 2)
        
        score = unique_ratio * 0.6 + repetition_penalty * 0.4
        
        return min(score, 1.0)
    
    @staticmethod
    def coherence_score(text: str) -> float:
        """Simple coherence heuristic (0-1)"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Has proper capitalization
        if text[0].isupper():
            score += 0.3
        
        # Has complete sentences
        if text.strip().endswith(('.', '!', '?')):
            score += 0.3
        
        # Not too short
        if len(text.split()) >= 10:
            score += 0.2
        
        # Has connective words
        connectives = ['and', 'but', 'however', 'therefore', 'because', 'since', 'although']
        if any(conn in text.lower() for conn in connectives):
            score += 0.2
        
        return min(score, 1.0)
    
    def score(self, text: str) -> float:
        """Overall quality score (0-1)"""
        if not text or len(text.strip()) < 20:
            return 0.0
        
        readability = self.readability_score(text)
        density = self.information_density(text)
        coherence = self.coherence_score(text)
        
        # Weighted average
        overall = readability * 0.3 + density * 0.4 + coherence * 0.3
        
        return overall


class DataWrangler:
    """Main data wrangling pipeline"""
    
    def __init__(
        self,
        enable_dedup: bool = True,
        min_quality_score: float = 0.3
    ):
        """
        Initialize data wrangler
        
        Args:
            enable_dedup: Enable deduplication
            min_quality_score: Minimum quality score threshold
        """
        self.cleaner = TextCleaner()
        self.extractor = StructureExtractor()
        self.deduplicator = Deduplicator()
        self.quality_scorer = QualityScorer()
        self.enable_dedup = enable_dedup
        self.min_quality_score = min_quality_score
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text through full wrangling pipeline
        
        Args:
            text: Raw text to process
        
        Returns:
            Dict with cleaned text, extracted structures, and metadata
        """
        # Clean text
        cleaned_text = self.cleaner.clean(text)
        
        # Extract structures
        text_without_tables, tables = self.extractor.extract_tables(cleaned_text)
        text_without_code, code_blocks = self.extractor.extract_code_blocks(text_without_tables)
        lists = self.extractor.extract_lists(text_without_code)
        
        # Extract metadata
        metadata = self.extractor.extract_metadata(cleaned_text)
        
        # Quality score
        quality_score = self.quality_scorer.score(cleaned_text)
        
        result = {
            'cleaned_text': cleaned_text,
            'text_only': text_without_code,
            'tables': tables,
            'code_blocks': code_blocks,
            'lists': lists,
            'metadata': metadata,
            'quality_score': quality_score,
            'passes_quality': quality_score >= self.min_quality_score
        }
        
        logger.debug(f"Processed text: quality={quality_score:.2f}, tables={len(tables)}, code={len(code_blocks)}")
        
        return result
    
    def process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chunks with deduplication
        
        Args:
            chunks: List of chunk dicts with 'content' field
            embeddings: Optional embeddings for semantic dedup
        
        Returns:
            Filtered and deduplicated chunks
        """
        # Extract content
        texts = [chunk['content'] for chunk in chunks]
        
        # Deduplicate
        if self.enable_dedup:
            texts = self.deduplicator.exact_dedup(texts)
            texts = self.deduplicator.fuzzy_dedup(texts, threshold=0.95)
        
        # Rebuild chunks
        processed_chunks = []
        for text in texts:
            # Find original chunk
            original_chunk = next((c for c in chunks if c['content'] == text), None)
            if original_chunk:
                # Score quality
                quality = self.quality_scorer.score(text)
                original_chunk['quality_score'] = quality
                
                if quality >= self.min_quality_score:
                    processed_chunks.append(original_chunk)
        
        # Semantic dedup if embeddings provided
        if self.enable_dedup and embeddings and len(embeddings) == len(processed_chunks):
            processed_chunks = self.deduplicator.semantic_dedup(
                processed_chunks,
                embeddings,
                threshold=0.95
            )
        
        logger.info(f"Processed {len(chunks)} chunks → {len(processed_chunks)} final chunks")
        
        return processed_chunks
