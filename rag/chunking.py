"""
Chunking Strategies
Semantic, recursive, and fixed-size chunking for optimal retrieval
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import re
import logging

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces"""
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_metadata = {
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end,
                "chunk_size": len(chunk_text),
                **(metadata or {})
            }
            
            chunks.append({
                "content": chunk_text.strip(),
                "metadata": chunk_metadata
            })
            
            start = end - self.overlap
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Recursive chunking respecting document structure"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        chunks = self._recursive_split(text, 0)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": i,
                "chunk_size": len(chunk_text),
                **(metadata or {})
            }
            
            result.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        logger.info(f"Created {len(result)} recursive chunks")
        return result
    
    def _recursive_split(self, text: str, separator_index: int) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        
        if separator_index >= len(self.separators):
            return self._force_split(text)
        
        separator = self.separators[separator_index]
        
        if separator == "":
            return self._force_split(text)
        
        splits = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, separator_index + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split + separator
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _force_split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on embedding similarity"""
    
    def __init__(
        self,
        embedding_service,
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        sentences = self._split_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        if len(text) < self.min_chunk_size:
            return [{
                "content": text,
                "metadata": {**(metadata or {}), "chunk_id": 0}
            }]
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_batch(sentences)
        
        # Group into semantic chunks
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        chunk_id = 0
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            similarity = self._cosine_similarity(current_embedding, sentence_embedding)
            current_size = sum(len(s) for s in current_chunk)
            
            if (similarity >= self.similarity_threshold and 
                current_size + len(sentence) <= self.max_chunk_size):
                current_chunk.append(sentence)
                current_embedding = self._average_embeddings([current_embedding, sentence_embedding])
            else:
                if current_size >= self.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, chunk_id, metadata))
                    chunk_id += 1
                    current_chunk = [sentence]
                    current_embedding = sentence_embedding
                else:
                    current_chunk.append(sentence)
                    current_embedding = self._average_embeddings([current_embedding, sentence_embedding])
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, chunk_id, metadata))
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        num_embeddings = len(embeddings)
        dim = len(embeddings[0])
        avg_embedding = [0.0] * dim
        
        for embedding in embeddings:
            for i in range(dim):
                avg_embedding[i] += embedding[i]
        
        return [x / num_embeddings for x in avg_embedding]
    
    def _create_chunk(self, sentences: List[str], chunk_id: int, base_metadata: Optional[Dict]) -> Dict[str, Any]:
        content = " ".join(sentences)
        
        chunk_metadata = {
            "chunk_id": chunk_id,
            "num_sentences": len(sentences),
            "chunk_size": len(content),
            **(base_metadata or {})
        }
        
        return {
            "content": content,
            "metadata": chunk_metadata
        }
