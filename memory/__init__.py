"""
Memory module for MemGPT
Handles memory operations, embeddings, and memory management
"""

from .manager import MemoryManager
from .embeddings import EmbeddingService

__all__ = ['MemoryManager', 'EmbeddingService']
