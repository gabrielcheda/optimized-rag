"""
Memory Manager
High-level interface for all memory operations
"""

from typing import List, Dict, Any, Optional
import logging

from database.operations import DatabaseOperations
from .embeddings import EmbeddingService
from config import (
    ARCHIVAL_SEARCH_RESULTS,
    RECALL_SEARCH_RESULTS,
    DEFAULT_HUMAN_PERSONA,
    DEFAULT_AGENT_PERSONA
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level memory management interface for MemGPT
    Handles all memory operations: archival, recall, and core memory
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize memory manager for specific agent
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.db = DatabaseOperations()
        self.embeddings = EmbeddingService()
        
        # Initialize core memory if not exists
        self._ensure_core_memory()
        
        logger.info(f"MemoryManager initialized for agent: {agent_id}")
    
    def _ensure_core_memory(self):
        """Ensure core memory exists for this agent"""
        core_memory = self.db.get_core_memory(self.agent_id)
        if core_memory is None:
            self.db.initialize_core_memory(
                self.agent_id,
                DEFAULT_HUMAN_PERSONA,
                DEFAULT_AGENT_PERSONA
            )
            logger.info(f"Initialized default core memory for agent {self.agent_id}")
    
    # ==================== CORE MEMORY ====================
    
    def get_core_memory(self) -> Dict[str, Any]:
        """
        Retrieve core memory (persona and facts)
        
        Returns:
            Dict with human_persona, agent_persona, and facts
        """
        core_memory = self.db.get_core_memory(self.agent_id)
        if not core_memory:
            raise ValueError(f"No core memory found for agent {self.agent_id}")
        
        return core_memory
    
    def core_memory_append(self, field: str, content: str) -> bool:
        """
        Append content to a core memory field
        
        Args:
            field: 'human' or 'agent' persona
            content: Content to append
        
        Returns:
            True if successful
        """
        field_name = f"{field}_persona" if field in ['human', 'agent'] else field
        
        if field_name not in ['human_persona', 'agent_persona']:
            raise ValueError(f"Invalid field: {field}. Must be 'human' or 'agent'")
        
        # Get current value
        core_memory = self.get_core_memory()
        current_value = core_memory[field_name]
        
        # Append new content
        new_value = f"{current_value}\n{content}"
        
        success = self.db.update_core_memory_field(
            self.agent_id,
            field_name,
            new_value
        )
        
        if success:
            logger.info(f"Appended to {field_name} for agent {self.agent_id}")
            self.db.log_memory_operation(
                self.agent_id,
                "core_memory_append",
                {"field": field_name, "content": content},
                success=True
            )
        
        return success
    
    def core_memory_replace(self, field: str, old_content: str, new_content: str) -> bool:
        """
        Replace specific content in core memory
        
        Args:
            field: 'human' or 'agent' persona
            old_content: Content to replace
            new_content: New content
        
        Returns:
            True if successful
        """
        field_name = f"{field}_persona" if field in ['human', 'agent'] else field
        
        if field_name not in ['human_persona', 'agent_persona']:
            raise ValueError(f"Invalid field: {field}. Must be 'human' or 'agent'")
        
        # Get current value
        core_memory = self.get_core_memory()
        current_value = core_memory[field_name]
        
        # Count occurrences
        count = current_value.count(old_content)
        
        if count == 0:
            logger.warning(f"Content to replace not found in {field_name}")
            return False
        
        if count > 1:
            logger.warning(
                f"Found {count} occurrences of content in {field_name}. "
                f"Replacing all instances."
            )
        
        # Replace content
        new_value = current_value.replace(old_content, new_content)
        
        success = self.db.update_core_memory_field(
            self.agent_id,
            field_name,
            new_value
        )
        
        if success:
            logger.info(
                f"Replaced {count} occurrence(s) in {field_name} for agent {self.agent_id}"
            )
            self.db.log_memory_operation(
                self.agent_id,
                "core_memory_replace",
                {
                    "field": field_name,
                    "old": old_content,
                    "new": new_content,
                    "count": count
                },
                success=True
            )
        
        return success
    
    def add_core_fact(self, fact: str) -> bool:
        """
        Add a fact to core memory
        
        Args:
            fact: Fact to add
        
        Returns:
            True if successful
        """
        success = self.db.append_core_memory_fact(self.agent_id, fact)
        
        if success:
            logger.info(f"Added fact to core memory for agent {self.agent_id}")
            self.db.log_memory_operation(
                self.agent_id,
                "add_core_fact",
                {"fact": fact},
                success=True
            )
        
        return success
    
    # ==================== ARCHIVAL MEMORY ====================
    
    def archival_memory_insert(self, content: str, metadata: Optional[Dict] = None) -> int:
        """
        Insert content into archival memory with automatic embedding generation
        
        Args:
            content: Text content to store
            metadata: Optional metadata
        
        Returns:
            ID of inserted record
        
        Raises:
            ValueError: If content or embedding is empty
        """
        # Validate content
        if not content or not content.strip():
            raise ValueError("Content cannot be empty or whitespace-only")
        
        try:
            # Generate embedding
            embedding = self.embeddings.generate_embedding(content)
            
            # Validate embedding
            if not embedding or len(embedding) == 0:
                raise ValueError("Generated embedding is empty")
            
            # Insert into database
            record_id = self.db.insert_archival_memory(
                self.agent_id,
                content,
                embedding,
                metadata
            )
            
            logger.info(f"Inserted archival memory {record_id} for agent {self.agent_id}")
            self.db.log_memory_operation(
                self.agent_id,
                "archival_memory_insert",
                {"record_id": record_id, "content_length": len(content)},
                success=True
            )
            
            return record_id
        except Exception as e:
            logger.error(f"Failed to insert archival memory: {e}")
            self.db.log_memory_operation(
                self.agent_id,
                "archival_memory_insert",
                {"error": str(e)},
                success=False,
                error_message=str(e)
            )
            raise
    
    def archival_memory_search(
        self,
        query: str,
        top_k: int = ARCHIVAL_SEARCH_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in archival memory
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: id, content, similarity, metadata, created_at
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.generate_embedding(query)
            
            # Search in database
            results = self.db.search_archival_memory(
                self.agent_id,
                query_embedding,
                limit=top_k
            )
            
            logger.info(
                f"Archival search returned {len(results)} results for agent {self.agent_id}"
            )
            self.db.log_memory_operation(
                self.agent_id,
                "archival_memory_search",
                {"query": query, "results_count": len(results)},
                success=True
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to search archival memory: {e}")
            self.db.log_memory_operation(
                self.agent_id,
                "archival_memory_search",
                {"query": query, "error": str(e)},
                success=False,
                error_message=str(e)
            )
            raise
    
    # ==================== RECALL MEMORY ====================
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: int = 0
    ) -> int:
        """
        Save a conversation message to recall memory
        
        Args:
            conversation_id: Conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            tokens_used: Token count
        
        Returns:
            ID of inserted record
        """
        record_id = self.db.insert_recall_memory(
            self.agent_id,
            conversation_id,
            role,
            content,
            tokens_used
        )
        
        logger.debug(f"Saved message to recall memory: {record_id}")
        return record_id
    
    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = RECALL_SEARCH_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent conversation messages
        
        Args:
            conversation_id: Conversation identifier
            limit: Number of messages to retrieve
        
        Returns:
            List of dicts with keys: id, role, content, timestamp, tokens_used
        """
        messages = self.db.get_recent_conversation(
            self.agent_id,
            conversation_id,
            limit
        )
        
        logger.debug(f"Retrieved {len(messages)} recent messages")
        return messages
    
    def conversation_search(
        self,
        conversation_id: str,
        query: str,
        limit: int = RECALL_SEARCH_RESULTS
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history using database query
        
        Args:
            conversation_id: Conversation identifier
            query: Search query
            limit: Number of results
        
        Returns:
            List of matching messages
        """
        try:
            results = self.db.search_conversation_history(
                self.agent_id,
                conversation_id,
                query,
                limit
            )
            logger.info(f"Conversation search found {len(results)} matches")
            return results
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []
    
    # ==================== BULK OPERATIONS ====================
    
    def bulk_insert_archival(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Bulk insert multiple items into archival memory with transaction
        
        Args:
            contents: List of text contents
            metadatas: Optional list of metadata dicts
        
        Returns:
            List of inserted record IDs
        
        Raises:
            ValueError: If contents and metadatas have different lengths
        """
        if metadatas is None:
            metadatas = [{}] * len(contents)
        
        if len(contents) != len(metadatas):
            raise ValueError("Contents and metadatas must have same length")
        
        # Validate all contents before processing
        for i, content in enumerate(contents):
            if not content or not content.strip():
                raise ValueError(f"Content at index {i} is empty or whitespace-only")
        
        try:
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(contents)} items...")
            embeddings = self.embeddings.generate_embeddings_batch(contents)
            
            # Use atomic bulk insert from DB
            record_ids = self.db.bulk_insert_archival_memory(
                self.agent_id,
                contents,
                embeddings,
                metadatas
            )
            
            logger.info(f"Bulk inserted {len(record_ids)} items into archival memory")
            self.db.log_memory_operation(
                self.agent_id,
                "bulk_insert_archival",
                {"count": len(record_ids)},
                success=True
            )
            
            return record_ids
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            self.db.log_memory_operation(
                self.agent_id,
                "bulk_insert_archival",
                {"error": str(e), "count": len(contents)},
                success=False,
                error_message=str(e)
            )
            raise
