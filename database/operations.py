"""
Database Operations
CRUD operations for archival, recall, and core memory tables
"""

import json
from typing import List, Dict, Any, Optional, Literal
import logging

from psycopg2 import sql
from .connection import db

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Handles all database operations for MemGPT memory management"""
    
    # ==================== ARCHIVAL MEMORY ====================
    
    @staticmethod
    def insert_archival_memory(
        agent_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Insert content with embedding into archival memory
        
        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            embedding: Vector embedding (1536 dimensions for ada-002)
            metadata: Optional metadata as JSON
        
        Returns:
            Inserted record ID
        """
        query = """
        INSERT INTO archival_memory (agent_id, content, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    query,
                    (agent_id, content, embedding, json.dumps(metadata or {}))
                )
                record_id = cursor.fetchone()[0]
                logger.info(f"Inserted archival memory ID {record_id} for agent {agent_id}")
                return record_id
        except Exception as e:
            logger.error(f"Failed to insert archival memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def bulk_insert_archival_memory(
        agent_id: str,
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict]
    ) -> List[int]:
        """
        Bulk insert multiple items into archival memory with transaction
        
        Args:
            agent_id: Unique identifier for the agent
            contents: List of text contents to store
            embeddings: List of vector embeddings
            metadatas: List of metadata dicts
        
        Returns:
            List of inserted record IDs
        
        Raises:
            ValueError: If list lengths don't match
        """
        if not (len(contents) == len(embeddings) == len(metadatas)):
            raise ValueError("Contents, embeddings, and metadatas must have same length")
        
        query = """
        INSERT INTO archival_memory (agent_id, content, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        
        record_ids = []
        
        try:
            with db.get_cursor() as cursor:
                # Executa todas as inserções dentro de uma transação
                for content, embedding, metadata in zip(contents, embeddings, metadatas):
                    cursor.execute(
                        query,
                        (agent_id, content, embedding, json.dumps(metadata))
                    )
                    record_id = cursor.fetchone()[0]
                    record_ids.append(record_id)
                
                logger.info(f"Bulk inserted {len(record_ids)} items for agent {agent_id}")
                return record_ids
        except Exception as e:
            logger.error(f"Failed to bulk insert archival memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def search_archival_memory(
        agent_id: str,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in archival memory using cosine similarity
        
        Args:
            agent_id: Agent identifier
            query_embedding: Query vector embedding
            limit: Number of results to return
        
        Returns:
            List of matching records with content and similarity scores
        """
        query = """
        SELECT 
            id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) as similarity,
            created_at
        FROM archival_memory
        WHERE agent_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    query,
                    (query_embedding, agent_id, query_embedding, limit)
                )
                results = cursor.fetchall()
                
                return [
                    {
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "similarity": float(row[3]),
                        "created_at": row[4]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Failed to search archival memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def delete_archival_memory(agent_id: str, memory_id: int) -> bool:
        """Delete specific archival memory by ID"""
        query = "DELETE FROM archival_memory WHERE id = %s AND agent_id = %s;"
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(query, (memory_id, agent_id))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete archival memory: {e}", exc_info=True)
            raise
    
    # ==================== RECALL MEMORY ====================
    
    @staticmethod
    def insert_recall_memory(
        agent_id: str,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: int = 0
    ) -> int:
        """
        Insert conversation message into recall memory
        
        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            tokens_used: Token count for the message
        
        Returns:
            Inserted record ID
        """
        query = """
        INSERT INTO recall_memory (agent_id, conversation_id, role, content, tokens_used)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    query,
                    (agent_id, conversation_id, role, content, tokens_used)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to insert recall memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def get_recent_conversation(
        agent_id: str,
        conversation_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent conversation messages
        
        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            limit: Number of messages to retrieve
        
        Returns:
            List of recent messages ordered by timestamp
        """
        query = """
        SELECT id, role, content, tokens_used, created_at
        FROM recall_memory
        WHERE agent_id = %s AND conversation_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(query, (agent_id, conversation_id, limit))
                results = cursor.fetchall()
                
                # Return in chronological order
                return [
                    {
                        "id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "tokens_used": row[3],
                        "created_at": row[4]
                    }
                    for row in reversed(results)
                ]
        except Exception as e:
            logger.error(f"Failed to get recent conversation: {e}", exc_info=True)
            raise
    
    @staticmethod
    def search_conversation_history(
        agent_id: str,
        conversation_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history using SQL LIKE
        
        Args:
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            query: Search query text
            limit: Number of results to return
        
        Returns:
            List of matching messages ordered by relevance (newest first)
        """
        sql_query = """
        SELECT id, role, content, tokens_used, created_at
        FROM recall_memory
        WHERE agent_id = %s 
          AND conversation_id = %s
          AND content ILIKE %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        
        try:
            with db.get_cursor() as cursor:
                # Add wildcards for LIKE search
                search_pattern = f"%{query}%"
                cursor.execute(sql_query, (agent_id, conversation_id, search_pattern, limit))
                results = cursor.fetchall()
                
                return [
                    {
                        "id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "tokens_used": row[3],
                        "created_at": row[4]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Failed to search conversation history: {e}", exc_info=True)
            raise
    
    # ==================== CORE MEMORY ====================
    
    @staticmethod
    def initialize_core_memory(
        agent_id: str,
        human_persona: str,
        agent_persona: str,
        facts: Optional[List[str]] = None
    ) -> None:
        """
        Initialize core memory for a new agent
        
        Args:
            agent_id: Agent identifier
            human_persona: Description of the human
            agent_persona: Description of the agent
            facts: List of important facts
        """
        query = """
        INSERT INTO core_memory (agent_id, human_persona, agent_persona, facts)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (agent_id) DO UPDATE
        SET human_persona = EXCLUDED.human_persona,
            agent_persona = EXCLUDED.agent_persona,
            facts = EXCLUDED.facts;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    query,
                    (agent_id, human_persona, agent_persona, json.dumps(facts or []))
                )
                logger.info(f"Initialized core memory for agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize core memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def get_core_memory(agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve core memory for an agent"""
        query = """
        SELECT human_persona, agent_persona, facts, created_at, updated_at
        FROM core_memory
        WHERE agent_id = %s;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(query, (agent_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "human_persona": result[0],
                        "agent_persona": result[1],
                        "facts": result[2],
                        "created_at": result[3],
                        "updated_at": result[4]
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get core memory: {e}", exc_info=True)
            raise
    
    # Whitelist of allowed fields for dynamic updates
    ALLOWED_CORE_MEMORY_FIELDS = frozenset(['human_persona', 'agent_persona'])

    @staticmethod
    def update_core_memory_field(
        agent_id: str,
        field: Literal['human_persona', 'agent_persona'],
        value: str
    ) -> bool:
        """
        Update specific field in core memory

        Args:
            agent_id: Agent identifier
            field: Field to update (human_persona or agent_persona)
            value: New value

        Returns:
            True if updated successfully

        Raises:
            ValueError: If field is not in allowed whitelist
        """
        if field not in DatabaseOperations.ALLOWED_CORE_MEMORY_FIELDS:
            raise ValueError(f"Invalid field: {field}. Allowed: {DatabaseOperations.ALLOWED_CORE_MEMORY_FIELDS}")

        query = sql.SQL("UPDATE core_memory SET {} = %s WHERE agent_id = %s;").format(
            sql.Identifier(field)
        )

        try:
            with db.get_cursor() as cursor:
                cursor.execute(query, (value, agent_id))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update core memory: {e}", exc_info=True)
            raise
    
    @staticmethod
    def append_core_memory_fact(agent_id: str, fact: str) -> bool:
        """Append a new fact to core memory facts array"""
        query = """
        UPDATE core_memory
        SET facts = facts || %s::jsonb
        WHERE agent_id = %s;
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(query, (json.dumps([fact]), agent_id))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to append core memory fact: {e}", exc_info=True)
            raise
    
    # ==================== MEMORY OPERATIONS LOG ====================
    
    @staticmethod
    def log_memory_operation(
        agent_id: str,
        operation_type: str,
        operation_details: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log memory operation for debugging and analytics"""
        query = """
        INSERT INTO memory_operations 
        (agent_id, operation_type, operation_details, success, error_message)
        VALUES (%s, %s, %s, %s, %s);
        """
        
        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        agent_id,
                        operation_type,
                        json.dumps(operation_details),
                        success,
                        error_message
                    )
                )
        except Exception as e:
            logger.error(f"Failed to log memory operation: {e}")
            # Don't raise - logging failure shouldn't break main operations
