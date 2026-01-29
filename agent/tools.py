"""
Memory Tools
LangChain tool definitions for memory operations
"""

from typing import List, Dict, Any, Optional
from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)


def create_memory_tools(memory_manager):
    """
    Create LangChain tools for memory operations
    
    Args:
        memory_manager: MemoryManager instance
    
    Returns:
        List of LangChain tools
    """
    
    @tool
    def core_memory_append(field: str, content: str) -> str:
        """
        Append content to core memory (human or agent persona).
        Use this to add new information that should always be remembered.
        
        Args:
            field: Either 'human' or 'agent' to specify which persona to update
            content: The content to append
        
        Returns:
            Success message
        """
        try:
            success = memory_manager.core_memory_append(field, content)
            if success:
                return f"Successfully appended to {field} persona in core memory"
            else:
                return f"Failed to append to {field} persona"
        except Exception as e:
            logger.error(f"core_memory_append error: {e}")
            return f"Error: {str(e)}"
    
    @tool
    def core_memory_replace(field: str, old_content: str, new_content: str) -> str:
        """
        Replace specific content in core memory.
        Use this to update or correct information in the persona.
        
        Args:
            field: Either 'human' or 'agent' to specify which persona to update
            old_content: The exact content to replace
            new_content: The new content
        
        Returns:
            Success message
        """
        try:
            success = memory_manager.core_memory_replace(field, old_content, new_content)
            if success:
                return f"Successfully replaced content in {field} persona"
            else:
                return f"Failed to replace content - old content not found"
        except Exception as e:
            logger.error(f"core_memory_replace error: {e}")
            return f"Error: {str(e)}"
    
    @tool
    def archival_memory_insert(content: str) -> str:
        """
        Insert content into archival memory for long-term storage.
        Use this to store information that doesn't fit in core memory but should be retrievable later.
        
        Args:
            content: The content to store in archival memory
        
        Returns:
            Success message with record ID
        """
        try:
            record_id = memory_manager.archival_memory_insert(content)
            return f"Successfully inserted into archival memory with ID {record_id}"
        except Exception as e:
            logger.error(f"archival_memory_insert error: {e}")
            return f"Error: {str(e)}"
    
    @tool
    def archival_memory_search(query: str, top_k: int = 5) -> str:
        """
        Search archival memory using semantic similarity.
        Use this to retrieve relevant information from past conversations or stored knowledge.
        
        Args:
            query: The search query
            top_k: Number of results to return (default: 5)
        
        Returns:
            Formatted search results
        """
        try:
            results = memory_manager.archival_memory_search(query, top_k)
            
            if not results:
                return "No results found in archival memory"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                similarity = result['similarity']
                content = result['content']
                formatted_results.append(
                    f"{i}. [Similarity: {similarity:.3f}] {content}"
                )
            
            return "Archival Memory Search Results:\n" + "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"archival_memory_search error: {e}")
            return f"Error: {str(e)}"
    
    @tool
    def conversation_search(query: str, limit: int = 10) -> str:
        """
        Search recent conversation history for specific content.
        Use this to recall what was discussed earlier in the conversation.
        
        Args:
            query: The search query
            limit: Number of messages to return (default: 10)
        
        Returns:
            Formatted conversation history
        """
        try:
            conversation_id = memory_manager.agent_id  # Using agent_id as conversation_id for now
            results = memory_manager.conversation_search(conversation_id, query, limit)
            
            if not results:
                return "No matching messages found in conversation history"
            
            formatted_results = []
            for msg in results:
                role = msg['role']
                content = msg['content']
                timestamp = msg['created_at']
                formatted_results.append(
                    f"[{timestamp}] {role}: {content}"
                )
            
            return "Conversation History:\n" + "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"conversation_search error: {e}")
            return f"Error: {str(e)}"
    
    @tool
    def add_core_fact(fact: str) -> str:
        """
        Add an important fact to core memory.
        Use this for key information that should always be accessible.
        
        Args:
            fact: The fact to remember
        
        Returns:
            Success message
        """
        try:
            success = memory_manager.add_core_fact(fact)
            if success:
                return f"Successfully added fact to core memory: {fact}"
            else:
                return "Failed to add fact to core memory"
        except Exception as e:
            logger.error(f"add_core_fact error: {e}")
            return f"Error: {str(e)}"
    
    return [
        core_memory_append,
        core_memory_replace,
        archival_memory_insert,
        archival_memory_search,
        conversation_search,
        add_core_fact
    ]
