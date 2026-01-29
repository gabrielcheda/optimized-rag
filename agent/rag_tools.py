"""
RAG Tools
LangChain tools for document management and web search
"""

from typing import Optional
from langchain.tools import tool


def create_rag_tools(document_store, web_search_tool):
    """
    Create RAG tools for agent
    
    Args:
        document_store: DocumentStore instance
        web_search_tool: WebSearchTool instance
    
    Returns:
        List of LangChain tools
    """
    
    @tool
    def upload_document(agent_id: str, file_path: str, metadata: Optional[dict] = None) -> str:
        """
        Upload and index a document for future retrieval.
        
        Args:
            agent_id: The agent ID
            file_path: Path to the document file (PDF, DOCX, TXT, MD, HTML supported)
            metadata: Optional metadata dictionary
        
        Returns:
            Upload result message
        
        Example:
            upload_document("agent_123", "report.pdf", {"category": "research"})
        """
        result = document_store.upload_and_index(agent_id, file_path, metadata=metadata)
        
        if result['success']:
            return (
                f"Document '{result['filename']}' uploaded successfully. "
                f"Created {result['chunk_count']} chunks. "
                f"Quality score: {result.get('quality_score', 'N/A')}"
            )
        else:
            return f"Upload failed: {result.get('error', 'Unknown error')}"
    
    @tool
    def search_documents(agent_id: str, query: str, max_results: int = 5) -> str:
        """
        Search indexed documents using semantic search.
        
        Args:
            agent_id: The agent ID
            query: Search query
            max_results: Maximum number of results (default: 5)
        
        Returns:
            Formatted search results
        
        Example:
            search_documents("agent_123", "machine learning algorithms", 3)
        """
        results = document_store.search(agent_id, query, top_k=max_results)
        
        if not results:
            return "No documents found matching the query."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result['filename']} (score: {result['score']:.3f})\n"
                f"   {result['content'][:200]}..."
            )
        
        return "\n\n".join(formatted)
    
    @tool
    def list_documents(agent_id: str) -> str:
        """
        List all uploaded documents for the agent.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            Formatted list of documents
        
        Example:
            list_documents("agent_123")
        """
        documents = document_store.list_documents(agent_id)
        
        if not documents:
            return "No documents uploaded yet."
        
        formatted = []
        for doc in documents:
            formatted.append(
                f"- {doc['filename']} ({doc['file_type']})\n"
                f"  Chunks: {doc['chunk_count']}, "
                f"Quality: {doc.get('quality_score', 'N/A')}, "
                f"Uploaded: {doc.get('uploaded_at', 'N/A')}"
            )
        
        return "\n".join(formatted)
    
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """
        Search the web for current information.
        
        Args:
            query: Search query
            max_results: Maximum results (default: 5)
        
        Returns:
            Formatted search results
        
        Example:
            web_search("latest AI developments 2024", 3)
        """
        results = web_search_tool.search(query, max_results=max_results)
        
        if not results:
            return "No web results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['content'][:200]}..."
            )
        
        return "\n\n".join(formatted)
    
    return [upload_document, search_documents, list_documents, web_search]
