"""
Document Store
Manages documents in PostgreSQL with chunking and indexing
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentStore:
    """Persistent document storage with chunking"""
    
    def __init__(
        self,
        database_ops,
        embedding_service,
        chunking_strategy,
        data_wrangler=None,
        kg_extractor=None
    ):
        """
        Initialize document store
        
        Args:
            database_ops: DatabaseOperations instance
            embedding_service: EmbeddingService instance
            chunking_strategy: ChunkingStrategy instance
            data_wrangler: Optional DataWrangler instance
        """
        self.db = database_ops
        self.embeddings = embedding_service
        self.chunker = chunking_strategy
        self.wrangler = data_wrangler
        self.kg_extractor = kg_extractor
        
        # Get embedding dimension from service
        self.embedding_dim = self.embeddings.get_embedding_dimension()
        
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if not exist"""
        try:
            from database.connection import db
            
            with db.get_cursor() as cur:
                # Try to increase maintenance_work_mem for index creation
                try:
                    cur.execute("SET LOCAL maintenance_work_mem = '64MB'")
                    logger.info("Temporarily increased maintenance_work_mem to 64MB for index creation")
                except Exception as mem_e:
                    logger.warning(f"Could not increase maintenance_work_mem: {mem_e}. Using default.")
                
                # Documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_type TEXT,
                        content TEXT,
                        metadata JSONB,
                        quality_score REAL,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(agent_id, filename)
                    )
                """)
                
                # Check if document_chunks table exists with wrong dimensions
                cur.execute("""
                    SELECT column_name, udt_name, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = 'document_chunks' AND column_name = 'embedding'
                """)
                existing_col = cur.fetchone()
                
                needs_recreation = False
                if existing_col:
                    # Check dimension from column type using format_type (more reliable)
                    cur.execute("""
                        SELECT 
                            format_type(atttypid, atttypmod) as formatted_type,
                            atttypmod - 4 as raw_dimension
                        FROM pg_attribute
                        WHERE attrelid = 'document_chunks'::regclass
                        AND attname = 'embedding'
                    """)
                    type_result = cur.fetchone()
                    
                    if type_result:
                        formatted_type, raw_dim = type_result
                        logger.info(f"Existing table embedding type: {formatted_type} (raw_dim={raw_dim})")
                        logger.info(f"Required embedding dimension: {self.embedding_dim}")
                        
                        # Extract dimension from formatted type (e.g., "vector(1536)" -> 1536)
                        import re
                        match = re.search(r'vector\((\d+)\)', formatted_type)
                        if match:
                            existing_dim = int(match.group(1))
                            if existing_dim != self.embedding_dim:
                                logger.warning(
                                    f"⚠️  document_chunks table has wrong embedding dimension "
                                    f"({existing_dim} instead of {self.embedding_dim}). "
                                    f"Recreating table to fix dimension mismatch..."
                                )
                                needs_recreation = True
                            else:
                                logger.info("✓ Table dimension matches, no recreation needed")
                        else:
                            logger.warning(f"Could not parse dimension from type: {formatted_type}")
                
                # Recreate table if dimension mismatch
                if needs_recreation:
                    # SAFETY: Check if table has data before dropping
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    chunk_count = cur.fetchone()[0]
                    
                    if chunk_count > 0:
                        logger.error(
                            f"⚠️  CANNOT DROP TABLE: {chunk_count} chunks would be lost! "
                            f"Please manually migrate data or use a different embedding dimension."
                        )
                        logger.error(
                            f"   Current table dimension: see above"
                            f"   Required dimension: {self.embedding_dim}"
                        )
                        raise ValueError(
                            f"Table dimension mismatch but table contains {chunk_count} chunks. "
                            f"Cannot auto-recreate to prevent data loss. "
                            f"Either: 1) Use same embedding dimension, or 2) Manually migrate data."
                        )
                    
                    cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
                    logger.info("Dropped document_chunks table for recreation (table was empty)")
                
                # Document chunks table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        agent_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector({self.embedding_dim}),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # OPTIMIZATION: lists=10 for compatibility with 32MB maintenance_work_mem
                # Note: lists=200 requires 63MB, lists=500 requires 157MB
                # Supabase free tier and most default PostgreSQL configs have 32MB limit
                # Drop old index if exists to recreate with new parameters
                cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_idx")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
                """)
                
                # OPTIMIZATION: Regular B-tree index for agent_id filtering
                # NOTE: Cannot use INCLUDE (embedding) because vector(1536) = 6144 bytes > 2704 byte limit
                cur.execute("DROP INDEX IF EXISTS document_chunks_agent_embedding_idx")  # Drop old problematic index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_agent_id_idx
                    ON document_chunks(agent_id)
                """)
                
                # OPTIMIZATION: Temporal index for recency boosting
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_created_at_idx
                    ON document_chunks(created_at DESC)
                """)
                
                cur.connection.commit()
                logger.info("Document tables and indexes created successfully")
        
        except Exception as e:
            logger.error(f"Table creation error: {e}")
            if "maintenance_work_mem" in str(e):
                logger.error(
                    "⚠️  PostgreSQL maintenance_work_mem is too low. "
                    "To fix, increase it in postgresql.conf or run: "
                    "ALTER SYSTEM SET maintenance_work_mem = '64MB'; "
                    "Then reload PostgreSQL configuration."
                )
            raise
    
    def upload_and_index(
        self,
        agent_id: str,
        file_path: str,
        file_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload document, chunk, embed, and index
        
        Args:
            agent_id: Agent ID
            file_path: Path to file
            file_content: Optional pre-loaded content
            metadata: Additional metadata
        
        Returns:
            Upload result with document_id and chunk count
        """
        try:
            from rag.document_processor import DocumentProcessor
            
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            file_type = file_path_obj.suffix.lower()
            
            # Process document
            if file_content:
                # Use provided content
                if self.wrangler:
                    wrangle_result = self.wrangler.process(file_content)
                    content = wrangle_result['cleaned_text']
                    quality_score = wrangle_result['quality_score']
                    extracted_metadata = wrangle_result.get('metadata', {})
                    logger.info(f"Content processed by wrangler: {len(content)} chars (quality: {quality_score})")
                else:
                    content = file_content
                    quality_score = None
                    extracted_metadata = {}
                    logger.info(f"Content used directly: {len(content)} chars")
            else:
                # Load and process file
                logger.info(f"Loading file from disk: {file_path}")
                processor = DocumentProcessor(self.wrangler)
                wrangle_result = processor.process_file(file_path)
                content = wrangle_result['cleaned_text']
                quality_score = wrangle_result['quality_score']
                extracted_metadata = wrangle_result.get('metadata', {})
                logger.info(f"File loaded and processed: {len(content)} chars (quality: {quality_score})")
            
            # Merge metadata
            full_metadata = {
                **extracted_metadata,
                **(metadata or {})
            }
            
            # Clean content: remove NUL characters (PostgreSQL incompatible)
            content = content.replace('\x00', '')
            
            # Insert document
            from database.connection import db
            
            with db.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (agent_id, filename, file_type, content, metadata, quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (agent_id, filename)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        quality_score = EXCLUDED.quality_score,
                        uploaded_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (agent_id, filename, file_type, content, json.dumps(full_metadata), quality_score))
                
                document_id = cur.fetchone()[0]
                cur.connection.commit()
            
            # Chunk document
            chunks = self.chunker.chunk(content)
            
            logger.info(f"Document chunked: {len(chunks)} chunks created from {len(content)} characters")
            
            if len(chunks) == 0:
                logger.error(f"⚠️  ZERO CHUNKS generated for {filename}!")
                logger.error(f"   Content length: {len(content)} chars")
                logger.error(f"   Content preview: {content[:200]}...")
                return {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_count": 0,
                    "chunks_created": 0,
                    "chunks_skipped": 0,
                    "quality_score": quality_score,
                    "success": True,
                    "error": "No chunks generated from document"
                }
            
            # Generate embeddings
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embeddings.generate_embeddings_batch(chunk_texts)
            
            logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks")
            
            # Insert chunks
            with db.get_cursor() as cur:
                # Clear existing chunks
                cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s",
                    (document_id,)
                )
                
                # Insert new chunks
                chunks_inserted = 0
                chunks_skipped = 0
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    # Edge case: validate embedding before insertion
                    import math
                    if not embedding or len(embedding) == 0:
                        logger.warning(f"Skipping chunk {i}: empty embedding")
                        chunks_skipped += 1
                        continue
                    if any(math.isnan(v) or math.isinf(v) for v in embedding):
                        logger.warning(f"Skipping chunk {i}: embedding contains NaN or Inf")
                        chunks_skipped += 1
                        continue
                    
                    # Merge chunk metadata with document metadata
                    chunk_metadata = {
                        **full_metadata,
                        **(chunk.get('metadata', {}))
                    }
                    
                    # Clean chunk content (remove NUL)
                    chunk_content = chunk['content'].replace('\x00', '')
                    
                    cur.execute("""
                        INSERT INTO document_chunks
                        (document_id, agent_id, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        document_id,
                        agent_id,
                        i,
                        chunk_content,
                        embedding,
                        json.dumps(chunk_metadata)
                    ))
                    chunks_inserted += 1
                
                cur.connection.commit()
                logger.info(f"✓ Inserted {chunks_inserted} chunks, skipped {chunks_skipped} (commit successful)")
            
            logger.info(f"Document indexed: {filename} ({chunks_inserted} chunks)")
            
            # Extract knowledge graph triples (Paper-compliant: knowledge graph)
            if self.kg_extractor:
                try:
                    triples = self.kg_extractor.extract_triples(
                        text=content,
                        source_doc_id=document_id,
                        max_triples=20
                    )
                    triples_stored = self.kg_extractor.store_triples(triples, agent_id)
                    logger.info(f"Stored {triples_stored} knowledge graph triples")
                except Exception as e:
                    logger.warning(f"KG extraction failed: {e}")
            
            return {
                "document_id": document_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "chunks_created": chunks_inserted,  # Return actual inserted count
                "chunks_skipped": chunks_skipped,
                "quality_score": quality_score,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Upload and index failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search document chunks
        
        Args:
            agent_id: Agent ID
            query: Search query
            top_k: Number of results
        
        Returns:
            List of matching chunks
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.generate_embedding(query)
            
            from database.connection import db
            
            with db.get_cursor() as cur:
                cur.execute("""
                    SELECT
                        dc.content,
                        dc.metadata,
                        d.filename,
                        d.file_type,
                        1 - (dc.embedding <=> %s::vector) as score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.agent_id = %s
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, agent_id, query_embedding, top_k))
                
                results = []
                for row in cur.fetchall():
                    content, metadata_json, filename, file_type, score = row
                    # PostgreSQL JSONB returns dict directly, not string
                    if isinstance(metadata_json, dict):
                        metadata = metadata_json
                    elif metadata_json:
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = {}
                    
                    results.append({
                        "content": content,
                        "filename": filename,
                        "file_type": file_type,
                        "score": float(score),
                        "metadata": metadata
                    })
                
                return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def list_documents(self, agent_id: str) -> List[Dict[str, Any]]:
        """List all documents for agent"""
        try:
            from database.connection import db
            
            with db.get_cursor() as cur:
                cur.execute("""
                    SELECT
                        id,
                        filename,
                        file_type,
                        quality_score,
                        uploaded_at,
                        (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) as chunk_count
                    FROM documents d
                    WHERE agent_id = %s
                    ORDER BY uploaded_at DESC
                """, (agent_id,))
                
                documents = []
                for row in cur.fetchall():
                    doc_id, filename, file_type, quality, uploaded, chunks = row
                    documents.append({
                        "id": doc_id,
                        "filename": filename,
                        "file_type": file_type,
                        "quality_score": float(quality) if quality else None,
                        "chunk_count": chunks,
                        "uploaded_at": uploaded.isoformat() if uploaded else None
                    })
                
                return documents
        
        except Exception as e:
            logger.error(f"List documents failed: {e}")
            return []
    
    def delete_document(self, agent_id: str, document_id: int) -> bool:
        """Delete document and chunks"""
        try:
            from database.connection import db
            
            with db.get_cursor() as cur:
                cur.execute("""
                    DELETE FROM documents
                    WHERE id = %s AND agent_id = %s
                """, (document_id, agent_id))
                
                cur.connection.commit()
                
                logger.info(f"Document deleted: {document_id}")
                return True
        
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
