"""
Document Store
Manages documents in PostgreSQL with chunking and indexing
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    FASE 6: Persistent document storage with optimized indexing

    Supports both IVFFlat (faster inserts) and HNSW (better recall) indexes.
    HNSW is recommended for precision-critical applications.
    """

    def __init__(
        self,
        database_ops,
        embedding_service,
        chunking_strategy,
        data_wrangler=None,
        kg_extractor=None,
        index_type: str = "hnsw",    # FASE 6: HNSW by default for precision
        ivfflat_lists: int = 100     # FASE 6: Increased from 10 for better quality
    ):
        """
        Initialize document store

        Args:
            database_ops: DatabaseOperations instance
            embedding_service: EmbeddingService instance
            chunking_strategy: ChunkingStrategy instance
            data_wrangler: Optional DataWrangler instance
            kg_extractor: Optional KnowledgeGraphExtractor instance
            index_type: FASE 6 - 'hnsw' (better recall) or 'ivfflat' (faster inserts)
            ivfflat_lists: FASE 6 - Number of lists for IVFFlat (default: 100)
        """
        self.db = database_ops
        self.embeddings = embedding_service
        self.chunker = chunking_strategy
        self.wrangler = data_wrangler
        self.kg_extractor = kg_extractor
        self.index_type = index_type.lower()
        self.ivfflat_lists = ivfflat_lists

        # Get embedding dimension from service
        self.embedding_dim = self.embeddings.get_embedding_dimension()

        logger.info(
            f"FASE 6 DocumentStore initialized: index_type={self.index_type}, "
            f"ivfflat_lists={self.ivfflat_lists}, embedding_dim={self.embedding_dim}"
        )

        self._ensure_tables()
    
    def _check_dimension_mismatch(self, cur) -> bool:
        """Check if document_chunks table has wrong embedding dimension
        
        Returns:
            True if table needs recreation, False otherwise
        """
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'document_chunks' AND column_name = 'embedding'
        """)
        
        if not cur.fetchone():
            return False
        
        # Get current dimension
        cur.execute("""
            SELECT format_type(atttypid, atttypmod) as formatted_type
            FROM pg_attribute
            WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding'
        """)
        result = cur.fetchone()
        
        if not result:
            return False
        
        formatted_type = result[0]
        logger.info(f"Existing embedding type: {formatted_type}, required: vector({self.embedding_dim})")
        
        # Extract dimension from formatted type (e.g., "vector(1536)" -> 1536)
        import re
        match = re.search(r'vector\((\d+)\)', formatted_type)
        
        if match:
            existing_dim = int(match.group(1))
            if existing_dim != self.embedding_dim:
                logger.warning(
                    f"⚠️  Dimension mismatch: {existing_dim} != {self.embedding_dim}"
                )
                return True
            else:
                logger.info("✓ Table dimension matches")
        
        return False
    
    def _recreate_chunks_table_if_needed(self, cur, needs_recreation: bool):
        """Recreate document_chunks table if dimension mismatch detected"""
        if not needs_recreation:
            return
        
        # Check if table has data
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        chunk_count = cur.fetchone()[0]
        
        if chunk_count > 0:
            raise ValueError(
                f"Cannot drop table with {chunk_count} chunks. "
                f"Manually migrate data or use same embedding dimension."
            )
        
        cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
        logger.info("Dropped empty document_chunks table for recreation")
    
    def _create_indexes(self, cur):
        """
        FASE 6: Create optimized indexes for document_chunks table

        Supports two index types:
        - HNSW: Better recall, recommended for precision (default in FASE 6)
        - IVFFlat: Faster inserts, good for high-volume scenarios
        """
        # Drop existing vector index to recreate with new settings
        cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_idx")
        cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx")

        if self.index_type == "hnsw":
            # FASE 6: HNSW index for better recall and precision
            # m=16: connections per layer (higher = better recall, more memory)
            # ef_construction=64: build-time search width (higher = better quality)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            logger.info("FASE 6: Created HNSW index (m=16, ef_construction=64)")
        else:
            # IVFFlat index with configurable lists
            # FASE 6: Increased default from 10 to 100 for better quality
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self.ivfflat_lists})
            """)
            logger.info(f"Created IVFFlat index (lists={self.ivfflat_lists})")

        # B-tree index for agent_id filtering
        cur.execute("DROP INDEX IF EXISTS document_chunks_agent_embedding_idx")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_agent_id_idx
            ON document_chunks(agent_id)
        """)

        # Temporal index for recency boosting
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_created_at_idx
            ON document_chunks(created_at DESC)
        """)

        # FASE 6: Composite index for common query pattern (agent_id + embedding)
        # This helps when filtering by agent before vector search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_agent_created_idx
            ON document_chunks(agent_id, created_at DESC)
        """)
    
    def _ensure_tables(self):
        """Create tables if not exist"""
        try:
            from database.connection import db
            
            with db.get_cursor() as cur:
                # Try to increase maintenance_work_mem for index creation
                try:
                    cur.execute("SET LOCAL maintenance_work_mem = '64MB'")
                    logger.info("Increased maintenance_work_mem to 64MB")
                except Exception as mem_e:
                    logger.warning(f"Could not increase maintenance_work_mem: {mem_e}")
                
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
                
                # Check dimension and recreate if needed
                needs_recreation = self._check_dimension_mismatch(cur)
                self._recreate_chunks_table_if_needed(cur, needs_recreation)
                
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
                
                # Create indexes
                self._create_indexes(cur)
                
                cur.connection.commit()
                logger.info("Document tables and indexes created successfully")
        
        except Exception as e:
            logger.error(f"Table creation error: {e}")
            if "maintenance_work_mem" in str(e):
                logger.error(
                    "⚠️  PostgreSQL maintenance_work_mem too low. "
                    "Run: ALTER SYSTEM SET maintenance_work_mem = '64MB';"
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
