-- Migration 001: Initial Database Schema
-- Creates all core tables for memGPT system

-- Document chunks table for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}',
    source_file VARCHAR(500),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Core memories table for agent context
CREATE TABLE IF NOT EXISTS core_memories (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL, -- 'human', 'persona', 'context'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Archival memories table for long-term storage
CREATE TABLE IF NOT EXISTS archival_memories (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph entities
CREATE TABLE IF NOT EXISTS kg_entities (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, entity_name)
);

-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS kg_relationships (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    source_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, source_entity_id, target_entity_id, relationship_type)
);

-- Conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    intent VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Basic indexes for initial setup
CREATE INDEX IF NOT EXISTS idx_document_chunks_agent ON document_chunks(agent_id);
CREATE INDEX IF NOT EXISTS idx_core_memories_agent ON core_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_archival_memories_agent ON archival_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_kg_entities_agent ON kg_entities(agent_id);
CREATE INDEX IF NOT EXISTS idx_kg_relationships_agent ON kg_relationships(agent_id);
CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent_id);

-- Enable pgvector extension (must be done by superuser)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Initial vector index for document chunks (basic setup)
-- Note: lists=10 (minimal) due to Supabase 32MB maintenance_work_mem limit
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

-- Initial vector index for archival memories
-- Note: lists=10 (minimal) due to 32MB maintenance_work_mem limit
-- Increased from lists=40 to ensure compatibility with default PostgreSQL config
CREATE INDEX IF NOT EXISTS archival_memories_embedding_idx
ON archival_memories
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

-- Analyze tables for query planner
ANALYZE document_chunks;
ANALYZE core_memories;
ANALYZE archival_memories;
ANALYZE kg_entities;
ANALYZE kg_relationships;
ANALYZE conversations;
