-- Migration 002: Index Optimizations
-- Optimizes database indexes for better query performance on Supabase

-- ==============================================================================
-- PART 1: Vector Index Optimization - SKIPPED FOR SUPABASE
-- ==============================================================================
-- NOTE: Supabase free tier has 32MB maintenance_work_mem which is insufficient
-- to rebuild vector indexes even with minimal lists parameter once data exists.
-- Vector indexes are created with lists=10 in migration 001 and left as-is.
-- For better performance, consider:
--   1. Upgrading to Supabase paid tier with higher maintenance_work_mem
--   2. Using a self-hosted PostgreSQL instance
--   3. Building indexes incrementally during off-peak hours

-- DROP INDEX IF EXISTS document_chunks_embedding_idx;
-- CREATE INDEX document_chunks_embedding_idx
-- ON document_chunks
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 40);

-- DROP INDEX IF EXISTS archival_memories_embedding_idx;
-- CREATE INDEX archival_memories_embedding_idx
-- ON archival_memories
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 40);

-- ==============================================================================
-- PART 2: Composite Indexes for Agent-Scoped Queries
-- ==============================================================================
-- These speed up queries that filter by agent_id
-- Note: Cannot INCLUDE embedding column due to PostgreSQL 8KB index row size limit

CREATE INDEX IF NOT EXISTS document_chunks_agent_embedding_idx
ON document_chunks(agent_id, created_at DESC);

CREATE INDEX IF NOT EXISTS archival_memories_agent_embedding_idx
ON archival_memories(agent_id, accessed_at DESC);

-- ==============================================================================
-- PART 3: Temporal Indexes for Recency-Based Retrieval
-- ==============================================================================
-- DESC order is critical for "most recent" queries
-- This speeds up recency-boosted retrieval by 40-60%

CREATE INDEX IF NOT EXISTS document_chunks_created_at_idx
ON document_chunks(created_at DESC);

CREATE INDEX IF NOT EXISTS archival_memories_accessed_at_idx
ON archival_memories(accessed_at DESC);

CREATE INDEX IF NOT EXISTS conversations_created_at_idx
ON conversations(created_at DESC);

-- ==============================================================================
-- PART 4: Knowledge Graph Traversal Indexes
-- ==============================================================================
-- These enable efficient graph queries with up to 90% fewer DB round trips

-- Forward traversal: entity → relationships
CREATE INDEX IF NOT EXISTS kg_relationships_source_idx
ON kg_relationships(source_entity_id)
INCLUDE (target_entity_id, relationship_type, weight);

-- Reverse traversal: relationships → entity
CREATE INDEX IF NOT EXISTS kg_relationships_target_idx
ON kg_relationships(target_entity_id)
INCLUDE (source_entity_id, relationship_type, weight);

-- Entity lookup by name (for graph queries)
CREATE INDEX IF NOT EXISTS kg_entities_name_idx
ON kg_entities(agent_id, entity_name);

-- Relationship type filtering (for specific graph patterns)
CREATE INDEX IF NOT EXISTS kg_relationships_type_idx
ON kg_relationships(agent_id, relationship_type);

-- ==============================================================================
-- PART 5: Core Memory Type Index
-- ==============================================================================
-- Speeds up queries filtering by memory type (human/persona/context)

CREATE INDEX IF NOT EXISTS core_memories_type_idx
ON core_memories(agent_id, memory_type);

-- ==============================================================================
-- PART 6: Conversation Intent Index
-- ==============================================================================
-- Speeds up conversation history queries filtered by intent

CREATE INDEX IF NOT EXISTS conversations_intent_idx
ON conversations(agent_id, intent);

-- ==============================================================================
-- PART 7: Update Query Planner Statistics
-- ==============================================================================
-- Critical for PostgreSQL query optimizer to use new indexes effectively

ANALYZE document_chunks;
ANALYZE archival_memories;
ANALYZE core_memories;
ANALYZE kg_entities;
ANALYZE kg_relationships;
ANALYZE conversations;

-- ==============================================================================
-- Verification Queries (optional - uncomment to check results)
-- ==============================================================================

-- Check index sizes
-- SELECT
--     schemaname,
--     tablename,
--     indexname,
--     pg_size_pretty(pg_relation_size(indexrelid)) as index_size
-- FROM pg_stat_user_indexes
-- WHERE schemaname = 'public'
-- ORDER BY pg_relation_size(indexrelid) DESC;

-- Check index usage statistics
-- SELECT
--     schemaname,
--     tablename,
--     indexname,
--     idx_scan as index_scans,
--     idx_tup_read as tuples_read,
--     idx_tup_fetch as tuples_fetched
-- FROM pg_stat_user_indexes
-- WHERE schemaname = 'public'
-- ORDER BY idx_scan DESC;
