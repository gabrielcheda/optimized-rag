-- Fix PostgreSQL maintenance_work_mem for IVFFlat index creation
-- This script increases maintenance_work_mem to support vector index creation

-- OPTION 1: Session-level (temporary, for current session only)
-- Use this when running migrations manually
SET maintenance_work_mem = '64MB';

-- OPTION 2: System-level (persistent, requires superuser)
-- Use this to permanently fix the issue
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- After ALTER SYSTEM, reload configuration:
-- SELECT pg_reload_conf();

-- OPTION 3: Transaction-level (for specific operations)
-- Use this in application code before creating indexes
-- BEGIN;
-- SET LOCAL maintenance_work_mem = '64MB';
-- CREATE INDEX ...
-- COMMIT;

-- Verify current setting:
SHOW maintenance_work_mem;

-- Check if you have permission to alter system:
-- SELECT * FROM pg_settings WHERE name = 'maintenance_work_mem';
