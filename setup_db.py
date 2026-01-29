"""
Database Setup Script
Run this script to initialize the PostgreSQL database with pgVector
"""

import psycopg2
import logging
from pathlib import Path

from config import POSTGRES_URI
from utils.logging_config import setup_logging

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def setup_database():
    """Initialize database with schema"""
    
    print("=" * 60)
    print("MemGPT Database Setup")
    print("=" * 60)
    print()
    
    # Read schema file
    schema_path = Path(__file__).parent / "database" / "schemas.sql"
    
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Connect and execute
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(POSTGRES_URI)
        cursor = conn.cursor()
        
        print("Executing schema SQL...")
        cursor.execute(schema_sql)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✓ Database setup completed successfully!")
        print()
        print("Created tables:")
        print("  - archival_memory (with pgVector)")
        print("  - recall_memory")
        print("  - core_memory")
        print("  - memory_operations")
        print()
        
        return True
    
    except Exception as e:
        logger.error(f"Database setup failed: {e}", exc_info=True)
        print(f"✗ Database setup failed: {e}")
        return False


def verify_setup():
    """Verify database setup"""
    
    try:
        conn = psycopg2.connect(POSTGRES_URI)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('archival_memory', 'recall_memory', 'core_memory', 'memory_operations')
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        
        print("Verification:")
        print(f"  Found {len(tables)} tables:")
        for table in tables:
            print(f"    ✓ {table[0]}")
        
        # Check pgVector extension
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        
        if vector_ext:
            print("  ✓ pgVector extension enabled")
        else:
            print("  ✗ pgVector extension not found")
        
        cursor.close()
        conn.close()
        
        print()
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        print(f"✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    if setup_database():
        verify_setup()
