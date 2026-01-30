"""
Database Connection Manager
Handles PostgreSQL connection pooling and context management
"""

import atexit
from psycopg2 import pool, OperationalError
from psycopg2.extensions import connection, cursor
from contextlib import contextmanager
from typing import Optional, Generator
import logging

from config import POSTGRES_URI

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages PostgreSQL connection pool for MemGPT"""
    
    _instance: Optional['DatabaseConnection'] = None
    _connection_pool: Optional[pool.ThreadedConnectionPool] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single connection pool"""
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection pool if not already created"""
        if self._connection_pool is None:
            self._create_pool()
    
    def _create_pool(self):
        """Create PostgreSQL connection pool"""
        try:
            self._connection_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=POSTGRES_URI
            )
            logger.info("Database connection pool created successfully")
            # Register cleanup on program exit
            atexit.register(self.close_all_connections)
        except OperationalError as e:
            logger.error(f"Failed to connect to database: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}", exc_info=True)
            raise
    
    @contextmanager
    def get_connection(self) -> Generator[connection, None, None]:
        """
        Context manager for getting database connection from pool

        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = None
        try:
            if self._connection_pool is None:
                raise RuntimeError("Connection pool not initialized")
            conn = self._connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            if conn and self._connection_pool is not None:
                self._connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self) -> Generator[cursor, None, None]:
        """
        Context manager for getting cursor directly

        Usage:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT ...")
                results = cursor.fetchall()
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("All database connections closed")
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None and result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}", exc_info=True)
            return False


# Global database connection instance
db = DatabaseConnection()
