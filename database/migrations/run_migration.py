"""
Database Migration Runner

Executes SQL migration files in order with proper error handling.
Works with both local PostgreSQL and remote databases (Supabase).

Usage:
    python database/migrations/run_migration.py [migration_number]
    
Examples:
    python database/migrations/run_migration.py           # Run all pending
    python database/migrations/run_migration.py 002       # Run specific migration
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.connection import DatabaseConnection


def get_migration_files():
    """Get all SQL migration files in order"""
    migrations_dir = Path(__file__).parent
    sql_files = sorted(migrations_dir.glob("*.sql"))
    return sql_files


def create_migrations_table(cur):
    """Create table to track applied migrations"""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def get_applied_migrations(cur):
    """Get list of already applied migrations"""
    cur.execute("SELECT migration_name FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def mark_migration_applied(cur, migration_name):
    """Record that a migration has been applied"""
    cur.execute(
        "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
        (migration_name,)
    )


def run_migration_file(cur, sql_file):
    """Execute a single migration file"""
    print(f"\n{'='*70}")
    print(f"Running: {sql_file.name}")
    print(f"{'='*70}")
    
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    # Split by statements (handle multi-statement files)
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    for i, statement in enumerate(statements, 1):
        # Skip comments-only statements
        if not statement or all(line.startswith('--') for line in statement.split('\n') if line.strip()):
            continue
        
        try:
            print(f"  Executing statement {i}/{len(statements)}...")
            cur.execute(statement)
        except Exception as e:
            print(f"  ✗ Error in statement {i}: {e}")
            # Show the problematic statement
            print(f"\n  Statement that failed:")
            print(f"  {statement[:200]}...")
            raise
    
    print(f"✓ {sql_file.name} completed successfully")


def run_migrations(specific_migration=None):
    """Run all pending migrations or a specific one"""
    db = DatabaseConnection()
    
    try:
        with db.get_cursor() as cur:
            # Create migrations tracking table
            create_migrations_table(cur)
            
            # Get applied migrations
            applied = get_applied_migrations(cur)
            print(f"\nAlready applied migrations: {len(applied)}")
            if applied:
                for name in sorted(applied):
                    print(f"  ✓ {name}")
            
            # Get all migration files
            migration_files = get_migration_files()
            
            if specific_migration:
                # Run specific migration
                migration_files = [
                    f for f in migration_files 
                    if f.stem.startswith(specific_migration)
                ]
                if not migration_files:
                    print(f"\n✗ Migration {specific_migration} not found")
                    return False
            
            # Filter out already applied
            pending_files = [
                f for f in migration_files 
                if f.stem not in applied
            ]
            
            if not pending_files:
                print("\n✓ No pending migrations")
                return True
            
            print(f"\nPending migrations: {len(pending_files)}")
            for f in pending_files:
                print(f"  → {f.name}")
            
            # Run each pending migration
            for sql_file in pending_files:
                run_migration_file(cur, sql_file)
                mark_migration_applied(cur, sql_file.stem)
            
            print(f"\n{'='*70}")
            print(f"✓ All migrations completed successfully!")
            print(f"{'='*70}")
            
            # Show final statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_migrations,
                    MAX(applied_at) as last_migration
                FROM schema_migrations
            """)
            total, last = cur.fetchone()
            print(f"\nTotal migrations applied: {total}")
            print(f"Last migration: {last}")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        return False


def show_status():
    """Show migration status without running anything"""
    db = DatabaseConnection()
    
    try:
        with db.get_cursor() as cur:
            # Check if migrations table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'schema_migrations'
                )
            """)
            if not cur.fetchone()[0]:
                print("No migrations have been run yet")
                return
            
            # Get applied migrations
            cur.execute("""
                SELECT migration_name, applied_at 
                FROM schema_migrations 
                ORDER BY applied_at
            """)
            applied = cur.fetchall()
            
            # Get all migration files
            all_migrations = get_migration_files()
            
            print(f"\n{'='*70}")
            print("Migration Status")
            print(f"{'='*70}")
            
            print(f"\nTotal migration files: {len(all_migrations)}")
            print(f"Applied migrations: {len(applied)}")
            print(f"Pending migrations: {len(all_migrations) - len(applied)}")
            
            print(f"\n{'Applied Migrations:':<50} Applied At")
            print(f"{'-'*70}")
            for name, applied_at in applied:
                print(f"✓ {name:<48} {applied_at}")
            
            # Show pending
            applied_names = {row[0] for row in applied}
            pending = [f for f in all_migrations if f.stem not in applied_names]
            
            if pending:
                print(f"\n{'Pending Migrations:'}")
                print(f"{'-'*70}")
                for f in pending:
                    print(f"→ {f.name}")
            
    except Exception as e:
        print(f"Error checking status: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run database migrations for memGPT system"
    )
    parser.add_argument(
        'migration',
        nargs='?',
        help='Specific migration number to run (e.g., "002")'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show migration status without running'
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    else:
        success = run_migrations(args.migration)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
