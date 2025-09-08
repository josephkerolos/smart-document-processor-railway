"""
Database State Manager for Smart Document Processor
Handles persistent storage of batch and processing states
"""

import os
import json
import asyncpg
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DBStateManager:
    def __init__(self):
        self.pool = None
        self.use_sqlite = False
        self.sqlite_manager = None
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
    
    async def initialize(self):
        """Initialize database connection pool with SQLite fallback"""
        # First try PostgreSQL
        if not os.getenv('USE_SQLITE', 'false').lower() == 'true':
            try:
                self.pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)
                logger.info("PostgreSQL connection pool initialized")
                
                # Create tables if they don't exist
                await self.create_tables()
                return
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL pool: {e}")
                logger.info("Falling back to SQLite...")
        
        # Fallback to SQLite
        try:
            from db_state_manager_sqlite import db_state_manager_sqlite
            self.sqlite_manager = db_state_manager_sqlite
            await self.sqlite_manager.initialize()
            self.use_sqlite = True
            self.pool = True  # Set to True to indicate initialization
            logger.info("Using SQLite state manager as fallback")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite fallback: {e}")
            raise
    
    async def create_tables(self):
        """Create state tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Read and execute the SQL file
            sql_file_path = os.path.join(os.path.dirname(__file__), 'create_state_tables.sql')
            if os.path.exists(sql_file_path):
                with open(sql_file_path, 'r') as f:
                    await conn.execute(f.read())
                logger.info("State tables created/verified")
    
    async def close(self):
        """Close database connection pool"""
        if self.use_sqlite and self.sqlite_manager:
            await self.sqlite_manager.close()
        elif self.pool and not self.use_sqlite:
            await self.pool.close()
    
    # Batch Processing State Management
    
    async def save_batch_state(self, batch_id: str, session_ids: List[str], status: str = 'active', metadata: Dict = None):
        """Save or update batch processing state"""
        if self.use_sqlite:
            return await self.sqlite_manager.save_batch_state(batch_id, session_ids, status, metadata)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO batch_processing_state (batch_id, session_ids, status, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (batch_id) DO UPDATE
                SET session_ids = $2, status = $3, metadata = $4, updated_at = CURRENT_TIMESTAMP
            """, batch_id, json.dumps(session_ids), status, json.dumps(metadata or {}))
            
            logger.info(f"Saved batch state for {batch_id}: {len(session_ids)} sessions, status={status}")
    
    async def get_batch_state(self, batch_id: str) -> Optional[Dict]:
        """Get batch processing state"""
        if self.use_sqlite:
            return await self.sqlite_manager.get_batch_state(batch_id)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT batch_id, session_ids, status, metadata, created_at, updated_at
                FROM batch_processing_state
                WHERE batch_id = $1
            """, batch_id)
            
            if row:
                return {
                    'batch_id': row['batch_id'],
                    'session_ids': json.loads(row['session_ids']),
                    'status': row['status'],
                    'metadata': json.loads(row['metadata']),
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }
            return None
    
    async def get_active_batches(self) -> List[Dict]:
        """Get all active batch states"""
        if self.use_sqlite:
            return await self.sqlite_manager.get_active_batches()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT batch_id, session_ids, status, metadata, created_at, updated_at
                FROM batch_processing_state
                WHERE status = 'active'
                ORDER BY created_at DESC
            """)
            
            return [{
                'batch_id': row['batch_id'],
                'session_ids': json.loads(row['session_ids']),
                'status': row['status'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at'].isoformat(),
                'updated_at': row['updated_at'].isoformat()
            } for row in rows]
    
    async def update_batch_status(self, batch_id: str, status: str):
        """Update batch status"""
        if self.use_sqlite:
            return await self.sqlite_manager.update_batch_status(batch_id, status)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE batch_processing_state
                SET status = $2, updated_at = CURRENT_TIMESTAMP
                WHERE batch_id = $1
            """, batch_id, status)
    
    async def delete_batch_state(self, batch_id: str):
        """Delete batch state (cascades to processing status)"""
        if self.use_sqlite:
            return await self.sqlite_manager.delete_batch_state(batch_id)
        
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM batch_processing_state WHERE batch_id = $1", batch_id)
    
    # Processing Status Management
    
    async def save_processing_status(self, session_id: str, batch_id: str, status: str, 
                                   processed_files: List[str] = None, extractions: Dict = None,
                                   total_files: int = 0, metadata: Dict = None):
        """Save or update processing session status"""
        if self.use_sqlite:
            return await self.sqlite_manager.save_processing_status(
                session_id, batch_id, status, processed_files, extractions, total_files, metadata
            )
        
        if self.pool is None:
            logger.warning(f"Database pool not initialized, skipping status save for session {session_id}")
            return
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO processing_status_db 
                (session_id, batch_id, status, processed_files, extractions, total_files, processed_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (session_id) DO UPDATE
                SET status = $3, processed_files = $4, extractions = $5, 
                    total_files = $6, processed_count = $7, metadata = $8,
                    updated_at = CURRENT_TIMESTAMP
            """, session_id, batch_id, status, 
                json.dumps(processed_files or []), 
                json.dumps(extractions or {}),
                total_files,
                len(processed_files) if processed_files else 0,
                json.dumps(metadata or {}))
    
    async def get_processing_status(self, session_id: str) -> Optional[Dict]:
        """Get processing session status"""
        if self.use_sqlite:
            return await self.sqlite_manager.get_processing_status(session_id)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT session_id, batch_id, status, processed_files, extractions, 
                       total_files, processed_count, metadata, created_at, updated_at
                FROM processing_status_db
                WHERE session_id = $1
            """, session_id)
            
            if row:
                return {
                    'session_id': row['session_id'],
                    'batch_id': row['batch_id'],
                    'status': row['status'],
                    'processed_files': json.loads(row['processed_files']),
                    'extractions': json.loads(row['extractions']),
                    'total_files': row['total_files'],
                    'processed_count': row['processed_count'],
                    'metadata': json.loads(row['metadata']),
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }
            return None
    
    async def get_batch_processing_status(self, batch_id: str) -> List[Dict]:
        """Get all processing sessions for a batch"""
        if self.use_sqlite:
            return await self.sqlite_manager.get_batch_processing_status(batch_id)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT session_id, status, processed_count, total_files, created_at, updated_at
                FROM processing_status_db
                WHERE batch_id = $1
                ORDER BY created_at
            """, batch_id)
            
            return [{
                'session_id': row['session_id'],
                'status': row['status'],
                'processed_count': row['processed_count'],
                'total_files': row['total_files'],
                'created_at': row['created_at'].isoformat(),
                'updated_at': row['updated_at'].isoformat()
            } for row in rows]
    
    async def cleanup_old_states(self, days_to_keep: int = 7) -> int:
        """Clean up old processing states"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT cleanup_old_processing_state($1)", days_to_keep)
            logger.info(f"Cleaned up {result} old processing states")
            return result

# Singleton instance
db_state_manager = DBStateManager()