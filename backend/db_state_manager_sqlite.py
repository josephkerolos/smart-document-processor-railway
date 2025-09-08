"""
SQLite State Manager for Smart Document Processor
Handles persistent storage using SQLite (no server required)
"""

import os
import json
import aiosqlite
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SQLiteStateManager:
    def __init__(self):
        self.db_path = os.getenv('SQLITE_DB_PATH', '/app/data/state.db')
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.pool = None
    
    async def initialize(self):
        """Initialize SQLite database and create tables"""
        try:
            # SQLite doesn't need a pool, but we'll maintain compatibility
            self.pool = True  # Flag to indicate initialization
            
            # Create tables if they don't exist
            await self.create_tables()
            logger.info(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    async def create_tables(self):
        """Create state tables if they don't exist"""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable foreign keys for SQLite
            await db.execute("PRAGMA foreign_keys = ON")
            
            # Create batch_processing_state table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS batch_processing_state (
                    batch_id TEXT PRIMARY KEY,
                    session_ids TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_batch_processing_status ON batch_processing_state(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_batch_processing_created ON batch_processing_state(created_at)")
            
            # Create processing_status_db table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS processing_status_db (
                    session_id TEXT PRIMARY KEY,
                    batch_id TEXT,
                    status TEXT DEFAULT 'processing',
                    processed_files TEXT DEFAULT '[]',
                    extractions TEXT DEFAULT '{}',
                    total_files INTEGER DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (batch_id) REFERENCES batch_processing_state(batch_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_processing_status_batch ON processing_status_db(batch_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_processing_status_status ON processing_status_db(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_processing_status_created ON processing_status_db(created_at)")
            
            await db.commit()
            logger.info("SQLite state tables created/verified")
    
    async def close(self):
        """Close database connection (compatibility method)"""
        self.pool = None
    
    # Batch Processing State Management
    
    async def save_batch_state(self, batch_id: str, session_ids: List[str], status: str = 'active', metadata: Dict = None):
        """Save or update batch processing state"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            
            # Check if batch exists
            cursor = await db.execute("SELECT batch_id FROM batch_processing_state WHERE batch_id = ?", (batch_id,))
            exists = await cursor.fetchone()
            
            if exists:
                # Update existing
                await db.execute("""
                    UPDATE batch_processing_state 
                    SET session_ids = ?, status = ?, metadata = ?, updated_at = ?
                    WHERE batch_id = ?
                """, (json.dumps(session_ids), status, json.dumps(metadata or {}), 
                     datetime.now().isoformat(), batch_id))
            else:
                # Insert new
                await db.execute("""
                    INSERT INTO batch_processing_state (batch_id, session_ids, status, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (batch_id, json.dumps(session_ids), status, json.dumps(metadata or {}),
                     datetime.now().isoformat(), datetime.now().isoformat()))
            
            await db.commit()
            logger.info(f"Saved batch state for {batch_id}: {len(session_ids)} sessions, status={status}")
    
    async def get_batch_state(self, batch_id: str) -> Optional[Dict]:
        """Get batch processing state"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT batch_id, session_ids, status, metadata, created_at, updated_at
                FROM batch_processing_state
                WHERE batch_id = ?
            """, (batch_id,))
            
            row = await cursor.fetchone()
            if row:
                return {
                    'batch_id': row[0],
                    'session_ids': json.loads(row[1]),
                    'status': row[2],
                    'metadata': json.loads(row[3]),
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            return None
    
    async def get_active_batches(self) -> List[Dict]:
        """Get all active batch states"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT batch_id, session_ids, status, metadata, created_at, updated_at
                FROM batch_processing_state
                WHERE status = 'active'
                ORDER BY created_at DESC
            """)
            
            rows = await cursor.fetchall()
            return [{
                'batch_id': row[0],
                'session_ids': json.loads(row[1]),
                'status': row[2],
                'metadata': json.loads(row[3]),
                'created_at': row[4],
                'updated_at': row[5]
            } for row in rows]
    
    async def update_batch_status(self, batch_id: str, status: str):
        """Update batch status"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE batch_processing_state
                SET status = ?, updated_at = ?
                WHERE batch_id = ?
            """, (status, datetime.now().isoformat(), batch_id))
            await db.commit()
    
    async def delete_batch_state(self, batch_id: str):
        """Delete batch state (cascades to processing status)"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("DELETE FROM batch_processing_state WHERE batch_id = ?", (batch_id,))
            await db.commit()
    
    # Processing Status Management
    
    async def save_processing_status(self, session_id: str, batch_id: str, status: str, 
                                   processed_files: List[str] = None, extractions: Dict = None,
                                   total_files: int = 0, metadata: Dict = None):
        """Save or update processing session status"""
        if self.pool is None:
            logger.warning(f"Database not initialized, skipping status save for session {session_id}")
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            
            # Check if session exists
            cursor = await db.execute("SELECT session_id FROM processing_status_db WHERE session_id = ?", (session_id,))
            exists = await cursor.fetchone()
            
            if exists:
                # Update existing
                await db.execute("""
                    UPDATE processing_status_db
                    SET status = ?, processed_files = ?, extractions = ?, 
                        total_files = ?, processed_count = ?, metadata = ?,
                        updated_at = ?
                    WHERE session_id = ?
                """, (status, json.dumps(processed_files or []), json.dumps(extractions or {}),
                     total_files, len(processed_files) if processed_files else 0,
                     json.dumps(metadata or {}), datetime.now().isoformat(), session_id))
            else:
                # Insert new
                await db.execute("""
                    INSERT INTO processing_status_db 
                    (session_id, batch_id, status, processed_files, extractions, total_files, 
                     processed_count, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, batch_id, status, json.dumps(processed_files or []), 
                     json.dumps(extractions or {}), total_files,
                     len(processed_files) if processed_files else 0,
                     json.dumps(metadata or {}), datetime.now().isoformat(), datetime.now().isoformat()))
            
            await db.commit()
    
    async def get_processing_status(self, session_id: str) -> Optional[Dict]:
        """Get processing session status"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT session_id, batch_id, status, processed_files, extractions, 
                       total_files, processed_count, created_at, updated_at, metadata
                FROM processing_status_db
                WHERE session_id = ?
            """, (session_id,))
            
            row = await cursor.fetchone()
            if row:
                return {
                    'session_id': row[0],
                    'batch_id': row[1],
                    'status': row[2],
                    'processed_files': json.loads(row[3]),
                    'extractions': json.loads(row[4]),
                    'total_files': row[5],
                    'processed_count': row[6],
                    'created_at': row[7],
                    'updated_at': row[8],
                    'metadata': json.loads(row[9])
                }
            return None
    
    async def get_batch_processing_status(self, batch_id: str) -> List[Dict]:
        """Get all processing sessions for a batch"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT session_id, batch_id, status, processed_files, extractions, 
                       total_files, processed_count, created_at, updated_at, metadata
                FROM processing_status_db
                WHERE batch_id = ?
                ORDER BY created_at DESC
            """, (batch_id,))
            
            rows = await cursor.fetchall()
            return [{
                'session_id': row[0],
                'batch_id': row[1],
                'status': row[2],
                'processed_files': json.loads(row[3]),
                'extractions': json.loads(row[4]),
                'total_files': row[5],
                'processed_count': row[6],
                'created_at': row[7],
                'updated_at': row[8],
                'metadata': json.loads(row[9])
            } for row in rows]
    
    async def update_processing_status(self, session_id: str, status: str):
        """Update processing session status"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE processing_status_db
                SET status = ?, updated_at = ?
                WHERE session_id = ?
            """, (status, datetime.now().isoformat(), session_id))
            await db.commit()
    
    async def delete_processing_status(self, session_id: str):
        """Delete processing session status"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM processing_status_db WHERE session_id = ?", (session_id,))
            await db.commit()

# Create singleton instance
db_state_manager_sqlite = SQLiteStateManager()