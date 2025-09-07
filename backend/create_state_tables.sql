-- Create tables for persisting smart document processor state
-- This prevents state loss on restart and fixes duplicate case creation

-- Table to store active batch processing state
CREATE TABLE IF NOT EXISTS batch_processing_state (
    batch_id VARCHAR(255) PRIMARY KEY,
    session_ids JSONB,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for status queries
CREATE INDEX IF NOT EXISTS idx_batch_processing_status ON batch_processing_state(status);
CREATE INDEX IF NOT EXISTS idx_batch_processing_created ON batch_processing_state(created_at);

-- Table to store processing session status
CREATE TABLE IF NOT EXISTS processing_status_db (
    session_id VARCHAR(255) PRIMARY KEY,
    batch_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'processing',
    processed_files JSONB DEFAULT '[]'::jsonb,
    extractions JSONB DEFAULT '{}'::jsonb,
    total_files INTEGER DEFAULT 0,
    processed_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_processing_status_batch ON processing_status_db(batch_id);
CREATE INDEX IF NOT EXISTS idx_processing_status_status ON processing_status_db(status);
CREATE INDEX IF NOT EXISTS idx_processing_status_created ON processing_status_db(created_at);

-- Add foreign key constraint if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_processing_batch' 
        AND table_name = 'processing_status_db'
    ) THEN
        ALTER TABLE processing_status_db 
        ADD CONSTRAINT fk_processing_batch 
        FOREIGN KEY (batch_id) 
        REFERENCES batch_processing_state(batch_id) 
        ON DELETE CASCADE;
    END IF;
END $$;

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers if they don't exist
DROP TRIGGER IF EXISTS update_batch_processing_state_updated_at ON batch_processing_state;
CREATE TRIGGER update_batch_processing_state_updated_at 
BEFORE UPDATE ON batch_processing_state 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_processing_status_db_updated_at ON processing_status_db;
CREATE TRIGGER update_processing_status_db_updated_at 
BEFORE UPDATE ON processing_status_db 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add cleanup function for old records
CREATE OR REPLACE FUNCTION cleanup_old_processing_state(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM batch_processing_state 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep
    AND status IN ('completed', 'failed', 'cancelled');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;