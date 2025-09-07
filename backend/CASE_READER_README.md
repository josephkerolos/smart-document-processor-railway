# Case Reader Documentation

## Overview

The Case Reader utility provides a comprehensive way to read and access case/session data from the Smart Document Processor system. It handles both in-memory and file-based case storage.

## Architecture

The system stores case data in two places:
1. **In-memory database**: `processing_status_db` dictionary in `google_drive_routes.py`
2. **File storage**: JSON files in the `processing_status/` directory

## Files Created

### 1. `case_reader.py`
The main utility class for reading cases.

**Key Features:**
- Load cases from both memory and files
- Filter cases by various criteria
- Get case summaries and statistics
- Both async and sync interfaces

**Main Classes:**
- `CaseReader`: Core class for case management

**Key Methods:**
- `load_all_cases()`: Load cases from all sources
- `filter_cases()`: Filter by company, form type, status, date range
- `get_case_summary()`: Get statistical summary
- `get_case()`: Get specific case by ID

### 2. `case_routes.py`
FastAPI router providing REST API endpoints for case access.

**Endpoints:**
- `GET /api/cases/` - List all cases with pagination
- `GET /api/cases/summary` - Get summary statistics
- `GET /api/cases/search` - Search with filters
- `GET /api/cases/{session_id}` - Get specific case
- `GET /api/cases/{session_id}/files` - Get case files
- `POST /api/cases/reload` - Reload cases from storage

### 3. `test_case_reader.py`
Test script demonstrating usage of the case reader.

## Usage Examples

### Basic Usage

```python
from case_reader import CaseReader

# Create reader instance
reader = CaseReader()

# Load all cases
cases = await reader.load_all_cases()

# Get case summary
summary = reader.get_case_summary()
print(f"Total cases: {summary['total_cases']}")
print(f"Status breakdown: {summary['status_breakdown']}")
```

### Filtering Cases

```python
# Filter by status
completed = reader.filter_cases(status="completed")

# Filter by company
acme_cases = reader.filter_cases(company_name="Acme Corp")

# Filter by date range
from datetime import datetime, timedelta
last_week = datetime.now() - timedelta(days=7)
recent = reader.filter_cases(date_from=last_week)

# Multiple filters
filtered = reader.filter_cases(
    company_name="Acme Corp",
    form_type="941-X",
    status="completed"
)
```

### Convenience Functions

```python
# Quick async read
cases = await read_all_cases()

# Synchronous read (files only)
cases = read_all_cases_sync()

# Get specific case
case = await get_case_by_id("session-123")
```

### API Integration

Add to your FastAPI application:

```python
from fastapi import FastAPI
from case_routes import router as case_router

app = FastAPI()
app.include_router(case_router)
```

### API Usage Examples

```bash
# Get all cases
curl http://localhost:8000/api/cases/

# Get summary
curl http://localhost:8000/api/cases/summary

# Search cases
curl "http://localhost:8000/api/cases/search?status=completed&company_name=Acme"

# Get specific case
curl http://localhost:8000/api/cases/abc123-def456

# Get case files
curl http://localhost:8000/api/cases/abc123-def456/files
```

## Case Data Structure

Each case contains:

```json
{
  "session_id": "unique-id",
  "created_at": "2024-01-17T10:30:00",
  "updated_at": "2024-01-17T10:45:00",
  "status": "completed",
  "progress": 100,
  "steps_completed": ["upload", "extract", "compile"],
  "current_step": null,
  "error": null,
  "gdrive_folder_id": "drive-folder-id",
  "gdrive_folder_path": "/ProcessedDocuments/AcmeCorp/2024/Q1",
  "files_uploaded": [
    {
      "file_id": "drive-file-id",
      "file_name": "941-X_compiled.json",
      "mime_type": "application/json"
    }
  ],
  "metadata": {
    "company_name": "Acme Corp",
    "form_type": "941-X",
    "quarter": "Q1",
    "year": "2024"
  }
}
```

## Testing

Run the test script to verify functionality:

```bash
python test_case_reader.py
```

This will:
1. Load all cases
2. Display summary statistics
3. Test filtering capabilities
4. Demonstrate API usage

## Notes

- Cases are loaded from both in-memory storage and files
- The in-memory database takes precedence for duplicate session IDs
- File storage is optional and controlled by configuration
- The system is designed to handle large numbers of cases efficiently
- All date/time values use ISO-8601 format