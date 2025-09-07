# Google Drive API Integration Guide

## Overview

The Smart Document Processor includes a comprehensive Google Drive integration API that allows you to:

- **Configure custom folder structures and naming templates**
- **Upload/download files to/from specific Google Drive paths**
- **Track processing status with real-time updates**
- **Query processed documents with advanced filters**
- **Create custom folder hierarchies programmatically**

## Quick Start

### 1. Basic Configuration

```python
import requests

# Base URL for the API
BASE_URL = "http://localhost:4830/api/gdrive"

# Get current configuration
response = requests.get(f"{BASE_URL}/config")
config = response.json()["config"]

# Update configuration
new_config = {
    "output_folder_path": "MyProcessedDocs",
    "company_folder_template": "{company}_Documents",
    "date_folder_template": "{date}_Processed",
    "create_year_folders": True,
    "create_quarter_folders": True
}

response = requests.post(f"{BASE_URL}/config", json=new_config)
```

### 2. Process Documents with Custom Paths

```python
# Process a document and specify custom output path
files = {'file': open('document.pdf', 'rb')}
data = {
    'selected_schema': '941-X',
    'custom_gdrive_path': 'TaxDocuments/2024/Q1'  # Optional custom path
}

response = requests.post("http://localhost:4830/api/process-enhanced", 
                        files=files, data=data)
```

### 3. Track Processing Status

```python
# Get processing status
session_id = response.json()["session_id"]
status = requests.get(f"{BASE_URL}/status/{session_id}")

# Status includes:
# - Current processing step
# - Progress percentage
# - Google Drive folder location
# - Files uploaded
# - Any errors
```

### 4. Query Processed Documents

```python
# Search for documents
params = {
    "company_name": "ABC Corp",
    "form_type": "941X",
    "date_from": "2024-01-01",
    "status": "completed",
    "limit": 50
}

results = requests.get(f"{BASE_URL}/query", params=params)
```

## API Reference

### Configuration Management

#### GET `/api/gdrive/config`
Get current Google Drive configuration.

**Response:**
```json
{
  "success": true,
  "config": {
    "root_folder_id": null,
    "input_folder_path": "InputDocuments",
    "output_folder_path": "ProcessedDocuments",
    "company_folder_template": "{company}",
    "form_folder_template": "{form_type}",
    "date_folder_template": "{date}",
    "batch_folder_template": "Batch_{date}_{time}_{quarters}",
    "auto_organize": true,
    "create_year_folders": true,
    "create_quarter_folders": true,
    "group_by_company": true,
    "group_by_form_type": true,
    "track_processing_status": true
  }
}
```

#### POST `/api/gdrive/config`
Update Google Drive configuration.

**Request Body:**
```json
{
  "output_folder_path": "CustomOutput",
  "company_folder_template": "{company}_TaxDocs",
  "create_year_folders": false
}
```

### Status Tracking

#### GET `/api/gdrive/status/{session_id}`
Get processing status for a specific session.

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "status": {
    "created_at": "2024-01-16T10:30:00",
    "updated_at": "2024-01-16T10:35:00",
    "status": "completed",
    "progress": 100,
    "steps_completed": ["upload", "clean", "split", "extract", "organize", "compress", "archive", "google_drive"],
    "current_step": "complete",
    "gdrive_folder_id": "1234567890",
    "gdrive_folder_path": "ProcessedDocuments/ABC Corp/941X/2024-01-16",
    "files_uploaded": ["all_extractions.json", "Master_Archive.zip", "Processing_Summary.txt"]
  }
}
```

#### POST `/api/gdrive/status/{session_id}`
Update processing status (used internally by the processor).

### File Operations

#### POST `/api/gdrive/upload-to-path`
Upload a file to a specific Google Drive path.

**Request Body:**
```json
{
  "file_path": "/path/to/local/file.pdf",
  "gdrive_path": "TaxDocuments/2024/Q1/Processed",
  "custom_name": "Q1_941X_Processed.pdf"
}
```

#### POST `/api/gdrive/download-from-path`
Download a file from Google Drive.

**Request Body:**
```json
{
  "file_id": "gdrive_file_id",
  "gdrive_path": "TaxDocuments/2024/Q1/file.pdf",
  "local_path": "downloads/tax_docs"
}
```

### Query and Search

#### GET `/api/gdrive/query`
Search for processed documents with filters.

**Query Parameters:**
- `company_name`: Filter by company name
- `form_type`: Filter by form type (e.g., "941X")
- `date_from`: Start date (ISO format)
- `date_to`: End date (ISO format)
- `status`: Filter by status ("processing", "completed", "failed")
- `limit`: Maximum results (1-200, default: 50)

**Response:**
```json
{
  "success": true,
  "total": 25,
  "results": [
    {
      "session_id": "abc123",
      "status": "completed",
      "created_at": "2024-01-16T10:30:00",
      "company_name": "ABC Corp",
      "form_type": "941X",
      "gdrive_folder_path": "ProcessedDocuments/ABC Corp/941X/2024-01-16",
      "gdrive_folder_id": "1234567890",
      "files_uploaded": 3
    }
  ]
}
```

### Folder Management

#### GET `/api/gdrive/folder-structure`
Get the current folder structure configuration.

**Query Parameters:**
- `parent_id`: Google Drive folder ID to start from
- `max_depth`: Maximum folder depth to return (1-5)

#### POST `/api/gdrive/create-custom-structure`
Create a custom folder hierarchy in Google Drive.

**Request Body:**
```json
{
  "parent_id": "root",
  "structure": {
    "TaxDocuments": {
      "2024": {
        "Q1": {},
        "Q2": {},
        "Q3": {},
        "Q4": {}
      },
      "2025": {
        "Q1": {}
      }
    }
  }
}
```

## Configuration Options

### Folder Templates

Templates support the following variables:
- `{company}`: Company name from extracted data
- `{form_type}`: Form type (e.g., 941X, 941)
- `{date}`: Processing date (YYYY-MM-DD)
- `{time}`: Processing time (HH-MM-SS)
- `{year}`: Year from document
- `{quarter}`: Quarter from document
- `{batch_id}`: Batch processing ID (first 8 chars)

### Example Configurations

#### Standard Tax Document Organization
```json
{
  "output_folder_path": "TaxDocuments",
  "company_folder_template": "{company}",
  "form_folder_template": "{form_type}",
  "date_folder_template": "{year}_Q{quarter}",
  "create_year_folders": false,
  "create_quarter_folders": false
}
```

#### Date-Based Organization
```json
{
  "output_folder_path": "ProcessedByDate",
  "company_folder_template": "{date}_{company}",
  "form_folder_template": "{form_type}",
  "create_year_folders": true,
  "create_quarter_folders": true
}
```

#### Minimal Organization
```json
{
  "output_folder_path": "AllDocuments",
  "group_by_company": false,
  "group_by_form_type": false,
  "date_folder_template": "{date}_{company}_{form_type}"
}
```

## Integration Examples

### Python Integration

```python
from smart_doc_client import SmartDocumentClient

# Initialize client
client = SmartDocumentClient(base_url="http://localhost:4830")

# Configure Google Drive settings
client.configure_gdrive({
    "output_folder_path": "MyCompany/TaxDocs",
    "batch_folder_template": "Batch_{date}_{company}_{quarters}"
})

# Process documents
results = client.process_documents(
    files=["q1.pdf", "q2.pdf", "q3.pdf", "q4.pdf"],
    form_type="941X",
    batch_mode=True
)

# Check status
status = client.get_status(results["session_id"])
print(f"Processing: {status['progress']}% - {status['current_step']}")

# Query results
docs = client.query_documents(
    company_name="MyCompany",
    date_from="2024-01-01"
)
```

### JavaScript/Node.js Integration

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class SmartDocumentClient {
    constructor(baseUrl = 'http://localhost:4830') {
        this.baseUrl = baseUrl;
    }

    async configure(config) {
        const response = await axios.post(`${this.baseUrl}/api/gdrive/config`, config);
        return response.data;
    }

    async processDocument(filePath, formType = '941-X') {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));
        form.append('selected_schema', formType);

        const response = await axios.post(
            `${this.baseUrl}/api/process-enhanced`,
            form,
            { headers: form.getHeaders() }
        );
        return response.data;
    }

    async getStatus(sessionId) {
        const response = await axios.get(`${this.baseUrl}/api/gdrive/status/${sessionId}`);
        return response.data;
    }

    async queryDocuments(filters) {
        const response = await axios.get(`${this.baseUrl}/api/gdrive/query`, { params: filters });
        return response.data;
    }
}

// Usage
const client = new SmartDocumentClient();

// Configure
await client.configure({
    output_folder_path: 'TaxDocuments/2024',
    create_quarter_folders: true
});

// Process
const result = await client.processDocument('./941x.pdf');
console.log('Session ID:', result.session_id);

// Check status
const status = await client.getStatus(result.session_id);
console.log('Status:', status.status.status);
console.log('Google Drive:', status.status.gdrive_folder_path);
```

### cURL Examples

```bash
# Configure Google Drive
curl -X POST http://localhost:4830/api/gdrive/config \
  -H "Content-Type: application/json" \
  -d '{
    "output_folder_path": "MyDocuments",
    "create_year_folders": true
  }'

# Process a document
curl -X POST http://localhost:4830/api/process-enhanced \
  -F "file=@document.pdf" \
  -F "selected_schema=941-X"

# Check status
curl http://localhost:4830/api/gdrive/status/SESSION_ID

# Query documents
curl "http://localhost:4830/api/gdrive/query?company_name=ABC%20Corp&form_type=941X"

# Upload to specific path
curl -X POST http://localhost:4830/api/gdrive/upload-to-path \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/local/path/file.pdf",
    "gdrive_path": "TaxDocs/2024/Q1"
  }'
```

## WebSocket Integration

For real-time status updates during processing:

```javascript
const ws = new WebSocket(`ws://localhost:4830/ws/enhanced/${sessionId}`);

ws.on('message', (data) => {
    const update = JSON.parse(data);
    
    if (update.type === 'workflow_plan') {
        console.log('Workflow steps:', update.data);
    } else if (update.type === 'step_update') {
        console.log(`Step ${update.data.step_id}: ${update.data.status}`);
        
        if (update.data.step_id === 'google_drive' && update.data.status === 'completed') {
            console.log('Uploaded to Google Drive!');
        }
    }
});
```

## Error Handling

The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (session or file not found)
- `500`: Internal Server Error

Error responses include details:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Best Practices

1. **Always check status before downloading**: Ensure processing is complete
2. **Use batch processing for multiple files**: More efficient than individual processing
3. **Configure paths before processing**: Set up your folder structure once
4. **Use query filters effectively**: Combine filters for precise results
5. **Handle WebSocket disconnections**: Implement reconnection logic
6. **Store session IDs**: Keep track of your processing sessions for later retrieval

## Security Considerations

1. **API Key**: Set `GOOGLE_DRIVE_API_KEY` environment variable
2. **File Access**: Only files in `processed_documents` directory are accessible
3. **Path Validation**: All paths are sanitized to prevent directory traversal
4. **Rate Limiting**: Implement rate limiting in production environments

## Troubleshooting

### Common Issues

1. **"Session not found"**: Session may have expired or invalid ID
2. **"Upload failed"**: Check Google Drive API credentials and permissions
3. **"Invalid configuration"**: Ensure all template variables are properly formatted
4. **"Connection refused"**: Verify the server is running on the correct port

### Debug Mode

Enable debug logging:
```python
response = requests.post(f"{BASE_URL}/config", json={
    "debug_mode": True
})
```

Check logs at: `document_processing.log`

## Support

For issues or feature requests, please contact the development team or create an issue in the project repository.