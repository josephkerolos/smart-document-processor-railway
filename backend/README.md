# Smart Document Processor - Backend API

A powerful document processing API with Google Drive integration, batch processing, and intelligent extraction capabilities.

## Features

- **Intelligent Document Processing**: Clean, split, extract, and organize documents automatically
- **Google Drive Integration**: Configurable upload paths, custom naming, and folder structures
- **Batch Processing**: Process multiple files in parallel with combined outputs
- **Real-time Status Tracking**: WebSocket updates and comprehensive status API
- **Advanced Querying**: Search processed documents by company, form type, date, and status
- **Flexible Configuration**: Customize folder structures, naming templates, and processing options

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd smart-document-processor/backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Server

```bash
# Start the backend server
python3 start.py
# Server runs on http://localhost:4830
```

### Basic Usage

```python
import requests

# Process a single document
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'selected_schema': '941-X'}
    response = requests.post('http://localhost:4830/api/process-enhanced', 
                           files=files, data=data)
    
session_id = response.json()['session_id']
print(f"Processing started: {session_id}")
```

## Google Drive Integration

### Configuration API

Configure how documents are organized in Google Drive:

```python
# Set custom folder structure
config = {
    "output_folder_path": "TaxDocuments/2024",
    "company_folder_template": "{company}_Docs",
    "create_year_folders": True,
    "create_quarter_folders": True
}

response = requests.post('http://localhost:4830/api/gdrive/config', json=config)
```

### Available Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `output_folder_path` | Base folder for processed documents | `"ProcessedDocuments"` |
| `company_folder_template` | Template for company folders | `"{company}"` |
| `form_folder_template` | Template for form type folders | `"{form_type}"` |
| `date_folder_template` | Template for date folders | `"{date}"` |
| `batch_folder_template` | Template for batch folders | `"Batch_{date}_{time}_{quarters}"` |
| `create_year_folders` | Create year subfolders | `true` |
| `create_quarter_folders` | Create quarter subfolders | `true` |
| `group_by_company` | Group by company name | `true` |
| `group_by_form_type` | Group by form type | `true` |
| `track_processing_status` | Enable status tracking | `true` |

### Template Variables

- `{company}` - Company name from extracted data
- `{form_type}` - Form type (e.g., 941X)
- `{date}` - Processing date (YYYY-MM-DD)
- `{time}` - Processing time (HH-MM-SS)
- `{year}` - Document year
- `{quarter}` - Document quarter
- `{batch_id}` - Batch ID (first 8 chars)

## API Endpoints

### Document Processing

#### `POST /api/process-enhanced`
Process a single document or part of a batch.

**Form Data:**
- `file`: PDF file to process
- `selected_schema`: Form type (e.g., "941-X", "941", "1040")
- `batch_id`: (Optional) Batch ID for batch processing
- `expected_value`: (Optional) Expected value for validation
- `target_size_mb`: (Optional) Target compression size

### Batch Processing

#### `POST /api/init-batch`
Initialize batch processing for multiple files.

```json
{
  "file_count": 4
}
```

#### `POST /api/finalize-batch`
Finalize batch and create combined outputs.

```json
{
  "batch_id": "uuid-here"
}
```

### Google Drive Management

#### `GET /api/gdrive/config`
Get current configuration.

#### `POST /api/gdrive/config`
Update configuration.

#### `GET /api/gdrive/status/{session_id}`
Get processing status with Google Drive info.

#### `GET /api/gdrive/query`
Query processed documents.

**Query Parameters:**
- `company_name` - Filter by company
- `form_type` - Filter by form type
- `date_from` - Start date
- `date_to` - End date
- `status` - Filter by status
- `limit` - Max results (1-200)

#### `POST /api/gdrive/upload-to-path`
Upload file to specific Google Drive path.

```json
{
  "file_path": "/local/path/file.pdf",
  "gdrive_path": "TaxDocs/2024/Q1",
  "custom_name": "Q1_Report.pdf"
}
```

## Integration Examples

### Python Client

```python
from examples.quick_integration import SmartDocumentClient

# Initialize client
client = SmartDocumentClient("http://localhost:4830")

# Configure Google Drive
client.configure_google_drive({
    "output_folder_path": "MyCompany/TaxDocs",
    "create_quarter_folders": True
})

# Process single document
result = client.process_single_document("document.pdf", "941-X")
status = client.wait_for_completion(result['session_id'])

# Process batch
batch_result = client.process_batch_documents(
    ["q1.pdf", "q2.pdf", "q3.pdf", "q4.pdf"],
    "941-X"
)

# Query documents
docs = client.query_documents(
    company_name="ABC Corp",
    date_from="2024-01-01"
)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Process document
const form = new FormData();
form.append('file', fs.createReadStream('document.pdf'));
form.append('selected_schema', '941-X');

const response = await axios.post(
    'http://localhost:4830/api/process-enhanced',
    form,
    { headers: form.getHeaders() }
);

// Check status
const status = await axios.get(
    `http://localhost:4830/api/gdrive/status/${response.data.session_id}`
);
```

### cURL

```bash
# Process document
curl -X POST http://localhost:4830/api/process-enhanced \
  -F "file=@document.pdf" \
  -F "selected_schema=941-X"

# Configure Google Drive
curl -X POST http://localhost:4830/api/gdrive/config \
  -H "Content-Type: application/json" \
  -d '{"output_folder_path": "TaxDocs/2024"}'

# Query documents
curl "http://localhost:4830/api/gdrive/query?company_name=ABC%20Corp"
```

## WebSocket Real-time Updates

Connect to WebSocket for real-time processing updates:

```javascript
const ws = new WebSocket(`ws://localhost:4830/ws/enhanced/${sessionId}`);

ws.on('message', (data) => {
    const update = JSON.parse(data);
    console.log(`${update.data.step_id}: ${update.data.status}`);
});
```

## Processing Workflow

1. **Upload** - File validation and preparation
2. **Clean** - Remove instruction pages using AI
3. **Split** - Separate multi-document PDFs
4. **Extract** - Extract data using Gemini AI
5. **Organize** - Structure files by metadata
6. **Compress** - Optimize file sizes
7. **Archive** - Create downloadable packages
8. **Google Drive** - Upload to configured location

## Status Tracking

Track processing status with detailed information:

```json
{
  "status": "completed",
  "progress": 100,
  "current_step": "complete",
  "steps_completed": ["upload", "clean", "split", "extract", "organize", "compress", "archive", "google_drive"],
  "gdrive_folder_path": "ProcessedDocuments/ABC Corp/941X/2024-01-16",
  "files_uploaded": ["all_extractions.json", "Master_Archive.zip"]
}
```

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

Error responses include details:
```json
{
  "detail": "Error description"
}
```

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_DRIVE_API_KEY=your_gdrive_api_key

# Optional
DEBUG_MODE=false
LOG_LEVEL=INFO
```

## Architecture

- **FastAPI** - High-performance async API framework
- **Google Gemini** - AI-powered extraction and cleaning
- **PyPDF2** - PDF manipulation
- **WebSockets** - Real-time updates
- **Async Processing** - Parallel document handling

## Performance

- Process multiple documents in parallel
- Batch processing for improved efficiency
- Configurable compression targets
- Async I/O for optimal throughput

## Security

- Sanitized file paths
- API key authentication
- Input validation
- Rate limiting ready

## Support

For detailed API documentation, see [GOOGLE_DRIVE_API_GUIDE.md](GOOGLE_DRIVE_API_GUIDE.md)

For integration examples, see the [examples](examples/) directory.

## License

[Your License Here]