# File Organizer API Retry Logic Implementation

## Changes Made

### 1. **Increased Timeout Settings**
- Added a 10-minute timeout for API requests (was using default ~2 minutes)
- Configuration: `ClientTimeout(total=600, connect=30, sock_read=600)`

### 2. **Retry Logic with Exponential Backoff**
- Maximum 5 retry attempts for failed requests
- Exponential backoff delays: 2s, 4s, 8s, 16s, 32s
- Total time with retries: up to ~62 seconds + request time

### 3. **Enhanced Error Handling**
- Detailed logging for each retry attempt
- Request timing information
- File size logging for debugging

### 4. **Applied to All API Methods**
- `clean_and_analyze_pdf()` - Main processing endpoint
- `split_pdf_with_structure()` - PDF splitting
- `compress_pdf()` - PDF compression (returns original on failure)

## Benefits

1. **No more timeouts**: 3rd quarter file that failed after 7 minutes will now have up to 10 minutes
2. **Automatic recovery**: Temporary network issues or API overload will be retried
3. **Better debugging**: Detailed logs show exactly what's happening
4. **Graceful degradation**: Compression failures return original file instead of crashing

## Testing

Run the test script to verify:
```bash
cd /Users/josephkerolos/smart-document-processor/backend
python3 test_retry_logic.py
```

## Configuration

To adjust settings, modify in `file_organizer_integration_v2.py`:
- `self.timeout` - Change timeout duration
- `self.max_retries` - Change number of retry attempts
- `self.retry_delay` - Change initial retry delay