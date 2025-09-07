# Technology Decisions for Enhanced Document Processor v11

## Key Enhancements from v10

### 1. Number Formatting
- **Decision**: Enforce comma-free number formatting throughout
- **Implementation**: Added explicit rules in prompts and post-processing validation
- **Rationale**: Ensures consistent data format for downstream processing

### 2. Schema Reference in Prompts
- **Decision**: Include concise schema reference for each page
- **Implementation**: `get_schema_reference()` provides page-specific schema structure
- **Rationale**: Guides AI to follow exact structure without overwhelming context

### 3. Clear Multi-Page Context
- **Decision**: Explicitly state that each page is part of a larger compilation
- **Implementation**: Added clear messaging in extraction prompts
- **Rationale**: Prevents AI from trying to extract complete form from single page

### 4. Gemini 2.5 Pro for Compilation
- **Decision**: Use Pro model for intelligent compilation and restructuring
- **Implementation**: `compile_results_with_pro_v2()` with full schema reference
- **Rationale**: Pro model better handles complex JSON merging and restructuring

### 5. Huge Token Limits
- **Decision**: Dramatically increase token limits
- **Implementation**: 
  - Extraction: 60,000 tokens
  - Compilation: 100,000 tokens
  - Schema reference: up to 100,000 chars
- **Rationale**: Prevents truncation and ensures complete data extraction

### 6. Fixed Retry Delays
- **Decision**: Use consistent 1-second retry delay
- **Implementation**: Changed from exponential backoff to fixed 1s delay
- **Rationale**: Faster recovery without overwhelming API

### 7. Intelligent JSON Restructuring
- **Decision**: Add Pro-powered JSON recovery and restructuring
- **Implementation**: `_recover_compilation_with_pro()` for malformed JSON
- **Rationale**: Handles edge cases where initial compilation produces invalid JSON

## Technical Stack (Unchanged)

- **Backend**: FastAPI with async/await
- **AI**: Google Gemini API (2.5 Flash for extraction, 2.5 Pro for compilation)
- **PDF Processing**: PyPDF2
- **WebSocket**: Native FastAPI WebSocket support
- **Concurrency**: ThreadPoolExecutor for parallel extraction

## Performance Optimizations

1. **Parallel Page Extraction**: All pages processed simultaneously
2. **Streaming Updates**: Real-time progress via WebSocket
3. **Smart Caching**: Debug mode saves intermediate results
4. **Fallback Strategies**: Multiple JSON parsing attempts

## Data Quality Measures

1. **Number Formatting Validation**: Post-processing to ensure no commas
2. **Schema Compliance**: Reference schema enforces structure
3. **Mathematical Validation**: ERC rate checks for 941-X forms
4. **Expected Null Handling**: Don't penalize legitimately empty fields

## Error Handling

1. **Graceful Degradation**: Falls back to simple merge if Pro compilation fails
2. **JSON Recovery**: Multiple strategies to parse malformed responses
3. **Timeout Protection**: 5-minute timeout for large compilations
4. **Detailed Logging**: Comprehensive extraction summaries

## Future Considerations

1. **Caching**: Could cache schema references to reduce prompt size
2. **Batch Processing**: Could process multiple documents in single API call
3. **Custom Models**: Could fine-tune models for specific form types
4. **Progressive Enhancement**: Could extract incrementally for very large documents