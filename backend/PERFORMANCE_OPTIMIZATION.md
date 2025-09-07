# Performance Optimization Guide

## Current Performance Issues

### Timing Breakdown (20-page document, 4 quarters):
- **Cleaning & Splitting**: 279.2s (4.6 min) - File-organizer API
- **Extraction**: 213.4s (3.6 min) - 24 Gemini API calls
- **Organization**: 0.0s - Fast
- **Compression**: 12.6s - Often fails
- **Archiving**: 1.6s - Fast
- **Total**: ~507s (8.5 minutes)

## Root Causes

1. **Too Many AI API Calls**: 24 Gemini calls (5 per document + compilation)
2. **File-Organizer Bottleneck**: Sequential page processing
3. **Network Overhead**: Separate services communicating over HTTP
4. **No Caching**: Reprocessing identical pages

## Optimization Strategies

### 1. Reduce Gemini API Calls (Save 70% time)
```python
# Current: 5 calls per document
for page in pages:
    extract_page()  # 5 separate calls
compile_results()   # 1 call

# Optimized: 1 call per document
extract_all_pages_at_once()  # 1 batched call
```

### 2. Implement Caching (Save 90% on repeated docs)
```python
# Add Redis or in-memory cache
cache_key = hashlib.md5(pdf_content).hexdigest()
if cache_key in cache:
    return cache[cache_key]
```

### 3. Bypass File-Organizer for Simple Cases
```python
# Quick check if document needs cleaning
if is_already_clean(pdf):
    skip_file_organizer()
    process_directly()
```

### 4. Batch Page Analysis
```python
# Send multiple pages in one API call
pages_batch = [page1, page2, page3, page4, page5]
results = gemini.analyze_batch(pages_batch)
```

### 5. Local Processing First
```python
# Do what we can locally before API calls
- Page counting
- Size checking  
- Basic structure detection
- Quarter identification from text
```

### 6. Parallel File-Organizer Operations
```python
# Run cleaning and compression in parallel
async with asyncio.TaskGroup() as tg:
    clean_task = tg.create_task(clean_pdf())
    compress_task = tg.create_task(prepare_compression())
```

### 7. Smart Compression Strategy
```python
# Only compress if needed
if total_size < target_size:
    skip_compression()
    
# Pre-calculate compression feasibility
if file_already_compressed():
    skip_compression()
```

### 8. Streaming Pipeline
```python
# Start next step before previous completes
async for cleaned_page in clean_pages():
    start_extraction(cleaned_page)  # Don't wait for all pages
```

### 9. Use Faster Models
```python
# Use Gemini Flash for simple extractions
if is_simple_form():
    use_gemini_flash()  # 10x faster
else:
    use_gemini_pro()
```

### 10. Infrastructure Improvements
- Run file-organizer on same machine (eliminate network latency)
- Use Unix sockets instead of HTTP
- Implement connection pooling
- Add request queuing

## Quick Wins (Implement First)

1. **Batch Gemini Calls**: Reduce from 24 to 4-6 calls
2. **Skip Compression When Unnecessary**: Check size first
3. **Cache Gemini Responses**: For identical pages
4. **Use Gemini Flash**: For standard forms
5. **Local File-Organizer**: Eliminate network overhead

## Expected Performance After Optimization

- **Cleaning & Splitting**: 30s (was 280s)
- **Extraction**: 40s (was 213s) 
- **Compression**: 5s (was 12s)
- **Total**: ~80s (was 507s) - **6x faster!**

## Implementation Priority

1. Batch Gemini API calls (High impact, Medium effort)
2. Implement caching (High impact, Low effort)
3. Skip unnecessary compression (Medium impact, Low effort)
4. Use Gemini Flash (High impact, Low effort)
5. Optimize file-organizer (High impact, High effort)