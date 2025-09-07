#!/usr/bin/env python3
"""
Test script to verify single vs multi-file processing behavior
"""

import asyncio
import aiohttp
import os
import json
from pathlib import Path

async def test_single_file():
    """Test single file processing - should upload to Google Drive individually"""
    print("\n=== Testing Single File Processing ===")
    
    # Find a test PDF file
    test_file = None
    for file in Path("sourcefiles").glob("*.pdf"):
        test_file = file
        break
    
    if not test_file:
        print("No test PDF file found in sourcefiles/")
        return
    
    print(f"Using test file: {test_file}")
    
    async with aiohttp.ClientSession() as session:
        # Process single file
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=test_file.name, content_type='application/pdf')
            data.add_field('selected_schema', '941-X')
            data.add_field('target_size_mb', '9.0')
            
            async with session.post('http://localhost:4830/api/process-enhanced', data=data) as resp:
                result = await resp.json()
                print(f"Response status: {resp.status}")
                print(f"Session ID: {result.get('session_id')}")
                print(f"Success: {result.get('success')}")
                print(f"Google Drive Upload: {result.get('google_drive_upload', {}).get('success', 'Not included')}")
                
                if result.get('google_drive_upload', {}).get('success'):
                    print(f"Google Drive Link: {result['google_drive_upload'].get('folder_link')}")
                
                return result

async def test_multiple_files():
    """Test multiple file processing - should use batch and upload once"""
    print("\n\n=== Testing Multiple File Processing ===")
    
    # Find test PDF files
    test_files = list(Path("sourcefiles").glob("*.pdf"))[:2]  # Get first 2 files
    
    if len(test_files) < 2:
        print("Need at least 2 PDF files in sourcefiles/ for multi-file test")
        return
    
    print(f"Using test files: {[f.name for f in test_files]}")
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Initialize batch
        batch_data = {"file_count": len(test_files)}
        async with session.post('http://localhost:4830/api/init-batch', json=batch_data) as resp:
            batch_result = await resp.json()
            batch_id = batch_result['batch_id']
            print(f"Initialized batch: {batch_id}")
        
        # Step 2: Process files with batch ID
        session_ids = []
        for i, test_file in enumerate(test_files):
            print(f"\nProcessing file {i+1}/{len(test_files)}: {test_file.name}")
            
            with open(test_file, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=test_file.name, content_type='application/pdf')
                data.add_field('selected_schema', '941-X')
                data.add_field('target_size_mb', '9.0')
                data.add_field('batch_id', batch_id)
                
                async with session.post('http://localhost:4830/api/process-enhanced', data=data) as resp:
                    result = await resp.json()
                    session_ids.append(result.get('session_id'))
                    print(f"  Session ID: {result.get('session_id')}")
                    print(f"  Success: {result.get('success')}")
                    print(f"  Google Drive Upload: {result.get('google_drive_upload', {}).get('success', 'Not included (expected for batch)')}")
        
        # Step 3: Finalize batch
        print(f"\nFinalizing batch {batch_id}...")
        await asyncio.sleep(1)  # Give time for processing to complete
        
        finalize_data = {"batch_id": batch_id}
        async with session.post('http://localhost:4830/api/finalize-batch', json=finalize_data) as resp:
            finalize_result = await resp.json()
            print(f"Batch finalization status: {resp.status}")
            print(f"Success: {finalize_result.get('success')}")
            print(f"Total files: {finalize_result.get('total_files')}")
            print(f"Total documents: {finalize_result.get('total_documents')}")
            print(f"Google Drive Upload: {finalize_result.get('google_drive_upload', {}).get('success')}")
            
            if finalize_result.get('google_drive_upload', {}).get('success'):
                print(f"Google Drive Link: {finalize_result['google_drive_upload'].get('folder_link')}")
            
            if finalize_result.get('all_extractions_path'):
                print(f"All extractions JSON: {finalize_result['all_extractions_path']}")
            
            return finalize_result

async def main():
    """Run all tests"""
    print("Starting batch processing tests...")
    print("Make sure the backend server is running on port 4830")
    
    try:
        # Test single file processing
        single_result = await test_single_file()
        
        # Test multiple file processing
        multi_result = await test_multiple_files()
        
        # Summary
        print("\n\n=== Test Summary ===")
        print("Single file processing:")
        print(f"  - Google Drive upload: {'✓' if single_result and single_result.get('google_drive_upload', {}).get('success') else '✗'}")
        
        print("\nMultiple file processing:")
        if multi_result:
            print(f"  - Batch processing: {'✓' if multi_result.get('success') else '✗'}")
            print(f"  - Combined Google Drive upload: {'✓' if multi_result.get('google_drive_upload', {}).get('success') else '✗'}")
            print(f"  - All extractions JSON created: {'✓' if multi_result.get('all_extractions_path') else '✗'}")
        else:
            print("  - Failed to complete multi-file test")
            
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())