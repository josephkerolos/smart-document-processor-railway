#!/usr/bin/env python3
"""
Test script to verify retry logic works correctly
"""

import asyncio
import logging
from file_organizer_integration_v2 import FileOrganizerIntegrationV2

# Set up logging to see retry attempts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_retry_logic():
    """Test the retry logic with a sample PDF"""
    
    # Create test PDF content (small file for testing)
    test_pdf_content = b"%PDF-1.4\n%Test PDF content for retry logic testing"
    
    async with FileOrganizerIntegrationV2() as file_organizer:
        print("\n=== Testing File Organizer Integration with Retry Logic ===")
        print(f"Timeout: {file_organizer.timeout.total}s")
        print(f"Max retries: {file_organizer.max_retries}")
        print(f"Initial retry delay: {file_organizer.retry_delay}s")
        print(f"Retry delays: {[file_organizer.retry_delay * (2**i) for i in range(file_organizer.max_retries)]}")
        
        try:
            # Test clean_and_analyze_pdf
            print("\n1. Testing clean_and_analyze_pdf...")
            result = await file_organizer.clean_and_analyze_pdf(
                test_pdf_content, 
                "test.pdf",
                compress=False
            )
            print(f"   ✓ Success! Result keys: {list(result.keys())}")
            
        except Exception as e:
            print(f"   ✗ Failed after all retries: {str(e)}")
            
        print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(test_retry_logic())