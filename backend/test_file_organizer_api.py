"""
Test File Organizer API integration
"""

import asyncio
import aiohttp

async def test_file_organizer():
    """Test if File Organizer API is accessible"""
    async with aiohttp.ClientSession() as session:
        try:
            # Test health endpoint
            async with session.get('http://localhost:2499') as response:
                if response.status == 200:
                    print("✅ File Organizer API is running")
                else:
                    print(f"❌ File Organizer API returned status: {response.status}")
                    
            # Test a simple PDF compression
            with open('/Users/josephkerolos/smart-document-processor/sourcefiles/941-X 2nd Qtr copy.pdf', 'rb') as f:
                pdf_content = f.read()
                
            form = aiohttp.FormData()
            form.add_field('pdf', pdf_content, filename='test.pdf', content_type='application/pdf')
            form.add_field('target_size_mb', '2.0')
            
            async with session.post('http://localhost:2499/api/compress-pdf', data=form) as response:
                if response.status == 200:
                    compressed = await response.read()
                    print(f"✅ PDF compression works! Original: {len(pdf_content)} bytes, Compressed: {len(compressed)} bytes")
                else:
                    error = await response.text()
                    print(f"❌ PDF compression failed: {error}")
                    
        except Exception as e:
            print(f"❌ Error connecting to File Organizer API: {e}")

if __name__ == "__main__":
    asyncio.run(test_file_organizer())