"""
Test script for Google Drive integration
"""

import asyncio
from google_drive_integration import GoogleDriveIntegration

async def test_integration():
    """Test the Google Drive integration"""
    
    async with GoogleDriveIntegration() as gdrive:
        try:
            # Test 1: Search for ProcessedDocuments folder
            print("Test 1: Searching for ProcessedDocuments folder...")
            folder_id = await gdrive.search_folder("ProcessedDocuments")
            if folder_id:
                print(f"✅ Found ProcessedDocuments folder: {folder_id}")
            else:
                print("📁 ProcessedDocuments folder not found, will be created on first upload")
            
            # Test 2: Create a test folder structure
            print("\nTest 2: Creating test folder structure...")
            test_folder_id = await gdrive.create_document_folder_structure(
                "Test_Company_Inc",
                "941X",
                "2025-06-14"
            )
            print(f"✅ Created folder structure with ID: {test_folder_id}")
            
            # Test 3: Upload test content
            print("\nTest 3: Uploading test file...")
            test_content = b"This is a test file for Google Drive integration"
            result = await gdrive.upload_file_content(
                test_content,
                "test_file.txt",
                test_folder_id
            )
            print(f"✅ Uploaded test file: {result['file']['name']} (ID: {result['file']['id']})")
            
            print("\n✅ All tests passed! Google Drive integration is working correctly.")
            print(f"View the test folder at: https://drive.google.com/drive/folders/{test_folder_id}")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())