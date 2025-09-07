"""
Test the enhanced processing summary format
"""

import asyncio
from google_drive_integration import GoogleDriveIntegration
from datetime import datetime

async def test_enhanced_summary():
    """Test the enhanced processing summary"""
    
    # Mock processing stats with detailed information
    mock_stats = {
        "documents_processed": 4,
        "pages_removed": 8,
        "total_pages": 24,
        "pages_kept": 16,
        "compressions_successful": 4,
        "extractions_successful": 4,
        "total_elapsed_time": 125.7,
        "average_time_per_document": 31.4,
        "start_time": "2025-06-14 15:30:45",
        "end_time": "2025-06-14 15:32:51",
        "step_timings": {
            "cleaning": 12.5,
            "splitting": 5.3,
            "extraction": 95.2,
            "organization": 2.1,
            "compression": 8.4,
            "archiving": 2.2
        },
        "document_details": [
            {
                "filename": "David_Anthony_Salon_LLC_941X_Q1_2021.pdf",
                "quarter": 1,
                "year": 2021,
                "pages": 5
            },
            {
                "filename": "David_Anthony_Salon_LLC_941X_Q2_2020.pdf",
                "quarter": 2,
                "year": 2020,
                "pages": 5
            },
            {
                "filename": "David_Anthony_Salon_LLC_941X_Q3_2020.pdf",
                "quarter": 3,
                "year": 2020,
                "pages": 5
            },
            {
                "filename": "David_Anthony_Salon_LLC_941X_Q4_2020.pdf",
                "quarter": 4,
                "year": 2020,
                "pages": 5
            }
        ]
    }
    
    async with GoogleDriveIntegration() as gdrive:
        try:
            # Test creating summary content
            print("Testing enhanced processing summary generation...\n")
            
            result = await gdrive.upload_processed_documents(
                session_id="test-session-123",
                master_archive_path="/tmp/test_archive.zip",
                company_name="David Anthony Salon LLC",
                form_type="941X",
                processing_stats=mock_stats
            )
            
            if result["success"]:
                print("✅ Successfully created enhanced processing summary!")
                print(f"📁 Google Drive folder: {result['folder_path']}")
                print(f"🔗 Direct link: {result['google_drive_url']}")
                print(f"\n📄 Summary file: {result['uploads']['summary']['name']}")
            else:
                print("❌ Failed to create summary")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Create a dummy archive file for testing
    with open("/tmp/test_archive.zip", "wb") as f:
        f.write(b"Test archive content")
    
    asyncio.run(test_enhanced_summary())