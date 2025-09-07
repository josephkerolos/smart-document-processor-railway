"""
Google Drive Manager API Integration
Handles automatic upload of processed documents to Google Drive
"""

import aiohttp
import asyncio
import logging
import os
from typing import Dict, Any, Optional, BinaryIO, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class GoogleDriveIntegration:
    """Integration with Google Drive Manager API for document uploads"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        # Use environment variable or real Google Drive Manager URL, not mock
        self.api_url = api_url or os.getenv("GOOGLE_DRIVE_MANAGER_URL", "http://localhost:4831") + "/api"
        self.api_key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY", "gdrive_api_key_2024")
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def create_folder(self, folder_name: str, parent_id: str = "root") -> Dict[str, Any]:
        """
        Create a new folder in Google Drive
        
        Args:
            folder_name: Name of the folder to create
            parent_id: Parent folder ID (default: root of master folder)
            
        Returns:
            Dict with folder information including ID
        """
        await self.ensure_session()
        
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "name": folder_name,
            "parentId": parent_id
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/folders",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to create folder: {error_text}")
                    
                result = await response.json()
                logger.info(f"Created folder '{folder_name}' with ID: {result['folder']['id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating folder: {e}")
            raise
            
    async def upload_file(self, file_path: str, folder_id: str = "root", 
                         custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to Google Drive
        
        Args:
            file_path: Path to the file to upload
            folder_id: Destination folder ID (default: root of master folder)
            custom_name: Custom name for the file (optional)
            
        Returns:
            Dict with upload result including file ID
        """
        await self.ensure_session()
        
        headers = {
            "X-API-Key": self.api_key
        }
        
        # Prepare multipart form data
        form = aiohttp.FormData()
        
        # Read file content
        file_name = custom_name or os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            form.add_field('file', f.read(), filename=file_name, content_type='application/octet-stream')
        
        if folder_id and folder_id != "root":
            form.add_field('folderId', folder_id)
        
        try:
            async with self.session.post(
                f"{self.api_url}/upload",
                headers=headers,
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to upload file: {error_text}")
                    
                result = await response.json()
                logger.info(f"Uploaded file '{file_name}' with ID: {result['file']['id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
            
    async def upload_file_content(self, content: bytes, filename: str, 
                                 folder_id: str = "root") -> Dict[str, Any]:
        """
        Upload file content directly without saving to disk
        
        Args:
            content: File content as bytes
            filename: Name for the file
            folder_id: Destination folder ID (default: root of master folder)
            
        Returns:
            Dict with upload result including file ID
        """
        await self.ensure_session()
        
        headers = {
            "X-API-Key": self.api_key
        }
        
        # Prepare multipart form data
        form = aiohttp.FormData()
        form.add_field('file', content, filename=filename, content_type='application/octet-stream')
        
        if folder_id and folder_id != "root":
            form.add_field('folderId', folder_id)
        
        try:
            async with self.session.post(
                f"{self.api_url}/upload",
                headers=headers,
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to upload file: {error_text}")
                    
                result = await response.json()
                logger.info(f"Uploaded file '{filename}' with ID: {result['file']['id']}")
                return result
                
        except Exception as e:
            logger.error(f"Error uploading file content: {e}")
            raise
            
    async def search_folder(self, folder_name: str, parent_id: str = None) -> Optional[str]:
        """
        Search for a folder by name
        
        Args:
            folder_name: Name of the folder to search for
            parent_id: Parent folder ID to search within (optional)
            
        Returns:
            Folder ID if found, None otherwise
        """
        await self.ensure_session()
        
        headers = {
            "X-API-Key": self.api_key
        }
        
        # When searching with a parent_id, we need to list files in that folder
        # and filter by name, not use search which searches globally
        if parent_id:
            params = {
                "folderId": parent_id,
                "pageSize": 50
            }
        else:
            params = {
                "search": folder_name,
                "pageSize": 50
            }
        
        try:
            async with self.session.get(
                f"{self.api_url}/files",
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to search for folder: {error_text}")
                    
                result = await response.json()
                
                # Look for folder in results
                for file in result.get("files", []):
                    if (file.get("mimeType") == "application/vnd.google-apps.folder" and 
                        file.get("name") == folder_name):
                        return file.get("id")
                        
                return None
                
        except Exception as e:
            logger.error(f"Error searching for folder: {e}")
            return None
    
    async def download_file(self, file_id: str) -> bytes:
        """
        Download a file from Google Drive
        
        Args:
            file_id: Google Drive file ID to download
            
        Returns:
            File content as bytes
        """
        await self.ensure_session()
        
        headers = {
            "X-API-Key": self.api_key
        }
        
        try:
            async with self.session.get(
                f"{self.api_url}/download/{file_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to download file: {error_text}")
                    
                content = await response.read()
                logger.info(f"Downloaded file {file_id} ({len(content)} bytes)")
                return content
                
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise
            
    async def create_document_folder_structure(self, company_name: str, 
                                             form_type: str, 
                                             processing_date: str) -> str:
        """
        Create a folder structure for processed documents
        Format: ProcessedDocuments/CompanyName/FormType/Date
        
        Args:
            company_name: Name of the company
            form_type: Type of form (e.g., 941X, 941)
            processing_date: Date of processing (YYYY-MM-DD format)
            
        Returns:
            ID of the final folder where documents should be uploaded
        """
        try:
            # Create or find ProcessedDocuments folder
            processed_folder_id = await self.search_folder("ProcessedDocuments")
            if not processed_folder_id:
                result = await self.create_folder("ProcessedDocuments")
                processed_folder_id = result["folder"]["id"]
                logger.info("Created ProcessedDocuments folder")
            
            # Create or find company folder
            clean_company_name = company_name.replace(",", "").replace(".", "")
            company_folder_id = await self.search_folder(clean_company_name, processed_folder_id)
            if not company_folder_id:
                result = await self.create_folder(clean_company_name, processed_folder_id)
                company_folder_id = result["folder"]["id"]
                logger.info(f"Created company folder: {clean_company_name}")
            
            # Create or find form type folder
            form_folder_id = await self.search_folder(form_type, company_folder_id)
            if not form_folder_id:
                result = await self.create_folder(form_type, company_folder_id)
                form_folder_id = result["folder"]["id"]
                logger.info(f"Created form type folder: {form_type}")
            
            # Create date/time folder (always create new)
            # Include time to make folders unique and easier to track
            processing_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            date_folder_name = f"Processed_{processing_datetime}"
            result = await self.create_folder(date_folder_name, form_folder_id)
            date_folder_id = result["folder"]["id"]
            logger.info(f"Created date/time folder: {date_folder_name}")
            
            return date_folder_id
            
        except Exception as e:
            logger.error(f"Error creating folder structure: {e}")
            raise
            
    async def upload_to_custom_folder(self, file_path: str, folder_id: str, custom_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to a specific Google Drive folder
        
        Args:
            file_path: Path to the file to upload
            folder_id: Google Drive folder ID where the file should be uploaded
            custom_name: Optional custom name for the file
            
        Returns:
            Dict with upload result
        """
        try:
            # Use the existing upload_file method
            result = await self.upload_file(file_path, folder_id, custom_name)
            return {
                "success": True,
                "file": result.get("file"),
                "folder_id": folder_id
            }
        except Exception as e:
            logger.error(f"Error uploading to custom folder: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def upload_processed_documents(self, session_id: str, 
                                       master_archive_path: str,
                                       company_name: str,
                                       form_type: str,
                                       processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload processed documents to Google Drive with proper organization
        
        Args:
            session_id: Processing session ID
            master_archive_path: Path to the master archive file
            company_name: Name of the company from extracted data
            form_type: Type of form processed
            processing_stats: Statistics about the processing
            
        Returns:
            Dict with upload results and Google Drive links
        """
        try:
            # Create folder structure with timestamp
            processing_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_id = await self.create_document_folder_structure(
                company_name, 
                form_type, 
                processing_datetime
            )
            
            # Upload master archive
            logger.info(f"Uploading master archive: {master_archive_path}")
            archive_result = await self.upload_file(
                master_archive_path,
                folder_id
            )
            
            # Create processing summary with detailed information
            # Format processing time
            total_seconds = processing_stats.get('total_elapsed_time', 0)
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            formatted_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            # Get current timestamp with timezone
            import time
            timezone = time.strftime('%Z')
            if not timezone:  # Fallback if timezone is empty
                timezone = time.tzname[time.daylight]
            timestamp = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S {timezone}")
            
            # Get step timings if available
            step_timings = processing_stats.get('step_timings', {})
            step_timing_details = ""
            if step_timings:
                step_timing_details = "\nDetailed Step Timings:\n"
                for step, duration in step_timings.items():
                    step_name = step.replace('_', ' ').title()
                    step_timing_details += f"- {step_name}: {duration:.1f}s\n"
            
            # Get document details if available
            document_details = processing_stats.get('document_details', [])
            doc_details_section = ""
            if document_details:
                doc_details_section = "\nProcessed Documents:\n"
                for doc in document_details:
                    doc_details_section += f"- {doc.get('filename', 'Unknown')}\n"
                    if 'quarter' in doc and 'year' in doc:
                        doc_details_section += f"  Quarter: {doc['quarter']}, Year: {doc['year']}\n"
                    if 'pages' in doc:
                        doc_details_section += f"  Pages: {doc['pages']}\n"
            
            summary_content = f"""Document Processing Summary
=====================================
Generated: {timestamp}
Session ID: {session_id}

Company Information:
-------------------
Company Name: {company_name}
Form Type: {form_type}

Processing Information:
----------------------
Start Time: {processing_stats.get('start_time', 'N/A')}
End Time: {processing_stats.get('end_time', 'N/A')}
Total Duration: {formatted_time} ({total_seconds:.1f} seconds)

Processing Statistics:
---------------------
- Documents Processed: {processing_stats.get('documents_processed', 0)}
- Total Pages Analyzed: {processing_stats.get('total_pages', 'N/A')}
- Pages Removed (Instructions): {processing_stats.get('pages_removed', 0)}
- Pages Kept: {processing_stats.get('pages_kept', 'N/A')}
- Successful Extractions: {processing_stats.get('extractions_successful', processing_stats.get('documents_processed', 0))}
- Successful Compressions: {processing_stats.get('compressions_successful', 0)}
- Average Time per Document: {processing_stats.get('average_time_per_document', 0):.1f}s
{step_timing_details}
{doc_details_section}
Files Included in Archive:
-------------------------
1. Master Archive Contents:
   - raw_cleaned/ : Original cleaned documents (uncompressed)
   - compressed_cleaned/ : Compressed versions
   - archives_cleaned/ : Pre-packaged collections
   - extractions/ : JSON extraction data for each document
   - Year/Quarter folders (e.g., 2020/Q3/)

2. Archive Types:
   - individual_uncompressed.zip : All split documents at original quality
   - individual_compressed.zip : All split documents compressed
   - combined_uncompressed.pdf : All documents in one PDF (if multiple)
   - combined_compressed.pdf : All documents in one compressed PDF (if multiple)

3. Data Formats:
   - PDF documents with cleaned content (instruction pages removed)
   - JSON files with extracted structured data
   - Both compressed and uncompressed versions

Processing Features Used:
------------------------
- AI-powered page cleaning (removed instruction pages)
- Intelligent document splitting
- Parallel data extraction using Gemini AI
- Metadata-based file organization
- Smart compression to target size
- Automatic Google Drive upload

Upload Information:
------------------
Upload Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Google Drive Path: ProcessedDocuments/{company_name}/{form_type}/Processed_{processing_datetime}/
Master Archive: {master_archive_path.split('/')[-1] if '/' in master_archive_path else master_archive_path}

=====================================
End of Processing Summary
"""
            
            # Upload summary
            summary_result = await self.upload_file_content(
                summary_content.encode('utf-8'),
                f"Processing_Summary_{session_id}.txt",
                folder_id
            )
            
            # Generate Google Drive folder link
            folder_link = f"https://drive.google.com/drive/folders/{folder_id}"
            
            # Build result
            result = {
                "success": True,
                "folder_id": folder_id,
                "folder_link": folder_link,
                "folder_path": f"ProcessedDocuments/{company_name}/{form_type}/Processed_{processing_datetime}",
                "uploads": {
                    "master_archive": {
                        "id": archive_result["file"]["id"],
                        "name": archive_result["file"]["name"],
                        "size": archive_result["file"].get("size")
                    },
                    "summary": {
                        "id": summary_result["file"]["id"],
                        "name": summary_result["file"]["name"]
                    }
                },
                "google_drive_url": f"https://drive.google.com/drive/folders/{folder_id}"
            }
            
            logger.info(f"Successfully uploaded documents to Google Drive: {result['folder_path']}")
            return result
            
        except Exception as e:
            logger.error(f"Error uploading processed documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def upload_batch_documents(self, batch_dir: str, master_archive_path: str,
                                   company_name: str, form_type: str, quarters_str: str,
                                   session_ids: List[str], processing_stats: Dict[str, Any]):
        """
        Upload batch processed documents to Google Drive
        Creates a single folder with all quarters combined
        """
        try:
            # Create folder structure: ProcessedDocuments/CompanyName/FormType/Batch_YYYY-MM-DD_HH-MM-SS_Quarters
            processing_date = datetime.now()
            date_str = processing_date.strftime("%Y-%m-%d")
            time_str = processing_date.strftime("%H-%M-%S")
            
            # Create base folder structure
            folder_structure = await self.create_document_folder_structure(
                company_name,
                form_type,
                date_str
            )
            
            # Create batch folder with all quarters
            batch_folder_name = f"Batch_{date_str}_{time_str}"
            if quarters_str:
                batch_folder_name += f"_{quarters_str}"
            
            batch_folder = await self.create_folder(batch_folder_name, folder_structure)
            batch_folder_id = batch_folder["folder"]["id"]
            logger.info(f"Created batch folder: {batch_folder_name}")
            
            # Upload master archive
            archive_name = os.path.basename(master_archive_path)
            with open(master_archive_path, 'rb') as f:
                archive_file_id = await self.upload_file_content(
                    f.read(),
                    archive_name,
                    batch_folder_id
                )
            logger.info(f"Uploaded master archive: {archive_name}")
            
            # Upload all_extractions.json
            all_extractions_path = os.path.join(batch_dir, "all_extractions.json")
            if os.path.exists(all_extractions_path):
                with open(all_extractions_path, 'r') as f:
                    extractions_content = f.read()
                
                extractions_file_id = await self.upload_file_content(
                    extractions_content.encode('utf-8'),
                    'all_extractions.json',
                    batch_folder_id
                )
                logger.info("Uploaded all_extractions.json")
            
            # Create and upload batch summary
            summary_content = f"""Batch Processing Summary
========================
Company: {company_name}
Form Type: {form_type}
Quarters: {quarters_str}
Processing Date: {date_str} {time_str}
Total Files: {processing_stats.get('total_files', 0)}
Total Documents: {processing_stats.get('total_documents', 0)}
Batch ID: {processing_stats.get('batch_id', 'N/A')}

Session IDs:
{chr(10).join(f'- {sid}' for sid in session_ids)}

Files Included:
- Master Archive: {archive_name}
- All Extractions: all_extractions.json
"""
            
            summary_file_id = await self.upload_file_content(
                summary_content.encode('utf-8'),
                f'Batch_Summary_{processing_stats.get("batch_id", "")[:8]}.txt',
                batch_folder_id
            )
            
            # Create individual session folders and organize files
            for session_id in session_ids:
                session_dir = f"processed_documents/{session_id}"
                if os.path.exists(session_dir):
                    # Create session subfolder
                    session_folder = await self.create_folder(f"Session_{session_id[:8]}", batch_folder_id)
                    session_folder_id = session_folder["folder"]["id"]
                    
                    # Upload organized files from each session
                    for root, dirs, files in os.walk(session_dir):
                        # Skip certain directories
                        if "original" in root or "temp" in root:
                            continue
                            
                        # Create matching folder structure
                        rel_path = os.path.relpath(root, session_dir)
                        if rel_path != ".":
                            current_folder_id = session_folder_id
                            for folder_part in rel_path.split(os.sep):
                                if folder_part:
                                    sub_folder = await self.create_folder(folder_part, current_folder_id)
                                    current_folder_id = sub_folder["folder"]["id"]
                        else:
                            current_folder_id = session_folder_id
                        
                        # Upload files
                        for file in files[:10]:  # Limit files to prevent timeout
                            if file.endswith(('.pdf', '.json', '.txt')):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'rb') as f:
                                        content = f.read()
                                    
                                    await self.upload_file_content(
                                        content,
                                        file,
                                        current_folder_id
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to upload {file}: {e}")
            
            # Get shareable link
            folder_link = f"https://drive.google.com/drive/folders/{batch_folder_id}"
            
            return {
                "success": True,
                "folder_name": f"ProcessedDocuments/{company_name}/{form_type}/{date_str}/{batch_folder_name}",
                "folder_id": batch_folder_id,
                "folder_link": folder_link,
                "files_uploaded": ["Master Archive", "all_extractions.json", "Batch Summary"] + 
                                [f"Session {sid[:8]}" for sid in session_ids]
            }
            
        except Exception as e:
            logger.error(f"Error uploading batch to Google Drive: {e}")
            return {
                "success": False,
                "error": str(e)
            }