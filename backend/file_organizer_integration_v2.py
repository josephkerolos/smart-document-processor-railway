"""
File Organizer API Integration v2 - Proper implementation
Correctly uses the file-organizer API endpoints
"""

import os
import io
import json
import base64
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import zipfile
from pathlib import Path
import time
from pdf_compression_service import pdf_compression_service

logger = logging.getLogger(__name__)

class FileOrganizerIntegrationV2:
    """Integration with File Organizer API for document processing workflow"""
    
    def __init__(self, api_url: str = "http://localhost:2499"):
        self.api_url = api_url
        self.session = None
        # Increased timeout for large files (10 minutes)
        self.timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_read=600)
        self.max_retries = 5
        self.retry_delay = 2  # Initial delay in seconds
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                    delay = self.retry_delay * (2 ** attempt)
                    error_msg = str(e) if str(e) else "Connection/timeout error"
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    error_msg = str(e) if str(e) else "Connection/timeout error"
                    logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
        
        raise last_exception
            
    async def clean_and_analyze_pdf(self, pdf_content: bytes, filename: str, 
                                   compress: bool = False, target_size_mb: float = 5.0) -> Dict[str, Any]:
        """
        Clean and analyze PDF using File Organizer API with retry logic
        This is the primary method that combines cleaning and structure analysis
        """
        await self.ensure_session()
        
        async def _make_request():
            # Use the analyze-and-split endpoint which does both cleaning and analysis
            form = aiohttp.FormData()
            form.add_field('pdf', pdf_content, filename=filename, content_type='application/pdf')
            
            params = {
                'compress': str(compress).lower(),
                'target_size_mb': str(target_size_mb),
                'auto_split': 'true'  # Explicitly enable auto-splitting
            }
            
            logger.info(f"Sending request to {self.api_url}/api/analyze-and-split for {filename} ({len(pdf_content)/(1024*1024):.1f}MB)")
            start_time = time.time()
            
            async with self.session.post(
                f"{self.api_url}/api/analyze-and-split",
                data=form,
                params=params
            ) as response:
                elapsed = time.time() - start_time
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Request failed after {elapsed:.1f}s with status {response.status}")
                    raise Exception(f"Analyze and split failed (status {response.status}): {error_text}")
                    
                result = await response.json()
                logger.info(f"Request succeeded after {elapsed:.1f}s")
                return result
        
        try:
            logger.info(f"Starting clean_and_analyze_pdf for {filename} with retry logic")
            return await self._retry_with_backoff(_make_request)
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown error - likely timeout or connection issue"
            logger.error(f"Error in clean_and_analyze_pdf after all retries: {error_msg}")
            # Provide more context for debugging
            if not str(e):
                logger.error(f"Empty error after processing {filename} ({len(pdf_content)/(1024*1024):.1f}MB)")
                logger.error("This often indicates a timeout. Consider processing files sequentially.")
            raise Exception(f"Failed to process {filename}: {error_msg}")
            
    async def split_pdf_with_structure(self, cleaned_pdf_base64: str, documents: List[Dict[str, Any]], 
                                      compress: bool = False, target_size_mb: float = 5.0) -> bytes:
        """
        Split PDF based on analyzed structure with retry logic
        Returns ZIP file content
        """
        await self.ensure_session()
        
        async def _make_request():
            request_data = {
                "cleanedPdfBase64": cleaned_pdf_base64,
                "documents": documents,
                "compress": compress,
                "targetSizeMb": target_size_mb
            }
            
            async with self.session.post(
                f"{self.api_url}/api/split-pdf",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Split PDF failed (status {response.status}): {error_text}")
                    
                # Response is a ZIP file
                zip_content = await response.read()
                return zip_content
        
        try:
            logger.info("Starting split_pdf_with_structure with retry logic")
            return await self._retry_with_backoff(_make_request)
        except Exception as e:
            logger.error(f"Error in split_pdf_with_structure after all retries: {e}")
            raise
            
    async def compress_pdf(self, pdf_content: bytes, target_size_mb: float = 2.0) -> bytes:
        """
        Compress PDF to target size using algorithmic compression with retry logic
        """
        await self.ensure_session()
        
        async def _make_request():
            form = aiohttp.FormData()
            form.add_field('pdf', pdf_content, filename='document.pdf', content_type='application/pdf')
            form.add_field('target_size_mb', str(target_size_mb))
            
            async with self.session.post(
                f"{self.api_url}/api/compress-pdf",
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Compression failed (status {response.status}): {error_text}")
                    
                compressed_content = await response.read()
                original_size = len(pdf_content) / (1024 * 1024)
                compressed_size = len(compressed_content) / (1024 * 1024)
                
                logger.info(f"Compressed PDF from {original_size:.2f}MB to {compressed_size:.2f}MB")
                return compressed_content
        
        try:
            logger.info(f"Starting compress_pdf with retry logic (target: {target_size_mb}MB)")
            return await self._retry_with_backoff(_make_request)
        except Exception as e:
            logger.error(f"Error compressing PDF after all retries: {e}")
            logger.warning("Falling back to embedded compression service")
            # Use embedded compression service as fallback
            try:
                compressed_content = await pdf_compression_service.compress_pdf(pdf_content, target_size_mb)
                logger.info("Successfully compressed using embedded service")
                return compressed_content
            except Exception as fallback_error:
                logger.error(f"Embedded compression also failed: {fallback_error}")
                logger.warning("Returning original PDF content due to compression failure")
                return pdf_content  # Return original if all compression fails
    
    async def compress_pdf_with_monitoring(self, pdf_content: bytes, filename: str, target_size_mb: float = 2.0) -> dict:
        """
        Compress PDF with monitoring wrapper - returns dict with compressed content and metadata
        """
        try:
            logger.info(f"Starting compression monitoring for {filename} (target: {target_size_mb}MB)")
            
            # Calculate original size
            original_size_mb = len(pdf_content) / (1024 * 1024)
            
            # Compress the PDF
            compressed_content = await self.compress_pdf(pdf_content, target_size_mb)
            
            # Calculate compressed size
            compressed_size_mb = len(compressed_content) / (1024 * 1024)
            
            # Calculate reduction percentage
            reduction_percent = ((original_size_mb - compressed_size_mb) / original_size_mb) * 100
            
            logger.info(f"Compression complete for {filename}: {original_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB ({reduction_percent:.1f}% reduction)")
            
            return {
                "content": compressed_content,
                "filename": filename,
                "original_size_mb": original_size_mb,
                "compressed_size_mb": compressed_size_mb,
                "reduction_percent": reduction_percent,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in compress_pdf_with_monitoring for {filename}: {e}")
            # Return original content if compression fails
            return {
                "content": pdf_content,
                "filename": filename,
                "original_size_mb": len(pdf_content) / (1024 * 1024),
                "compressed_size_mb": len(pdf_content) / (1024 * 1024),
                "reduction_percent": 0,
                "success": False,
                "error": str(e)
            }
            
    async def process_document_workflow(self, pdf_content: bytes, filename: str, 
                                      output_dir: str, expected_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete document processing workflow:
        1. Clean and analyze PDF
        2. Split if multiple documents detected
        3. Extract and organize documents
        """
        results = {
            "success": False,
            "cleaned": False,
            "split_count": 0,
            "documents": [],
            "cleaning_report": {},
            "analysis_results": {},
            "errors": []
        }
        
        try:
            logger.info(f"Starting workflow for {filename}")
            
            # Step 1: Clean and analyze the PDF
            analysis_result = await self.clean_and_analyze_pdf(pdf_content, filename)
            
            if not analysis_result.get("success"):
                raise Exception(analysis_result.get("error", "Analysis failed"))
            
            results["cleaned"] = True
            results["analysis_results"] = analysis_result
            
            # Extract cleaning information - check different possible field names
            cleaning_data = None
            if "cleaningResults" in analysis_result:
                cleaning_data = analysis_result["cleaningResults"]
            elif "cleaning" in analysis_result:
                cleaning_data = analysis_result["cleaning"]
            
            if cleaning_data:
                results["cleaning_report"] = {
                    "removed_pages": cleaning_data.get("removedPages", 0),
                    "total_pages": cleaning_data.get("originalPages", 0),
                    "kept_pages": cleaning_data.get("keptPages", 0),
                    "page_decisions": cleaning_data.get("analysisResults", [])
                }
                logger.info(f"Cleaning results: Removed {results['cleaning_report']['removed_pages']} pages out of {results['cleaning_report']['total_pages']}")
            else:
                # Fallback if no cleaning results
                results["cleaning_report"] = {
                    "removed_pages": 0,
                    "total_pages": 0,
                    "kept_pages": 0,
                    "page_decisions": []
                }
                logger.warning("No cleaning results found in API response")
            
            # Get the cleaned PDF
            cleaned_pdf_base64 = analysis_result.get("cleanedPdfBase64", "")
            if not cleaned_pdf_base64:
                raise Exception("No cleaned PDF returned")
                
            cleaned_pdf_content = base64.b64decode(cleaned_pdf_base64)
            
            # Log the full analysis result to debug
            logger.info(f"Analysis result keys: {list(analysis_result.keys())}")
            logger.info(f"Can split: {analysis_result.get('canSplit', False)}")
            logger.info(f"Split preview: {analysis_result.get('splitPreview', [])}")
            
            # Step 2: Check if we can split the document
            can_split = analysis_result.get("canSplit", False)
            split_preview = analysis_result.get("splitPreview", [])
            
            if can_split and split_preview:
                logger.info(f"Document can be split into {len(split_preview)} parts")
                
                # Convert split preview to proper format for split endpoint
                documents_for_split = []
                for doc in split_preview:
                    # Parse page range (e.g., "1-5" -> startPage: 1, endPage: 5)
                    page_range = doc.get('pageRange', '1-1')
                    start_page, end_page = page_range.split('-')
                    
                    documents_for_split.append({
                        'startPage': int(start_page),
                        'endPage': int(end_page),
                        'suggestedFilename': doc.get('filename', 'document.pdf'),
                        'formType': doc.get('formType', ''),
                        'identifier': doc.get('identifier', '')
                    })
                
                # Split the PDF
                split_zip_content = await self.split_pdf_with_structure(
                    cleaned_pdf_base64,
                    documents_for_split,
                    compress=False
                )
                
                # Extract split PDFs from ZIP
                split_documents = []
                with zipfile.ZipFile(io.BytesIO(split_zip_content)) as zip_file:
                    for name in zip_file.namelist():
                        if name.endswith('.pdf'):
                            pdf_data = zip_file.read(name)
                            # Add "cleaned_" prefix if pages were removed
                            if results["cleaning_report"]["removed_pages"] > 0:
                                display_name = f"cleaned_{name}"
                            else:
                                display_name = name
                            
                            split_documents.append({
                                "filename": display_name,
                                "content": pdf_data,
                                "metadata": {
                                    "was_split": True,
                                    "was_cleaned": results["cleaning_report"]["removed_pages"] > 0,
                                    "pages_removed": results["cleaning_report"]["removed_pages"],
                                    "original_filename": filename
                                }
                            })
                
                results["documents"] = split_documents
                results["split_count"] = len(split_documents)
            else:
                # No splitting needed, use the cleaned PDF as is
                logger.info("Document does not need splitting")
                # Add "cleaned_" prefix only if pages were actually removed
                if results["cleaning_report"]["removed_pages"] > 0:
                    output_filename = f"cleaned_{filename}"
                else:
                    output_filename = filename
                    
                results["documents"] = [{
                    "filename": output_filename,
                    "content": cleaned_pdf_content,
                    "metadata": {
                        "was_split": False,
                        "was_cleaned": results["cleaning_report"]["removed_pages"] > 0,
                        "pages_removed": results["cleaning_report"]["removed_pages"],
                        "original_filename": filename
                    }
                }]
                results["split_count"] = 1
            
            results["success"] = True
            logger.info(f"Workflow completed successfully for {filename}")
            
        except Exception as e:
            logger.error(f"Error in workflow for {filename}: {e}")
            results["errors"].append(str(e))
            
        return results
        
    def organize_by_quarter(self, documents: List[Dict[str, Any]], base_path: str) -> Dict[str, List[str]]:
        """
        Organize documents by quarter based on extracted metadata
        Creates folder structure: base_path/YYYY/Q[1-4]/
        """
        organized = {}
        
        for doc in documents:
            # Extract quarter and year from filename or metadata
            filename = doc["filename"]
            
            # Look for patterns like "Q1_2023" or "2023_Q1" in filename
            import re
            quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
            year_match = re.search(r'(20\d{2})', filename)
            
            if quarter_match:
                quarter = f"Q{quarter_match.group(1)}"
            else:
                # Default to current quarter
                current_month = datetime.now().month
                quarter = f"Q{(current_month - 1) // 3 + 1}"
                
            if year_match:
                year = year_match.group(1)
            else:
                year = str(datetime.now().year)
            
            # Create folder structure
            folder_path = os.path.join(base_path, year, quarter)
            os.makedirs(folder_path, exist_ok=True)
            
            # Save document
            file_path = os.path.join(folder_path, doc["filename"])
            with open(file_path, 'wb') as f:
                f.write(doc["content"])
            
            # Track organization
            key = f"{year}/{quarter}"
            if key not in organized:
                organized[key] = []
            organized[key].append(file_path)
            
        logger.info(f"Organized {len(documents)} documents into {len(organized)} quarters")
        return organized
        
    def create_zip_archive(self, files: List[Tuple[str, bytes]], zip_name: str) -> bytes:
        """
        Create a ZIP archive containing multiple files
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in files:
                zip_file.writestr(filename, content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    async def combine_pdfs(self, pdf_list: List[Tuple[str, bytes]], compress: bool = False, 
                          target_size_mb: float = 5.0) -> bytes:
        """
        Combine multiple PDFs into one using PyPDF2, optionally with compression
        """
        try:
            # Use PyPDF2 to combine PDFs locally
            from PyPDF2 import PdfWriter, PdfReader
            
            pdf_writer = PdfWriter()
            
            # Add all PDFs to the writer
            for filename, content in pdf_list:
                pdf_reader = PdfReader(io.BytesIO(content))
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_writer.add_page(page)
            
            # Write combined PDF to bytes
            output_buffer = io.BytesIO()
            pdf_writer.write(output_buffer)
            output_buffer.seek(0)
            combined_content = output_buffer.getvalue()
            
            logger.info(f"Combined {len(pdf_list)} PDFs into one using PyPDF2")
            
            # If compression is requested, compress the combined PDF
            if compress:
                compressed_content = await self.compress_pdf(combined_content, target_size_mb)
                return compressed_content
            else:
                return combined_content
                
        except Exception as e:
            logger.error(f"Error combining PDFs with PyPDF2: {e}")
            # Fallback to API method if PyPDF2 fails
            await self.ensure_session()
            
            # Convert PDFs to base64 for the API
            pdfs_base64 = []
            for filename, content in pdf_list:
                pdfs_base64.append({
                    "fileName": filename,
                    "pdfBase64": base64.b64encode(content).decode('utf-8')
                })
            
            request_data = {
                "pdfs": pdfs_base64,
                "compress": compress,
                "targetSizeMb": target_size_mb
            }
            
            try:
                async with self.session.post(
                    f"{self.api_url}/api/combine-cleaned-pdfs-base64",
                    json=request_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API combining failed: {error_text}")
                        
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/pdf' in content_type:
                        # Response is PDF directly
                        combined_content = await response.read()
                        logger.info(f"Combined {len(pdf_list)} PDFs via API (direct PDF response)")
                        return combined_content
                    else:
                        # Response is JSON with base64 PDF
                        result = await response.json()
                        if result.get("success") and result.get("combinedPdf"):
                            combined_content = base64.b64decode(result["combinedPdf"])
                            logger.info(f"Combined {len(pdf_list)} PDFs via API (JSON response)")
                            return combined_content
                        else:
                            raise Exception("No combined PDF in response")
                    
            except Exception as e:
                # Truncate error message if it contains base64 data
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... (truncated)"
                logger.error(f"Error combining PDFs: {error_msg}")
                raise Exception(f"PDF combination failed: {error_msg[:200]}")