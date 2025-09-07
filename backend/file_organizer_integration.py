"""
File Organizer API Integration for Smart Document Processor
Handles document cleaning, splitting, organization, and compression
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

logger = logging.getLogger(__name__)

class FileOrganizerIntegration:
    """Integration with File Organizer API for document processing workflow"""
    
    def __init__(self, api_url: str = "http://localhost:2499"):
        self.api_url = api_url
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
            
    async def clean_pdf(self, pdf_content: bytes, filename: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Clean PDF using File Organizer API's analyze-and-split endpoint
        Returns cleaned PDF content and cleaning report
        """
        # Use the analyze-and-split endpoint which also cleans PDFs
        split_results = await self.split_pdf(pdf_content, filename)
        
        # If we got split documents, combine them back
        if len(split_results) > 0:
            # Get the first document (or combine if multiple)
            cleaned_content = split_results[0]["content"]
            
            # Create cleaning report
            cleaning_report = {
                "removed_pages": 0,  # We don't have this info from split
                "total_pages": 0,
                "kept_pages": 0,
                "split_count": len(split_results)
            }
            
            return cleaned_content, cleaning_report
        else:
            # Return original if no results
            return pdf_content, {"error": "No results from cleaning", "removed_pages": 0}
            
    async def split_pdf(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Split PDF into individual documents using AI-powered analysis
        Returns list of split documents with metadata
        """
        await self.ensure_session()
        
        # Create form data - the API expects multipart form data
        form = aiohttp.FormData()
        form.add_field('pdf', pdf_content, filename=filename, content_type='application/pdf')
        form.add_field('compress', 'false')
        form.add_field('auto_split', 'true')
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/analyze-and-split",
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Splitting failed: {error_text}")
                    
                result = await response.json()
                
                if result.get("success"):
                    split_documents = []
                    
                    # Handle cleaned PDF (if no splitting occurred)
                    if result.get("cleanedPdf"):
                        split_documents.append({
                            "filename": result.get("cleanedFilename", filename),
                            "content": base64.b64decode(result["cleanedPdf"]),
                            "metadata": {
                                "was_split": False,
                                "page_count": result.get("pageCount", 0),
                                "removed_pages": result.get("removedPages", 0)
                            }
                        })
                    
                    # Handle split PDFs
                    if result.get("splitPdfs"):
                        for split_pdf in result["splitPdfs"]:
                            split_documents.append({
                                "filename": split_pdf["filename"],
                                "content": base64.b64decode(split_pdf["content"]),
                                "metadata": {
                                    "was_split": True,
                                    "pages": split_pdf.get("pages", []),
                                    "extracted_info": split_pdf.get("extractedInfo", {}),
                                    "confidence": split_pdf.get("confidence", 0)
                                }
                            })
                    
                    logger.info(f"Split {filename} into {len(split_documents)} documents")
                    return split_documents
                else:
                    raise Exception("Splitting API returned failure")
                    
        except Exception as e:
            logger.error(f"Error splitting PDF {filename}: {e}")
            # Return original as single document if splitting fails
            return [{
                "filename": filename,
                "content": pdf_content,
                "metadata": {"error": str(e), "was_split": False}
            }]
            
    async def compress_pdf(self, pdf_content: bytes, target_size_mb: float = 2.0) -> bytes:
        """
        Compress PDF to target size using algorithmic compression
        """
        await self.ensure_session()
        
        # Create form data
        form = aiohttp.FormData()
        form.add_field('pdf', pdf_content, filename='document.pdf', content_type='application/pdf')
        form.add_field('target_size_mb', str(target_size_mb))
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/compress-pdf",
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Compression failed: {error_text}")
                    
                compressed_content = await response.read()
                original_size = len(pdf_content) / (1024 * 1024)
                compressed_size = len(compressed_content) / (1024 * 1024)
                
                logger.info(f"Compressed PDF from {original_size:.2f}MB to {compressed_size:.2f}MB")
                return compressed_content
                
        except Exception as e:
            logger.error(f"Error compressing PDF: {e}")
            # Return original if compression fails
            return pdf_content
            
    async def combine_pdfs(self, pdf_list: List[Tuple[str, bytes]], compress: bool = False) -> bytes:
        """
        Combine multiple PDFs into one, optionally with compression
        """
        await self.ensure_session()
        
        # Convert PDFs to base64
        pdfs_base64 = [
            {
                "filename": filename,
                "content": base64.b64encode(content).decode('utf-8')
            }
            for filename, content in pdf_list
        ]
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/combine-cleaned-pdfs-base64",
                json={
                    "pdfs": pdfs_base64,
                    "compress": compress
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Combining failed: {error_text}")
                    
                result = await response.json()
                
                if result.get("success") and result.get("combinedPdf"):
                    combined_content = base64.b64decode(result["combinedPdf"])
                    logger.info(f"Combined {len(pdf_list)} PDFs into one")
                    return combined_content
                else:
                    raise Exception("Combining API returned no result")
                    
        except Exception as e:
            logger.error(f"Error combining PDFs: {e}")
            raise
            
    def organize_by_quarter(self, documents: List[Dict[str, Any]], base_path: str) -> Dict[str, List[str]]:
        """
        Organize documents by quarter based on extracted metadata
        Creates folder structure: base_path/YYYY/Q[1-4]/
        """
        organized = {}
        
        for doc in documents:
            # Extract quarter and year from metadata or filename
            metadata = doc.get("metadata", {})
            extracted_info = metadata.get("extracted_info", {})
            
            # Try to get quarter and year from metadata
            quarter = extracted_info.get("quarter", "")
            year = extracted_info.get("year", "")
            
            # Fallback to filename parsing
            if not quarter or not year:
                filename = doc["filename"]
                # Look for patterns like "Q1_2023" or "2023_Q1"
                import re
                quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
                year_match = re.search(r'(20\d{2})', filename)
                
                if quarter_match:
                    quarter = f"Q{quarter_match.group(1)}"
                if year_match:
                    year = year_match.group(1)
            
            # Default to current year/quarter if not found
            if not year:
                year = str(datetime.now().year)
            if not quarter:
                current_month = datetime.now().month
                quarter = f"Q{(current_month - 1) // 3 + 1}"
            
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
        
    async def process_document_workflow(self, pdf_content: bytes, filename: str, 
                                      output_dir: str, expected_value: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete document processing workflow:
        1. Clean PDF
        2. Split if needed
        3. Process extractions (placeholder for integration with existing extractor)
        4. Organize by quarter
        5. Create compressed and uncompressed versions
        6. Create combined versions
        """
        results = {
            "cleaned": False,
            "split_count": 0,
            "documents": [],
            "organized_paths": {},
            "archives": {},
            "errors": []
        }
        
        try:
            # Step 1: Clean the PDF
            logger.info(f"Starting workflow for {filename}")
            cleaned_content, cleaning_report = await self.clean_pdf(pdf_content, filename)
            results["cleaned"] = True
            results["cleaning_report"] = cleaning_report
            
            # Step 2: Split the cleaned PDF if needed
            split_documents = await self.split_pdf(cleaned_content, filename)
            results["split_count"] = len(split_documents)
            
            # Step 3: Process each split document
            processed_documents = []
            for split_doc in split_documents:
                # Here we would integrate with the existing extraction process
                # For now, we'll just prepare the document structure
                processed_doc = {
                    "filename": split_doc["filename"],
                    "content": split_doc["content"],
                    "metadata": split_doc["metadata"],
                    "extracted_data": {}  # Placeholder for extraction results
                }
                processed_documents.append(processed_doc)
            
            results["documents"] = processed_documents
            
            # Step 4: Organize by quarter
            organized = self.organize_by_quarter(processed_documents, output_dir)
            results["organized_paths"] = organized
            
            # Step 5: Create compressed versions
            compressed_documents = []
            for doc in processed_documents:
                compressed_content = await self.compress_pdf(doc["content"], target_size_mb=2.0)
                compressed_documents.append({
                    **doc,
                    "content": compressed_content,
                    "filename": doc["filename"].replace(".pdf", "_compressed.pdf")
                })
            
            # Organize compressed versions
            compressed_dir = os.path.join(output_dir, "compressed")
            self.organize_by_quarter(compressed_documents, compressed_dir)
            
            # Step 6: Create archives
            # Individual documents (uncompressed)
            individual_files = [(doc["filename"], doc["content"]) for doc in processed_documents]
            results["archives"]["individual_uncompressed"] = self.create_zip_archive(
                individual_files, "individual_uncompressed.zip"
            )
            
            # Individual documents (compressed)
            compressed_files = [(doc["filename"], doc["content"]) for doc in compressed_documents]
            results["archives"]["individual_compressed"] = self.create_zip_archive(
                compressed_files, "individual_compressed.zip"
            )
            
            # Combined documents
            if len(processed_documents) > 1:
                # Combine uncompressed
                combined_uncompressed = await self.combine_pdfs(
                    [(doc["filename"], doc["content"]) for doc in processed_documents],
                    compress=False
                )
                results["archives"]["combined_uncompressed"] = combined_uncompressed
                
                # Combine compressed
                combined_compressed = await self.combine_pdfs(
                    [(doc["filename"], doc["content"]) for doc in processed_documents],
                    compress=True
                )
                results["archives"]["combined_compressed"] = combined_compressed
            
            logger.info(f"Workflow completed for {filename}")
            
        except Exception as e:
            logger.error(f"Error in workflow for {filename}: {e}")
            results["errors"].append(str(e))
            
        return results