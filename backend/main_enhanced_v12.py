import io
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from fastapi import FastAPI, UploadFile, File, Body, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from PyPDF2 import PdfReader, PdfWriter
import logging
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from pdf_to_png_converter import PDFToPNGConverter
from json_fixer import parse_json_safely, fix_json_string, extract_json_from_llm_response, advanced_json_reconstruction

# Import modules for v12 enhancements
from file_organizer_integration_v2 import FileOrganizerIntegrationV2
from google_drive_integration import GoogleDriveIntegration
from db_state_manager import db_state_manager

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from React build
frontend_build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "build")
if os.path.exists(frontend_build_path):
    app.mount("/static", StaticFiles(directory=frontend_build_path, html=False), name="static")
    logger.info(f"Serving static files from {frontend_build_path}")
else:
    logger.warning(f"Frontend build directory not found at {frontend_build_path}")

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# WebSocket connections for real-time updates
active_connections: Dict[str, WebSocket] = {}

# Active processing sessions that can be cancelled
active_sessions: Set[str] = set()

# Track confirmation status to prevent duplicates
session_confirmations: Dict[str, bool] = {}

# Thread pool for parallel API calls
executor = ThreadPoolExecutor(max_workers=10)

# Initialize PDF to PNG converter
png_converter = PDFToPNGConverter(use_external_service=True)

# Debug mode flag
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# Dictionary to track active batch processing
active_batches = {}

# Active document processors for status tracking
active_processors = {}

# Compatibility wrapper for processing_status_db
class ProcessingStatusDB:
    """Wrapper to make db updates compatible with existing code"""
    def __init__(self):
        self._cache = {}
        self._pending_saves = []
    
    def __setitem__(self, key, value):
        self._cache[key] = value
        
        # Check if database pool is ready
        if db_state_manager.pool is None:
            # Queue for later processing
            self._pending_saves.append((key, value))
            logger.debug(f"Queued status save for session {key} (pool not ready)")
        else:
            # Save immediately
            asyncio.create_task(self._save_status(key, value))
    
    async def _save_status(self, key, value):
        """Save status to database"""
        await db_state_manager.save_processing_status(
            key, 
            value.get('metadata', {}).get('batch_id', ''),
            value.get('status', 'processing'),
            value.get('files', []),
            value.get('extractions', {}),
            value.get('total_files', 0),
            value
        )
    
    async def process_pending(self):
        """Process all queued saves after database initialization"""
        if not self._pending_saves:
            return
            
        logger.info(f"Processing {len(self._pending_saves)} pending status saves")
        for key, value in self._pending_saves:
            await self._save_status(key, value)
        self._pending_saves.clear()
    
    def __getitem__(self, key):
        return self._cache.get(key, {})
    
    def __contains__(self, key):
        return key in self._cache
    
    def get(self, key, default=None):
        return self._cache.get(key, default)

# Global instance for compatibility
processing_status_db = ProcessingStatusDB()

# Helper functions for batch state management
async def get_batch_info(batch_id: str) -> Optional[Dict]:
    """Get batch info from database"""
    batch_state = await db_state_manager.get_batch_state(batch_id)
    if batch_state:
        # Get processing status for all sessions
        sessions = await db_state_manager.get_batch_processing_status(batch_id)
        
        # Get totals from batch metadata
        metadata = batch_state.get('metadata', {})
        total_files = metadata.get('total_files', len(sessions))
        # Use metadata completed_files if available, otherwise fall back to summing sessions
        completed_files = metadata.get('completed_files', sum(s['processed_count'] for s in sessions))
        
        return {
            "batch_id": batch_id,
            "total_files": total_files,
            "completed_files": completed_files,
            "session_ids": batch_state['session_ids'],
            "skip_individual_gdrive": metadata.get('skip_individual_gdrive', True),
            "output_folder": metadata.get('output_folder'),
            "processing_session_ids": metadata.get('processing_session_ids', []),
            "created_at": batch_state['created_at'],
            "google_drive_folder_id": metadata.get('google_drive_folder_id'),
            "case_id": metadata.get('case_id')
        }
    return None

async def create_batch(batch_id: str, total_files: int, **kwargs):
    """Create a new batch in database"""
    metadata = {
        "total_files": total_files,
        "skip_individual_gdrive": kwargs.get('skip_individual_gdrive', True),
        "output_folder": kwargs.get('output_folder'),
        "case_id": kwargs.get('case_id'),
        "google_drive_folder_id": kwargs.get('google_drive_folder_id'),
        "created_at": datetime.now().isoformat()
    }
    await db_state_manager.save_batch_state(batch_id, [], "active", metadata)
    active_batches[batch_id] = {"session_ids": [], "metadata": metadata}

async def add_session_to_batch(batch_id: str, session_id: str):
    """Add session to batch"""
    batch_state = await db_state_manager.get_batch_state(batch_id)
    if batch_state:
        session_ids = batch_state['session_ids']
        if session_id not in session_ids:
            session_ids.append(session_id)
            await db_state_manager.save_batch_state(batch_id, session_ids, batch_state['status'], batch_state['metadata'])

async def update_batch_completion(batch_id: str, session_id: str, files_completed: int = 1):
    """Update batch completion count"""
    # Get current batch state
    batch_state = await db_state_manager.get_batch_state(batch_id)
    if batch_state:
        metadata = batch_state.get('metadata', {})
        # Update the completed files count
        current_completed = metadata.get('completed_files', 0)
        metadata['completed_files'] = current_completed + files_completed
        
        # Save updated batch state
        await db_state_manager.save_batch_state(
            batch_id, 
            batch_state['session_ids'], 
            batch_state['status'],
            metadata
        )
        
        logger.info(f"Updated batch {batch_id} completion: {metadata['completed_files']}/{metadata.get('total_files', 0)} files completed")

async def delete_batch(batch_id: str):
    """Delete batch from database"""
    await db_state_manager.delete_batch_state(batch_id)

@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool and recover state"""
    await db_state_manager.initialize()
    
    # Process any pending status saves that were queued before database initialization
    await processing_status_db.process_pending()
    
    # Recover active batches from database
    active_batches_list = await db_state_manager.get_active_batches()
    for batch in active_batches_list:
        batch_id = batch['batch_id']
        active_batches[batch_id] = {
            "session_ids": batch['session_ids'],
            "metadata": batch['metadata']
        }
    
    logger.info(f"Recovered {len(active_batches)} active batches from database")
    
    # Log batch details
    for batch_id, batch_data in active_batches.items():
        logger.info(f"Recovered batch {batch_id} with {len(batch_data['session_ids'])} sessions")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup database connections"""
    await db_state_manager.close()

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Helper function for status updates
async def update_session_status(session_id: str, status_update: Dict[str, Any]):
    """Update session status via database"""
    # Get current status from DB
    current_status = await db_state_manager.get_processing_status(session_id)
    
    if not current_status:
        # Create new status - get batch_id from status_update or metadata
        batch_id = status_update.get('batch_id', status_update.get('metadata', {}).get('batch_id', ''))
        await db_state_manager.save_processing_status(
            session_id, batch_id, 'initialized',
            metadata={
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": "initialized",
                "progress": 0,
                "steps_completed": [],
                "current_step": None,
                "error": None,
                "gdrive_folder_id": None,
                "gdrive_folder_path": None,
                "files_uploaded": [],
                "case_id": None
            }
        )
        current_status = {"metadata": {}}
    
    # Update with new values
    metadata = current_status.get('metadata', {})
    if isinstance(metadata, dict):
        for key, value in status_update.items():
            metadata[key] = value
        metadata["updated_at"] = datetime.now().isoformat()
        
        # Save updated status
        await db_state_manager.save_processing_status(
            session_id, 
            current_status.get('batch_id', ''),
            metadata.get('status', current_status.get('status', 'processing')),
            current_status.get('processed_files', []),
            current_status.get('extractions', {}),
            current_status.get('total_files', 0),
            metadata
        )

class ProcessingStatus(Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    SPLITTING = "splitting"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingStep:
    step_name: str
    status: str
    message: str
    duration: Optional[float] = None
    timestamp: str = None
    progress: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class DocumentProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.processing_steps: List[ProcessingStep] = []
        self.start_time = time.time()
        self.current_plan = []
        self.completed_steps = []
        self.waiting_for_confirmation = False
        self.debug_data = {}  # Store debug information
        self.expected_value = None
        logger.info(f"Created processor for session {session_id}")
        
    async def should_cancel(self) -> bool:
        """Check if this session should be cancelled"""
        return self.session_id not in active_sessions
        
    async def send_update(self, update_type: str, data: Dict[str, Any]):
        """Send update to connected client"""
        if self.session_id in active_connections:
            try:
                await active_connections[self.session_id].send_json({
                    "type": update_type,
                    "data": data
                })
            except Exception as e:
                logger.error(f"Failed to send update to {self.session_id}: {e}")
                
    async def send_time_update(self):
        """Send elapsed time update"""
        elapsed = int(time.time() - self.start_time)
        await self.send_update("time_update", {"elapsed_seconds": elapsed})
                
    async def create_processing_plan(self, file_count: int, detected_forms: List[str]):
        """Create a generative processing plan"""
        plan_steps = [
            {"id": "detect", "name": "🔍 Analyze document types", "status": "pending"},
            {"id": "validate", "name": "✓ Validate form selection", "status": "pending"},
        ]
        
        if file_count > 1:
            plan_steps.append({"id": "queue", "name": f"📋 Queue {file_count} files for processing", "status": "pending"})
            
        plan_steps.extend([
            {"id": "split", "name": "📄 Split document into pages", "status": "pending"},
            {"id": "extract", "name": "📑 Extract data directly from PDF", "status": "pending"},
            {"id": "compile", "name": "🔄 Compile and structure results", "status": "pending"},
            {"id": "quality", "name": "🔍 Verify extraction quality", "status": "pending"},
        ])
        
        # Add value matching step if expected value is provided
        if self.expected_value:
            plan_steps.append({"id": "match", "name": "🎯 Match expected value", "status": "pending"})
        
        plan_steps.append({"id": "complete", "name": "✅ Finalize results", "status": "pending"})
        
        self.current_plan = plan_steps
        await self.send_update("processing_plan", {"steps": plan_steps})
        logger.info(f"Created processing plan with {len(plan_steps)} steps")
        
    async def update_plan_step(self, step_id: str, status: str, message: Optional[str] = None):
        """Update a step in the processing plan"""
        for step in self.current_plan:
            if step["id"] == step_id:
                step["status"] = status
                if message:
                    step["message"] = message
                logger.info(f"Step {step_id}: {status} - {message}")
                break
                
        await self.send_update("processing_plan", {"steps": self.current_plan})
        await self.send_time_update()
        
    async def send_status_update(self, step_id: str, status_message: str):
        """Send a status update for a specific step"""
        await self.send_update("status_update", {
            "step_id": step_id,
            "message": status_message
        })
        logger.debug(f"Status update for {step_id}: {status_message}")

    async def detect_form_type_with_confirmation(self, pdf_content: bytes, selected_schema: str) -> Dict[str, Any]:
        """Detect form type and ask for confirmation if needed"""
        start_time = time.time()
        await self.update_plan_step("detect", "in_progress")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            if not pdf_reader.pages:
                raise ValueError("PDF has no pages")
                
            first_page = pdf_reader.pages[0]
            
            pdf_writer = PdfWriter()
            pdf_writer.add_page(first_page)
            page_stream = io.BytesIO()
            pdf_writer.write(page_stream)
            page_stream.seek(0)
            
            file_part = types.Part.from_bytes(
                data=page_stream.getvalue(),
                mime_type="application/pdf"
            )
            
            detection_prompt = """
            Analyze this document and identify what type of IRS form or document this is.
            Look for form numbers like "941", "941-X", "1040", etc.
            
            Return a JSON object with:
            {
                "detected_form_type": "the exact form number (e.g., '941', '941-X', '1040')",
                "confidence": "high/medium/low",
                "form_title": "the full title of the form",
                "tax_year": "if visible",
                "quarter": "if it's a quarterly form"
            }
            """
            
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-05-20",
                contents=[detection_prompt, file_part],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json"
                }
            )
            
            # Use safe JSON parser for detection result
            detection_result = parse_json_safely(response.text)
            
            if not detection_result:
                logger.warning("Failed to parse detection response, using empty result")
                detection_result = {"detected_form_type": None, "confidence": "low"}
            
            detected_form = detection_result.get("detected_form_type")
            
            logger.info(f"Detected form type: {detected_form} with {detection_result.get('confidence')} confidence")
            
            # Only ask for confirmation if not already confirmed for this session
            if (selected_schema == "generic" and detected_form and 
                detected_form in ["941", "941-X", "1040", "2848", "8821"] and
                self.session_id not in session_confirmations):
                
                await self.update_plan_step("detect", "completed", f"Detected: {detected_form}")
                
                # Ask for confirmation
                await self.send_update("form_confirmation_needed", {
                    "detected_form": detected_form,
                    "form_title": detection_result.get("form_title", ""),
                    "confidence": detection_result.get("confidence", "medium"),
                    "message": f"Detected form {detected_form}. Would you like to use the specific template for better extraction?"
                })
                
                detection_result["needs_confirmation"] = True
                detection_result["selected_schema"] = selected_schema
                self.waiting_for_confirmation = True
                
            else:
                await self.update_plan_step("detect", "completed", f"Using: {selected_schema}")
                detection_result["needs_confirmation"] = False
                
            return detection_result
            
        except Exception as e:
            logger.error(f"Error detecting form type: {e}\n{traceback.format_exc()}")
            await self.update_plan_step("detect", "failed", str(e)[:100])
            return {"detected_form_type": None, "confidence": "low", "error": str(e)}

    async def split_pdf_pages(self, pdf_content: bytes) -> List[Tuple[int, bytes]]:
        """Split PDF into individual pages and return page number and content"""
        await self.update_plan_step("split", "in_progress")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"Splitting PDF into {total_pages} individual pages")
            
            pages = []
            for page_idx in range(total_pages):
                if await self.should_cancel():
                    raise Exception("Processing cancelled")
                
                await self.send_status_update("split", f"Splitting page {page_idx + 1} of {total_pages}...")
                
                # Extract single page
                pdf_writer = PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_idx])
                page_stream = io.BytesIO()
                pdf_writer.write(page_stream)
                page_stream.seek(0)
                
                pages.append((page_idx, page_stream.getvalue()))
                
                # Update progress
                progress = int((page_idx / total_pages) * 100)
                await self.send_update("progress", {
                    "message": f"Split page {page_idx + 1} of {total_pages}",
                    "percentage": progress
                })
            
            await self.update_plan_step("split", "completed", f"Split into {total_pages} pages")
            return pages
            
        except Exception as e:
            logger.error(f"Error splitting PDF: {e}\n{traceback.format_exc()}")
            await self.update_plan_step("split", "failed", str(e)[:100])
            raise

    def extract_single_page(self, page_num: int, page_content: bytes, schema: str, total_pages: int) -> Dict[str, Any]:
        """Extract data from a single page - runs in thread pool"""
        logger.info(f"Starting extraction for page {page_num + 1} of {total_pages}")
        
        # First, try direct PDF extraction
        result = self._extract_from_pdf(page_num, page_content, schema, total_pages)
        
        # Check if extraction failed or quality is too low
        if result["status"] == "error" or self._should_fallback_to_image(result):
            logger.info(f"Direct PDF extraction failed or low quality for page {page_num + 1}, falling back to image conversion")
            result = self._extract_from_image_fallback(page_num, page_content, schema, total_pages)
        
        return result
    
    def _should_fallback_to_image(self, result: Dict[str, Any]) -> bool:
        """Determine if we should fallback to image extraction based on result quality"""
        if result["status"] != "success":
            return True
            
        data = result.get("data", {})
        
        # Check for extraction errors
        if "_extraction_error" in data or "_extraction_failed" in data:
            return True
            
        # Check if data is mostly empty
        non_null_count = self._count_non_null_values(data)
        if non_null_count < 3:  # Too few fields extracted
            return True
            
        return False
    
    def _extract_from_pdf(self, page_num: int, page_content: bytes, schema: str, total_pages: int) -> Dict[str, Any]:
        """Direct extraction from PDF"""
        max_retries = 2  # Fewer retries since we have fallback
        
        for attempt in range(max_retries):
            try:
                file_part = types.Part.from_bytes(
                    data=page_content,
                    mime_type="application/pdf"
                )
                
                # Get page-specific context and schema reference
                page_context = self.get_page_specific_context(schema, page_num)
                schema_reference = self.get_schema_reference(schema, page_num)
                
                # Enhanced extraction prompt with all requirements
                extraction_prompt = f"""Extract ALL data from page {page_num + 1} of {total_pages} of IRS Form {schema}.

This is ONE PAGE of a multi-page form. Your extraction will be compiled with other pages to create the complete form data.

{page_context}

CRITICAL FORMATTING RULES:
1. Amounts should NOT have commas - format as 1000.00 not 1,000.00
2. Preserve exact decimals (e.g., 5107.23)
3. EIN format must be XX-XXXXXXX (with hyphen)
4. Checkboxes: true=checked, false=unchecked, null=not visible on this page
5. Extract EVERY visible value - do not skip fields

REFERENCE SCHEMA for page {page_num + 1}:
{schema_reference}

Return ONLY valid JSON following the schema structure. Extract all data accurately as this will be combined with other pages."""
                
                # Use structured output with huge token limits
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20",
                    contents=[extraction_prompt, file_part],
                    config={
                        "max_output_tokens": 60000,  # Huge token limit for extraction
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                        "top_p": 0.1
                    }
                )
                
                # Check if response is valid
                if not response or not response.text:
                    raise ValueError("Empty response from Gemini API")
                
                response_text = response.text.strip()
                if not response_text:
                    raise ValueError("Empty response text from Gemini API")
                
                # Parse the response with enhanced error handling
                page_data = self.parse_extraction_response(response_text, page_num + 1)
                
                # Log extraction summary if we got valid data
                if page_data and "_extraction_error" not in page_data:
                    self._log_extraction_summary(page_num + 1, page_data)
                    
                    # Save debug file if enabled
                    if DEBUG_MODE:
                        self._save_debug_extraction(page_num + 1, page_data, schema)
                    
                    # Clean extracted data
                    page_data = self.clean_extracted_data(page_data)
                
                return {
                    "page_number": page_num + 1,
                    "data": page_data,
                    "status": "success"
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_num + 1}: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying page {page_num + 1} after 1 second...")
                    time.sleep(1)  # Fixed 1s retry delay as requested
                    continue
                else:
                    logger.error(f"All attempts failed for page {page_num + 1}")
                    return {
                        "page_number": page_num + 1,
                        "data": {
                            "_page": page_num + 1,
                            "_extraction_failed": True,
                            "_error": str(e)
                        },
                        "status": "error",
                        "error": str(e),
                        "retries": attempt + 1
                    }
    
    def _extract_from_image_fallback(self, page_num: int, page_content: bytes, schema: str, total_pages: int) -> Dict[str, Any]:
        """Fallback extraction using image conversion"""
        logger.info(f"Converting page {page_num + 1} to image for extraction")
        
        try:
            # Convert PDF page to PNG using high quality
            png_bytes = asyncio.run(pdf_converter.convert_pdf_page_to_png(
                page_content, page_num, total_pages, quality="high"
            ))
            
            if not png_bytes:
                raise ValueError("Failed to convert PDF page to image")
            
            # Create image part for Gemini
            image_part = types.Part.from_bytes(
                data=png_bytes,
                mime_type="image/png"
            )
            
            # Get page-specific context and schema reference
            page_context = self.get_page_specific_context(schema, page_num)
            schema_reference = self.get_schema_reference(schema, page_num)
            
            # Enhanced extraction prompt for image
            extraction_prompt = f"""Extract ALL data from this IMAGE of page {page_num + 1} of {total_pages} of IRS Form {schema}.

This is an IMAGE of ONE PAGE of a multi-page form. Use OCR to read all text and extract data accurately.

{page_context}

CRITICAL FORMATTING RULES:
1. Amounts should NOT have commas - format as 1000.00 not 1,000.00
2. Preserve exact decimals (e.g., 5107.23)
3. EIN format must be XX-XXXXXXX (with hyphen)
4. Checkboxes: true=checked, false=unchecked, null=not visible on this page
5. Extract EVERY visible value - do not skip fields
6. This is an IMAGE so use OCR to read all text carefully

REFERENCE SCHEMA for page {page_num + 1}:
{schema_reference}

Return ONLY valid JSON following the schema structure. Use OCR to extract all data accurately."""
            
            # Use Gemini with image
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=[extraction_prompt, image_part],
                config={
                    "max_output_tokens": 60000,
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                    "top_p": 0.1
                }
            )
            
            # Check if response is valid
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API for image extraction")
            
            response_text = response.text.strip()
            if not response_text:
                raise ValueError("Empty response text from Gemini API for image extraction")
            
            # Parse the response
            page_data = self.parse_extraction_response(response_text, page_num + 1)
            
            # Log extraction summary
            if page_data and "_extraction_error" not in page_data:
                self._log_extraction_summary(page_num + 1, page_data)
                logger.info(f"Successfully extracted data from image for page {page_num + 1}")
                
                # Save debug file if enabled
                if DEBUG_MODE:
                    self._save_debug_extraction(page_num + 1, page_data, schema, suffix="_image")
                
                # Clean extracted data
                page_data = self.clean_extracted_data(page_data)
            
            return {
                "page_number": page_num + 1,
                "data": page_data,
                "status": "success",
                "extraction_method": "image_fallback"
            }
            
        except Exception as e:
            logger.error(f"Image extraction failed for page {page_num + 1}: {e}")
            return {
                "page_number": page_num + 1,
                "data": {
                    "_page": page_num + 1,
                    "_extraction_failed": True,
                    "_error": f"Image extraction failed: {str(e)}"
                },
                "status": "error",
                "error": str(e),
                "extraction_method": "image_fallback"
            }

    def get_page_specific_context(self, schema: str, page_num: int) -> str:
        """Get page-specific extraction context"""
        if schema == "941-X":
            contexts = {
                0: """FOCUS ON PAGE 1:
- Employer Info: EIN (XX-XXXXXXX format), Name, Address (complete)
- Correction Info: Quarter (1-4), Year
- Correction Type checkboxes
- Return you're correcting date
- Remember: amounts without commas (1000.00 not 1,000.00)""",
                1: """FOCUS ON PAGE 2 (Part 2):
Extract ALL lines 1-27 with THREE columns each:
- Column 1: Corrected amount (no commas: 1000.00)
- Column 2: Amount originally reported (no commas: 1000.00)
- Column 3: Difference (no commas: 1000.00)
Include ALL amounts even if 0.00""",
                2: """FOCUS ON PAGE 3 (Part 3):
Extract lines 28-40 especially:
- Line 30: Qualified wages for ERC (no commas)
- Line 31a: Qualified health plan expenses (no commas)
- Lines 33a, 34, 35a, 36a: Additional qualified wages (no commas)
- refundableEmployeeRetention under refundableCredits""",
                3: """FOCUS ON PAGE 4:
- Part 4 if present
- Signature section
- Paid preparer information""",
                4: """FOCUS ON PAGE 5:
- Any continuation data
- Additional calculations
- Supporting information""",
                5: """FOCUS ON PAGE 6:
- Instructions or additional data if form fields present"""
            }
            return contexts.get(page_num, "Extract all visible form data - amounts without commas")
        else:
            return "Extract all visible form data with complete accuracy - amounts without commas (1000.00 not 1,000.00)"

    def get_schema_reference(self, schema: str, page_num: int) -> str:
        """Get concise schema reference for the specific page"""
        if schema == "941-X":
            if page_num == 0:  # Page 1
                return """{
  "employerInfo": {
    "ein": "XX-XXXXXXX",
    "name": "string",
    "tradeName": "string",
    "address": {
      "street": "string",
      "city": "string",
      "state": "string",
      "zipCode": "string"
    }
  },
  "correctionInfo": {
    "quarter": 1-4,
    "year": "YYYY"
  },
  "correctionType": {
    "isAdjustedReturn": true/false,
    "checkbox_X": true/false
  }
}"""
            elif page_num == 1:  # Page 2
                return """{
  "lines": {
    "line1": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    "line2": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    "line3": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    ... continue for all lines 1-27 with same structure
  }
}"""
            elif page_num == 2:  # Page 3
                return """{
  "refundableCredits": {
    "refundableEmployeeRetention": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00}
  },
  "lines": {
    "line30": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    "line31a": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    "line33a": {"correctedAmount": 0.00, "originalAmount": 0.00, "difference": 0.00},
    ... continue for ERC-related lines
  }
}"""
            else:
                return "Extract following the standard 941-X structure"
        else:
            # Load actual schema if available
            try:
                schema_json = self.load_schema(schema)
                if schema_json:
                    # Return a simplified version
                    return json.dumps(self._simplify_schema(schema_json), indent=2)[:2000]
            except:
                pass
            return "Extract all form fields following standard IRS structure"

    def _simplify_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simplified schema showing structure without all the JSON Schema keywords"""
        def simplify_node(node):
            if isinstance(node, dict):
                if "properties" in node:
                    return {k: simplify_node(v) for k, v in node["properties"].items() if not k.startswith("$")}
                elif "type" in node:
                    if node["type"] == "object":
                        return {}
                    elif node["type"] == "array":
                        return []
                    elif node["type"] == "string":
                        return "string"
                    elif node["type"] in ["number", "integer"]:
                        return 0.00
                    elif node["type"] == "boolean":
                        return False
                return node
            return node
        
        return simplify_node(schema)

    def parse_extraction_response(self, response_text: str, page_num: int) -> Dict[str, Any]:
        """Parse extraction response with multiple fallback strategies"""
        try:
            # First attempt - direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for page {page_num}: {e}")
            
            # Try various recovery strategies
            strategies = [
                ("parse_json_safely", lambda: parse_json_safely(response_text, page_num)),
                ("extract_json_from_llm_response", lambda: extract_json_from_llm_response(response_text)),
                ("advanced_json_reconstruction", lambda: advanced_json_reconstruction(response_text)),
                ("fix_json_string", lambda: json.loads(fix_json_string(response_text)))
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    result = strategy_func()
                    if result and not result.get("_parse_error"):
                        logger.info(f"Successfully parsed using {strategy_name} for page {page_num}")
                        return result
                except Exception as strategy_error:
                    logger.debug(f"{strategy_name} failed: {strategy_error}")
                    continue
            
            # All strategies failed
            logger.error(f"All JSON parsing attempts failed for page {page_num}")
            return {"_extraction_error": "JSON parse failed", "_page": page_num}

    def _log_extraction_summary(self, page_num: int, data: Dict[str, Any]):
        """Log what was extracted from a page"""
        fields_extracted = []
        nulls_found = []
        
        def analyze_dict(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not key.startswith('_'):
                        full_key = f"{prefix}{key}" if prefix else key
                        if value is None or value == "" or value == "null":
                            nulls_found.append(full_key)
                        elif isinstance(value, dict):
                            analyze_dict(value, f"{full_key}.")
                        elif isinstance(value, list) and len(value) > 0:
                            fields_extracted.append(f"{full_key}[{len(value)} items]")
                        else:
                            fields_extracted.append(f"{full_key}: {str(value)[:50]}")
        
        analyze_dict(data)
        
        logger.info(f"Page {page_num} extraction summary:")
        logger.info(f"  - Fields with values: {len(fields_extracted)}")
        logger.info(f"  - Null/empty fields: {len(nulls_found)}")
        
        if fields_extracted:
            logger.info(f"  - Sample extracted: {', '.join(fields_extracted[:5])}")
        if nulls_found and len(nulls_found) < 20:
            logger.info(f"  - Null fields: {', '.join(nulls_found[:10])}")

    def _save_debug_extraction(self, page_num: int, data: Dict[str, Any], schema: str, suffix: str = ""):
        """Save debug extraction data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_extractions"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"{debug_dir}/{schema}_page{page_num}_{timestamp}{suffix}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "page": page_num,
                    "schema": schema,
                    "timestamp": timestamp,
                    "extracted_data": data,
                    "field_count": self._count_fields(data),
                    "non_null_count": self._count_non_null_values(data)
                }, f, indent=2)
            
            logger.debug(f"Saved debug extraction to {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug extraction: {e}")

    async def parallel_extract_pages(self, pages: List[Tuple[int, bytes]], schema: str) -> List[Dict[str, Any]]:
        """Extract data from all pages in parallel"""
        await self.update_plan_step("extract", "in_progress", "Extracting directly from PDF...")
        total_pages = len(pages)
        
        await self.send_status_update("extract", f"Launching parallel extraction for {total_pages} pages...")
        
        # Create futures for all pages
        futures = []
        for page_num, page_content in pages:
            future = executor.submit(
                self.extract_single_page,
                page_num,
                page_content,
                schema,
                total_pages
            )
            futures.append(future)
        
        # Wait for all extractions to complete
        results = []
        completed = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Log the result for each page
            page_num = result["page_number"]
            status = result["status"]
            extraction_method = result.get("extraction_method", "pdf")
            
            if status == "error":
                logger.warning(f"Page {page_num} extraction failed: {result.get('error', 'Unknown error')}")
            else:
                if extraction_method == "image_fallback":
                    logger.info(f"Page {page_num} extracted successfully (via image conversion)")
                    await self.send_status_update("extract", 
                        f"Page {page_num} required image conversion")
                else:
                    logger.info(f"Page {page_num} extracted successfully (direct PDF)")
            
            await self.send_status_update("extract", 
                f"Completed {completed} of {total_pages} pages")
            
            # Update progress
            progress = int((completed / total_pages) * 100)
            await self.send_update("progress", {
                "message": f"Extracted {completed} of {total_pages} pages",
                "percentage": progress
            })
        
        # Sort results by page number
        results.sort(key=lambda x: x["page_number"])
        
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        
        await self.update_plan_step("extract", "completed", 
            f"Extracted {successful} of {total_pages} pages")
        
        return results

    async def compile_results_with_pro_v2(self, page_results: List[Dict[str, Any]], schema: str) -> Dict[str, Any]:
        """Enhanced compilation using Gemini Pro with full schema and intelligent restructuring"""
        try:
            await self.update_plan_step("compile", "in_progress")
            logger.info(f"Starting enhanced compilation for {len(page_results)} pages")
            
            # Extract data from all pages
            all_page_data = []
            failed_pages = []
            
            for result in page_results:
                if result["status"] == "success" and result.get("data"):
                    all_page_data.append({
                        "page": result["page_number"],
                        "data": result["data"]
                    })
                else:
                    failed_pages.append(result["page_number"])
            
            if not all_page_data:
                logger.error("No successful page extractions to compile")
                raise Exception("No data to compile")
            
            logger.info(f"Compiling data from {len(all_page_data)} successful pages")
            
            # Load full schema for reference
            schema_json = self.load_schema(schema) or {}
            
            # Create enhanced compilation prompt with full schema
            compilation_prompt = f"""You are compiling extracted data from {len(all_page_data)} pages of IRS Form {schema} into a single, complete JSON object.

FULL SCHEMA REFERENCE:
{json.dumps(schema_json, indent=2)[:100000]}  # Huge limit for schema

PAGE DATA TO MERGE:
{json.dumps(all_page_data, indent=2)}

INTELLIGENT RESTRUCTURING RULES:
1. Combine all page data into one unified structure following the schema exactly
2. When the same field appears on multiple pages:
   - Use the non-null, non-empty value
   - If multiple valid values exist, use the most complete one
   - For amounts, ensure no commas (1000.00 not 1,000.00)
3. Preserve ALL extracted values - do not discard any data
4. Maintain the exact schema structure and field names
5. Remove any page-specific metadata (like _page)
6. Ensure EIN format is XX-XXXXXXX
7. For 941-X: Ensure all line items have correctedAmount, originalAmount, and difference
8. Clean up any extraction artifacts or schema remnants

Return ONLY the merged JSON object that perfectly matches the schema structure."""

            logger.info(f"Sending enhanced compilation request (prompt length: {len(compilation_prompt)} chars)")
            
            # Use Gemini Pro with huge token limits
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-pro-preview-06-05",  # Using Pro as requested
                    contents=[compilation_prompt],
                    config={
                        "max_output_tokens": 100000,  # Huge output limit
                        "response_mime_type": "application/json",
                        "temperature": 0.0
                    }
                ),
                timeout=300.0  # 5 minute timeout for large compilations
            )
            
            if response and hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
                logger.info(f"Received compilation response: {len(response_text)} chars")
                
                # Parse the response
                try:
                    compiled_data = json.loads(response_text)
                    logger.info("Successfully parsed compilation response")
                    
                    # Log post-compilation data
                    if DEBUG_MODE:
                        self._save_debug_compilation("post_compilation_v2", compiled_data, schema)
                    
                    # Clean up any remaining schema artifacts
                    compiled_data = self.clean_extracted_data(compiled_data)
                    
                    # Final validation of number formatting
                    compiled_data = self._ensure_number_formatting(compiled_data)
                    
                    await self.update_plan_step("compile", "completed", 
                        f"Intelligently compiled {len(all_page_data)} pages")
                    return compiled_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error in compilation: {e}")
                    # Try recovery with Gemini Pro
                    return await self._recover_compilation_with_pro(response_text, all_page_data, schema)
            else:
                logger.error("Empty response from compilation")
                raise Exception("Empty compilation response")
                
        except asyncio.TimeoutError:
            logger.error("Compilation timeout")
            await self.update_plan_step("compile", "failed", "Timeout")
            # Fallback to simple merge
            return self._simple_merge_fallback(all_page_data)
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            await self.update_plan_step("compile", "failed", str(e))
            # Fallback to simple merge
            return self._simple_merge_fallback(all_page_data)

    async def _recover_compilation_with_pro(self, malformed_json: str, page_data: List[Dict], schema: str) -> Dict[str, Any]:
        """Use Gemini Pro to fix malformed JSON compilation"""
        logger.info("Attempting to recover malformed compilation with Gemini Pro")
        
        recovery_prompt = f"""Fix this malformed JSON and ensure it matches the {schema} schema structure:

MALFORMED JSON:
{malformed_json[:50000]}

REQUIREMENTS:
1. Fix any JSON syntax errors
2. Ensure all amounts have no commas (1000.00 not 1,000.00)
3. Ensure EIN format is XX-XXXXXXX
4. Return valid JSON only

Return ONLY the fixed JSON."""

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-pro-preview-06-05",
                contents=[recovery_prompt],
                config={
                    "max_output_tokens": 100000,
                    "response_mime_type": "application/json",
                    "temperature": 0.0
                }
            )
            
            if response and response.text:
                return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
        
        # Final fallback
        return self._simple_merge_fallback(page_data)

    def _simple_merge_fallback(self, page_data: List[Dict]) -> Dict[str, Any]:
        """Simple merge strategy as final fallback"""
        logger.info("Using simple merge fallback strategy")
        merged = {}
        for page_info in page_data:
            if isinstance(page_info.get("data"), dict):
                merged = self.deep_merge(merged, page_info["data"])
        return self._ensure_number_formatting(merged)

    def _ensure_number_formatting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all numbers are formatted without commas"""
        def format_value(value):
            if isinstance(value, str) and re.match(r'^\d{1,3}(,\d{3})*(\.\d+)?$', value):
                # Remove commas from number strings
                return value.replace(',', '')
            elif isinstance(value, dict):
                return {k: format_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [format_value(item) for item in value]
            return value
        
        return format_value(data)

    def _save_debug_compilation(self, stage: str, data: Any, schema: str):
        """Save debug compilation data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_extractions"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"{debug_dir}/{schema}_{stage}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "stage": stage,
                    "schema": schema,
                    "timestamp": timestamp,
                    "data": data
                }, f, indent=2)
            
            logger.debug(f"Saved {stage} debug data to {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug compilation: {e}")

    async def process_with_parallel_extraction(self, file_content: bytes, schema: str, filename: str) -> List[Dict[str, Any]]:
        """Process document with parallel page extraction"""
        results = []
        
        try:
            # Split PDF into pages
            pages = await self.split_pdf_pages(file_content)
            
            # Extract from all pages in parallel
            page_results = await self.parallel_extract_pages(pages, schema)
            
            # Use enhanced Pro compilation
            compiled_json = await self.compile_results_with_pro_v2(page_results, schema)
            
            # Validate extraction
            await self.update_plan_step("quality", "in_progress")
            validation_result = self.validate_extraction_result_v2(compiled_json, schema)
            quality = validation_result["quality_score"]
            
            # Create detailed quality message
            quality_message = f"Quality: {quality:.0f}%"
            if validation_result.get("expected_nulls_info"):
                quality_message += f" ({validation_result['expected_nulls_info']})"
            
            if validation_result["math_validation_passed"]:
                await self.update_plan_step("quality", "completed", quality_message)
            else:
                errors_preview = "; ".join(validation_result["math_errors"][:3])
                await self.update_plan_step("quality", "completed", 
                    f"{quality_message} (Math issues: {errors_preview}...)")
            
            # Check which extraction methods were used
            pdf_extractions = sum(1 for r in page_results if r.get("extraction_method") != "image_fallback")
            image_extractions = sum(1 for r in page_results if r.get("extraction_method") == "image_fallback")
            
            extraction_method_summary = "direct_pdf"
            if image_extractions > 0 and pdf_extractions > 0:
                extraction_method_summary = f"mixed (PDF: {pdf_extractions}, Image: {image_extractions})"
            elif image_extractions > 0:
                extraction_method_summary = "image_conversion"
            
            # Add metadata
            compiled_json["_metadata"] = {
                "filename": filename,
                "total_pages": len(pages),
                "extracted_pages": len([r for r in page_results if r["status"] == "success"]),
                "selected_schema": schema,
                "quality_score": quality,
                "populated_fields": validation_result["populated_count"],
                "total_fields": validation_result["total_fields"],
                "expected_nulls": validation_result.get("expected_null_count", 0),
                "math_validation_passed": validation_result["math_validation_passed"],
                "processing_time": time.time() - self.start_time,
                "extraction_method": extraction_method_summary,
                "pages_converted_to_images": image_extractions
            }
            
            results.append(compiled_json)
            
            await self.update_plan_step("complete", "completed")
            
        except Exception as e:
            logger.error(f"Error in parallel document processing: {e}\n{traceback.format_exc()}")
            if "cancelled" in str(e).lower():
                await self.update_plan_step("complete", "cancelled", "Processing stopped by user")
            else:
                await self.update_plan_step("complete", "failed", str(e)[:100])
            raise
            
        return results

    def validate_extraction_result_v2(self, json_data: Dict[str, Any], form_type: str) -> Dict[str, Any]:
        """Enhanced validation that accounts for expected nulls"""
        
        # Get expected null fields for this form type
        expected_nulls = self.get_expected_null_fields(form_type)
        
        def count_values(obj, path=""):
            null_count = 0
            expected_null_count = 0
            zero_count = 0
            total_count = 0
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not key.startswith('_'):  # Skip metadata
                        current_path = f"{path}.{key}" if path else key
                        n, en, z, t = count_values(value, current_path)
                        null_count += n
                        expected_null_count += en
                        zero_count += z
                        total_count += t
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    n, en, z, t = count_values(item, f"{path}[{idx}]")
                    null_count += n
                    expected_null_count += en
                    zero_count += z
                    total_count += t
            else:
                total_count = 1
                if obj is None or obj == "" or obj == "null":
                    null_count = 1
                    # Check if this null is expected
                    if path in expected_nulls:
                        expected_null_count = 1
                elif obj == 0 or obj == "0" or obj == 0.0:
                    zero_count = 1
                    
            return null_count, expected_null_count, zero_count, total_count
        
        null_count, expected_null_count, zero_count, total_count = count_values(json_data)
        
        # Calculate quality score accounting for expected nulls
        unexpected_nulls = null_count - expected_null_count
        empty_fields = unexpected_nulls + zero_count
        
        # Quality calculation that doesn't penalize expected nulls
        if total_count > 0:
            populated_fields = total_count - empty_fields
            quality_score = (populated_fields / total_count) * 100
        else:
            quality_score = 0
        
        # Check critical fields
        has_critical = False
        math_validation_passed = True
        math_errors = []
        
        if form_type == "941-X":
            # Critical field check
            if "employerInfo" in json_data:
                has_critical = bool(json_data.get("employerInfo", {}).get("ein"))
            
            # Run mathematical validation
            math_validation_passed, math_errors = self.validate_941x_mathematics(json_data)
            
        elif form_type == "941" and "employerInfo" in json_data:
            has_critical = bool(json_data.get("employerInfo", {}).get("ein"))
        
        return {
            "is_valid": quality_score >= 30 and math_validation_passed,  # Lower threshold
            "quality_score": quality_score,
            "empty_percentage": (empty_fields / total_count * 100) if total_count > 0 else 0,
            "populated_count": total_count - empty_fields,
            "total_fields": total_count,
            "expected_null_count": expected_null_count,
            "unexpected_null_count": unexpected_nulls,
            "has_critical_fields": has_critical,
            "math_validation_passed": math_validation_passed,
            "math_errors": math_errors,
            "expected_nulls_info": f"{expected_null_count} fields expected empty"
        }

    def get_expected_null_fields(self, form_type: str) -> Set[str]:
        """Get fields that are expected to be null for a form type"""
        expected_nulls = set()
        
        if form_type == "941-X":
            # Fields that might not be used in all corrections
            expected_nulls.update([
                "lines.line11d", "lines.line11e", "lines.line11f",  # Sick leave credits
                "lines.line21", "lines.line22", "lines.line23",  # COBRA credits
                "lines.line24", "lines.line25",  # Special government credits
                "part4",  # Part 4 might be empty
                "signatureSection.paidPreparer",  # Might not have paid preparer
            ])
            
        elif form_type == "941":
            expected_nulls.update([
                "part3",  # Often not applicable
                "schedule_b",  # Only for semiweekly depositors
                "lines.line11c", "lines.line11d",  # COVID credits
            ])
            
        return expected_nulls

    def validate_941x_mathematics(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Comprehensive mathematical validation for Form 941-X"""
        errors = []
        
        def get_line_value(line_num: str, column: str = "difference") -> float:
            """Safely get a line value from the data"""
            try:
                lines = data.get("lines", {})
                line_data = lines.get(f"line{line_num}", {})
                if isinstance(line_data, dict):
                    if column == "difference":
                        return float(line_data.get("difference", 0) or 0)
                    elif column == "corrected":
                        return float(line_data.get("correctedAmount", 0) or 0)
                    elif column == "original":
                        return float(line_data.get("originalAmount", 0) or 0)
                return 0.0
            except:
                return 0.0
        
        def get_quarter_year() -> tuple[int, int]:
            """Get quarter and year from the form"""
            try:
                quarter = int(data.get("correctionInfo", {}).get("quarter", 0))
                year = int(data.get("correctionInfo", {}).get("year", 0))
                return quarter, year
            except:
                return 0, 0
        
        # 1. ERC Total Coherence
        line_18a = get_line_value("18a")
        line_26a = get_line_value("26a")
        total_erc = line_18a + line_26a
        
        # 2. Quarter-specific ERC rate validation
        quarter, year = get_quarter_year()
        if year == 2020 and quarter in [2, 3, 4]:
            qualified_wages = (get_line_value("30") + get_line_value("31a") + 
                             get_line_value("33a") + get_line_value("34"))
            if qualified_wages > 0:
                erc_rate = total_erc / qualified_wages
                if abs(erc_rate - 0.50) > 0.01:  # 1% tolerance
                    errors.append(f"ERC rate {erc_rate:.2%} != 50% for 2020 Q{quarter}")
        
        elif year == 2021 and quarter in [1, 2, 3, 4]:
            qualified_wages = get_line_value("30") + get_line_value("31a")
            if qualified_wages > 0:
                erc_rate = total_erc / qualified_wages
                if abs(erc_rate - 0.70) > 0.01:  # 1% tolerance
                    errors.append(f"ERC rate {erc_rate:.2%} != 70% for 2021 Q{quarter}")
        
        # Return validation results
        passed = len(errors) == 0
        if not passed:
            logger.warning(f"941-X mathematical validation failed with {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
        
        return passed, errors

    def load_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Load schema from file"""
        schema_mapping = {
            "1040": "1040.json",
            "2848": "2848.json", 
            "8821": "8821.json",
            "941": "941.json",
            "941-X": "941-X.json",
            "payroll": "payroll.json",
            "generic": None
        }
        
        filename = schema_mapping.get(schema_id)
        if not filename:
            return None
            
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            schema_path = os.path.join(project_root, filename)
            
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema {schema_id}: {e}")
            return None
    
    def clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove schema artifacts from extracted data"""
        if not isinstance(data, dict):
            return data
            
        cleaned = {}
        for key, value in data.items():
            # Skip schema keywords
            if key in ["type", "properties", "description", "$schema", "required", "const", "enum"]:
                continue
                
            # Recursively clean nested structures
            if isinstance(value, dict):
                # Check if this looks like a schema node
                if "type" in value and "properties" in value:
                    # Try to extract actual values from properties
                    if isinstance(value.get("properties"), dict):
                        cleaned[key] = self.clean_extracted_data(value["properties"])
                    else:
                        continue
                else:
                    cleaned_value = self.clean_extracted_data(value)
                    if cleaned_value:  # Only include non-empty dicts
                        cleaned[key] = cleaned_value
            elif isinstance(value, list):
                cleaned[key] = [self.clean_extracted_data(item) if isinstance(item, dict) else item for item in value]
            else:
                # Include actual values (not schema type names)
                if value not in ["string", "number", "integer", "boolean", "array", "object"]:
                    cleaned[key] = value
                    
        return cleaned
    
    def _count_non_null_values(self, data: Dict[str, Any]) -> int:
        """Count non-null values in a dictionary"""
        count = 0
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, dict):
                    count += self._count_non_null_values(value)
                elif value is not None and value != "" and value != [] and value != {}:
                    count += 1
        return count
    
    def deep_merge(self, base: Dict, addition: Dict) -> Dict:
        """Deep merge dictionaries"""
        result = base.copy()
        
        for key, value in addition.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            elif result[key] is None and value is not None:
                result[key] = value
                
        return result
    
    def _count_fields(self, data: Dict[str, Any], prefix: str = "") -> int:
        """Count total number of fields in a nested dictionary"""
        count = 0
        if isinstance(data, dict):
            for key, value in data.items():
                if not key.startswith('_'):  # Skip metadata fields
                    if isinstance(value, dict):
                        count += self._count_fields(value, f"{prefix}{key}.")
                    elif isinstance(value, list):
                        count += 1  # Count list as one field
                        for item in value:
                            if isinstance(item, dict):
                                count += self._count_fields(item, f"{prefix}{key}[].")
                    else:
                        count += 1
        return count

    async def match_expected_value(self, extracted_data: Dict[str, Any], expected_value: str) -> Dict[str, Any]:
        """Use AI to match expected value against extracted data"""
        try:
            # For 941-X, we specifically look for refundable employee retention credit
            matching_prompt = f"""
You are comparing an expected Employee Retention Credit (ERC) value with extracted data from a 941-X form.

Expected Value: ${expected_value}

Extracted Data (relevant sections):
{json.dumps(extracted_data, indent=2)}

Task: Find if the expected value matches any ERC amount in the extracted data.
Focus on:
1. refundableEmployeeRetention under refundableCredits
2. The "difference" field (which represents the credit amount)
3. Consider both exact matches and close matches (within $0.01 due to rounding)

Return a JSON response:
{{
    "is_match": true/false,
    "expected_value": "{expected_value}",
    "found_value": "the actual value found or null",
    "field_path": "path to the field where found (e.g., refundableCredits.refundableEmployeeRetention.difference)",
    "confidence": "high/medium/low",
    "explanation": "brief explanation of the match or why no match was found"
}}
"""
            
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-05-20",
                contents=[matching_prompt],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json",
                    "temperature": 0.0
                }
            )
            
            if response and hasattr(response, 'text') and response.text:
                match_result = json.loads(response.text.strip())
                logger.info(f"Value match result: {match_result}")
                return match_result
            else:
                logger.error("Empty response from value matching")
                return {
                    "is_match": False,
                    "expected_value": expected_value,
                    "found_value": None,
                    "confidence": "low",
                    "explanation": "Unable to perform matching due to API error"
                }
                
        except Exception as e:
            logger.error(f"Error in value matching: {e}")
            return {
                "is_match": False,
                "expected_value": expected_value,
                "found_value": None,
                "confidence": "low",
                "explanation": f"Error during matching: {str(e)}"
            }

class EnhancedDocumentProcessor(DocumentProcessor):
    """Enhanced processor with file organizer integration"""
    
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.file_organizer = FileOrganizerIntegrationV2()
        self.workflow_results = {}
        # Use absolute path for output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "processed_documents", session_id)
        
        # Add logging for directory creation
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if os.path.exists(self.output_dir):
                logger.info(f"Created/verified output directory: {self.output_dir}")
                logger.info(f"Absolute path: {os.path.abspath(self.output_dir)}")
            else:
                logger.error(f"Failed to create output directory: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory {self.output_dir}: {e}")
            raise
            
        self._timer_task = None
        self._timer_running = False
        self.step_timings = {}
        self.batch_id = None  # Will be set for batch processing
        
    async def send_elapsed_time_updates(self, start_time: float):
        """Send elapsed time updates every second"""
        self._timer_running = True
        try:
            while self._timer_running:
                elapsed = time.time() - start_time
                await self.send_update("time_update", {"elapsed_seconds": int(elapsed)})
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Timer error: {e}")
    
    def start_step_timer(self, step_name: str):
        """Start timing a specific step"""
        self.step_timings[step_name] = {"start": time.time()}
    
    def end_step_timer(self, step_name: str):
        """End timing a specific step"""
        if step_name in self.step_timings:
            self.step_timings[step_name]["end"] = time.time()
            self.step_timings[step_name]["duration"] = self.step_timings[step_name]["end"] - self.step_timings[step_name]["start"]
            return self.step_timings[step_name]["duration"]
        return 0
    
    async def create_workflow_plan(self, file_count: int, enable_google_drive: bool = True, skip_individual_gdrive: bool = False):
        """Create enhanced workflow plan including cleaning and organization"""
        plan_steps = [
            {"id": "upload", "name": "📤 Upload documents", "status": "completed"},
            {"id": "clean", "name": "🧹 Clean documents (remove instruction pages)", "status": "pending"},
            {"id": "split", "name": "✂️ Split into individual documents", "status": "pending"},
            {"id": "extract", "name": "📊 Extract data from each document", "status": "pending"},
            {"id": "organize", "name": "📁 Organize by quarter", "status": "pending"},
            {"id": "compress", "name": "🗜️ Create compressed versions", "status": "pending"},
            {"id": "archive", "name": "📦 Create archives", "status": "pending"},
        ]
        
        if enable_google_drive and not skip_individual_gdrive:
            plan_steps.append({"id": "google_drive", "name": "☁️ Upload to Google Drive", "status": "pending"})
        
        plan_steps.append({"id": "complete", "name": "✅ Process complete", "status": "pending"})
        
        self.current_plan = plan_steps
        await self.send_update("workflow_plan", {"steps": plan_steps})
        logger.info(f"Created enhanced workflow plan with {len(plan_steps)} steps")
        
    async def process_with_file_organizer(self, pdf_content: bytes, filename: str, 
                                        selected_schema: str, expected_value: Optional[str] = None,
                                        target_size_mb: float = 9.0, skip_individual_gdrive: bool = False):
        """Process document through complete workflow with file organizer"""
        try:
            # Start workflow and timer
            start_time = time.time()
            await self.create_workflow_plan(1, enable_google_drive=True, skip_individual_gdrive=skip_individual_gdrive)
            
            # Send initial timer
            asyncio.create_task(self.send_elapsed_time_updates(start_time))
            
            # Save original upload
            original_dir = os.path.join(self.output_dir, "original")
            os.makedirs(original_dir, exist_ok=True)
            original_path = os.path.join(original_dir, filename)
            with open(original_path, 'wb') as f:
                f.write(pdf_content)
            logger.info(f"Saved original upload: {original_path}")
            
            # Use the new integrated workflow
            async with self.file_organizer:
                # Step 1 & 2: Clean and analyze the document (combined)
                self.start_step_timer("cleaning")
                await self.update_plan_step("clean", "in_progress", "Cleaning and analyzing document...")
                
                workflow_result = await self.file_organizer.process_document_workflow(
                    pdf_content, 
                    filename, 
                    self.output_dir,
                    expected_value
                )
                
                if not workflow_result["success"]:
                    raise Exception(f"Workflow failed: {workflow_result.get('errors', ['Unknown error'])}")
                
                # Update cleaning status
                cleaning_duration = self.end_step_timer("cleaning")
                cleaning_report = workflow_result["cleaning_report"]
                await self.send_update("cleaning_complete", {
                    "removed_pages": cleaning_report.get("removed_pages", 0),
                    "total_pages": cleaning_report.get("total_pages", 0),
                    "kept_pages": cleaning_report.get("kept_pages", 0)
                })
                await self.update_plan_step("clean", "completed", 
                                          f"Removed {cleaning_report.get('removed_pages', 0)} pages ({cleaning_duration:.1f}s)")
                
                # Update splitting status
                self.start_step_timer("splitting")
                split_documents = workflow_result["documents"]
                await self.update_plan_step("split", "in_progress", "Analyzing document structure...")
                
                splitting_duration = self.end_step_timer("splitting")
                await self.update_plan_step("split", "completed", f"Split into {len(split_documents)} documents ({splitting_duration:.1f}s)")
                
                # Step 3: Extract data from each document in parallel
                self.start_step_timer("extraction")
                await self.update_plan_step("extract", "in_progress", f"Extracting data from {len(split_documents)} documents in parallel...")
                
                # Prepare extraction plan
                logger.info(f"Extraction plan: {len(split_documents)} documents will be processed in parallel")
                for idx, doc in enumerate(split_documents):
                    logger.info(f"  Document {idx + 1}: {doc['filename']} ({len(doc['content'])} bytes)")
                
                # Parallel extraction
                extracted_documents = []
                total_docs = len(split_documents)
                
                # Send start status
                await self.send_update("extraction_started", {
                    "total_documents": total_docs,
                    "message": f"Starting parallel extraction of {total_docs} documents..."
                })
                
                logger.info(f"Starting parallel extraction of {total_docs} documents...")
                
                # Process all documents in parallel
                extraction_tasks = []
                for idx, doc in enumerate(split_documents):
                    task = asyncio.create_task(
                        self.extract_document_with_tracking(doc, selected_schema, expected_value, idx)
                    )
                    extraction_tasks.append(task)
                
                # Wait for all extractions
                extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                
                # Process results
                for idx, result in enumerate(extraction_results):
                    if isinstance(result, Exception):
                        logger.error(f"Document {idx + 1} extraction failed: {result}")
                        # Create error result
                        extracted_documents.append({
                            "filename": split_documents[idx]["filename"],
                            "content": split_documents[idx]["content"],
                            "extracted_data": {"_extraction_error": str(result)},
                            "metadata": {"extraction_failed": True, "error": str(result)}
                        })
                    else:
                        # Ensure the result has the PDF content from the original document
                        result["content"] = split_documents[idx]["content"]
                        extracted_documents.append(result)
                
                extraction_duration = self.end_step_timer("extraction")
                successful_extractions = len([d for d in extracted_documents if not d.get("metadata", {}).get("extraction_failed", False)])
                await self.update_plan_step("extract", "completed", 
                                          f"Extracted data from {successful_extractions}/{total_docs} documents ({extraction_duration:.1f}s)")
                
                # Step 4: Organize by quarter
                self.start_step_timer("organizing")
                await self.update_plan_step("organize", "in_progress", "Organizing documents by quarter...")
                
                organized_paths = self.organize_by_quarter_with_metadata(extracted_documents, self.output_dir)
                organizing_duration = self.end_step_timer("organizing")
                
                quarters_count = len(organized_paths)
                await self.update_plan_step("organize", "completed", f"Organized into {quarters_count} quarters ({organizing_duration:.1f}s)")
                
                # Step 5: Create compressed versions
                self.start_step_timer("compression")
                await self.update_plan_step("compress", "in_progress", f"Creating compressed versions for {len(extracted_documents)} documents in parallel...")
                
                # Calculate target size per document
                total_size_mb = sum(len(doc["content"]) for doc in extracted_documents) / (1024 * 1024)
                per_doc_target_mb = target_size_mb / len(extracted_documents) if len(extracted_documents) > 0 else target_size_mb
                
                logger.info(f"Total size: {total_size_mb:.2f}MB, Target: {target_size_mb}MB, Per-doc target: {per_doc_target_mb:.2f}MB")
                
                # Initialize failed_count for tracking compression failures
                failed_count = 0
                
                # Check if compression is needed
                if total_size_mb <= target_size_mb:
                    logger.info(f"Total size {total_size_mb:.2f}MB is already within target {target_size_mb}MB, skipping compression")
                    await self.update_plan_step("compress", "completed", f"Compression not needed (already within {target_size_mb}MB target) ({0:.1f}s)")
                    
                    # Create "compressed" versions even when skipping compression
                    # This ensures the compressed folder has files
                    compressed_documents = []
                    for doc in extracted_documents:
                        # Create a copy with _compressed suffix
                        compressed_filename = doc["filename"].replace(".pdf", "_compressed.pdf")
                        compressed_doc = {
                            **doc,
                            "filename": compressed_filename,
                            "compressed": True,
                            "compression_ratio": 1.0  # No compression, same size
                        }
                        compressed_documents.append(compressed_doc)
                    
                    # Save compressed versions to disk even when compression is skipped
                    # This ensures files are available for batch collection
                    for compressed_doc in compressed_documents:
                        # Save compressed PDF to disk
                        compressed_path = os.path.join(self.output_dir, compressed_doc["filename"])
                        with open(compressed_path, 'wb') as f:
                            f.write(compressed_doc["content"])
                        logger.info(f"Saved compressed file (no compression): {compressed_path}")
                        
                        # Also organize the compressed version by quarter
                        self.organize_by_quarter_with_metadata([compressed_doc], self.output_dir)
                else:
                    # Compress documents in parallel
                    compression_tasks = []
                    for doc in extracted_documents:
                        task = asyncio.create_task(
                            self.file_organizer.compress_pdf_with_monitoring(
                                doc["content"],
                                doc["filename"],
                                per_doc_target_mb
                            )
                        )
                        compression_tasks.append(task)
                    
                    compression_results = await asyncio.gather(*compression_tasks, return_exceptions=True)
                    
                    # Process compression results
                    compressed_documents = []
                    failed_count = 0
                    
                    for idx, (doc, result) in enumerate(zip(extracted_documents, compression_results)):
                        if isinstance(result, Exception):
                            logger.error(f"Compression failed for {doc['filename']}: {result}")
                            # Use original if compression failed but still create a _compressed version
                            compressed_filename = doc["filename"].replace(".pdf", "_compressed.pdf")
                            compressed_doc = {
                                **doc,
                                "filename": compressed_filename,
                                "compressed": False,
                                "compression_error": str(result),
                                "compression_ratio": 1.0  # No compression, same size
                            }
                            compressed_documents.append(compressed_doc)
                            
                            # Save the file with _compressed suffix even though it's not actually compressed
                            compressed_path = os.path.join(self.output_dir, compressed_filename)
                            with open(compressed_path, 'wb') as f:
                                f.write(doc["content"])
                            logger.info(f"Saved uncompressed file as compressed (compression failed): {compressed_path}")
                            
                            failed_count += 1
                        else:
                            compressed_content = result["compressed_content"]
                            compression_ratio = result["compression_ratio"]
                            logger.info(f"Compressed {doc['filename']}: {compression_ratio:.1%} of original")
                            
                            # Organize compressed file
                            compressed_filename = doc["filename"].replace(".pdf", "_compressed.pdf")
                            compressed_doc = {
                                **doc,
                                "content": compressed_content,
                                "filename": compressed_filename,
                                "compressed": True,
                                "compression_ratio": compression_ratio
                            }
                            compressed_documents.append(compressed_doc)
                            
                            # Save compressed PDF to disk
                            compressed_path = os.path.join(self.output_dir, compressed_filename)
                            with open(compressed_path, 'wb') as f:
                                f.write(compressed_content)
                            logger.info(f"Saved compressed file: {compressed_path}")
                            
                            # Also organize the compressed version
                            if doc.get("extracted_data"):
                                self.organize_by_quarter_with_metadata([compressed_doc], self.output_dir)
                
                compression_duration = self.end_step_timer("compression")
                
                if failed_count > 0:
                    await self.update_plan_step("compress", "completed", 
                                              f"Created {len(compressed_documents) - failed_count}/{len(compressed_documents)} compressed versions ({compression_duration:.1f}s)")
                else:
                    await self.update_plan_step("compress", "completed", 
                                              f"Created {len(compressed_documents)} compressed versions ({compression_duration:.1f}s)")
                
                # Step 6: Create archives (prepare for upload)
                self.start_step_timer("archiving")
                await self.update_plan_step("archive", "in_progress", "Preparing files for upload...")
                
                # Prepare files for upload
                files_to_upload = {
                    "original": [],
                    "cleaned": [],
                    "individual_uncompressed": [],
                    "individual_compressed": [],
                    "combined_uncompressed": None,
                    "combined_compressed": None,
                    "all_extractions": None
                }
                
                # Add individual documents
                for doc in extracted_documents:
                    if doc.get("content"):
                        # Determine the original filename
                        if "cleaned_" in doc["filename"]:
                            pdf_filename = doc["filename"]
                        else:
                            pdf_filename = doc["filename"]
                        
                        # Store PDF reference
                        files_to_upload["individual_uncompressed"].append({
                            "filename": pdf_filename,
                            "content": doc["content"],
                            "type": "pdf"
                        })
                        
                        # Store JSON extraction
                        if "extracted_data" in doc:
                            if doc["filename"].endswith("_extraction.json"):
                                json_filename = doc["filename"]
                            else:
                                json_filename = pdf_filename.replace(".pdf", "_extraction.json")
                            json_content = json.dumps(doc["extracted_data"], indent=2).encode('utf-8')
                            files_to_upload["individual_uncompressed"].append({
                                "filename": json_filename,
                                "content": json_content,
                                "type": "json"
                            })
                
                # Add compressed documents
                for doc in compressed_documents:
                    if doc.get("compressed", False) and doc.get("content"):
                        # Determine the compressed filename
                        if "_compressed" in doc["filename"]:
                            pdf_filename = doc["filename"]
                        else:
                            pdf_filename = doc["filename"]
                        
                        # Store PDF reference
                        files_to_upload["individual_compressed"].append({
                            "filename": pdf_filename,
                            "content": doc["content"],
                            "type": "pdf"
                        })
                        
                        # Store JSON extraction
                        if "extracted_data" in doc:
                            if doc["filename"].endswith("_extraction.json"):
                                json_filename = doc["filename"]
                            else:
                                json_filename = pdf_filename.replace(".pdf", "_extraction.json")
                            json_content = json.dumps(doc["extracted_data"], indent=2).encode('utf-8')
                            files_to_upload["individual_compressed"].append({
                                "filename": json_filename,
                                "content": json_content,
                                "type": "json"
                            })
                
                # Try to create combined PDFs if requested
                if len(extracted_documents) > 1:
                    try:
                        # Combined uncompressed
                        combined_uncompressed = await self.file_organizer.combine_pdfs(
                            [doc["content"] for doc in extracted_documents if doc.get("content")],
                            "combined_uncompressed.pdf"
                        )
                        if combined_uncompressed:
                            files_to_upload["combined_uncompressed"] = {
                                "filename": "combined_uncompressed.pdf",
                                "content": combined_uncompressed,
                                "type": "pdf"
                            }
                        
                        # Combined compressed
                        compressed_pdfs = [doc["content"] for doc in compressed_documents if doc.get("compressed", False) and doc.get("content")]
                        if compressed_pdfs:
                            combined_compressed = await self.file_organizer.combine_pdfs(
                                compressed_pdfs,
                                "combined_compressed.pdf"
                            )
                            if combined_compressed:
                                files_to_upload["combined_compressed"] = {
                                    "filename": "combined_compressed.pdf",
                                    "content": combined_compressed,
                                    "type": "pdf"
                                }
                    except Exception as e:
                        logger.warning(f"Failed to create combined PDFs: {e}")
                        # Continue without combined versions
                
                archiving_duration = self.end_step_timer("archiving")
                await self.update_plan_step("archive", "completed", f"Prepared files for upload ({archiving_duration:.1f}s)")
                
                # Step 7: Complete
                await self.update_plan_step("complete", "completed", "Workflow complete!")
                
                # Stop the timer
                self._timer_running = False
                total_elapsed = time.time() - start_time
                
                # Prepare detailed statistics (needed for both batch and individual processing)
                processing_stats = {
                    "documents_processed": len(extracted_documents),
                    "pages_removed": cleaning_report.get("removed_pages", 0),
                    "total_pages": cleaning_report.get("total_pages", 0),
                    "pages_kept": cleaning_report.get("kept_pages", 0),
                    "compressions_successful": len(compressed_documents) - failed_count,
                    "extractions_successful": len([d for d in extracted_documents if not d.get("extraction_failed", False)]),
                    "total_elapsed_time": round(total_elapsed, 1),
                    "average_time_per_document": round(total_elapsed / len(extracted_documents), 1) if extracted_documents else 0,
                    "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "step_timings": {
                        step: round(timing.get("duration", 0), 1) 
                        for step, timing in self.step_timings.items()
                    },
                    "document_details": [
                        {
                            "filename": doc["filename"],
                            "quarter": doc.get("extracted_data", {}).get("correctionInfo", {}).get("quarter") or 
                                      doc.get("extracted_data", {}).get("quarterInfo", {}).get("quarter", "Unknown"),
                            "year": doc.get("extracted_data", {}).get("correctionInfo", {}).get("year") or 
                                   doc.get("extracted_data", {}).get("quarterInfo", {}).get("year", "Unknown"),
                            "extraction_time": doc.get("metadata", {}).get("processing_time", 0)
                        }
                        for doc in extracted_documents
                    ]
                }
                
                # Step 8: Prepare all files for Google Drive upload
                self.start_step_timer("master_archive")
                await self.update_plan_step("archive", "in_progress", "Preparing all files for upload...")
                
                # Save extraction data as JSON files
                extractions_dir = os.path.join(self.output_dir, "extractions")
                os.makedirs(extractions_dir, exist_ok=True)
                logger.info(f"Created extractions directory: {extractions_dir}")
                
                for doc in extracted_documents:
                    # Save individual extraction JSON
                    json_filename = doc["filename"].replace(".pdf", "_extraction.json")
                    json_path = os.path.join(extractions_dir, json_filename)
                    with open(json_path, 'w') as f:
                        json.dump(doc["extracted_data"], f, indent=2)
                    logger.info(f"Saved extraction to: {json_path}")
                
                # Create combined extractions file
                all_extractions = {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "total_documents": len(extracted_documents),
                    "documents": [
                        {
                            "filename": doc["filename"],
                            "data": doc["extracted_data"]
                        }
                        for doc in extracted_documents
                    ]
                }
                
                all_extractions_path = os.path.join(extractions_dir, "all_extractions.json")
                with open(all_extractions_path, 'w') as f:
                    json.dump(all_extractions, f, indent=2)
                logger.info(f"Saved all_extractions.json to: {all_extractions_path} with {len(extracted_documents)} documents")
                
                # Store all_extractions.json in files_to_upload
                files_to_upload["all_extractions"] = {
                    "filename": "all_extractions.json",
                    "content": json.dumps(all_extractions, indent=2).encode('utf-8'),
                    "type": "json"
                }
                
                master_archive_duration = self.end_step_timer("master_archive")
                await self.update_plan_step("archive", "completed", f"Prepared all files for upload ({master_archive_duration:.1f}s)")
                
                # Step 8: Upload to Google Drive (skip if part of batch)
                google_drive_result = None
                if not skip_individual_gdrive:
                    try:
                        self.start_step_timer("google_drive_upload")
                        await self.update_plan_step("google_drive", "in_progress", "Uploading files to Google Drive...")
                        
                        # Extract company name and form type for folder organization
                        company_name = None
                        form_type = None
                        if extracted_documents:
                            first_doc_data = extracted_documents[0].get("extracted_data", {})
                            if "employerInfo" in first_doc_data:
                                company_name = first_doc_data["employerInfo"].get("name", "Unknown")
                            
                            # Detect form type
                            form_match = re.search(r'941-?X?', extracted_documents[0]["filename"], re.IGNORECASE)
                            if form_match:
                                form_type = form_match.group(0).upper().replace('-', '')
                            else:
                                form_type = "941X"
                        
                        if not company_name:
                            company_name = "Unknown_Company"
                        
                        # Upload to Google Drive using configuration
                        async with GoogleDriveIntegration() as google_drive:
                            # Update status tracking
                            if True:  # Always track processing status
                                await update_session_status(self.session_id, {
                                    "status": "uploading",
                                    "current_step": "google_drive",
                                    "progress": 90,
                                    "metadata": {
                                        "company_name": company_name,
                                        "form_type": form_type,
                                        "documents_count": len(extracted_documents)
                                    }
                                })
                            
                            # Upload files to Google Drive folder structure
                            google_drive_result = await self.upload_files_to_google_drive(
                                google_drive,
                                files_to_upload,
                                organized_paths,
                                company_name,
                                form_type,
                                processing_stats,
                                cleaning_report
                            )
                        
                        if google_drive_result and google_drive_result.get("success"):
                            google_drive_duration = self.end_step_timer("google_drive_upload")
                            await self.update_plan_step("google_drive", "completed", 
                                                      f"Uploaded to Google Drive ({google_drive_duration:.1f}s)")
                            await self.send_update("google_drive_complete", google_drive_result)
                        else:
                            raise Exception(google_drive_result.get("error", "Upload failed"))
                            
                    except Exception as e:
                        logger.warning(f"Google Drive upload failed: {e}")
                        await self.update_plan_step("google_drive", "failed", f"Upload failed: {str(e)[:50]}...")
                        # Continue without failing the entire workflow
                else:
                    logger.info(f"Skipping Google Drive upload for batch processing (session {self.session_id})")
                
                # Prepare final results
                workflow_results = {
                    "success": True,
                    "cleaning_report": cleaning_report,
                    "split_count": len(split_documents),
                    "extracted_documents": [
                        {
                            "filename": doc["filename"],
                            "metadata": doc.get("_metadata", doc.get("metadata", {})),
                            "extracted_data": doc["extracted_data"]
                        }
                        for doc in extracted_documents
                    ],
                    "organized_paths": organized_paths,
                    "google_drive_folders": google_drive_result if google_drive_result else None,
                    "output_directory": self.output_dir,
                    "total_elapsed_time": round(total_elapsed, 1),
                    "processing_stats": {
                        "documents_processed": len(extracted_documents),
                        "pages_removed": cleaning_report.get("removed_pages", 0),
                        "compressions_successful": len(compressed_documents) - failed_count,
                        "average_time_per_document": round(total_elapsed / len(extracted_documents), 1) if extracted_documents else 0
                    },
                    "step_timings": {
                        step: round(timing.get("duration", 0), 1) 
                        for step, timing in self.step_timings.items()
                    },
                    "google_drive_upload": google_drive_result
                }
                
                await self.send_update("workflow_complete", workflow_results)
                
                # Update session status to completed with processed_count
                # This is critical for batch processing to track completion
                await db_state_manager.save_processing_status(
                    self.session_id,
                    self.batch_id or "",  # Use batch_id if part of a batch
                    "completed",
                    [filename],  # processed_files
                    {},  # extractions
                    1,  # total_files
                    {  # metadata
                        "workflow_complete": True,
                        "processing_stats": processing_stats,
                        "google_drive_result": google_drive_result
                    }
                )
                
                return workflow_results
                
        except Exception as e:
            logger.error(f"Error in enhanced workflow: {e}")
            await self.update_plan_step("complete", "failed", str(e))
            
            # Update session status to failed
            await db_state_manager.save_processing_status(
                self.session_id,
                self.batch_id or "",  # Use batch_id if part of a batch
                "failed",
                [],  # processed_files
                {},  # extractions
                1,  # total_files
                {"error": str(e)}  # metadata
            )
            
            raise

    async def extract_document_with_tracking(self, split_doc: Dict[str, Any], selected_schema: str,
                                           expected_value: Optional[str], doc_index: int) -> Dict[str, Any]:
        """Extract data from a FULL document (all pages) with real-time progress tracking"""
        filename = split_doc["filename"]
        
        # Send start notification
        await self.send_update("document_extraction_started", {
            "index": doc_index,
            "filename": filename,
            "status": "extracting",
            "message": "Starting full document extraction..."
        })
        
        try:
            # Extract data from the ENTIRE document (not just individual pages)
            start_time = time.time()
            
            # Create a dedicated processor for this specific document
            # Create a temporary session ID and add it to active sessions
            temp_session_id = f"{self.session_id}_doc_{doc_index}"
            active_sessions.add(temp_session_id)
            
            doc_processor = DocumentProcessor(temp_session_id)
            doc_processor.expected_value = expected_value
            
            # Process the entire document with its own parallel page extraction
            logger.info(f"Starting full document extraction for {filename} (document {doc_index + 1})")
            
            # The V11 processor will handle:
            # 1. Splitting into pages
            # 2. Extracting all pages in parallel 
            # 3. Running compilation immediately after
            extraction_results = await doc_processor.process_with_parallel_extraction(
                split_doc["content"],
                selected_schema,
                filename
            )
            
            extraction_time = time.time() - start_time
            
            # Get the first (and should be only) result
            if extraction_results and len(extraction_results) > 0:
                extraction_result = extraction_results[0]
                
                # Send completion notification
                await self.send_update("document_extraction_completed", {
                    "index": doc_index,
                    "filename": filename,
                    "status": "completed",
                    "extraction_time": extraction_time,
                    "has_errors": "error" in extraction_result,
                    "pages_processed": extraction_result.get("compilation_metadata", {}).get("total_pages", 0),
                    "message": f"Extracted and compiled {extraction_result.get('compilation_metadata', {}).get('total_pages', 0)} pages"
                })
                
                logger.info(f"Document {filename} extraction complete in {extraction_time:.1f}s")
                
                # Add filename at top level for compatibility with organize function
                # The extraction result has filename in _metadata.filename
                # but organize expects it at the top level
                extraction_result["filename"] = extraction_result.get("_metadata", {}).get("filename", filename)
                
                # Also ensure we have the original filename from the split document
                extraction_result["original_filename"] = filename
                
                # Map the extracted data to the expected format
                # The extraction returns data directly, but organize expects it under "extracted_data"
                if "extracted_data" not in extraction_result:
                    # Create a copy without metadata for extracted_data
                    extracted_data = {k: v for k, v in extraction_result.items() 
                                    if k not in ["_metadata", "filename", "original_filename", "compilation_metadata"]}
                    extraction_result["extracted_data"] = extracted_data
                
                # CRITICAL: Add the PDF content to the extraction result
                # The organize function needs this to save the PDF files
                extraction_result["content"] = split_doc["content"]
                
                return extraction_result
            else:
                raise Exception("No extraction results returned")
            
        except Exception as e:
            logger.error(f"Error extracting document {filename}: {e}")
            # Send error notification
            await self.send_update("document_extraction_completed", {
                "index": doc_index,
                "filename": filename,
                "status": "failed",
                "error": str(e)
            })
            raise
        finally:
            # Clean up temporary session
            if 'temp_session_id' in locals():
                active_sessions.discard(temp_session_id)
    
    def organize_by_quarter_with_metadata(self, documents: List[Dict[str, Any]], base_path: str) -> Dict[str, List[str]]:
        """Organize documents by quarter using extracted metadata instead of filename parsing"""
        organized = {}
        
        for idx, doc in enumerate(documents):
            try:
                # Handle both formats: direct filename or nested in _metadata
                original_filename = None
                if "filename" in doc:
                    original_filename = doc["filename"]
                elif "_metadata" in doc and "filename" in doc["_metadata"]:
                    original_filename = doc["_metadata"]["filename"]
                else:
                    # Log the document structure for debugging
                    logger.error(f"Document {idx} missing filename. Keys: {list(doc.keys())}")
                    if "_metadata" in doc:
                        logger.error(f"  _metadata keys: {list(doc['_metadata'].keys())}")
                    raise KeyError(f"Document {idx} missing filename field")
                
                logger.info(f"Organizing document {idx}: {original_filename}")
                
                extracted_data = doc.get("extracted_data", {})
                
                # Try to get quarter and year from extracted data
                quarter = None
                year = None
                company_name = None
                form_type = None
                
                # Extract company name
                if "employerInfo" in extracted_data:
                    company_name = extracted_data["employerInfo"].get("name", "Unknown")
                
                # Extract quarter and year
                if "correctionInfo" in extracted_data:
                    quarter = extracted_data["correctionInfo"].get("quarter")
                    year = extracted_data["correctionInfo"].get("year")
                elif "quarterInfo" in extracted_data:
                    quarter = extracted_data["quarterInfo"].get("quarter")
                    year = extracted_data["quarterInfo"].get("year")
                elif "quarterBeingCorrected" in extracted_data:
                    quarter = extracted_data["quarterBeingCorrected"].get("quarter")
                    year = extracted_data["quarterBeingCorrected"].get("year")
                
                # Extract form type from filename
                form_match = re.search(r'941-?X?', original_filename, re.IGNORECASE)
                if form_match:
                    form_type = form_match.group(0).upper().replace('-', '')
                else:
                    form_type = "941X"
                
                # Convert quarter number to string
                quarter_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
                quarter_str = quarter_map.get(quarter, f"Q{quarter}" if quarter else "Unknown")
                
                # Create organized filename
                if company_name and year and quarter:
                    # Clean company name for filename
                    clean_company = re.sub(r'[^\w\s-]', '', company_name).strip()
                    clean_company = re.sub(r'[-\s]+', '_', clean_company).upper()
                    
                    # Determine file type (compressed or uncompressed)
                    file_type = "compressed" if "_compressed" in original_filename else "uncompressed"
                    
                    # Create new organized filename
                    organized_filename = f"{clean_company}_{quarter_str}_{year}_{form_type}_{file_type}.pdf"
                    
                    # Create directory structure
                    year_dir = os.path.join(base_path, str(year))
                    quarter_dir = os.path.join(year_dir, quarter_str)
                    os.makedirs(quarter_dir, exist_ok=True)
                    
                    # Save the file
                    output_path = os.path.join(quarter_dir, organized_filename)
                    if "content" not in doc:
                        logger.error(f"Document {idx} missing content field. Keys: {list(doc.keys())}")
                        raise KeyError(f"Document {idx} missing content field")
                    with open(output_path, 'wb') as f:
                        f.write(doc["content"])
                    
                    # Track in organized dict
                    quarter_key = f"{year}/{quarter_str}"
                    if quarter_key not in organized:
                        organized[quarter_key] = []
                    organized[quarter_key].append(organized_filename)
                    
                    logger.info(f"Organized {original_filename} as {organized_filename} in {quarter_key}")
                else:
                    # If we can't organize by quarter, save in an "unorganized" folder
                    unorganized_dir = os.path.join(base_path, "unorganized")
                    os.makedirs(unorganized_dir, exist_ok=True)
                    
                    output_path = os.path.join(unorganized_dir, original_filename)
                    if "content" not in doc:
                        logger.error(f"Document {idx} missing content field for unorganized save. Keys: {list(doc.keys())}")
                        raise KeyError(f"Document {idx} missing content field")
                    with open(output_path, 'wb') as f:
                        f.write(doc["content"])
                    
                    if "unorganized" not in organized:
                        organized["unorganized"] = []
                    organized["unorganized"].append(original_filename)
                    
                    logger.warning(f"Could not organize {original_filename} - missing metadata")
                
            except KeyError as e:
                logger.error(f"KeyError processing document {idx}: {e}")
                logger.error(f"Document structure: {json.dumps({k: type(v).__name__ for k, v in doc.items()}, indent=2)}")
                # Save to unorganized folder with error prefix
                unorganized_dir = os.path.join(base_path, "unorganized")
                os.makedirs(unorganized_dir, exist_ok=True)
                
                error_filename = f"error_{idx}_{doc.get('original_filename', 'unknown.pdf')}"
                output_path = os.path.join(unorganized_dir, error_filename)
                
                if "content" in doc:
                    with open(output_path, 'wb') as f:
                        f.write(doc["content"])
                    
                    if "unorganized" not in organized:
                        organized["unorganized"] = []
                    organized["unorganized"].append(error_filename)
                    logger.warning(f"Saved error document as {error_filename}")
                else:
                    logger.error(f"Document {idx} has no content to save")
                
            except Exception as e:
                logger.error(f"Unexpected error organizing document {idx}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Document keys: {list(doc.keys()) if isinstance(doc, dict) else 'Not a dict'}")
                # Continue processing other documents
                continue
        
        logger.info(f"Organized {len(documents)} documents into {len(organized)} categories")
        return organized

    async def upload_files_to_google_drive(self, google_drive, files_to_upload: Dict[str, Any],
                                         organized_paths: Dict[str, List[str]], company_name: str,
                                         form_type: str, processing_stats: Dict[str, Any],
                                         cleaning_report: Dict[str, Any]) -> Dict[str, Any]:
        """Upload all files to Google Drive in a folder structure instead of ZIP archives"""
        try:
            # Always define processing_datetime for use in summaries
            processing_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            date_str = datetime.now().strftime("%B %d, %Y")
            time_str = datetime.now().strftime("%I:%M %p")
            
            # Initialize main_folder_name to avoid UnboundLocalError
            main_folder_name = f"{company_name}_{form_type}_{processing_datetime}"
            main_folder_name = re.sub(r'[<>:"/\\|?*]', '_', main_folder_name)
            main_folder_name = main_folder_name[:255]  # Google Drive name limit
            
            # Use the provided google_drive_folder_id if available (case folder)
            # Otherwise create a new folder structure
            if hasattr(self, 'google_drive_folder_id') and self.google_drive_folder_id:
                # Use the existing case folder
                batch_folder_id = self.google_drive_folder_id
                logger.info(f"Using existing Google Drive folder: {batch_folder_id}")
            else:
                # Create new folder structure
                batch_folder_result = await google_drive.create_folder(main_folder_name)
                batch_folder_id = batch_folder_result["folder"]["id"]
                logger.info(f"Created main Google Drive folder: {main_folder_name}")
            
            # Always create the output folder (specified by user)
            output_folder_name = getattr(self, 'output_folder', None) or "files_produced"
            files_produced_folder_result = await google_drive.create_folder(output_folder_name, batch_folder_id)
            files_produced_folder_id = files_produced_folder_result["folder"]["id"]
            logger.info(f"Created output folder: {output_folder_name}")
            
            # Keep track of uploaded files
            uploaded_files = []
            
            # Upload all_extractions.json to files_produced folder
            if files_to_upload.get("all_extractions"):
                all_extractions_file = files_to_upload["all_extractions"]
                await google_drive.upload_file_content(
                    all_extractions_file["content"],
                    all_extractions_file["filename"],
                    files_produced_folder_id
                )
                uploaded_files.append(all_extractions_file["filename"])
                logger.info(f"Uploaded all_extractions.json to {output_folder_name} folder")
            
            # Create archives folder structure
            archives_folder_result = await google_drive.create_folder("archives", files_produced_folder_id)
            archives_folder_id = archives_folder_result["folder"]["id"]
            individual_folder_result = await google_drive.create_folder("individual", archives_folder_id)
            individual_folder_id = individual_folder_result["folder"]["id"]
            # Create uncompressed folder with cleaned/raw subdirectories
            individual_uncomp_folder_result = await google_drive.create_folder("uncompressed", individual_folder_id)
            individual_uncomp_folder_id = individual_uncomp_folder_result["folder"]["id"]
            uncomp_cleaned_folder_result = await google_drive.create_folder("cleaned", individual_uncomp_folder_id)
            uncomp_cleaned_folder_id = uncomp_cleaned_folder_result["folder"]["id"]
            uncomp_raw_folder_result = await google_drive.create_folder("raw", individual_uncomp_folder_id)
            uncomp_raw_folder_id = uncomp_raw_folder_result["folder"]["id"]
            
            # Create compressed folder with cleaned/raw subdirectories
            individual_comp_folder_result = await google_drive.create_folder("compressed", individual_folder_id)
            individual_comp_folder_id = individual_comp_folder_result["folder"]["id"]
            comp_cleaned_folder_result = await google_drive.create_folder("cleaned", individual_comp_folder_id)
            comp_cleaned_folder_id = comp_cleaned_folder_result["folder"]["id"]
            comp_raw_folder_result = await google_drive.create_folder("raw", individual_comp_folder_id)
            comp_raw_folder_id = comp_raw_folder_result["folder"]["id"]
            
            # Upload individual uncompressed files to cleaned/raw subdirectories
            if files_to_upload["individual_uncompressed"]:
                logger.info(f"Uploading {len(files_to_upload['individual_uncompressed'])} uncompressed files to cleaned/raw subdirectories...")
                cleaned_count = 0
                raw_count = 0
                
                for file_info in files_to_upload["individual_uncompressed"]:
                    # Determine target folder based on file classification
                    if is_cleaned_file(file_info):
                        target_folder_id = uncomp_cleaned_folder_id
                        subfolder = "cleaned"
                        cleaned_count += 1
                    else:
                        target_folder_id = uncomp_raw_folder_id
                        subfolder = "raw"
                        raw_count += 1
                    
                    await google_drive.upload_file_content(
                        file_info["content"],
                        file_info["filename"],
                        target_folder_id
                    )
                    uploaded_files.append(f"individual_uncompressed/{subfolder}/{file_info['filename']}")
                
                logger.info(f"Uploaded {cleaned_count} cleaned and {raw_count} raw uncompressed files")
            
            # Upload individual compressed files to cleaned/raw subdirectories
            if files_to_upload["individual_compressed"]:
                logger.info(f"Uploading {len(files_to_upload['individual_compressed'])} compressed files to cleaned/raw subdirectories...")
                cleaned_count = 0
                raw_count = 0
                
                for file_info in files_to_upload["individual_compressed"]:
                    # Determine target folder based on file classification
                    if is_cleaned_file(file_info):
                        target_folder_id = comp_cleaned_folder_id
                        subfolder = "cleaned"
                        cleaned_count += 1
                    else:
                        target_folder_id = comp_raw_folder_id
                        subfolder = "raw"
                        raw_count += 1
                    
                    await google_drive.upload_file_content(
                        file_info["content"],
                        file_info["filename"],
                        target_folder_id
                    )
                    uploaded_files.append(f"individual_compressed/{subfolder}/{file_info['filename']}")
                
                logger.info(f"Uploaded {cleaned_count} cleaned and {raw_count} raw compressed files")
            
            # Upload combined files if they exist
            if files_to_upload["combined_uncompressed"]:
                file_info = files_to_upload["combined_uncompressed"]
                await google_drive.upload_file_content(
                    file_info["content"],
                    file_info["filename"],
                    archives_folder_id
                )
                uploaded_files.append(file_info["filename"])
                logger.info("Uploaded combined_uncompressed.pdf")
            
            if files_to_upload["combined_compressed"]:
                file_info = files_to_upload["combined_compressed"]
                await google_drive.upload_file_content(
                    file_info["content"],
                    file_info["filename"],
                    archives_folder_id
                )
                uploaded_files.append(file_info["filename"])
                logger.info("Uploaded combined_compressed.pdf")
            
            # Upload organized files by quarter
            if organized_paths:
                organized_folder_result = await google_drive.create_folder("organized_by_quarter", files_produced_folder_id)
                organized_folder_id = organized_folder_result["folder"]["id"]
                
                for quarter_path, filenames in organized_paths.items():
                    if quarter_path == "unorganized":
                        # Create unorganized folder
                        quarter_folder_result = await google_drive.create_folder("unorganized", organized_folder_id)
                        quarter_folder_id = quarter_folder_result["folder"]["id"]
                    else:
                        # Create year/quarter folder structure
                        year, quarter = quarter_path.split("/")
                        year_folder_result = await google_drive.create_folder(year, organized_folder_id)
                        year_folder_id = year_folder_result["folder"]["id"]
                        quarter_folder_result = await google_drive.create_folder(quarter, year_folder_id)
                        quarter_folder_id = quarter_folder_result["folder"]["id"]
                    
                    # Upload files for this quarter
                    quarter_dir = os.path.join(self.output_dir, quarter_path.replace("/", os.sep))
                    if os.path.exists(quarter_dir):
                        file_paths = [os.path.join(quarter_dir, f) for f in filenames if os.path.exists(os.path.join(quarter_dir, f))]
                        
                        for file_path in file_paths:
                            filename = os.path.basename(file_path)
                            with open(file_path, 'rb') as f:
                                await google_drive.upload_file_content(
                                    f.read(),
                                    filename,
                                    quarter_folder_id
                                )
                            uploaded_files.append(f"{year}/{quarter}/{filename}")
                    
                    # Also upload JSON extractions for this quarter
                    quarter_dir = os.path.dirname(file_paths[0]) if file_paths else None
                    if quarter_dir and os.path.exists(quarter_dir):
                        json_files = [f for f in os.listdir(quarter_dir) if f.endswith('_extraction.json')]
                        for json_file in json_files:
                            json_path = os.path.join(quarter_dir, json_file)
                            if os.path.exists(json_path):
                                with open(json_path, 'rb') as f:
                                    await google_drive.upload_file_content(
                                        f.read(),
                                        json_file,
                                        quarter_folder_id
                                    )
                                uploaded_files.append(f"{year}/{quarter}/{json_file}")
            
            # Upload processing summary
            summary_content = self.create_processing_summary(
                self.session_id,
                company_name,
                form_type,
                processing_stats,
                cleaning_report,
                uploaded_files,
                processing_datetime
            )
            
            await google_drive.upload_file_content(
                summary_content.encode('utf-8'),
                "processing_summary.txt",
                files_produced_folder_id
            )
            uploaded_files.append("processing_summary.txt")
            
            # Get the web link for the folder
            folder_link = f"https://drive.google.com/drive/folders/{batch_folder_id}"
            
            # Collect all unique quarters
            quarters_list = []
            for quarter_path in organized_paths.keys():
                if quarter_path != "unorganized" and "/" in quarter_path:
                    year, quarter = quarter_path.split("/")
                    quarters_list.append(f"{quarter} {year}")
            
            quarters_str = ", ".join(sorted(set(quarters_list))) if quarters_list else "N/A"
            
            # Update session status with folder info
            if True:  # Always track processing status
                await update_session_status(self.session_id, {
                    "status": "completed",
                    "current_step": "google_drive_upload_complete",
                    "progress": 100,
                    "gdrive_folder_id": batch_folder_id,
                    "gdrive_folder_path": folder_link,
                    "files_uploaded": uploaded_files,
                    "metadata": {
                        "upload_timestamp": datetime.now().isoformat(),
                        "folder_structure": {
                            "main": main_folder_name,
                            "files_produced": output_folder_name,
                            "has_archives": bool(files_to_upload.get("individual_uncompressed") or files_to_upload.get("combined_uncompressed")),
                            "has_organized": bool(organized_paths)
                        }
                    }
                })
            
            return {
                "success": True,
                "folder_id": batch_folder_id,
                "folder_link": folder_link,
                "folder_name": main_folder_name,
                "files_uploaded": len(uploaded_files),
                "upload_summary": {
                    "uncompressed_files": len(files_to_upload.get("individual_uncompressed", [])),
                    "compressed_files": len(files_to_upload.get("individual_compressed", [])),
                    "combined_files": 2 if (files_to_upload.get("combined_uncompressed") and files_to_upload.get("combined_compressed")) else 0,
                    "quarters_organized": len(organized_paths),
                    "all_extractions": 1 if files_to_upload.get("all_extractions") else 0
                },
                "quarters": quarters_str
            }
            
        except Exception as e:
            logger.error(f"Error uploading files to Google Drive: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_processing_summary(self, session_id: str, company_name: str, form_type: str,
                                processing_stats: Dict[str, Any], cleaning_report: Dict[str, Any],
                                uploaded_files: List[str], processing_datetime: str) -> str:
        """Create a detailed processing summary"""
        total_seconds = processing_stats.get('total_elapsed_time', 0)
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        formatted_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        # Get the output folder name from instance or default
        output_folder_name = getattr(self, 'output_folder', None) or "files_produced"
        
        # Get current timestamp with timezone
        import time as time_module
        timezone = time_module.strftime('%Z')
        date_str = datetime.now().strftime("%B %d, %Y")
        time_str = datetime.now().strftime("%I:%M %p") + f" {timezone}"
        
        # Count uploaded files by type
        uploaded_count = len(uploaded_files)
        
        # Prepare quarters string from document details
        quarters_set = set()
        for doc_detail in processing_stats.get('document_details', []):
            quarter = doc_detail.get('quarter', 'Unknown')
            year = doc_detail.get('year', 'Unknown')
            if quarter != 'Unknown' and year != 'Unknown':
                quarters_set.add(f"Q{quarter} {year}")
        
        quarters_str = ", ".join(sorted(quarters_set)) if quarters_set else "N/A"
        
        summary = f"""Smart Document Processing Summary
{'=' * 50}

Company: {company_name}
Form Type: {form_type}
Quarters: {quarters_str}
Processing Date: {date_str} {time_str}
Total Files: {processing_stats.get('total_files', 0)}
Total Documents: {processing_stats.get('documents_processed', 0)}
Batch ID: {processing_stats.get('batch_id', 'N/A')}
Files Uploaded: {uploaded_count}

Processing Statistics:
---------------------
Total Processing Time: {formatted_time}
Average Time per Document: {processing_stats.get('average_time_per_document', 0):.1f}s
Pages Removed: {cleaning_report.get('removed_pages', 0)} of {cleaning_report.get('total_pages', 0)}
Pages Kept: {cleaning_report.get('kept_pages', 0)}
Successful Extractions: {processing_stats.get('extractions_successful', 0)}
Successful Compressions: {processing_stats.get('compressions_successful', 0)}

Step Timing Breakdown:
--------------------"""
        
        # Add step timings
        for step, duration in processing_stats.get('step_timings', {}).items():
            step_name = step.replace('_', ' ').title()
            summary += f"\n{step_name}: {duration}s"
        
        summary += f"""

Files Uploaded to Google Drive:
-----------------------------
Output Folder: {output_folder_name}
"""
        
        # Categorize uploaded files
        extraction_files = [f for f in uploaded_files if f.endswith('.json')]
        pdf_files = [f for f in uploaded_files if f.endswith('.pdf')]
        other_files = [f for f in uploaded_files if not f.endswith('.json') and not f.endswith('.pdf')]
        
        summary += f"\n- Extraction Files (.json): {len(extraction_files)}"
        summary += f"\n- PDF Files: {len(pdf_files)}"
        if other_files:
            summary += f"\n- Other Files: {len(other_files)}"
        
        summary += f"""

Processing Details:
-----------------
Session ID: {session_id}
Processor Version: Enhanced v12
Document Cleaning: Enabled
Parallel Extraction: Enabled
Google Drive Integration: Enabled
Batch Processing: {'Yes' if hasattr(self, 'batch_id') and self.batch_id else 'No'}

Notes:
------
- All files are organized in the '{output_folder_name}' folder
- Individual documents are in 'archives/individual' subfolder:
  - uncompressed/cleaned: Files with pages removed during cleaning
  - uncompressed/raw: Original files without cleaning
  - compressed/cleaned: Compressed versions of cleaned files
  - compressed/raw: Compressed versions of raw files
- Files organized by quarter are in 'organized_by_quarter' subfolder
- JSON extractions are included with their corresponding PDFs
- The 'all_extractions.json' file contains all extracted data

End of Processing Summary
{'=' * 50}
"""
        return summary
    
    async def upload_batch_files_to_google_drive(self, google_drive, batch_files: Dict[str, Any],
                                               all_extractions: List[Dict], company_name: str,
                                               form_type: str, quarters_str: str, session_ids: List[str],
                                               processing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Upload batch files to Google Drive in a folder structure"""
        try:
            # Initialize main_folder_name at the beginning to avoid scope issues
            processing_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_folder_name = getattr(self, 'output_folder', None) or "941x_forms"
            main_folder_name = f"{company_name}_{form_type}_Batch_{processing_datetime}"
            # Clean up folder name
            main_folder_name = re.sub(r'[<>:"/\\|?*]', '_', main_folder_name)
            main_folder_name = main_folder_name[:255]  # Google Drive name limit
            
            # Use the provided google_drive_folder_id if available (case folder)
            if hasattr(self, 'google_drive_folder_id') and self.google_drive_folder_id:
                case_folder_id = self.google_drive_folder_id
                logger.info(f"Using case folder: {case_folder_id}")
                
                # Navigate to customer_uploaded_docs/941x_forms
                output_folder_name = getattr(self, 'output_folder', None) or "941x_forms"
                logger.info(f"Looking for output folder: {output_folder_name}")
                
                # First, find customer_uploaded_docs folder
                customer_docs_folder_id = await google_drive.search_folder("customer_uploaded_docs", case_folder_id)
                if not customer_docs_folder_id:
                    logger.error("customer_uploaded_docs folder not found in case folder")
                    # Create it if it doesn't exist
                    customer_docs_result = await google_drive.create_folder("customer_uploaded_docs", case_folder_id)
                    customer_docs_folder_id = customer_docs_result["folder"]["id"]
                    logger.info(f"Created customer_uploaded_docs folder: {customer_docs_folder_id}")
                else:
                    logger.info(f"Found customer_uploaded_docs folder: {customer_docs_folder_id}")
                
                # Find or create the output folder (941x_forms) inside customer_uploaded_docs
                target_folder_id = await google_drive.search_folder(output_folder_name, customer_docs_folder_id)
                if not target_folder_id:
                    logger.info(f"Creating {output_folder_name} folder in customer_uploaded_docs")
                    target_folder_result = await google_drive.create_folder(output_folder_name, customer_docs_folder_id)
                    target_folder_id = target_folder_result["folder"]["id"]
                else:
                    logger.info(f"Found existing {output_folder_name} folder: {target_folder_id}")
                
                # Create batch folder with timestamp inside the target folder
                processing_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                batch_folder_name = f"Batch_{processing_datetime}"
                batch_folder_result = await google_drive.create_folder(batch_folder_name, target_folder_id)
                batch_folder_id = batch_folder_result["folder"]["id"]
                files_produced_folder_id = batch_folder_id  # The batch folder is where files go
                logger.info(f"Created batch folder: {batch_folder_name} in {output_folder_name}")
            else:
                # Create new folder structure if no case folder
                # main_folder_name already defined at the beginning of the function
                batch_folder_result = await google_drive.create_folder(main_folder_name)
                batch_folder_id = batch_folder_result["folder"]["id"]
                files_produced_folder_id = batch_folder_id
                logger.info(f"Created standalone batch folder: {main_folder_name}")
            
            # Upload all_extractions.json from the batch_files parameter
            if batch_files.get("all_extractions"):
                all_extractions_content = json.dumps(batch_files["all_extractions"], indent=2)
                await google_drive.upload_file_content(
                    all_extractions_content.encode('utf-8'),
                    "all_extractions.json",
                    files_produced_folder_id
                )
            else:
                # Fallback to original method if not provided
                all_extractions_content = json.dumps({
                    "batch_timestamp": datetime.now().isoformat(),
                    "total_files": len(session_ids),
                    "total_documents": len(all_extractions),
                    "company_name": company_name,
                    "form_type": form_type,
                    "quarters": quarters_str,
                    "documents": all_extractions
                }, indent=2)
                
                await google_drive.upload_file_content(
                    all_extractions_content.encode('utf-8'),
                    "all_extractions.json",
                    files_produced_folder_id
                )
            
            uploaded_count = 1  # Start with all_extractions.json
            
            # Create archives folder structure
            archives_folder_result = await google_drive.create_folder("archives", files_produced_folder_id)
            archives_folder_id = archives_folder_result["folder"]["id"]
            
            # Create individual folders
            individual_folder_result = await google_drive.create_folder("individual", archives_folder_id)
            individual_folder_id = individual_folder_result["folder"]["id"]
            
            # Create uncompressed folder with cleaned/raw subdirectories
            uncompressed_folder_result = await google_drive.create_folder("uncompressed", individual_folder_id)
            uncompressed_folder_id = uncompressed_folder_result["folder"]["id"]
            uncomp_cleaned_folder_result = await google_drive.create_folder("cleaned", uncompressed_folder_id)
            uncomp_cleaned_folder_id = uncomp_cleaned_folder_result["folder"]["id"]
            uncomp_raw_folder_result = await google_drive.create_folder("raw", uncompressed_folder_id)
            uncomp_raw_folder_id = uncomp_raw_folder_result["folder"]["id"]
            
            # Create compressed folder with cleaned/raw subdirectories
            compressed_folder_result = await google_drive.create_folder("compressed", individual_folder_id)
            compressed_folder_id = compressed_folder_result["folder"]["id"]
            comp_cleaned_folder_result = await google_drive.create_folder("cleaned", compressed_folder_id)
            comp_cleaned_folder_id = comp_cleaned_folder_result["folder"]["id"]
            comp_raw_folder_result = await google_drive.create_folder("raw", compressed_folder_id)
            comp_raw_folder_id = comp_raw_folder_result["folder"]["id"]
            
            # Upload individual uncompressed files to cleaned/raw subdirectories
            if batch_files.get("individual_uncompressed"):
                logger.info(f"Uploading {len(batch_files['individual_uncompressed'])} uncompressed files to cleaned/raw subdirectories...")
                cleaned_count = 0
                raw_count = 0
                
                for pdf_info in batch_files["individual_uncompressed"]:
                    # Determine target folder based on file classification
                    if is_cleaned_file(pdf_info):
                        target_folder_id = uncomp_cleaned_folder_id
                        cleaned_count += 1
                    else:
                        target_folder_id = uncomp_raw_folder_id
                        raw_count += 1
                    
                    await google_drive.upload_file_content(
                        pdf_info["content"],
                        pdf_info["filename"],
                        target_folder_id
                    )
                    uploaded_count += 1
                
                logger.info(f"Uploaded {cleaned_count} cleaned and {raw_count} raw uncompressed files")
            
            # Upload individual compressed files to cleaned/raw subdirectories
            if batch_files.get("individual_compressed"):
                logger.info(f"Uploading {len(batch_files['individual_compressed'])} compressed files to cleaned/raw subdirectories...")
                cleaned_count = 0
                raw_count = 0
                
                for pdf_info in batch_files["individual_compressed"]:
                    # Determine target folder based on file classification
                    if is_cleaned_file(pdf_info):
                        target_folder_id = comp_cleaned_folder_id
                        cleaned_count += 1
                    else:
                        target_folder_id = comp_raw_folder_id
                        raw_count += 1
                    
                    await google_drive.upload_file_content(
                        pdf_info["content"],
                        pdf_info["filename"],
                        target_folder_id
                    )
                    uploaded_count += 1
                
                logger.info(f"Uploaded {cleaned_count} cleaned and {raw_count} raw compressed files")
            
            # Upload extraction JSONs alongside compressed files
            if batch_files.get("extraction_jsons"):
                for json_info in batch_files["extraction_jsons"]:
                    await google_drive.upload_file_content(
                        json_info["content"],
                        json_info["filename"],
                        compressed_folder_id
                    )
                    uploaded_count += 1
            
            # Upload combined files to archives root
            if batch_files.get("combined_uncompressed"):
                await google_drive.upload_file_content(
                    batch_files["combined_uncompressed"],
                    "combined_uncompressed.pdf",
                    archives_folder_id
                )
                uploaded_count += 1
            
            if batch_files.get("combined_compressed"):
                await google_drive.upload_file_content(
                    batch_files["combined_compressed"],
                    "combined_compressed.pdf",
                    archives_folder_id
                )
                uploaded_count += 1
            
            # Create organized_by_quarter structure
            if batch_files.get("organized_paths"):
                organized_folder_result = await google_drive.create_folder("organized_by_quarter", files_produced_folder_id)
                organized_folder_id = organized_folder_result["folder"]["id"]
                
                # Read files from organized paths on disk
                for quarter_path in batch_files["organized_paths"]:
                    if quarter_path != "unorganized" and "/" in quarter_path:
                        year, quarter = quarter_path.split("/")
                        
                        # Create year folder
                        year_folder_id = await google_drive.search_folder(year, organized_folder_id)
                        if not year_folder_id:
                            year_folder_result = await google_drive.create_folder(year, organized_folder_id)
                            year_folder_id = year_folder_result["folder"]["id"]
                        
                        # Create quarter folder
                        quarter_folder_result = await google_drive.create_folder(quarter, year_folder_id)
                        quarter_folder_id = quarter_folder_result["folder"]["id"]
                        
                        # Upload files from this quarter
                        # Files are organized directly in year/quarter structure, not under organized_by_quarter/
                        quarter_dir = os.path.join(self.output_dir, quarter_path)
                        if os.path.exists(quarter_dir):
                            for file in os.listdir(quarter_dir):
                                if file.endswith(".pdf") or file.endswith(".json"):
                                    file_path = os.path.join(quarter_dir, file)
                                    with open(file_path, 'rb') as f:
                                        await google_drive.upload_file_content(
                                            f.read(),
                                            file,
                                            quarter_folder_id
                                        )
                                        uploaded_count += 1
                            logger.info(f"Uploaded {len(os.listdir(quarter_dir))} files from {quarter_path}")
                        else:
                            logger.warning(f"Quarter directory not found: {quarter_dir}")
            
            # Upload session files for debugging (keep existing functionality)
            if batch_files.get("sessions"):
                sessions_folder_result = await google_drive.create_folder("session_files", files_produced_folder_id)
                sessions_folder_id = sessions_folder_result["folder"]["id"]
                
                for session_info in batch_files["sessions"]:
                    session_id = session_info["session_id"]
                    session_folder_result = await google_drive.create_folder(f"session_{session_id[:8]}", sessions_folder_id)
                    session_folder_id = session_folder_result["folder"]["id"]
                    
                    for file_path, rel_path in session_info["files"][:20]:  # Limit to prevent timeout
                        try:
                            with open(file_path, 'rb') as f:
                                await google_drive.upload_file_content(
                                    f.read(),
                                    os.path.basename(file_path),
                                    session_folder_id
                                )
                                uploaded_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to upload session file {file_path}: {e}")
            
            # Create processing summary
            summary_content = f"""Batch Processing Summary
{'=' * 50}

Company: {company_name}
Form Type: {form_type}
Quarters: {quarters_str}
Processing Date: {datetime.now().strftime("%B %d, %Y %I:%M %p")}
Total Files: {processing_stats.get('total_files', len(session_ids))}
Total Documents: {processing_stats.get('total_documents', len(all_extractions))}
Files Uploaded: {uploaded_count}

Output Folder: {output_folder_name}

Folder Structure:
----------------
- all_extractions.json (consolidated data from all documents)
- archives/
  - individual/
    - uncompressed/
      - cleaned/ (files with pages removed)
      - raw/ (original files)
    - compressed/
      - cleaned/ (compressed versions of cleaned files)
      - raw/ (compressed versions of raw files)
  - combined_uncompressed.pdf {' (created)' if batch_files.get('combined_uncompressed') else ' (not created)'}
  - combined_compressed.pdf {' (created)' if batch_files.get('combined_compressed') else ' (not created)'}
- organized_by_quarter/
  - Files organized by year/quarter structure
- session_files/ (original processing sessions for debugging)

Processing Statistics:
---------------------
Total Processing Time: {processing_stats.get('total_processing_time', 0):.1f}s
Total Pages Removed: {processing_stats.get('total_pages_removed', 0)}
Average Time per File: {processing_stats.get('average_time_per_file', 0):.1f}s
Quarters Organized: {len(batch_files.get('organized_paths', {}))}

Notes:
------
- All files are organized in the '{output_folder_name}' folder
- The 'all_extractions.json' file contains all extracted data from all documents
- Individual files are in 'archives/individual' with cleaned/raw organization:
  - uncompressed/cleaned: Files with pages removed during cleaning
  - uncompressed/raw: Original files without cleaning
  - compressed/cleaned: Compressed versions of cleaned files
  - compressed/raw: Compressed versions of raw files
- Files are also organized by quarter in 'organized_by_quarter' folder
- Original session files are preserved in 'session_files' for debugging

End of Batch Processing Summary
{'=' * 50}
"""
            
            await google_drive.upload_file_content(
                summary_content.encode('utf-8'),
                "batch_processing_summary.txt",
                files_produced_folder_id
            )
            uploaded_count += 1
            
            # Get the web link for the folder
            folder_link = f"https://drive.google.com/drive/folders/{batch_folder_id}"
            
            return {
                "success": True,
                "folder_id": batch_folder_id,
                "folder_link": folder_link,
                "folder_name": main_folder_name,
                "files_uploaded": uploaded_count,
                "output_folder": output_folder_name
            }
            
        except Exception as e:
            logger.error(f"Error uploading batch files to Google Drive: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# New endpoints for v12

@app.post("/api/initiate-batch")
async def initiate_batch(request: dict = Body(...)):
    """Initiate a new batch for processing multiple files"""
    file_count = request.get("file_count", 0)
    
    if file_count <= 0:
        return JSONResponse(content={"error": "Invalid file count"}, status_code=400)
    
    batch_id = str(uuid.uuid4())
    await create_batch(batch_id, file_count, skip_individual_gdrive=True)
    
    logger.info(f"Initialized batch {batch_id} for {file_count} files")
    
    return JSONResponse(content={
        "batch_id": batch_id,
        "file_count": file_count,
        "status": "active"
    })

@app.post("/api/finalize-batch")
async def finalize_batch(request: dict = Body(...)):
    """Finalize batch processing - combine all extractions and upload to Google Drive"""
    batch_id = request.get("batch_id")
    
    batch_info = await get_batch_info(batch_id)
    if not batch_id or not batch_info:
        logger.error(f"Invalid batch ID for finalization: {batch_id}")
        active_batches = await db_state_manager.get_active_batches()
        logger.info(f"Active batches: {[b['batch_id'] for b in active_batches]}")
        return JSONResponse(content={"error": f"Invalid batch ID: {batch_id}"}, status_code=400)
    
    # Check if all files are completed
    if batch_info["completed_files"] < batch_info["total_files"]:
        return JSONResponse(content={
            "error": f"Batch not complete. {batch_info['completed_files']}/{batch_info['total_files']} files processed"
        }, status_code=400)
    
    # Check if we have the processing session IDs
    if "processing_session_ids" not in batch_info:
        logger.error(f"Batch {batch_id} missing processing_session_ids")
        return JSONResponse(content={"error": "Batch processing not properly completed"}, status_code=400)
    
    try:
        # Get the session IDs from the batch
        session_ids = batch_info.get("processing_session_ids", [])
        
        # Extract Google Drive and case information from batch info
        google_drive_folder_id = batch_info.get("google_drive_folder_id")
        case_id = batch_info.get("case_id")
        output_folder = batch_info.get("output_folder", "files_produced")
        
        logger.info(f"Finalizing batch {batch_id} with {len(session_ids)} sessions, folder_id: {google_drive_folder_id}, case_id: {case_id}")
        
        # Call combine_results to do the actual combining and Google Drive upload
        combine_request = {
            "session_ids": session_ids,
            "google_drive_folder_id": google_drive_folder_id,
            "case_id": case_id,
            "output_folder": output_folder
        }
        combine_response = await combine_results(combine_request)
        
        # Extract the response data - combine_results returns a JSONResponse
        if isinstance(combine_response, JSONResponse):
            combine_result = json.loads(combine_response.body.decode('utf-8'))
        else:
            combine_result = combine_response
        
        # Get Google Drive result from the combine response
        google_drive_result = combine_result.get("google_drive_upload", None)
        
        # Clean up batch tracking
        await delete_batch(batch_id)
        
        # Return the combined result with batch info
        return JSONResponse(content={
            "success": True,
            "batch_id": batch_id,
            "total_files": batch_info["total_files"],
            "total_documents": combine_result.get("total_documents", 0),
            "batch_folder": combine_result.get("batch_folder"),
            "company_names": combine_result.get("company_names", []),
            "form_types": combine_result.get("form_types", []),
            "google_drive_upload": google_drive_result,
            "files_location": combine_result.get("files_location", {})
        })
        
    except Exception as e:
        logger.error(f"Error finalizing batch: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get the status of a batch processing job"""
    batch_info = await get_batch_info(batch_id)
    
    if not batch_info:
        return JSONResponse(
            content={"error": f"Batch {batch_id} not found"},
            status_code=404
        )
    
    # Return batch status information
    return JSONResponse(content={
        "batch_id": batch_id,
        "total_files": batch_info["total_files"],
        "completed_files": batch_info["completed_files"],
        "is_complete": batch_info["completed_files"] >= batch_info["total_files"],
        "session_ids": batch_info.get("session_ids", []),
        "created_at": batch_info.get("created_at", ""),
        "status": "completed" if batch_info["completed_files"] >= batch_info["total_files"] else "processing"
    })

@app.post("/api/process-gdrive")
async def process_document_gdrive(request: dict = Body(...)):
    """Process document directly from Google Drive"""
    case_id = request.get("case_id")
    google_drive_folder_id = request.get("google_drive_folder_id")
    google_drive_file_id = request.get("google_drive_file_id")
    file_name = request.get("file_name", "document.pdf")
    selected_schema = request.get("selected_schema", "941-X")
    target_size_mb = float(request.get("target_size_mb", 9.0))
    output_folder = request.get("output_folder", "files_produced")
    process_from_gdrive = request.get("process_from_gdrive", True)
    
    session_id = str(uuid.uuid4())
    active_sessions.add(session_id)
    
    try:
        # Create processor with custom output folder
        processor = EnhancedDocumentProcessor(session_id)
        processor.case_id = case_id
        processor.google_drive_folder_id = google_drive_folder_id
        processor.output_folder = output_folder
        
        # Store in active processors for status tracking
        active_processors[session_id] = processor
        
        # Update status in gdrive router
        from google_drive_routes import processing_status_db
        processing_status_db[session_id] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "downloading",
            "progress": 0,
            "current_step": "downloading_from_gdrive",
            "gdrive_folder_id": google_drive_folder_id,
            "case_id": case_id,
            "metadata": {
                "output_folder": output_folder,
                "file_name": file_name,
                "google_drive_file_id": google_drive_file_id
            }
        }
        
        # Download file from Google Drive
        async with GoogleDriveIntegration() as google_drive:
            logger.info(f"Downloading file {google_drive_file_id} from Google Drive")
            pdf_content = await google_drive.download_file(google_drive_file_id)
            
            # Update status
            processing_status_db[session_id]["status"] = "processing"
            processing_status_db[session_id]["progress"] = 10
            processing_status_db[session_id]["current_step"] = "processing_document"
            
            # Process the document
            result = await processor.process_with_file_organizer(
                pdf_content,
                file_name,
                selected_schema,
                None,  # expected_value
                target_size_mb,
                False  # Don't skip Google Drive upload
            )
            
            # Update final status
            processing_status_db[session_id]["status"] = "completed"
            processing_status_db[session_id]["progress"] = 100
            processing_status_db[session_id]["current_step"] = "complete"
            processing_status_db[session_id]["google_drive_result"] = result.get("google_drive_upload")
            
            return JSONResponse(content={
                "session_id": session_id,
                "status": "completed",
                "result": result,
                "google_drive_upload": result.get("google_drive_upload")
            })
            
    except Exception as e:
        logger.error(f"Error processing from Google Drive: {e}")
        processing_status_db[session_id]["status"] = "failed"
        processing_status_db[session_id]["error"] = str(e)
        
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    finally:
        active_sessions.discard(session_id)

@app.post("/api/process-batch-gdrive")
async def process_batch_gdrive(request: dict = Body(...)):
    """Process multiple documents from Google Drive as a batch"""
    case_id = request.get("case_id")
    google_drive_folder_id = request.get("google_drive_folder_id")
    files = request.get("files", [])
    selected_schema = request.get("selected_schema", "941-X")
    target_size_mb = float(request.get("target_size_mb", 9.0))
    output_folder = request.get("output_folder", "files_produced")
    
    # Initialize batch
    batch_id = str(uuid.uuid4())
    
    # Create batch tracking
    await create_batch(batch_id, len(files), 
                      skip_individual_gdrive=True,
                      output_folder=output_folder,
                      case_id=case_id,
                      google_drive_folder_id=google_drive_folder_id)
    
    # Create a session for the batch
    session_id = str(uuid.uuid4())
    active_sessions.add(session_id)
    await add_session_to_batch(batch_id, session_id)
    
    # Update status in gdrive router
    from google_drive_routes import processing_status_db
    processing_status_db[session_id] = {
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": "processing_batch",
        "progress": 0,
        "current_step": "downloading_files",
        "gdrive_folder_id": google_drive_folder_id,
        "metadata": {
            "case_id": case_id,
            "batch_id": batch_id,
            "file_count": len(files),
            "output_folder": output_folder
        }
    }
    
    try:
        # Create processor for batch
        processor = EnhancedDocumentProcessor(session_id)
        processor.case_id = case_id
        processor.google_drive_folder_id = google_drive_folder_id
        processor.output_folder = output_folder
        
        # Store in active processors
        active_processors[session_id] = processor
        
        # Process files in background to avoid timeout
        async def process_batch_async():
            try:
                session_ids = []
                
                async with GoogleDriveIntegration() as google_drive:
                    for idx, file_info in enumerate(files):
                        file_id = file_info.get('googleDriveFileId', file_info.get('google_drive_file_id'))
                        file_name = file_info.get('fileName', file_info.get('file_name', f'document_{idx}.pdf'))
                        
                        # Update progress
                        progress = int((idx / len(files)) * 50)  # 0-50% for downloading
                        processing_status_db[session_id]["progress"] = progress
                        processing_status_db[session_id]["current_step"] = f"downloading_file_{idx+1}_of_{len(files)}"
                        
                        # Download file
                        logger.info(f"Downloading file {file_id} ({file_name}) from Google Drive")
                        try:
                            pdf_content = await google_drive.download_file(file_id)
                            logger.info(f"Successfully downloaded {len(pdf_content)} bytes for {file_name}")
                        except Exception as e:
                            logger.error(f"Error downloading file {file_id}: {e}")
                            continue
                        
                        # Create a sub-session for this file
                        file_session_id = str(uuid.uuid4())
                        active_sessions.add(file_session_id)
                        await add_session_to_batch(batch_id, file_session_id)
                        session_ids.append(file_session_id)
                        
                        # Create processor for this file
                        file_processor = EnhancedDocumentProcessor(file_session_id)
                        file_processor.case_id = case_id
                        file_processor.google_drive_folder_id = google_drive_folder_id
                        file_processor.output_folder = output_folder
                        file_processor.batch_id = batch_id  # Set batch_id for proper tracking
                        
                        # Save session metadata to database
                        await db_state_manager.save_processing_status(
                            file_session_id,
                            batch_id,
                            "processing",
                            [],  # processed_files
                            {},  # extractions
                            1,   # total_files
                            {
                                "case_id": case_id,
                                "google_drive_folder_id": google_drive_folder_id,
                                "output_folder": output_folder,
                                "file_name": file_name,
                                "created_at": datetime.now().isoformat()
                            }
                        )
                        
                        # Process file
                        logger.info(f"Processing file {file_name} with session {file_session_id}")
                        try:
                            result = await file_processor.process_with_file_organizer(
                                pdf_content,
                                file_name,
                                selected_schema,
                                None,  # expected_value
                                target_size_mb,
                                True   # Skip individual gdrive upload for batch
                            )
                            logger.info(f"Processing complete for {file_name}, result: {result.get('status', 'unknown') if result else 'None'}")
                        except Exception as e:
                            logger.error(f"Error processing file {file_name}: {e}")
                            logger.error(traceback.format_exc())
                            continue
                        
                        # Verify that the output directory and extractions exist
                        base_dir = os.path.dirname(os.path.abspath(__file__))
                        session_output_dir = os.path.join(base_dir, "processed_documents", file_session_id)
                        extractions_path = os.path.join(session_output_dir, "extractions", "all_extractions.json")
                        
                        if os.path.exists(extractions_path):
                            logger.info(f"Verified extraction file exists: {extractions_path}")
                        else:
                            logger.error(f"ERROR: Extraction file not found after processing: {extractions_path}")
                            logger.error(f"Session directory exists: {os.path.exists(session_output_dir)}")
                            if os.path.exists(session_output_dir):
                                logger.error(f"Directory contents: {os.listdir(session_output_dir)}")
                        
                        await update_batch_completion(batch_id, file_session_id)
                
                # Update status to indicate batch processing is complete
                # But NOT finalized - that happens in /api/finalize-batch
                processing_status_db[session_id]["progress"] = 100
                processing_status_db[session_id]["current_step"] = "batch_processing_complete"
                processing_status_db[session_id]["status"] = "completed"
                processing_status_db[session_id]["metadata"]["batch_ready_for_finalization"] = True
                
                # Store the session IDs in the batch for finalization
                batch_state = await db_state_manager.get_batch_state(batch_id)
                if batch_state:
                    batch_state['metadata']['processing_session_ids'] = session_ids
                    await db_state_manager.save_batch_state(batch_id, batch_state['session_ids'], 
                                                           batch_state['status'], batch_state['metadata'])
                
                logger.info(f"Batch {batch_id} processing complete, ready for finalization")
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                processing_status_db[session_id]["status"] = "failed"
                processing_status_db[session_id]["error"] = str(e)
        
        # Start async processing
        asyncio.create_task(process_batch_async())
        
        return JSONResponse(content={
            "batch_id": batch_id,
            "session_id": session_id,
            "status": "processing",
            "message": f"Processing {len(files)} files from Google Drive in background",
            "files": files
        })
        
    except Exception as e:
        logger.error(f"Error initializing batch: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

def is_cleaned_file(file_info):
    """Determine if a file is cleaned (had pages removed) or raw"""
    # Check filename for cleaned_ prefix
    if file_info.get("filename", "").startswith("cleaned_"):
        return True
    
    # Check metadata for was_cleaned flag
    metadata = file_info.get("metadata", {})
    if metadata.get("was_cleaned", False):
        return True
    
    # Check if pages were removed
    if metadata.get("pages_removed", 0) > 0:
        return True
    
    return False


@app.post("/api/combine-results")
async def combine_results(request: dict = Body(...)):
    """Combine results from multiple processed files"""
    session_ids = request.get("session_ids", [])
    
    # Extract Google Drive and case information from request
    request_google_drive_folder_id = request.get("google_drive_folder_id")
    request_case_id = request.get("case_id")
    request_output_folder = request.get("output_folder", "files_produced")
    
    if not session_ids:
        return JSONResponse(content={"error": "No session IDs provided"}, status_code=400)
    
    combined_session_id = str(uuid.uuid4())
    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined_output_dir = os.path.join(base_dir, "processed_documents", combined_session_id)
    os.makedirs(combined_output_dir, exist_ok=True)
    
    try:
        # Collect all extracted documents and statistics
        all_extracted_documents = []
        total_pages_removed = 0
        total_processing_time = 0
        all_company_names = set()
        all_form_types = set()
        all_quarters = set()
        
        # Collect all files from sessions
        all_cleaned_pdfs = []
        all_compressed_pdfs = []
        all_extraction_jsons = []
        
        # Process each session
        for session_id in session_ids:
            session_dir = os.path.join(base_dir, "processed_documents", session_id)
            logger.info(f"Checking session directory: {session_dir}")
            
            if not os.path.exists(session_dir):
                logger.error(f"Session directory does not exist: {session_dir}")
                continue
                
            # Look for extraction files
            extraction_dir = os.path.join(session_dir, "extractions")
            if os.path.exists(extraction_dir):
                logger.info(f"Found extraction directory: {extraction_dir}")
                # Read all_extractions.json if it exists
                all_extractions_path = os.path.join(extraction_dir, "all_extractions.json")
                if os.path.exists(all_extractions_path):
                    with open(all_extractions_path, 'r') as f:
                        extractions_data = json.load(f)
                        for doc in extractions_data.get("documents", []):
                            # Add metadata about which session this came from
                            doc["session_id"] = session_id
                            doc["original_filename"] = doc.get("filename", "unknown.pdf")
                            all_extracted_documents.append(doc)
                            
                            # Extract company name and form type
                            if "data" in doc and "employerInfo" in doc["data"]:
                                company_name = doc["data"]["employerInfo"].get("name")
                                if company_name:
                                    all_company_names.add(company_name)
                            
                            # Extract quarters
                            if "data" in doc:
                                correction_info = doc["data"].get("correctionInfo", {})
                                quarter = correction_info.get("quarter")
                                year = correction_info.get("year")
                                if quarter and year:
                                    all_quarters.add(f"Q{quarter} {year}")
            
            # Collect cleaned and compressed PDFs from session
            # Files are organized directly in year/quarter folders, not in cleaned/ or organized_by_quarter/
            logger.info(f"Scanning session directory for PDFs: {session_dir}")
            
            # Walk through the entire session directory to find PDFs
            for root, dirs, files in os.walk(session_dir):
                # Skip extraction and logs directories
                if "extractions" in root or "logs" in root:
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if it's a PDF file
                    if file.endswith(".pdf"):
                        # Read the file content
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Determine file type based on filename and path
                        if "_compressed" in file:
                            # This is explicitly a compressed file
                            logger.info(f"Found compressed PDF: {file}")
                            all_compressed_pdfs.append({
                                "filename": file,
                                "content": content,
                                "session_id": session_id,
                                "rel_path": os.path.relpath(file_path, session_dir)
                            })
                        elif "_uncompressed" in file:
                            # This is explicitly an uncompressed file
                            logger.info(f"Found uncompressed PDF: {file}")
                            all_cleaned_pdfs.append({
                                "filename": file,
                                "content": content,
                                "session_id": session_id,
                                "rel_path": os.path.relpath(file_path, session_dir)
                            })
                        elif "cleaned_" in file:
                            # Files starting with "cleaned_" are uncompressed
                            logger.info(f"Found cleaned PDF: {file}")
                            all_cleaned_pdfs.append({
                                "filename": file,
                                "content": content,
                                "session_id": session_id,
                                "rel_path": os.path.relpath(file_path, session_dir)
                            })
                        else:
                            # For other PDFs, check the directory structure
                            rel_path = os.path.relpath(root, session_dir)
                            
                            # If it's in year/quarter structure, it's an organized file
                            if re.match(r'\d{4}/Q\d', rel_path):
                                logger.info(f"Found organized PDF in {rel_path}: {file}")
                                # Add to cleaned PDFs (these are typically uncompressed)
                                all_cleaned_pdfs.append({
                                    "filename": file,
                                    "content": content,
                                    "session_id": session_id,
                                    "rel_path": os.path.relpath(file_path, session_dir)
                                })
                            elif "cleaned" in root or "uncompressed" in root:
                                # If in a cleaned or uncompressed directory
                                logger.info(f"Found PDF in cleaned/uncompressed dir: {file}")
                                all_cleaned_pdfs.append({
                                    "filename": file,
                                    "content": content,
                                    "session_id": session_id,
                                    "rel_path": os.path.relpath(file_path, session_dir)
                                })
                            elif "compressed" in root:
                                # If in a compressed directory
                                logger.info(f"Found PDF in compressed dir: {file}")
                                all_compressed_pdfs.append({
                                    "filename": file,
                                    "content": content,
                                    "session_id": session_id,
                                    "rel_path": os.path.relpath(file_path, session_dir)
                                })
                            elif not any(skip in root for skip in ["original", "temp", "cache", "logs"]):
                                # Default: treat as cleaned/uncompressed unless in a skip directory
                                logger.info(f"Found PDF (defaulting to cleaned): {file}")
                                all_cleaned_pdfs.append({
                                    "filename": file,
                                    "content": content,
                                    "session_id": session_id,
                                    "rel_path": os.path.relpath(file_path, session_dir)
                                })
                    elif file.endswith("_extraction.json"):
                        # Extraction JSON files
                        logger.info(f"Found extraction JSON: {file}")
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        all_extraction_jsons.append({
                            "filename": file,
                            "content": content,
                            "session_id": session_id,
                            "rel_path": os.path.relpath(file_path, session_dir)
                        })
        
        # Create organized structure at batch level
        logger.info(f"Collected {len(all_cleaned_pdfs)} cleaned PDFs and {len(all_compressed_pdfs)} compressed PDFs")
        
        # Create temporary processor for organization
        temp_processor = EnhancedDocumentProcessor(combined_session_id)
        temp_processor.output_dir = combined_output_dir
        
        # Prepare documents for organization (match extraction data with PDFs)
        batch_documents = []
        for pdf_info in all_cleaned_pdfs:
            # Find matching extraction data
            matching_doc = None
            for doc in all_extracted_documents:
                if doc.get("original_filename") == pdf_info["filename"] or \
                   doc.get("filename") == pdf_info["filename"]:
                    matching_doc = doc
                    break
            
            if matching_doc:
                batch_documents.append({
                    "filename": pdf_info["filename"],
                    "content": pdf_info["content"],
                    "extracted_data": matching_doc.get("data", {}),
                    "metadata": matching_doc.get("metadata", {})
                })
        
        # Organize by quarter at batch level
        organized_paths = {}
        if batch_documents:
            organized_paths = temp_processor.organize_by_quarter_with_metadata(batch_documents, combined_output_dir)
            logger.info(f"Organized {len(batch_documents)} documents into {len(organized_paths)} quarters")
        
        # Create combined PDFs if multiple documents
        combined_uncompressed_content = None
        combined_compressed_content = None
        
        if len(all_cleaned_pdfs) > 1:
            try:
                # Combine all cleaned PDFs
                file_organizer = FileOrganizerIntegrationV2()
                combined_uncompressed_content = await file_organizer.combine_pdfs(
                    [pdf["content"] for pdf in all_cleaned_pdfs],
                    "combined_uncompressed.pdf"
                )
                logger.info("Created combined uncompressed PDF")
                
                # Combine all compressed PDFs
                if all_compressed_pdfs:
                    combined_compressed_content = await file_organizer.combine_pdfs(
                        [pdf["content"] for pdf in all_compressed_pdfs],
                        "combined_compressed.pdf"
                    )
                    logger.info("Created combined compressed PDF")
            except Exception as e:
                logger.warning(f"Failed to create combined PDFs: {e}")
        
        # Create a unified extraction file with all data
        unified_extractions = {
            "batch_id": combined_session_id,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(session_ids),
            "total_documents": len(all_extracted_documents),
            "documents": all_extracted_documents,
            "processing_info": {
                "cleaned_pdfs": len(all_cleaned_pdfs),
                "compressed_pdfs": len(all_compressed_pdfs),
                "quarters_organized": list(organized_paths.keys()),
                "has_combined_files": combined_uncompressed_content is not None
            }
        }
        
        # Save unified extractions
        unified_path = os.path.join(combined_output_dir, "unified_extractions.json")
        with open(unified_path, 'w') as f:
            json.dump(unified_extractions, f, indent=2)
        
        # Create summary report
        summary = {
            "batch_id": combined_session_id,
            "timestamp": datetime.now().isoformat(),
            "total_files": len(session_ids),
            "successful_extractions": len(all_extracted_documents),
            "company_names": list(all_company_names),
            "form_types": list(all_form_types),
            "quarters": list(all_quarters)
        }
        
        # Try to upload to Google Drive if we have a case folder
        google_drive_result = None
        
        # Use provided values from request first, then fall back to active processors
        google_drive_folder_id = request_google_drive_folder_id
        case_id = request_case_id
        output_folder = request_output_folder
        
        # If not provided in request, try to get from active processors
        if not google_drive_folder_id:
            logger.info("Google Drive folder ID not provided in request, checking active processors")
            for session_id in session_ids:
                # Try to get from active processors
                if session_id in active_processors:
                    processor = active_processors[session_id]
                    if hasattr(processor, 'google_drive_folder_id') and processor.google_drive_folder_id:
                        google_drive_folder_id = processor.google_drive_folder_id
                        case_id = getattr(processor, 'case_id', None)
                        output_folder = getattr(processor, 'output_folder', "files_produced")
                        logger.info(f"Found folder ID from active processor: {google_drive_folder_id}")
                        break
        else:
            logger.info(f"Using Google Drive folder ID from request: {google_drive_folder_id}")
        
        if google_drive_folder_id:
            logger.info(f"Preparing Google Drive upload with folder_id: {google_drive_folder_id}, case_id: {case_id}")
            try:
                # Create a temporary processor for the upload
                temp_processor = EnhancedDocumentProcessor(combined_session_id)
                temp_processor.google_drive_folder_id = google_drive_folder_id
                temp_processor.case_id = case_id
                temp_processor.output_folder = output_folder
                temp_processor.output_dir = combined_output_dir
                
                # Prepare batch files for upload
                batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                batch_dir = os.path.join(combined_output_dir, f"batch_{batch_timestamp}")
                os.makedirs(batch_dir, exist_ok=True)
                
                # Create directory structure on disk for the batch
                # This is necessary because upload_batch_files_to_google_drive expects files on disk
                
                # Create archives directory structure with cleaned/raw subdirectories
                archives_dir = os.path.join(combined_output_dir, "archives")
                individual_dir = os.path.join(archives_dir, "individual")
                
                # Create subdirectories for cleaned and raw files
                uncompressed_cleaned_dir = os.path.join(individual_dir, "uncompressed", "cleaned")
                uncompressed_raw_dir = os.path.join(individual_dir, "uncompressed", "raw")
                compressed_cleaned_dir = os.path.join(individual_dir, "compressed", "cleaned")
                compressed_raw_dir = os.path.join(individual_dir, "compressed", "raw")
                
                # Create all directories
                os.makedirs(uncompressed_cleaned_dir, exist_ok=True)
                os.makedirs(uncompressed_raw_dir, exist_ok=True)
                os.makedirs(compressed_cleaned_dir, exist_ok=True)
                os.makedirs(compressed_raw_dir, exist_ok=True)
                
                # Save all uncompressed PDFs to disk in cleaned/raw subdirectories
                logger.info(f"Saving {len(all_cleaned_pdfs)} uncompressed PDFs to cleaned/raw subdirectories")
                for pdf_info in all_cleaned_pdfs:
                    # Determine if file is cleaned or raw
                    if is_cleaned_file(pdf_info):
                        target_dir = uncompressed_cleaned_dir
                        logger.debug(f"File {pdf_info['filename']} classified as cleaned")
                    else:
                        target_dir = uncompressed_raw_dir
                        logger.debug(f"File {pdf_info['filename']} classified as raw")
                    
                    file_path = os.path.join(target_dir, pdf_info["filename"])
                    with open(file_path, 'wb') as f:
                        f.write(pdf_info["content"])
                
                # Save all compressed PDFs to disk in cleaned/raw subdirectories
                logger.info(f"Saving {len(all_compressed_pdfs)} compressed PDFs to cleaned/raw subdirectories")
                for pdf_info in all_compressed_pdfs:
                    # Determine if file is cleaned or raw
                    if is_cleaned_file(pdf_info):
                        target_dir = compressed_cleaned_dir
                        logger.debug(f"Compressed file {pdf_info['filename']} classified as cleaned")
                    else:
                        target_dir = compressed_raw_dir
                        logger.debug(f"Compressed file {pdf_info['filename']} classified as raw")
                    
                    file_path = os.path.join(target_dir, pdf_info["filename"])
                    with open(file_path, 'wb') as f:
                        f.write(pdf_info["content"])
                
                # Save combined PDFs if they exist
                if combined_uncompressed_content:
                    combined_uncomp_path = os.path.join(archives_dir, "combined_uncompressed.pdf")
                    with open(combined_uncomp_path, 'wb') as f:
                        f.write(combined_uncompressed_content)
                    logger.info("Saved combined uncompressed PDF")
                
                if combined_compressed_content:
                    combined_comp_path = os.path.join(archives_dir, "combined_compressed.pdf")
                    with open(combined_comp_path, 'wb') as f:
                        f.write(combined_compressed_content)
                    logger.info("Saved combined compressed PDF")
                
                # Create organized_by_quarter structure on disk
                if organized_paths:
                    organized_dir = os.path.join(combined_output_dir, "organized_by_quarter")
                    os.makedirs(organized_dir, exist_ok=True)
                    
                    # The organized_paths dict contains quarter paths as keys (e.g., "2024/Q1")
                    # We need to recreate the files in this structure
                    for quarter_path, filenames in organized_paths.items():
                        if quarter_path != "unorganized" and "/" in quarter_path:
                            year, quarter = quarter_path.split("/")
                            quarter_dir = os.path.join(organized_dir, year, quarter)
                            os.makedirs(quarter_dir, exist_ok=True)
                            
                            # Find and copy files for this quarter
                            # Files are in all_cleaned_pdfs and all_compressed_pdfs with matching names
                            for filename in filenames:
                                # Look for the file in our collected PDFs
                                for pdf_list in [all_cleaned_pdfs, all_compressed_pdfs]:
                                    for pdf_info in pdf_list:
                                        if pdf_info["filename"] == filename:
                                            file_path = os.path.join(quarter_dir, filename)
                                            with open(file_path, 'wb') as f:
                                                f.write(pdf_info["content"])
                                            logger.info(f"Saved {filename} to {quarter_path}")
                                            break
                
                # Save extraction JSONs to compressed cleaned/raw subdirectories
                logger.info(f"Saving {len(all_extraction_jsons)} extraction JSONs to cleaned/raw subdirectories")
                for json_info in all_extraction_jsons:
                    # Determine if the corresponding PDF is cleaned or raw
                    # JSON files typically have _extraction.json suffix, find the base PDF name
                    base_filename = json_info["filename"].replace("_extraction.json", ".pdf")
                    
                    # Check if we can find metadata from the corresponding PDF
                    is_cleaned = False
                    for pdf_info in all_compressed_pdfs:
                        if pdf_info["filename"] == base_filename or pdf_info["filename"].replace("_compressed", "") == base_filename:
                            is_cleaned = is_cleaned_file(pdf_info)
                            break
                    
                    # If still not found, check filename pattern
                    if not is_cleaned and base_filename.startswith("cleaned_"):
                        is_cleaned = True
                    
                    # Save to appropriate subdirectory
                    if is_cleaned:
                        target_dir = compressed_cleaned_dir
                        logger.debug(f"JSON file {json_info['filename']} classified as cleaned")
                    else:
                        target_dir = compressed_raw_dir
                        logger.debug(f"JSON file {json_info['filename']} classified as raw")
                    
                    file_path = os.path.join(target_dir, json_info["filename"])
                    with open(file_path, 'wb') as f:
                        f.write(json_info["content"])
                
                # Save all_extractions.json
                all_extractions_path = os.path.join(combined_output_dir, "all_extractions.json")
                with open(all_extractions_path, 'w') as f:
                    json.dump(unified_extractions, f, indent=2)
                logger.info("Saved all_extractions.json")
                
                # Prepare files for upload with proper structure
                combined_files_to_upload = {
                    "sessions": [],  # Keep for debugging
                    "all_extractions": unified_extractions,
                    "individual_uncompressed": all_cleaned_pdfs,
                    "individual_compressed": all_compressed_pdfs,
                    "extraction_jsons": all_extraction_jsons,
                    "combined_uncompressed": combined_uncompressed_content,
                    "combined_compressed": combined_compressed_content,
                    "organized_paths": organized_paths
                }
                
                # Still collect session files for debugging
                for session_id in session_ids:
                    session_dir = os.path.join(base_dir, "processed_documents", session_id)
                    if os.path.exists(session_dir):
                        session_files = []
                        for root, dirs, files in os.walk(session_dir):
                            # Skip large directories to avoid duplicating files
                            if "cleaned" in root or "organized_by_quarter" in root:
                                continue
                            for file in files:
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, session_dir)
                                session_files.append((file_path, rel_path))
                        
                        if session_files:
                            combined_files_to_upload["sessions"].append({
                                "session_id": session_id,
                                "files": session_files
                            })
                
                # Calculate statistics
                batch_stats = {
                    "session_id": combined_session_id,
                    "batch_folder": batch_dir,
                    "batch_timestamp": batch_timestamp,
                    "total_files": len(session_ids),
                    "total_documents": len(all_extracted_documents),
                    "total_pages_removed": sum(doc.get("metadata", {}).get("pages_removed", 0) for doc in all_extracted_documents),
                    "total_processing_time": round(sum(doc.get("processing_time", 0) for doc in all_extracted_documents), 1),
                    "average_time_per_file": round(sum(doc.get("processing_time", 0) for doc in all_extracted_documents) / len(session_ids), 1) if session_ids else 0
                }
                
                # Extract first company name and form type for folder naming
                company_name = list(all_company_names)[0] if all_company_names else "Unknown_Company"
                form_type = "941X"  # Default form type
                quarters_str = ", ".join(sorted(all_quarters)) if all_quarters else "Multiple"
                
                # Upload to Google Drive
                async with GoogleDriveIntegration() as google_drive:
                    google_drive_result = await temp_processor.upload_batch_files_to_google_drive(
                        google_drive,
                        combined_files_to_upload,
                        all_extracted_documents,
                        company_name,
                        form_type,
                        quarters_str,
                        session_ids,
                        {
                            "batch_id": combined_session_id,
                            "total_files": len(session_ids),
                            "total_documents": len(all_extracted_documents),
                            "processing_timestamp": batch_timestamp
                        }
                    )
                    
                    if google_drive_result and google_drive_result.get("success"):
                        logger.info(f"Successfully uploaded batch to Google Drive: {google_drive_result.get('folder_link')}")
                
            except Exception as e:
                logger.error(f"Error uploading batch to Google Drive: {e}")
                logger.error(f"Folder ID: {google_drive_folder_id}, Case ID: {case_id}")
                google_drive_result = {"success": False, "error": str(e)}
        else:
            logger.warning(f"No Google Drive folder ID available for batch upload. Case ID: {case_id}")
            google_drive_result = {
                "success": False, 
                "error": "No Google Drive folder ID available",
                "reason": "folder_id_missing"
            }
        
        # Return combined results
        return JSONResponse(content={
            "success": True,
            "batch_id": combined_session_id,
            "session_ids": session_ids,
            "total_documents": len(all_extracted_documents),
            "company_names": list(all_company_names),
            "form_types": list(all_form_types),
            "quarters": list(all_quarters),
            "summary": summary,
            "google_drive_upload": google_drive_result,
            "files_location": {
                "unified_extractions": unified_path,
                "output_directory": combined_output_dir
            }
        })
        
    except Exception as e:
        logger.error(f"Error combining results: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/api/download-archive/{session_id}/{archive_name:path}")
async def download_archive(session_id: str, archive_name: str):
    """Download processed files - redirects to Google Drive folder"""
    # Since we no longer create local archives, return a message about Google Drive
    return JSONResponse(content={
        "message": "Files are now uploaded directly to Google Drive folders instead of ZIP archives.",
        "info": "Check the google_drive_upload field in the processing response for the folder link.",
        "deprecated": True
    })

@app.get("/api/gdrive/status/{session_id}")
async def get_gdrive_processing_status(session_id: str):
    """Get the status of Google Drive document processing"""
    from google_drive_routes import processing_status_db
    
    status = processing_status_db.get(session_id)
    if not status:
        # Try to get from database
        db_status = await db_state_manager.get_processing_status(session_id)
        if db_status:
            status = db_status.get('metadata', {})
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    return JSONResponse(content=status)

# Include the Google Drive routes
try:
    from google_drive_routes import router as gdrive_router
    app.include_router(gdrive_router, prefix="/api/gdrive")
    logger.info("Google Drive routes included successfully")
except ImportError:
    logger.warning("Google Drive routes module not found")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "status": "connected"
        })
        
        while True:
            # Use receive_json with a proper error handler
            try:
                data = await websocket.receive_text()
                if data:
                    # Handle incoming messages
                    message = json.loads(data)
                    
                    if message.get("type") == "confirm_form":
                        session_confirmations[session_id] = True
                    elif message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from {session_id}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
    finally:
        active_connections.pop(session_id, None)
        session_confirmations.pop(session_id, None)

@app.post("/process-documents-v3")
async def process_documents_v3(
    files: List[UploadFile] = File(...), 
    schema: str = Form("generic"),
    session_id: str = Form(None),
    expected_value: Optional[str] = Form(None)
):
    """Enhanced document processing with parallel extraction"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Add session to active sessions immediately
    active_sessions.add(session_id)
    
    logger.info(f"Starting processing for session {session_id} with {len(files)} files")
    processor = DocumentProcessor(session_id)
    
    # Set expected value if provided
    if expected_value:
        processor.expected_value = expected_value
        logger.info(f"Expected value set: {expected_value}")
    
    # Send initial time update
    await processor.send_time_update()
    
    # Start time update loop
    async def time_updater():
        while session_id in active_sessions:
            await asyncio.sleep(1)
            if session_id in active_connections:
                await processor.send_time_update()
    
    time_task = asyncio.create_task(time_updater())
    
    try:
        # Create processing plan
        detected_forms = []
        detection_results = []
        
        for file in files:
            content = await file.read()
            await file.seek(0)  # Reset for later use
            
            detection = await processor.detect_form_type_with_confirmation(content, schema)
            detection_results.append(detection)
            if detection.get("detected_form_type"):
                detected_forms.append(detection["detected_form_type"])
        
        await processor.create_processing_plan(len(files), detected_forms)
        
        all_results = []
        
        await processor.update_plan_step("validate", "in_progress")
        
        # Handle confirmation if needed
        needs_confirmation = any(d.get("needs_confirmation") for d in detection_results)
        auto_confirmed = False
        
        if needs_confirmation:
            # Check if any detection has high confidence for auto-confirmation
            high_confidence = any(d.get("confidence") == "high" for d in detection_results if d.get("detected_form_type"))
            
            if high_confidence:
                # Wait for auto-confirmation (5 seconds)
                await asyncio.sleep(5.5)
                auto_confirmed = True
                # Update schema if auto-confirmed
                for d in detection_results:
                    if d.get("confidence") == "high" and d.get("detected_form_type"):
                        schema = d["detected_form_type"]
                        break
                await processor.update_plan_step("validate", "completed", "Auto-confirmed template selection")
            else:
                # Wait for user response
                max_wait = 10  # seconds
                waited = 0
                
                while processor.waiting_for_confirmation and waited < max_wait:
                    await asyncio.sleep(0.5)
                    waited += 0.5
                    
                    # Check if session was confirmed
                    if session_id in session_confirmations:
                        processor.waiting_for_confirmation = False
                        break
                
                if processor.waiting_for_confirmation:
                    # Timeout - proceed with generic
                    logger.info(f"Confirmation timeout for session {session_id}")
                    await processor.update_plan_step("validate", "completed", "Using generic template")
                else:
                    await processor.update_plan_step("validate", "completed", "User confirmed selection")
        else:
            await processor.update_plan_step("validate", "completed")
        
        # Process each file with parallel extraction
        for file_idx, file in enumerate(files):
            content = await file.read()
            
            results = await processor.process_with_parallel_extraction(
                content, schema, file.filename
            )
            
            all_results.extend(results)
        
        # Perform value matching if expected value is provided
        if expected_value and all_results and schema == "941-X":
            await processor.update_plan_step("match", "in_progress", "Comparing values...")
            logger.info(f"Performing value matching for expected value: {expected_value}")
            # Use the first result for matching (assuming single document)
            if len(all_results) > 0:
                match_result = await processor.match_expected_value(all_results[0], expected_value)
                # Send match result via WebSocket
                await processor.send_update("value_match_result", match_result)
                
                # Update plan step based on result
                if match_result.get("is_match"):
                    await processor.update_plan_step("match", "completed", f"✅ Match found: ${match_result.get('found_value')}")
                else:
                    await processor.update_plan_step("match", "completed", "❌ No match found")
            
    except Exception as e:
        logger.error(f"Processing error for session {session_id}: {e}\n{traceback.format_exc()}")
        if "cancelled" not in str(e).lower():
            await processor.update_plan_step("complete", "failed", str(e)[:100])
    finally:
        time_task.cancel()
        active_sessions.discard(session_id)
        logger.info(f"Completed processing for session {session_id}")
    
    return JSONResponse({
        "session_id": session_id,
        "results": all_results,
        "total_documents": len(all_results),
        "status": "completed" if all_results else "failed",
        "processing_time": time.time() - processor.start_time
    })

@app.post("/cancel-processing/{session_id}")
async def cancel_processing(session_id: str):
    """Cancel an ongoing processing session"""
    if session_id in active_sessions:
        active_sessions.discard(session_id)
        logger.info(f"Cancelled processing for session {session_id}")
        
        # Notify the processor
        if session_id in active_connections:
            try:
                await active_connections[session_id].send_json({
                    "type": "cancelled",
                    "message": "Processing cancelled by user"
                })
            except:
                pass
                
        return {"status": "cancelled", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/confirm-form-selection")
async def confirm_form_selection(
    session_id: str = Body(...),
    use_detected_form: bool = Body(...),
    detected_form: str = Body(...),
    auto_confirmed: bool = Body(False)
):
    """Confirm whether to use detected form template"""
    session_confirmations[session_id] = True
    logger.info(f"Form selection confirmed for session {session_id}: {detected_form if use_detected_form else 'generic'}")
    
    return {
        "status": "confirmed",
        "use_detected_form": use_detected_form,
        "schema": detected_form if use_detected_form else "generic",
        "auto_confirmed": auto_confirmed
    }

# Legacy endpoint
@app.post("/process-pdf")
async def process_pdf_legacy(file: UploadFile = File(...), schema: str = Form("generic")):
    """Legacy endpoint"""
    results = await process_documents_v3([file], schema)
    response_data = results.body.decode('utf-8')
    parsed_data = json.loads(response_data)
    
    if parsed_data["results"]:
        return {"response": json.dumps(parsed_data["results"][0])}
    else:
        return {"response": json.dumps({"error": "No results"})}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "active_connections": len(active_connections),
        "debug_mode": DEBUG_MODE,
        "png_converter": {
            "available": png_converter._check_external_service(),
            "external_service_url": png_converter.external_service_url
        }
    }

@app.get("/png-converter-status")
async def png_converter_status():
    """Check PNG converter service status"""
    external_available = png_converter._check_external_service()
    return {
        "external_service_available": external_available,
        "external_service_url": png_converter.external_service_url,
        "fallback_method": "pdf2image (local)" if not external_available else "external service",
        "note": "PNG conversion is used as a last resort when PDF extraction fails"
    }

# Catch-all route for React app - must be last!
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serve React app for any unmatched routes"""
    # Skip API and WebSocket routes
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Serve index.html for client-side routing
    index_path = os.path.join(frontend_build_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # Fallback response if frontend not built
        return JSONResponse({"message": "Frontend not built. Please build the React app first."}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting enhanced document processing server v12 on port 4830")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info("Key improvements in v12:")
    logger.info("- Amounts formatted without commas (1000.00 not 1,000.00)")
    logger.info("- Reference schema included in extraction prompts")
    logger.info("- Clear indication that each page is part of compilation")
    logger.info("- Gemini 2.5 Pro for compilation with full schema")
    logger.info("- Huge token limits (60k extraction, 100k compilation)")
    logger.info("- Fixed 1s retry delay (not exponential)")
    logger.info("- Intelligent JSON restructuring with Gemini 2.5 Pro")
    logger.info("- Enhanced number formatting validation")
    uvicorn.run(app, host="0.0.0.0", port=4830)