import io
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from fastapi import FastAPI, UploadFile, File, Body, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
            {"id": "extract", "name": "🚀 Extract data from pages (parallel)", "status": "pending"},
            {"id": "compile", "name": "🔄 Compile and structure results", "status": "pending"},
            {"id": "quality", "name": "🔍 Verify extraction quality", "status": "pending"},
            {"id": "complete", "name": "✅ Finalize results", "status": "pending"}
        ])
        
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
                model="gemini-1.5-flash",
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
        max_retries = 2  # Reduced retries since we're doing parallel processing
        
        for attempt in range(max_retries):
            try:
                file_part = types.Part.from_bytes(
                    data=page_content,
                    mime_type="application/pdf"
                )
                
                # Load schema for context
                schema_json = self.load_schema(schema)
                if schema_json:
                    # Convert schema to example format for better extraction
                    schema_example = self.schema_to_example(schema_json)
                    schema_context = json.dumps(schema_example, indent=2)
                else:
                    schema_context = "Extract all data"
                
                # Page-specific context for critical pages
                page_context = ""
                if schema == "941-X":
                    if page_num == 0:  # Page 1
                        page_context = "\nIMPORTANT: This is the first page with employer info, EIN, and correction summary."
                    elif page_num == 1:  # Page 2 (0-indexed)
                        page_context = "\nIMPORTANT: This page typically contains Part 2 (ERC calculations) including lines 18a, 26a."
                    elif page_num == 2:  # Page 3
                        page_context = "\nIMPORTANT: This page typically contains qualified wages (lines 30-34) and supporting calculations."
                    elif page_num == 5:  # Page 6 (last page)
                        page_context = "\nNOTE: This may be an instructions page with no form fields to extract."
                
                # Comprehensive single-page extraction prompt
                extraction_prompt = f"""
You are extracting data from page {page_num + 1} of {total_pages} of a {schema} form.
{page_context}

TASK: Extract ACTUAL VALUES from the PDF document and fill them into this structure:

{schema_context}

CRITICAL INSTRUCTIONS:
1. EXTRACT VALUES: Look at the PDF and extract the actual text, numbers, and checkbox states you see
2. FILL THE STRUCTURE: Replace null values with the actual data from the document
3. PRESERVE NULLS: If a field is not visible on this page, keep it as null
4. DATA TYPES:
   - Numbers: Extract as numbers (123.45 not "123.45")
   - Text: Extract exactly as shown (preserve case, spacing)
   - Checkboxes: true if checked, false if not, null if not visible
   - Dates: Extract as strings in the format shown

EXAMPLES:
- If you see "EIN: 12-3456789" → {{"ein": "12-3456789"}}
- If you see "Business name: ABC Corporation" → {{"name": "ABC Corporation"}}
- If you see "Total: $50,000.00" → {{"total": 50000.00}}
- If you see a checked box next to "Quarterly return" → {{"quarterlyReturn": true}}

DO NOT:
- Return schema definitions (no "type", "properties", "description", etc.)
- Return empty objects {{}} - use null instead
- Make up data that's not visible on this page

For 941-X forms specifically:
- Extract three-column data (Column 1: Corrected Amount, Column 2: Original Amount, Column 3: Difference)
- Line numbers are critical - match them exactly

Return ONLY the JSON with extracted values.
"""
                
                # Try with text/plain first to get better control over output
                try:
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[extraction_prompt, file_part],
                        config={
                            "max_output_tokens": 8000,
                            "response_mime_type": "text/plain",
                            "temperature": 0.1
                        }
                    )
                    
                    # Check if response is valid
                    if not response or not response.text:
                        raise ValueError("Empty response from Gemini API")
                    
                    response_text = response.text.strip()
                    if not response_text:
                        raise ValueError("Empty response text from Gemini API")
                    
                except Exception as api_error:
                    logger.warning(f"API call failed for page {page_num + 1} (attempt {attempt + 1}): {api_error}")
                    raise api_error
                
                # Check if response is valid
                if not response_text:
                    raise ValueError("Empty response from Gemini API")
                
                # Try to clean up the response text before parsing
                response_text = response_text.strip()
                
                # Clean and extract JSON
                if response_text:
                    import re
                    
                    # Log first part of response for debugging
                    logger.debug(f"Raw response for page {page_num + 1} (first 100 chars): {response_text[:100]}")
                    
                    # Remove any markdown formatting
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].strip()
                    
                    # Basic cleanup - remove control characters
                    response_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_text)
                    
                    # Find JSON in the response
                    json_start = response_text.find('{') 
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        response_text = response_text[json_start:json_end]
                    
                    # Try to parse JSON using the robust parser
                    page_data = parse_json_safely(response_text, page_num + 1)
                    
                    if page_data:
                        logger.info(f"Successfully parsed JSON for page {page_num + 1}")
                    else:
                        logger.error(f"Failed to parse JSON for page {page_num + 1}, using empty structure")
                        page_data = {}
                else:
                    raise ValueError("Empty response text")
                # Check if this is likely an instructions page (minimal extracted data)
                if page_num == 5 and schema == "941-X":
                    # Count non-null values in the extracted data
                    non_null_count = sum(1 for v in str(page_data).split('"') if v and v != 'null' and v != '{' and v != '}')
                    if non_null_count < 10:  # Very few values extracted
                        logger.info(f"Page {page_num + 1} appears to be instructions/informational page with minimal data")
                
                return {
                    "page_number": page_num + 1,
                    "data": page_data,
                    "status": "success"
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_num + 1}: {e}")
                
                if attempt < max_retries - 1:
                    # Don't sleep in thread pool - just retry immediately
                    logger.info(f"Retrying page {page_num + 1}...")
                    continue
                else:
                    # Last resort: Try text extraction
                    logger.warning(f"Falling back to text extraction for page {page_num + 1}")
                    try:
                        # Use a simpler prompt for fallback
                        fallback_response = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=[f"Extract ALL text content from this PDF page. Return the complete text.", file_part],
                            config={
                                "max_output_tokens": 4000,
                                "response_mime_type": "text/plain",
                                "temperature": 0.1
                            }
                        )
                        
                        if fallback_response and fallback_response.text:
                            # Return a minimal structure with the raw text
                            logger.info(f"Fallback extraction succeeded for page {page_num + 1}")
                            return {
                                "page_number": page_num + 1,
                                "data": {
                                    "_raw_text": fallback_response.text[:2000],  # Limit text size
                                    "_extraction_method": "fallback_text",
                                    "_page_info": f"Page {page_num + 1} of {total_pages}",
                                    "_note": "Page extracted as text due to JSON parsing issues"
                                },
                                "status": "partial",
                                "error": f"JSON extraction failed: {str(e)}",
                                "retries": attempt + 1
                            }
                        else:
                            # If even fallback fails, return minimal data
                            logger.error(f"Even fallback extraction returned empty for page {page_num + 1}")
                            return {
                                "page_number": page_num + 1,
                                "data": {
                                    "_page": page_num + 1,
                                    "_extraction_failed": True,
                                    "_note": "Page could not be extracted"
                                },
                                "status": "partial",
                                "error": "Both JSON and text extraction failed",
                                "retries": attempt + 1
                            }
                    except Exception as fallback_error:
                        logger.error(f"Fallback extraction exception for page {page_num + 1}: {fallback_error}")
                        
                        # PNG conversion as absolute last resort
                        logger.warning(f"Attempting PNG conversion fallback for page {page_num + 1}")
                        try:
                            # Convert to PNG using asyncio in thread
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            png_bytes = loop.run_until_complete(
                                png_converter.convert_pdf_page_to_png(
                                    page_content, page_num, total_pages, quality="high"
                                )
                            )
                            
                            if png_bytes:
                                logger.info(f"PNG conversion successful for page {page_num + 1}, attempting extraction")
                                
                                # Create PNG part for Gemini
                                png_part = types.Part.from_bytes(
                                    data=png_bytes,
                                    mime_type="image/png"
                                )
                                
                                # Need to get schema context for PNG extraction
                                schema_json = self.load_schema(schema)
                                if schema_json:
                                    schema_example = self.schema_to_example(schema_json)
                                    png_schema_context = json.dumps(schema_example, indent=2)
                                else:
                                    png_schema_context = "Extract all data"
                                
                                # Try extraction with PNG
                                png_prompt = f"""Analyze this PNG image of a {schema} form page and extract actual data values.

TARGET STRUCTURE (fill with actual values from the image):
{png_schema_context}
{page_context}

This is page {page_num + 1} of {total_pages}. The image is a converted PDF page.

EXTRACTION RULES:
1. Extract ACTUAL VALUES visible in the image, not schema definitions
2. Replace null values with the actual data you see
3. For fields not visible, keep as null
4. Extract numbers as numbers, text as text
5. For checkboxes, use true/false
6. CRITICAL: Return actual data, NOT "type", "properties", etc.

Return ONLY the JSON with extracted values. If a field is not visible on this page, use null."""
                                
                                png_response = client.models.generate_content(
                                    model="gemini-1.5-flash",
                                    contents=[png_prompt, png_part],
                                    config={
                                        "max_output_tokens": 4000,
                                        "response_mime_type": "application/json",
                                        "temperature": 0.1
                                    }
                                )
                                
                                if png_response and png_response.text:
                                    try:
                                        png_data = json.loads(png_response.text)
                                        logger.info(f"PNG extraction successful for page {page_num + 1}")
                                        return {
                                            "page_number": page_num + 1,
                                            "data": png_data,
                                            "status": "success",
                                            "extraction_method": "png_conversion",
                                            "note": "Extracted via PNG conversion"
                                        }
                                    except json.JSONDecodeError:
                                        logger.error(f"PNG extraction returned invalid JSON for page {page_num + 1}")
                            
                            loop.close()
                            
                        except Exception as png_error:
                            logger.error(f"PNG conversion fallback failed for page {page_num + 1}: {png_error}")
                        
                        # Return minimal structure if all methods fail
                        return {
                            "page_number": page_num + 1,
                            "data": {
                                "_page": page_num + 1,
                                "_extraction_failed": True,
                                "_error": str(fallback_error),
                                "_attempted_png_conversion": True
                            },
                            "status": "partial",
                            "error": str(fallback_error),
                            "retries": attempt + 1
                        }

    async def parallel_extract_pages(self, pages: List[Tuple[int, bytes]], schema: str) -> List[Dict[str, Any]]:
        """Extract data from all pages in parallel"""
        await self.update_plan_step("extract", "in_progress")
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
            if status == "error":
                logger.warning(f"Page {page_num} extraction failed: {result.get('error', 'Unknown error')}")
            elif status == "partial":
                logger.info(f"Page {page_num} extracted with fallback method")
            else:
                logger.info(f"Page {page_num} extracted successfully")
            
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
        
        successful = sum(1 for r in results if r["status"] in ["success", "partial"])
        failed = sum(1 for r in results if r["status"] == "error")
        partial = sum(1 for r in results if r["status"] == "partial")
        
        if partial > 0:
            await self.update_plan_step("extract", "completed", 
                f"Extracted {successful} of {total_pages} pages ({partial} partial)")
        else:
            await self.update_plan_step("extract", "completed", 
                f"Extracted {successful} of {total_pages} pages")
        
        return results

    async def compile_results_with_pro(self, page_results: List[Dict[str, Any]], schema: str) -> Dict[str, Any]:
        """Compile all page results into final structured JSON using Pro model"""
        await self.update_plan_step("compile", "in_progress")
        
        try:
            # Prepare all page data
            all_page_data = []
            failed_pages = []
            partial_pages = []
            
            for result in page_results:
                if result["status"] == "success":
                    all_page_data.append({
                        "page": result["page_number"],
                        "data": result["data"]
                    })
                elif result["status"] == "partial":
                    # Include partial extractions
                    all_page_data.append({
                        "page": result["page_number"],
                        "data": result["data"],
                        "partial": True
                    })
                    partial_pages.append(result["page_number"])
                else:
                    failed_pages.append(result["page_number"])
            
            # If too many pages failed, try fallback approach
            if len(failed_pages) > len(page_results) * 0.3:  # More than 30% failed
                logger.warning(f"High failure rate: {len(failed_pages)} of {len(page_results)} pages failed")
                await self.send_status_update("compile", f"Warning: {len(failed_pages)} pages failed extraction")
            
            # Log extraction summary
            logger.info(f"Compilation summary: {len(all_page_data)} successful pages, {len(failed_pages)} failed, {len(partial_pages)} partial")
            logger.debug(f"Page data sizes: {[len(str(pd.get('data', {}))) for pd in all_page_data]}")
            
            # Load schema
            schema_json = self.load_schema(schema)
            if schema_json:
                # Convert schema to example format for better merging
                schema_example = self.schema_to_example(schema_json)
                schema_context = json.dumps(schema_example, indent=2)
            else:
                schema_context = ""
            
            # Compilation prompt for Pro model
            failed_pages_note = f"\nNOTE: Pages {failed_pages} failed extraction. Please infer missing data from context if possible." if failed_pages else ""
            partial_pages_note = f"\nNOTE: Pages {partial_pages} had partial extraction - raw text included. Extract structured data from the raw text." if partial_pages else ""
            
            compilation_prompt = f"""
TASK: Merge extracted data from {len(page_results)} pages of a {schema} form into ONE complete JSON document.
{failed_pages_note}{partial_pages_note}

PAGE DATA TO MERGE:
{json.dumps(all_page_data, indent=2)}

TARGET FORMAT:
{schema_context}

MERGING RULES:
1. COMBINE ALL DATA: Take all non-null values from all pages and merge them
2. PREFER NON-NULL: When the same field appears on multiple pages, use the non-null value
3. PRESERVE VALUES: Keep all extracted numbers, text, and boolean values exactly as they are
4. ORGANIZE BY STRUCTURE: Follow the target format's hierarchical organization
5. DO NOT include schema keywords like "type", "properties", "description"

SPECIAL RULES FOR 941-X:
- Preserve three-column data (corrected, original, difference)
- If you have corrected and original but not difference, calculate: difference = corrected - original
- Group all lines by their numbers (line1, line2, etc.)

OUTPUT REQUIREMENTS:
- Return ONLY the merged JSON combining all page data
- No explanations, no markdown, just pure JSON
- All values should be actual extracted data, not schema definitions

EXAMPLE:
If page 1 has: {{"employerInfo": {{"ein": "12-3456789"}}}}
And page 2 has: {{"employerInfo": {{"name": "ABC Corp"}}}}
Result should be: {{"employerInfo": {{"ein": "12-3456789", "name": "ABC Corp"}}}}
"""
            
            # Log prompt size for debugging
            prompt_size = len(compilation_prompt)
            logger.info(f"Compilation prompt size: {prompt_size} characters")
            
            # Save prompt to debug file if very large
            if prompt_size > 50000:
                debug_filename = f"compilation_prompt_{self.session_id}_{int(time.time())}.txt"
                debug_path = os.path.join(os.path.dirname(__file__), "debug", debug_filename)
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                with open(debug_path, 'w') as f:
                    f.write(compilation_prompt)
                logger.warning(f"Large prompt saved to {debug_path}")
            
            # Add timeout for compilation with retry logic
            response = None
            api_attempts = 0
            max_api_attempts = 2
            
            while api_attempts < max_api_attempts and not response:
                try:
                    api_attempts += 1
                    logger.info(f"Sending compilation request to Gemini Pro (attempt {api_attempts}/{max_api_attempts})...")
                    
                    # Try different response formats based on attempt
                    if api_attempts == 1:
                        # First attempt: text/plain for flexibility
                        response_config = {
                            "max_output_tokens": 30000,
                            "response_mime_type": "text/plain",
                            "temperature": 0.1
                        }
                    else:
                        # Second attempt: force JSON response
                        response_config = {
                            "max_output_tokens": 30000,
                            "response_mime_type": "application/json",
                            "temperature": 0.05  # Lower temperature for more consistent output
                        }
                        # Also modify prompt for JSON-only response
                        compilation_prompt = compilation_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no text before or after."
                    
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.models.generate_content,
                            model="gemini-1.5-pro",  # Use Pro for compilation
                            contents=[compilation_prompt],
                            config=response_config
                        ),
                        timeout=120.0  # 2 minute timeout
                    )
                    
                    if response and hasattr(response, 'text') and response.text:
                        logger.info(f"Received response from Gemini Pro on attempt {api_attempts}")
                        break
                    else:
                        logger.warning(f"Empty or invalid response on attempt {api_attempts}")
                        response = None
                        
                except asyncio.TimeoutError:
                    logger.error(f"Compilation timed out on attempt {api_attempts}")
                    if api_attempts >= max_api_attempts:
                        raise Exception("Compilation timeout - API not responding after retries")
                    await asyncio.sleep(2)  # Brief pause before retry
                    
                except Exception as api_error:
                    logger.error(f"API call failed on attempt {api_attempts}: {api_error}")
                    logger.error(f"Error type: {type(api_error).__name__}")
                    logger.error(f"Error details: {str(api_error)}")
                    if hasattr(api_error, 'response'):
                        logger.error(f"API response status: {getattr(api_error.response, 'status_code', 'N/A')}")
                    if api_attempts >= max_api_attempts:
                        raise Exception(f"Gemini API error after {max_api_attempts} attempts: {str(api_error)}")
                    await asyncio.sleep(2)  # Brief pause before retry
            
            # Check if response is valid
            if not response:
                logger.error("No response object from Gemini")
                raise Exception("No response from Gemini API")
            
            # Log response structure for debugging
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response attributes: {dir(response)}")
            
            if not hasattr(response, 'text'):
                logger.error("Response has no 'text' attribute")
                # Try to find the actual content
                if hasattr(response, 'candidates') and response.candidates:
                    logger.info("Found candidates in response")
                    if hasattr(response.candidates[0], 'content'):
                        response_text = str(response.candidates[0].content)
                        logger.info(f"Extracted text from candidates: {len(response_text)} chars")
                    else:
                        raise Exception("Could not extract text from response candidates")
                else:
                    raise Exception("Invalid response format from Gemini API")
            else:
                response_text = response.text
            
            logger.info(f"Compilation response length: {len(response_text) if response_text else 0} chars")
            
            # Check for empty response
            if not response_text or response_text.strip() == "":
                logger.error("Response text is empty or contains only whitespace!")
                raise Exception("Empty response from Gemini API")
            
            # Log first 500 chars of response for debugging
            logger.info(f"Response preview: {response_text[:500]}...")
            
            # Save raw response for debugging
            debug_filename = f"compilation_response_{self.session_id}_{int(time.time())}.json"
            debug_path = os.path.join(os.path.dirname(__file__), "debug", debug_filename)
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, 'w') as f:
                f.write(response_text)
            logger.info(f"Raw response saved to {debug_path}")
            
            # Try multiple parsing strategies
            compiled_data = None
            
            # Strategy 1: Use the LLM response extractor
            try:
                compiled_data = extract_json_from_llm_response(response_text)
                if compiled_data and len(compiled_data) > 1:
                    logger.info("Successfully extracted JSON using LLM response parser")
                else:
                    compiled_data = None
            except Exception as e:
                logger.debug(f"LLM response extraction failed: {e}")
            
            # Strategy 2: Use the safe parser directly
            if not compiled_data:
                compiled_data = parse_json_safely(response_text)
                if compiled_data and not compiled_data.get("_parse_error"):
                    logger.info("Successfully parsed JSON using safe parser")
                else:
                    compiled_data = None
            
            # Strategy 3: Advanced reconstruction
            if not compiled_data:
                logger.warning("Attempting advanced JSON reconstruction")
                compiled_data = advanced_json_reconstruction(response_text)
                if compiled_data:
                    logger.info(f"Advanced reconstruction recovered {len(compiled_data)} fields")
            
            if not compiled_data:
                logger.error("Failed to parse compilation response, attempting advanced recovery")
                
                # Try alternative parsing strategies
                # 1. Check if response contains multiple JSON objects
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
                if json_objects:
                    logger.info(f"Found {len(json_objects)} potential JSON objects in response")
                    for i, json_obj in enumerate(json_objects):
                        try:
                            parsed = json.loads(json_obj)
                            if parsed and isinstance(parsed, dict) and len(parsed) > 1:
                                logger.info(f"Successfully parsed JSON object {i+1}")
                                compiled_data = parsed
                                break
                        except:
                            continue
                
                # 2. Try to extract JSON from between specific markers
                if not compiled_data:
                    json_match = re.search(r'(\{[\s\S]*\})', response_text)
                    if json_match:
                        try:
                            compiled_data = json.loads(json_match.group(1))
                            logger.info("Successfully extracted JSON from response")
                        except:
                            pass
                
                # 3. Manual construction as last resort
                if not compiled_data:
                    logger.warning("Using manual fallback merge due to parsing failure")
                    compiled_data = self.simple_merge(page_results)
                    compiled_data["_compilation_method"] = "fallback_merge"
                    compiled_data["_parsing_error"] = "Failed to parse Gemini response"
            
            # Validate compiled data structure
            if compiled_data:
                field_count = self._count_fields(compiled_data)
                logger.info(f"Compiled data contains {field_count} fields")
                
                # Check if we got schema instead of data
                if any(key in str(compiled_data) for key in ["type", "properties", "description", "$schema"]):
                    logger.warning("Response may contain schema definitions instead of extracted data")
                    # Try to extract actual data if possible
                    if "properties" in compiled_data:
                        logger.info("Attempting to extract data from schema-like response")
                        compiled_data = self._extract_data_from_schema_response(compiled_data)
            
            # Add metadata about failed pages
            if failed_pages:
                if "_warnings" not in compiled_data:
                    compiled_data["_warnings"] = []
                compiled_data["_warnings"].append(f"Failed to extract pages: {failed_pages}")
            
            # Add debug metadata
            compiled_data["_debug"] = {
                "compilation_timestamp": datetime.now().isoformat(),
                "response_length": len(response_text),
                "successful_pages": len(all_page_data),
                "failed_pages": len(failed_pages),
                "partial_pages": len(partial_pages)
            }
            
            await self.update_plan_step("compile", "completed", "Data compiled and structured")
            return compiled_data
            
        except Exception as e:
            logger.error(f"Error compiling results: {e}\n{traceback.format_exc()}")
            await self.update_plan_step("compile", "failed", str(e)[:100])
            
            # Enhanced fallback with better error reporting
            logger.info("Using enhanced fallback merge")
            fallback_result = self.simple_merge(page_results)
            fallback_result["_error"] = str(e)
            fallback_result["_compilation_method"] = "error_fallback"
            return fallback_result

    def simple_merge(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple merge as fallback if compilation fails"""
        merged = {}
        for result in page_results:
            if result["status"] == "success" and result["data"]:
                merged = self.deep_merge(merged, result["data"])
        return merged

    async def process_with_parallel_extraction(self, file_content: bytes, schema: str, filename: str) -> List[Dict[str, Any]]:
        """Process document with parallel page extraction"""
        results = []
        
        try:
            # Split PDF into pages
            pages = await self.split_pdf_pages(file_content)
            
            # Extract from all pages in parallel
            page_results = await self.parallel_extract_pages(pages, schema)
            
            # Compile results with Pro model
            compiled_json = await self.compile_results_with_pro(page_results, schema)
            
            # Validate extraction
            await self.update_plan_step("quality", "in_progress")
            validation_result = self.validate_extraction_result(compiled_json, schema)
            quality = 100 - validation_result["empty_percentage"]
            
            if validation_result["math_validation_passed"]:
                await self.update_plan_step("quality", "completed", f"Quality: {quality:.0f}%")
            else:
                errors_preview = "; ".join(validation_result["math_errors"][:3])
                await self.update_plan_step("quality", "completed", 
                    f"Quality: {quality:.0f}% (Math issues: {errors_preview}...)")
            
            # Add metadata
            compiled_json["_metadata"] = {
                "filename": filename,
                "total_pages": len(pages),
                "extracted_pages": len([r for r in page_results if r["status"] == "success"]),
                "selected_schema": schema,
                "quality_score": quality,
                "math_validation_passed": validation_result["math_validation_passed"],
                "processing_time": time.time() - self.start_time,
                "extraction_method": "parallel"
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

    def validate_extraction_result(self, json_data: Dict[str, Any], form_type: str) -> Dict[str, Any]:
        """Validate extraction quality with form-specific mathematical checks"""
        def count_values(obj):
            null_count = 0
            zero_count = 0
            total_count = 0
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not key.startswith('_'):  # Skip metadata
                        n, z, t = count_values(value)
                        null_count += n
                        zero_count += z
                        total_count += t
            elif isinstance(obj, list):
                for item in obj:
                    n, z, t = count_values(item)
                    null_count += n
                    zero_count += z
                    total_count += t
            else:
                total_count = 1
                if obj is None or obj == "" or obj == "null":
                    null_count = 1
                elif obj == 0 or obj == "0" or obj == 0.0:
                    zero_count = 1
                    
            return null_count, zero_count, total_count
        
        null_count, zero_count, total_count = count_values(json_data)
        empty_percentage = ((null_count + zero_count) / total_count * 100) if total_count > 0 else 0
        
        # Check critical fields
        has_critical = False
        math_validation_passed = True
        math_errors = []
        
        if form_type == "941-X":
            # Critical field check
            if "employerInfo" in json_data:
                has_critical = bool(json_data.get("employerInfo", {}).get("ein"))
            
            # Run comprehensive 941-X mathematical validation
            math_validation_passed, math_errors = self.validate_941x_mathematics(json_data)
            
        elif form_type == "941" and "employerInfo" in json_data:
            has_critical = bool(json_data.get("employerInfo", {}).get("ein"))
        
        return {
            "is_valid": empty_percentage < 70 and math_validation_passed,
            "empty_percentage": empty_percentage,
            "populated_count": total_count - null_count - zero_count,
            "total_fields": total_count,
            "has_critical_fields": has_critical,
            "math_validation_passed": math_validation_passed,
            "math_errors": math_errors
        }

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
        
        # 3. Recovery startup ceiling for Q4 2021
        if year == 2021 and quarter == 4:
            is_recovery_startup = data.get("lines", {}).get("line31b", {}).get("checked", False)
            if is_recovery_startup and total_erc > 50000:
                errors.append(f"Recovery startup ERC ${total_erc:,.2f} exceeds $50,000 limit")
        
        # 4. Q2 2020 inclusion rule
        if not (year == 2020 and quarter == 2):
            if get_line_value("33a") != 0 or get_line_value("34") != 0:
                errors.append("Lines 33a/34 should only have values for Q2 2020")
        
        # 5. Social Security column formulas (6.2% each side = 12.4% total)
        ss_lines = [
            ("8", 0.124),   # Social security tax
            ("9", 0.062),   # Employer share FICA
            ("10", 0.062),  # Employee share FICA
            ("11", 0.124)   # Total social security
        ]
        
        for line_num, rate in ss_lines:
            col3 = get_line_value(line_num, "corrected") - get_line_value(line_num, "original")
            col4 = get_line_value(line_num)
            expected = col3 * rate
            if abs(col4 - expected) > 0.01:  # $0.01 tolerance
                errors.append(f"Line {line_num} col 4 (${col4:,.2f}) != col 3 × {rate} (${expected:,.2f})")
        
        # 6. Medicare column formulas
        medicare_lines = [
            ("12", 0.029),  # Medicare tax (1.45% × 2)
            ("13", 0.009)   # Additional Medicare tax
        ]
        
        for line_num, rate in medicare_lines:
            col3 = get_line_value(line_num, "corrected") - get_line_value(line_num, "original")
            col4 = get_line_value(line_num)
            expected = col3 * rate
            if abs(col4 - expected) > 0.01:
                errors.append(f"Line {line_num} col 4 (${col4:,.2f}) != col 3 × {rate} (${expected:,.2f})")
        
        # 7. Subtotal check (Line 23 = sum of lines 7-22)
        subtotal = sum(get_line_value(str(i)) for i in range(7, 23))
        line_23 = get_line_value("23")
        if abs(line_23 - subtotal) > 0.01:
            errors.append(f"Line 23 (${line_23:,.2f}) != sum of lines 7-22 (${subtotal:,.2f})")
        
        # 8. Grand total check (Line 27)
        grand_total = (get_line_value("23") + get_line_value("24") + get_line_value("25") +
                      get_line_value("26a") + get_line_value("26b") + get_line_value("26c"))
        line_27 = get_line_value("27")
        if abs(line_27 - grand_total) > 0.01:
            errors.append(f"Line 27 (${line_27:,.2f}) != sum of lines 23-26c (${grand_total:,.2f})")
        
        # 9. Column 3 difference rule
        for line_num in ["6", "7", "8", "9", "10", "11", "12", "13"] + [str(i) for i in range(28, 41)]:
            corrected = get_line_value(line_num, "corrected")
            original = get_line_value(line_num, "original")
            difference = corrected - original
            recorded_diff = get_line_value(line_num, "difference")
            if abs(recorded_diff - difference) > 0.01:
                errors.append(f"Line {line_num} col 3 calculation error")
        
        # 10. Date window enforcement
        restricted_lines = {
            (2020, 2, 2021, 1): ["9", "10", "17", "25", "28", "29"],
            (2021, 2, 2021, 3): ["18b", "26b", "35", "36", "37", "38", "39", "40"],
            (2020, 2, 2021, 4): ["18a", "26a", "30", "31a"]
        }
        
        for (start_year, start_q, end_year, end_q), lines in restricted_lines.items():
            if not ((year > start_year or (year == start_year and quarter >= start_q)) and
                   (year < end_year or (year == end_year and quarter <= end_q))):
                for line in lines:
                    if get_line_value(line) != 0:
                        errors.append(f"Line {line} invalid for {year} Q{quarter}")
        
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
    
    def schema_to_example(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON Schema to example structure with null values (not empty objects)"""
        def process_schema_node(node):
            if not isinstance(node, dict):
                return None
                
            node_type = node.get("type")
            
            if node_type == "object":
                # Process object properties
                result = {}
                properties = node.get("properties", {})
                for key, value in properties.items():
                    # Skip metadata fields in the schema
                    if key not in ["$schema", "title", "description", "required"]:
                        processed = process_schema_node(value)
                        # Always include the key with appropriate value
                        if isinstance(processed, dict) and processed:
                            # If it has nested structure, include it
                            result[key] = processed
                        else:
                            # For leaf nodes, use null
                            result[key] = None
                return result
                
            elif node_type == "array":
                # Return empty array as placeholder
                return []
                
            elif node_type in ["string", "number", "integer", "boolean"]:
                # Return null for all primitive types
                return None
                
            else:
                # For any other type or if type is not specified
                return None
        
        # Process the entire schema
        if "properties" in schema:
            return process_schema_node(schema)
        else:
            # If the schema itself doesn't have properties at root, return as is
            return process_schema_node({"type": "object", "properties": schema})
    
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
    
    def _extract_data_from_schema_response(self, schema_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual data from a response that contains schema definitions"""
        result = {}
        
        def extract_values(node, target):
            if isinstance(node, dict):
                if "properties" in node:
                    # This is a schema node, process its properties
                    for key, value in node.get("properties", {}).items():
                        if key not in ["type", "description", "$schema", "required"]:
                            target[key] = {}
                            extract_values(value, target[key])
                else:
                    # Look for actual values
                    for key, value in node.items():
                        if key not in ["type", "properties", "description", "$schema", "required", "items"]:
                            if isinstance(value, dict):
                                target[key] = {}
                                extract_values(value, target[key])
                            elif value is not None and value != "" and not isinstance(value, str) or (isinstance(value, str) and value.lower() not in ["string", "number", "integer", "boolean", "array", "object"]):
                                target[key] = value
        
        extract_values(schema_response, result)
        
        # Clean up empty nested dictionaries
        def clean_empty(d):
            if isinstance(d, dict):
                return {k: clean_empty(v) for k, v in d.items() if v and (not isinstance(v, dict) or clean_empty(v))}
            return d
        
        cleaned = clean_empty(result)
        return cleaned if cleaned else {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    # Don't add to active_sessions here - let the processing endpoint manage it
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
        # Don't remove from active_sessions here - let processing complete
        session_confirmations.pop(session_id, None)

@app.post("/process-documents-v3")
async def process_documents_v3(
    files: List[UploadFile] = File(...), 
    schema: str = Form("generic"),
    session_id: str = Form(None)
):
    """Enhanced document processing with parallel extraction"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Add session to active sessions immediately
    active_sessions.add(session_id)
    
    logger.info(f"Starting processing for session {session_id} with {len(files)} files")
    processor = DocumentProcessor(session_id)
    
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting document processing server on port 4830")
    uvicorn.run(app, host="0.0.0.0", port=4830)