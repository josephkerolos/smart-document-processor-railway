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

# Debug mode flag
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

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
            {"id": "extract", "name": "🚀 Extract data from pages (parallel)", "status": "pending"},
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
        max_retries = 3  # Increased retries
        
        for attempt in range(max_retries):
            try:
                file_part = types.Part.from_bytes(
                    data=page_content,
                    mime_type="application/pdf"
                )
                
                # Get page-specific extraction instructions
                page_context = self.get_page_specific_context(schema, page_num)
                
                # Simplified extraction prompt for better performance
                extraction_prompt = f"""Extract ALL data from page {page_num + 1} of {total_pages} of IRS Form {schema}.

{page_context}

RULES:
1. Extract EVERY visible value - text, numbers, checkboxes
2. Checkboxes: true=checked, false=unchecked, null=not visible
3. Include all amounts with exact decimals
4. Preserve EIN format (XX-XXXXXXX)
5. For 941-X: extract all 3 columns per line (corrected, original, difference)

Return ONLY valid JSON with extracted data. No explanations or markdown."""
                
                # Use structured output with higher token limits
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20",
                    contents=[extraction_prompt, file_part],
                    config={
                        "max_output_tokens": 30000,  # Significantly increased
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
                    logger.info(f"Retrying page {page_num + 1} after {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
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

    def get_page_specific_context(self, schema: str, page_num: int) -> str:
        """Get page-specific extraction context"""
        if schema == "941-X":
            contexts = {
                0: """FOCUS ON PAGE 1:
- Employer Info: EIN, Name, Address (complete)
- Correction Info: Quarter (1-4), Year
- Correction Type checkboxes
- Return you're correcting date""",
                1: """FOCUS ON PAGE 2 (Part 2):
Extract ALL lines 1-27 with THREE columns each:
- Column 1: Corrected amount
- Column 2: Amount originally reported
- Column 3: Difference
Include ALL amounts even if 0.00""",
                2: """FOCUS ON PAGE 3 (Part 3):
Extract lines 28-40 especially:
- Line 30: Qualified wages for ERC
- Line 31a: Qualified health plan expenses
- Lines 33a, 34, 35a, 36a: Additional qualified wages
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
            return contexts.get(page_num, "Extract all visible form data")
        else:
            return "Extract all visible form data with complete accuracy"

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

    def _save_debug_extraction(self, page_num: int, data: Dict[str, Any], schema: str):
        """Save debug extraction data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = "debug_extractions"
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"{debug_dir}/{schema}_page{page_num}_{timestamp}.json"
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
        
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        
        await self.update_plan_step("extract", "completed", 
            f"Extracted {successful} of {total_pages} pages")
        
        return results

    async def compile_results_optimized(self, page_results: List[Dict[str, Any]], schema: str) -> Dict[str, Any]:
        """Optimized compilation that handles large documents better"""
        try:
            await self.update_plan_step("compile", "in_progress")
            logger.info(f"Starting compilation for {len(page_results)} pages")
            
            # Simple merge strategy for better performance
            compiled_data = {}
            
            for result in page_results:
                if result["status"] == "success" and result.get("data"):
                    page_data = result["data"]
                    # Remove page metadata
                    page_data.pop("_page", None)
                    page_data.pop("_extraction_failed", None)
                    page_data.pop("_error", None)
                    
                    # Merge data
                    compiled_data = self.deep_merge(compiled_data, page_data)
            
            if not compiled_data:
                logger.error("No successful page extractions to compile")
                raise Exception("No data to compile")
            
            logger.info("Compilation completed successfully")
            await self.update_plan_step("compile", "completed", 
                f"Compiled {len(page_results)} pages")
            
            return compiled_data
            
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            await self.update_plan_step("compile", "failed", str(e))
            raise

    async def process_with_parallel_extraction(self, file_content: bytes, schema: str, filename: str) -> List[Dict[str, Any]]:
        """Process document with parallel page extraction"""
        results = []
        
        try:
            # Split PDF into pages
            pages = await self.split_pdf_pages(file_content)
            
            # Extract from all pages in parallel
            page_results = await self.parallel_extract_pages(pages, schema)
            
            # Use optimized compilation
            compiled_json = await self.compile_results_optimized(page_results, schema)
            
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
                "extraction_method": "parallel_optimized"
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting optimized document processing server on port 4830")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info("Key improvements in v10:")
    logger.info("- Increased token limits (30k for extraction, 50k for compilation)")
    logger.info("- 3-minute timeout for compilation")
    logger.info("- Better retry logic with exponential backoff")
    logger.info("- Simplified prompts for better performance")
    logger.info("- Optimized compilation without truncation")
    logger.info("- Enhanced JSON parsing with multiple fallback strategies")
    uvicorn.run(app, host="0.0.0.0", port=4830)