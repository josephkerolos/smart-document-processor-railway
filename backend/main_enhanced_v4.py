import io
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
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
            {"id": "split", "name": "📄 Detect document boundaries", "status": "pending"},
            {"id": "extract", "name": "📖 Extract data from each document", "status": "pending"},
            {"id": "quality", "name": "🔍 Verify extraction quality", "status": "pending"},
            {"id": "enhance", "name": "✨ Enhance low-quality extractions", "status": "pending"},
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
                model="gemini-2.5-flash-preview-05-20",
                contents=[detection_prompt, file_part],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json"
                }
            )
            
            detection_result = json.loads(response.text)
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

    async def smart_document_splitting(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Enhanced document splitting with comprehensive boundary detection"""
        await self.update_plan_step("split", "in_progress")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"Starting document splitting for {total_pages} pages")
            
            # For single page or very short documents
            if total_pages <= 2:
                await self.update_plan_step("split", "completed", "Single document")
                return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]
            
            # Analyze ALL pages to find form boundaries
            await self.send_status_update("split", f"Analyzing {total_pages} pages for document boundaries...")
            
            page_analyses = []
            
            # Check every page for comprehensive boundary detection
            for page_idx in range(total_pages):
                if await self.should_cancel():
                    raise Exception("Processing cancelled")
                    
                await self.send_status_update("split", f"Deep analysis of page {page_idx + 1} of {total_pages}...")
                
                page = pdf_reader.pages[page_idx]
                pdf_writer = PdfWriter()
                pdf_writer.add_page(page)
                page_stream = io.BytesIO()
                pdf_writer.write(page_stream)
                page_stream.seek(0)
                
                file_part = types.Part.from_bytes(
                    data=page_stream.getvalue(),
                    mime_type="application/pdf"
                )
                
                # Comprehensive boundary detection prompt
                detect_prompt = """
                COMPREHENSIVE DOCUMENT BOUNDARY ANALYSIS
                
                Analyze this page to determine if it's the start of a new form/document.
                
                STRONG INDICATORS OF A NEW DOCUMENT:
                1. Form Numbers/Headers:
                   - "Form 941" or "Form 941-X" at the top
                   - "Rev. [date]" indicating form version
                   - OMB control numbers in top-right corner
                   
                2. Page Numbering Patterns:
                   - "Page 1 of 5" or "Page 1"
                   - Page numbering RESETS (not continues)
                   - Absence of "Continued from previous page"
                   
                3. Required Fields Starting Fresh:
                   - Employer Identification Number (EIN) field
                   - Business name/address fields
                   - Tax period/quarter selection boxes
                   - All fields are EMPTY (not pre-filled)
                   
                4. Structural Elements:
                   - Official government header/seal
                   - Thick border around entire page
                   - Instructions block "Read the separate instructions"
                   - Barcode or scan line at bottom
                   
                5. Filing Period Indicators:
                   - Different quarter checkboxes (Q1, Q2, Q3, Q4)
                   - Different tax year
                   - "Amended Return" checkbox
                
                CONTINUATION INDICATORS (NOT a new document):
                - "Schedule B (Form 941)" or similar schedule notation
                - "Worksheet" or "Continuation Sheet"
                - Page numbers like "Page 2 of 5", "Page 3 of 5"
                - "See instructions" without full form header
                - Calculations carrying over from previous pages
                
                Analyze and return JSON:
                {
                    "is_first_page": true/false,
                    "confidence_score": 0-100,
                    "form_type": "941/941-X/1040/etc or null",
                    "form_version": "Rev date if visible",
                    "page_indicator": "e.g., 'Page 1 of 5'",
                    "has_form_header": true/false,
                    "has_ein_field": true/false,
                    "has_period_selection": true/false,
                    "has_barcode": true/false,
                    "is_schedule_or_worksheet": true/false,
                    "boundary_indicators_found": ["list of specific indicators"]
                }
                """
                
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model="gemini-2.5-flash-preview-05-20",
                        contents=[detect_prompt, file_part],
                        config={
                            "max_output_tokens": 500,
                            "response_mime_type": "application/json"
                        }
                    )
                    
                    page_info = json.loads(response.text)
                    
                    # Apply comprehensive decision logic
                    is_new_document = False
                    confidence = page_info.get("confidence_score", 0)
                    
                    # High confidence new document if multiple strong indicators
                    if page_info.get("is_first_page", False) and confidence >= 70:
                        indicator_count = 0
                        if page_info.get("has_form_header"): indicator_count += 1
                        if page_info.get("has_ein_field"): indicator_count += 1
                        if page_info.get("has_period_selection"): indicator_count += 1
                        if page_info.get("has_barcode"): indicator_count += 1
                        if "Page 1" in str(page_info.get("page_indicator", "")): indicator_count += 1
                        
                        # Require at least 3 strong indicators
                        if indicator_count >= 3 and not page_info.get("is_schedule_or_worksheet", False):
                            is_new_document = True
                    
                    page_analyses.append({
                        "page_idx": page_idx,
                        "is_first_page": is_new_document,
                        "form_type": page_info.get("form_type"),
                        "confidence": confidence,
                        "indicators": page_info.get("boundary_indicators_found", []),
                        "page_indicator": page_info.get("page_indicator", "")
                    })
                    
                    logger.info(f"Page {page_idx + 1}: {'NEW DOCUMENT' if is_new_document else 'continuation'} "
                              f"(confidence: {confidence}%, form: {page_info.get('form_type', 'unknown')})")
                    
                except Exception as e:
                    logger.error(f"Error analyzing page {page_idx}: {e}")
                    page_analyses.append({
                        "page_idx": page_idx,
                        "is_first_page": False,
                        "form_type": None,
                        "confidence": 0
                    })
                
                # Update progress
                progress = int((page_idx / total_pages) * 100)
                await self.send_update("progress", {
                    "message": f"Analyzing page {page_idx + 1} of {total_pages}...",
                    "percentage": progress
                })
            
            # Find document boundaries with validation
            documents = []
            current_doc_start = 0
            current_form_type = page_analyses[0].get("form_type") if page_analyses else None
            
            for i in range(1, len(page_analyses)):
                if page_analyses[i]["is_first_page"]:
                    # Validate this is a logical split
                    pages_in_section = i - current_doc_start
                    
                    # Check if this makes sense (e.g., Form 941 is typically 3-5 pages)
                    if pages_in_section >= 2:  # Reasonable document size
                        documents.append({
                            "start_page": current_doc_start,
                            "end_page": i - 1,
                            "document_index": len(documents),
                            "form_type": current_form_type,
                            "page_count": pages_in_section
                        })
                        current_doc_start = i
                        current_form_type = page_analyses[i].get("form_type")
                        
                        logger.info(f"Document {len(documents)}: pages {documents[-1]['start_page']+1}-{documents[-1]['end_page']+1} "
                                  f"({documents[-1]['page_count']} pages, type: {documents[-1]['form_type']})")
            
            # Add the last document
            final_pages = total_pages - current_doc_start
            documents.append({
                "start_page": current_doc_start,
                "end_page": total_pages - 1,
                "document_index": len(documents),
                "form_type": current_form_type,
                "page_count": final_pages
            })
            
            logger.info(f"Found {len(documents)} documents in PDF with comprehensive boundary detection")
            await self.update_plan_step("split", "completed", f"Found {len(documents)} document(s)")
            
            # Log summary for debugging
            for doc in documents:
                logger.info(f"Document {doc['document_index'] + 1}: "
                          f"Pages {doc['start_page'] + 1}-{doc['end_page'] + 1} "
                          f"({doc['page_count']} pages), Type: {doc['form_type']}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in document splitting: {e}\n{traceback.format_exc()}")
            await self.update_plan_step("split", "failed", str(e)[:100])
            # Return single document as fallback
            return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]

    async def process_with_generative_updates(self, file_content: bytes, schema: str, filename: str) -> List[Dict[str, Any]]:
        """Process a file with generative status updates"""
        results = []
        
        try:
            # Detect multiple documents
            documents = await self.smart_document_splitting(file_content)
            
            await self.update_plan_step("extract", "in_progress")
            
            for doc_idx, doc_info in enumerate(documents):
                if await self.should_cancel():
                    raise Exception("Processing cancelled")
                    
                await self.send_status_update("extract", 
                    f"Extracting from document {doc_idx + 1} of {len(documents)}...")
                
                # Extract pages for this document
                pdf_reader = PdfReader(io.BytesIO(file_content))
                doc_pages = []
                for page_idx in range(doc_info['start_page'], doc_info['end_page'] + 1):
                    if page_idx < len(pdf_reader.pages):
                        doc_pages.append(pdf_reader.pages[page_idx])
                
                # Process pages
                merged_json = await self.process_document_pages(doc_pages, schema)
                
                if merged_json:
                    # Quality check
                    await self.update_plan_step("quality", "in_progress")
                    validation_result = self.validate_extraction_result(merged_json, schema)
                    quality = 100 - validation_result["empty_percentage"]
                    
                    if quality < 70 and not validation_result.get("has_critical_fields", False):
                        await self.update_plan_step("enhance", "in_progress")
                        await self.send_status_update("enhance", "Retrying with enhanced prompts...")
                        
                        # Simple retry for first page
                        if doc_pages:
                            retry_json = await self.process_document_pages([doc_pages[0]], schema)
                            if retry_json:
                                merged_json = self.deep_merge(merged_json, retry_json)
                                validation_result = self.validate_extraction_result(merged_json, schema)
                                quality = 100 - validation_result["empty_percentage"]
                        
                        await self.update_plan_step("enhance", "completed", f"Quality: {quality:.0f}%")
                    else:
                        await self.update_plan_step("quality", "completed", f"Quality: {quality:.0f}%")
                    
                    # Add metadata
                    merged_json["_metadata"] = {
                        "filename": filename,
                        "document_index": doc_info['document_index'],
                        "pages": f"{doc_info['start_page'] + 1}-{doc_info['end_page'] + 1}",
                        "page_count": doc_info['page_count'],
                        "detected_form_type": doc_info.get('form_type'),
                        "selected_schema": schema,
                        "quality_score": quality,
                        "processing_time": time.time() - self.start_time
                    }
                    
                    results.append(merged_json)
            
            await self.update_plan_step("extract", "completed", f"Extracted {len(results)} documents")
            await self.update_plan_step("complete", "completed")
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}\n{traceback.format_exc()}")
            if "cancelled" in str(e).lower():
                await self.update_plan_step("complete", "cancelled", "Processing stopped by user")
            else:
                await self.update_plan_step("complete", "failed", str(e)[:100])
            raise
            
        return results

    async def process_document_pages(self, pages: List[Any], schema: str) -> Dict[str, Any]:
        """Process pages of a document"""
        page_results = []
        
        for page_num, page in enumerate(pages):
            if await self.should_cancel():
                raise Exception("Processing cancelled")
                
            try:
                # Extract from each page
                pdf_writer = PdfWriter()
                pdf_writer.add_page(page)
                page_stream = io.BytesIO()
                pdf_writer.write(page_stream)
                page_stream.seek(0)
                
                file_part = types.Part.from_bytes(
                    data=page_stream.getvalue(),
                    mime_type="application/pdf"
                )
                
                # Raw extraction
                raw_response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-preview-05-20",
                    contents=["Extract all text and data from this document page", file_part],
                    config={
                        "max_output_tokens": 40000,
                        "response_mime_type": "text/plain"
                    }
                )
                
                # Convert to JSON
                json_prompt = self.generate_schema_prompt(schema, raw_response.text)
                json_response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-preview-05-20",
                    contents=[json_prompt],
                    config={
                        "max_output_tokens": 40000,
                        "response_mime_type": "application/json"
                    }
                )
                
                page_data = json.loads(json_response.text)
                page_results.append(page_data)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
        
        # Merge results
        if not page_results:
            return {}
            
        merged = page_results[0]
        for result in page_results[1:]:
            merged = self.deep_merge(merged, result)
            
        return merged

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

    def generate_schema_prompt(self, schema_id: str, extracted_text: str) -> str:
        """Generate schema-specific prompt"""
        schema = self.load_schema(schema_id)
        
        if schema:
            schema_json = json.dumps(schema, indent=2)
            
            if schema_id == "941-X":
                return f"""
                Extract data from this Form 941-X text into the schema.
                IMPORTANT: This is a CORRECTION form with three columns:
                - Column 1: CORRECTED amount
                - Column 2: ORIGINAL amount  
                - Column 3: DIFFERENCE
                
                Schema: {schema_json}
                Text: {extracted_text}
                """
            else:
                return f"""
                Extract all data from this form into the schema.
                Schema: {schema_json}
                Text: {extracted_text}
                """
        return f"Extract data into JSON: {extracted_text}"

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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    # Don't add to active_sessions here - let the processing endpoint manage it
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            message = json.loads(data)
            
            if message.get("type") == "confirm_form":
                session_confirmations[session_id] = True
                
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
    """Enhanced document processing with real-time updates"""
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
        
        # Process each file
        for file_idx, file in enumerate(files):
            content = await file.read()
            
            results = await processor.process_with_generative_updates(
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
        "active_connections": len(active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting document processing server on port 4830")
    uvicorn.run(app, host="127.0.0.1", port=4830)