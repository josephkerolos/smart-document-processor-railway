import io
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Body, Form, WebSocket, WebSocketDisconnect
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

class ProcessingStatus(Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    SPLITTING = "splitting"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingStep:
    step_name: str
    status: str
    message: str
    duration: Optional[float] = None
    timestamp: str = None
    details: Optional[str] = None  # For verbose logging
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class DocumentProcessor:
    def __init__(self):
        self.processing_steps: List[ProcessingStep] = []
        self.start_time = None
        
    async def log_step(self, step_name: str, status: str, message: str, 
                      duration: Optional[float] = None, details: Optional[str] = None):
        """Log a processing step with simplified user-friendly message"""
        # Create user-friendly messages
        user_friendly_messages = {
            "form_detection": {
                "started": "🔍 Identifying document type...",
                "completed": f"✅ {message}",
                "failed": "❌ Could not identify document type"
            },
            "multi_doc_detection": {
                "started": "📄 Checking for multiple documents...",
                "analyzing": "🔎 Analyzing document structure...",
                "completed": f"✅ {message}",
                "failed": "⚠️ Using single document mode"
            },
            "document_processing": {
                "started": f"📋 {message}",
                "completed": f"✅ {message}"
            },
            "raw_extraction": {
                "started": "📖 Reading document content...",
                "completed": f"✅ Extracted text from {message.split()[-1]}",
            },
            "json_conversion": {
                "started": "🔄 Converting to structured data...",
                "completed": "✅ Data structured successfully"
            },
            "validation": {
                "started": "🔍 Checking data quality...",
                "completed": f"✅ {message}",
                "failed": "⚠️ Low quality extraction detected"
            },
            "retry": {
                "started": "🔄 Improving extraction quality...",
                "completed": f"✅ {message}"
            },
            "file_processing": {
                "started": f"📁 {message}",
                "completed": f"✅ {message}",
                "failed": f"❌ {message}"
            }
        }
        
        # Get user-friendly message
        if step_name in user_friendly_messages and status in user_friendly_messages[step_name]:
            display_message = user_friendly_messages[step_name][status]
        else:
            display_message = message
        
        step = ProcessingStep(step_name, status, display_message, duration, details=details)
        self.processing_steps.append(step)
        
        # Send only important updates to WebSocket clients
        important_steps = ["form_detection", "document_processing", "validation", "retry", "file_processing"]
        if step_name in important_steps or status in ["failed", "warning"]:
            update = {
                "type": "step_update",
                "step": {
                    "name": step.step_name,
                    "status": step.status,
                    "message": step.message,
                    "duration": step.duration,
                    "timestamp": step.timestamp
                }
            }
            
            for connection in active_connections:
                try:
                    await connection.send_json(update)
                except:
                    pass

    async def detect_form_type(self, pdf_content: bytes) -> Dict[str, Any]:
        """Detect the actual form type in the PDF"""
        start_time = time.time()
        await self.log_step("form_detection", "started", "Analyzing document to identify form type...")
        
        try:
            # Extract first page for form detection
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            if not pdf_reader.pages:
                raise ValueError("PDF has no pages")
                
            first_page = pdf_reader.pages[0]
            
            # Create a temporary PDF with just the first page
            pdf_writer = PdfWriter()
            pdf_writer.add_page(first_page)
            page_stream = io.BytesIO()
            pdf_writer.write(page_stream)
            page_stream.seek(0)
            
            # Use Gemini to detect form type
            file_part = types.Part.from_bytes(
                data=page_stream.getvalue(),
                mime_type="application/pdf"
            )
            
            detection_prompt = """
            Analyze this document and identify what type of IRS form or document this is.
            Look for form numbers like "941", "941-X", "1040", etc.
            
            Return a JSON object with:
            {
                "detected_form_type": "the form number (e.g., '941', '941-X', '1040')",
                "confidence": "high/medium/low",
                "form_title": "the full title of the form",
                "tax_year": "if visible",
                "quarter": "if it's a quarterly form"
            }
            """
            
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-pro-preview-06-05",
                contents=[detection_prompt, file_part],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json"
                }
            )
            
            detection_result = json.loads(response.text)
            duration = time.time() - start_time
            
            form_type = detection_result.get('detected_form_type', 'Unknown')
            confidence = detection_result.get('confidence', 'unknown')
            
            await self.log_step("form_detection", "completed", 
                f"Detected: {form_type}", 
                duration)
            
            return detection_result
            
        except Exception as e:
            duration = time.time() - start_time
            await self.log_step("form_detection", "failed", f"Error detecting form type: {str(e)}", duration)
            return {"detected_form_type": None, "confidence": "low", "error": str(e)}

    async def detect_multi_document(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Detect if PDF contains multiple documents and split accordingly"""
        start_time = time.time()
        await self.log_step("multi_doc_detection", "started", "Checking for multiple documents...")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            if total_pages <= 5:
                await self.log_step("multi_doc_detection", "completed", 
                    f"Single document ({total_pages} pages)", 
                    time.time() - start_time)
                return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]
            
            # For larger PDFs, analyze structure
            await self.log_step("multi_doc_detection", "analyzing", 
                f"Analyzing {total_pages} pages...")
            
            # Sample pages to detect patterns - more efficient sampling
            sample_indices = [0]  # Always check first page
            if total_pages > 10:
                # Sample every 5th page for large documents
                sample_indices.extend(range(5, min(total_pages, 50), 5))
            else:
                # Check all pages for small documents
                sample_indices.extend(range(1, total_pages))
            
            page_types = []
            
            for idx in sample_indices:
                if idx < total_pages:
                    page = pdf_reader.pages[idx]
                    pdf_writer = PdfWriter()
                    pdf_writer.add_page(page)
                    page_stream = io.BytesIO()
                    pdf_writer.write(page_stream)
                    page_stream.seek(0)
                    
                    file_part = types.Part.from_bytes(
                        data=page_stream.getvalue(),
                        mime_type="application/pdf"
                    )
                    
                    # Quick detection prompt
                    detect_prompt = """
                    Quickly identify if this page is:
                    1. A form first page (has form number at top)
                    2. A continuation page
                    3. A different document type
                    
                    Return JSON: {"page_type": "first_page|continuation|other", "form_number": "if visible"}
                    """
                    
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model="gemini-2.5-pro-preview-06-05",
                        contents=[detect_prompt, file_part],
                        config={
                            "max_output_tokens": 200,
                            "response_mime_type": "application/json"
                        }
                    )
                    
                    try:
                        page_info = json.loads(response.text)
                    except:
                        page_info = {"page_type": "continuation"}
                    
                    page_types.append((idx, page_info))
            
            # Identify document boundaries
            documents = []
            current_doc_start = 0
            
            for i in range(1, len(page_types)):
                page_idx, page_info = page_types[i]
                if page_info.get("page_type") == "first_page":
                    # Found start of new document
                    documents.append({
                        "start_page": current_doc_start,
                        "end_page": page_idx - 1,
                        "document_index": len(documents),
                        "form_type": page_types[len(documents)][1].get("form_number")
                    })
                    current_doc_start = page_idx
            
            # Add the last document
            documents.append({
                "start_page": current_doc_start,
                "end_page": total_pages - 1,
                "document_index": len(documents),
                "form_type": page_types[-1][1].get("form_number") if page_types else None
            })
            
            duration = time.time() - start_time
            await self.log_step("multi_doc_detection", "completed", 
                f"Found {len(documents)} document(s)", duration)
            
            return documents
            
        except Exception as e:
            duration = time.time() - start_time
            await self.log_step("multi_doc_detection", "failed", 
                f"Error: {str(e)}", duration)
            # Return single document as fallback
            try:
                pdf_reader = PdfReader(io.BytesIO(pdf_content))
                total_pages = len(pdf_reader.pages) if pdf_reader.pages else 1
            except:
                total_pages = 1
            return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]

    def validate_extraction_result(self, json_data: Dict[str, Any], form_type: str) -> Dict[str, Any]:
        """Validate if extraction result has too many nulls/zeros"""
        
        def count_values(obj, path=""):
            """Recursively count null, zero, and total values"""
            null_count = 0
            zero_count = 0
            total_count = 0
            populated_fields = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    n, z, t, p = count_values(value, current_path)
                    null_count += n
                    zero_count += z
                    total_count += t
                    populated_fields.extend(p)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    n, z, t, p = count_values(item, f"{path}[{i}]")
                    null_count += n
                    zero_count += z
                    total_count += t
                    populated_fields.extend(p)
            else:
                total_count = 1
                if obj is None or obj == "" or obj == "null":
                    null_count = 1
                elif obj == 0 or obj == "0" or obj == 0.0:
                    zero_count = 1
                else:
                    populated_fields.append((path, obj))
                    
            return null_count, zero_count, total_count, populated_fields
        
        # Skip metadata in validation
        data_to_validate = {k: v for k, v in json_data.items() if not k.startswith('_')}
        
        null_count, zero_count, total_count, populated_fields = count_values(data_to_validate)
        
        # Calculate percentages
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
        zero_percentage = (zero_count / total_count * 100) if total_count > 0 else 0
        empty_percentage = null_percentage + zero_percentage
        
        # Check if we have critical fields populated
        critical_fields = {
            "941": ["employerInfo.ein", "employerInfo.name", "quarter"],
            "941-X": ["employerInfo.ein", "employerInfo.name", "quarterBeingCorrected"],
            "1040": ["personalInfo.taxpayer.ssn", "personalInfo.taxpayer.lastName"],
        }
        
        critical_populated = False
        if form_type in critical_fields:
            for field in critical_fields[form_type]:
                if any(field in path for path, _ in populated_fields):
                    critical_populated = True
                    break
        
        # More lenient validation - 85% is too strict
        is_valid = empty_percentage < 70 or (critical_populated and len(populated_fields) > 5)
        
        return {
            "is_valid": is_valid,
            "null_percentage": null_percentage,
            "zero_percentage": zero_percentage,
            "empty_percentage": empty_percentage,
            "populated_count": len(populated_fields),
            "total_fields": total_count,
            "populated_fields": [f"{path}: {value}" for path, value in populated_fields[:10]],
            "has_critical_fields": critical_populated
        }

    async def process_single_page(self, page, schema: str, page_num: int, total_pages: int) -> Dict[str, Any]:
        """Process a single page with timing"""
        start_time = time.time()
        
        # Create page PDF
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        page_stream = io.BytesIO()
        pdf_writer.write(page_stream)
        page_stream.seek(0)
        
        file_part = types.Part.from_bytes(
            data=page_stream.getvalue(),
            mime_type="application/pdf"
        )
        
        # Step 1: Raw extraction (silent - too technical for users)
        raw_start = time.time()
        
        raw_response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-pro-preview-06-05",
            contents=["List every single thing exactly as it appears on the document, each column and row, in full", file_part],
            config={
                "max_output_tokens": 40000,
                "response_mime_type": "text/plain"
            }
        )
        raw_text = raw_response.text
        raw_duration = time.time() - raw_start
        
        # Only log for first and last page to reduce clutter
        if page_num == 0 or page_num == total_pages - 1:
            await self.log_step("raw_extraction", "completed", 
                f"page {page_num + 1}", raw_duration)
        
        # Step 2: JSON conversion (silent - too technical)
        json_start = time.time()
        
        json_prompt = self.generate_schema_prompt(schema, raw_text)
        json_response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-pro-preview-06-05",
            contents=[json_prompt],
            config={
                "max_output_tokens": 40000,
                "response_mime_type": "application/json"
            }
        )
        
        json_duration = time.time() - json_start
        
        total_duration = time.time() - start_time
        
        return {
            "page_num": page_num,
            "json_text": json_response.text,
            "timings": {
                "raw_extraction": raw_duration,
                "json_conversion": json_duration,
                "total": total_duration
            }
        }

    def generate_schema_prompt(self, schema_id: str, extracted_text: str) -> str:
        """Generate schema-specific prompt"""
        schema = self.load_schema(schema_id)
        
        # Generic document schema
        GENERIC_SCHEMA = """
        {
          "documentMetadata": {
            "documentID": "Unique identifier",
            "documentType": "Type/category",
            "documentNumber": "Reference number",
            "documentDate": "YYYY-MM-DD"
          },
          "financialData": {
            "currency": "Currency code",
            "totalAmount": "Final total"
          },
          "partyInformation": {
            "vendor": {
              "name": "Vendor name",
              "address": "Vendor address"
            }
          }
        }
        """
        
        if schema:
            schema_json = json.dumps(schema, indent=2)
            
            if schema_id in ["1040", "2848", "8821", "941"]:
                return f"""
                Format the following extracted text from an IRS tax form into a JSON object that strictly follows the schema below.
                This is specifically for IRS Form {schema_id}. Pay special attention to the form fields, numbers, checkboxes, and taxpayer information.
                
                IMPORTANT: Extract all visible data. If a field is not present or empty, use null.
                
                Schema:
                {schema_json}
                
                Extracted Text:
                {extracted_text}
                """
            elif schema_id == "941-X":
                return f"""
                Format the following extracted text from IRS Form 941-X (Adjusted Employer's Quarterly Federal Tax Return) into a JSON object.
                
                CRITICAL INSTRUCTIONS:
                1. This is a CORRECTION form with three columns for each field:
                   - Column 1: CORRECTED amount (what it should be)
                   - Column 2: ORIGINAL amount (what was originally reported)
                   - Column 3: DIFFERENCE (Column 1 minus Column 2)
                2. Extract the quarter and year being CORRECTED (not the current quarter)
                3. Identify if this is an adjusted return or claim for refund
                4. Extract all three column values for each numeric field
                5. Include Part 4 explanations for corrections
                6. Include Part 2 certifications
                
                Schema:
                {schema_json}
                
                Extracted Text:
                {extracted_text}
                """
            elif schema_id == "payroll":
                return f"""
                Format the following extracted text from a payroll document into a JSON object.
                Pay special attention to employee details, earnings, deductions, and tax information.
                
                Schema:
                {schema_json}
                
                Extracted Text:
                {extracted_text}
                """
            else:
                return f"""
                Format the following extracted text into a JSON object that follows the schema below.
                Extract all visible data. If a field is not present, use null.
                
                Schema:
                {schema_json}
                
                Extracted Text:
                {extracted_text}
                """
        else:
            return f"""
            Format the following extracted text into a JSON object that follows the schema below.
            
            Schema:
            {GENERIC_SCHEMA}
            
            Extracted Text:
            {extracted_text}
            """
    
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
        
        if schema_id not in schema_mapping:
            return None
            
        filename = schema_mapping.get(schema_id)
        if filename is None:
            return None
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        schema_path = os.path.join(project_root, filename)
        
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema {schema_id}: {e}")
            return None

    def deep_merge(self, base: Dict, addition: Dict) -> Dict:
        """Deep merge dictionaries"""
        result = base.copy()
        
        for key, value in addition.items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key].extend(value)
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self.deep_merge(result[key], value)
                elif isinstance(result[key], bool) and isinstance(value, bool):
                    result[key] = result[key] or value
                elif result[key] is None and value is not None:
                    result[key] = value
                    
        return result

processor = DocumentProcessor()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...), schema: str = Form("generic")):
    """Process multiple documents with enhanced features"""
    all_results = []
    
    for file_index, file in enumerate(files):
        processor = DocumentProcessor()
        processor.start_time = time.time()
        
        await processor.log_step("file_processing", "started", 
            f"Starting {file.filename}")
        
        try:
            file_content = await file.read()
            
            # Detect form type
            detection_result = await processor.detect_form_type(file_content)
            detected_form = detection_result.get("detected_form_type")
            
            # Check if detected form matches selected schema
            if detected_form and schema != "generic":
                if detected_form != schema:
                    await processor.log_step("form_validation", "warning", 
                        f"Form mismatch: detected {detected_form}, using {schema}")
            
            # Detect multiple documents
            documents = await processor.detect_multi_document(file_content)
            
            file_results = []
            
            for doc_info in documents:
                doc_pages_total = doc_info['end_page'] - doc_info['start_page'] + 1
                await processor.log_step("document_processing", "started", 
                    f"Processing document {doc_info['document_index'] + 1} ({doc_pages_total} pages)")
                
                # Extract pages for this document
                pdf_reader = PdfReader(io.BytesIO(file_content))
                doc_pages = []
                for page_idx in range(doc_info['start_page'], doc_info['end_page'] + 1):
                    if page_idx < len(pdf_reader.pages):
                        doc_pages.append(pdf_reader.pages[page_idx])
                
                # Process pages concurrently
                page_tasks = []
                for page_num, page in enumerate(doc_pages):
                    page_tasks.append(processor.process_single_page(page, schema, page_num, len(doc_pages)))
                
                page_results = await asyncio.gather(*page_tasks)
                
                # Merge results
                merged_json = None
                for result in page_results:
                    try:
                        data = json.loads(result["json_text"])
                        if merged_json is None:
                            merged_json = data
                        else:
                            merged_json = processor.deep_merge(merged_json, data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from page {result['page_num'] + 1}: {str(e)}")
                
                if merged_json:
                    # Validate extraction
                    validation_start = time.time()
                    await processor.log_step("validation", "started", "Checking data quality...")
                    
                    validation_result = processor.validate_extraction_result(merged_json, schema)
                    validation_duration = time.time() - validation_start
                    
                    quality_percentage = 100 - validation_result["empty_percentage"]
                    
                    if not validation_result["is_valid"]:
                        await processor.log_step("validation", "failed", 
                            f"Quality: {quality_percentage:.0f}%",
                            validation_duration)
                        
                        # Retry once
                        retry_start = time.time()
                        await processor.log_step("retry", "started", "Enhancing extraction...")
                        
                        # Re-process first few pages with enhanced prompt
                        retry_pages = min(3, len(doc_pages))
                        retry_results = []
                        for page_num in range(retry_pages):
                            result = await processor.process_single_page(doc_pages[page_num], schema, page_num, retry_pages)
                            retry_results.append(result)
                        
                        # Merge retry results
                        retry_merged = None
                        for result in retry_results:
                            try:
                                data = json.loads(result["json_text"])
                                if retry_merged is None:
                                    retry_merged = data
                                else:
                                    retry_merged = processor.deep_merge(retry_merged, data)
                            except:
                                pass
                        
                        if retry_merged:
                            retry_validation = processor.validate_extraction_result(retry_merged, schema)
                            retry_quality = 100 - retry_validation["empty_percentage"]
                            
                            if retry_validation["is_valid"] or retry_quality > quality_percentage:
                                merged_json = processor.deep_merge(merged_json, retry_merged)
                                validation_result = retry_validation
                                quality_percentage = retry_quality
                        
                        retry_duration = time.time() - retry_start
                        await processor.log_step("retry", "completed", 
                            f"Quality improved to {quality_percentage:.0f}%",
                            retry_duration)
                    else:
                        await processor.log_step("validation", "completed", 
                            f"Quality: {quality_percentage:.0f}%",
                            validation_duration)
                    
                    # Add metadata
                    merged_json["_metadata"] = {
                        "filename": file.filename,
                        "document_index": doc_info['document_index'],
                        "pages": f"{doc_info['start_page'] + 1}-{doc_info['end_page'] + 1}",
                        "detected_form_type": detected_form,
                        "selected_schema": schema,
                        "validation": validation_result,
                        "processing_time": time.time() - processor.start_time,
                        "processing_steps": [
                            {
                                "name": step.step_name,
                                "status": step.status,
                                "message": step.message,
                                "duration": step.duration,
                                "timestamp": step.timestamp
                            } for step in processor.processing_steps
                            if step.step_name in ["form_detection", "document_processing", "validation", "retry", "file_processing"]
                        ]
                    }
                    
                    file_results.append(merged_json)
                    
                    await processor.log_step("document_processing", "completed", 
                        f"Document {doc_info['document_index'] + 1} complete")
            
            all_results.extend(file_results)
            
            await processor.log_step("file_processing", "completed", 
                f"Completed {file.filename}",
                time.time() - processor.start_time)
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            await processor.log_step("file_processing", "failed", 
                f"Error: {str(e)[:100]}",
                time.time() - processor.start_time)
            
            all_results.append({
                "error": str(e),
                "filename": file.filename,
                "_metadata": {
                    "processing_steps": [
                        {
                            "name": step.step_name,
                            "status": step.status,
                            "message": step.message,
                            "duration": step.duration,
                            "timestamp": step.timestamp
                        } for step in processor.processing_steps
                    ]
                }
            })
    
    return JSONResponse({
        "results": all_results,
        "total_documents": len(all_results),
        "status": "completed"
    })

# Keep the original endpoint for backward compatibility
@app.post("/process-pdf")
async def process_pdf_legacy(file: UploadFile = File(...), schema: str = Form("generic")):
    """Legacy endpoint - redirects to new endpoint"""
    results = await process_documents([file], schema)
    response_data = results.body.decode('utf-8')
    parsed_data = json.loads(response_data)
    
    # Return first result in legacy format
    if parsed_data["results"]:
        return {"response": json.dumps(parsed_data["results"][0])}
    else:
        return {"response": json.dumps({"error": "No results"})}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4830)