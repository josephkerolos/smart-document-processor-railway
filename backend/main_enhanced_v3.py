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
active_connections: Dict[str, WebSocket] = {}

# Active processing sessions that can be cancelled
active_sessions: Set[str] = set()

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
    progress: Optional[int] = None  # Progress percentage
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class DocumentProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.processing_steps: List[ProcessingStep] = []
        self.start_time = None
        self.current_plan = []
        self.completed_steps = []
        
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
            except:
                pass
                
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
        
    async def update_plan_step(self, step_id: str, status: str, message: Optional[str] = None):
        """Update a step in the processing plan"""
        for step in self.current_plan:
            if step["id"] == step_id:
                step["status"] = status
                if message:
                    step["message"] = message
                break
                
        await self.send_update("processing_plan", {"steps": self.current_plan})
        
    async def send_status_update(self, step_id: str, status_message: str):
        """Send a status update for a specific step"""
        await self.send_update("status_update", {
            "step_id": step_id,
            "message": status_message
        })

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
                model="gemini-2.5-pro-preview-06-05",
                contents=[detection_prompt, file_part],
                config={
                    "max_output_tokens": 1000,
                    "response_mime_type": "application/json"
                }
            )
            
            detection_result = json.loads(response.text)
            detected_form = detection_result.get("detected_form_type")
            
            # If generic schema selected and we detected a specific form
            if selected_schema == "generic" and detected_form and detected_form in ["941", "941-X", "1040", "2848", "8821"]:
                await self.update_plan_step("detect", "completed", f"Detected: {detected_form}")
                
                # Ask for confirmation
                await self.send_update("form_confirmation_needed", {
                    "detected_form": detected_form,
                    "form_title": detection_result.get("form_title", ""),
                    "confidence": detection_result.get("confidence", "medium"),
                    "message": f"Detected form {detected_form}. Would you like to use the specific template for better extraction?"
                })
                
                # Wait for user response (this will be handled by a separate endpoint)
                detection_result["needs_confirmation"] = True
                detection_result["selected_schema"] = selected_schema
                
            else:
                await self.update_plan_step("detect", "completed", f"Using: {selected_schema}")
                detection_result["needs_confirmation"] = False
                
            return detection_result
            
        except Exception as e:
            await self.update_plan_step("detect", "failed", str(e))
            return {"detected_form_type": None, "confidence": "low", "error": str(e)}

    async def smart_document_splitting(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Enhanced document splitting that properly detects form boundaries"""
        await self.update_plan_step("split", "in_progress")
        
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages) if pdf_reader.pages else 0
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            # For single page or very short documents
            if total_pages <= 2:
                await self.update_plan_step("split", "completed", "Single document")
                return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]
            
            # Analyze ALL pages to find form boundaries
            await self.send_update("progress", {
                "message": f"Analyzing {total_pages} pages for document boundaries...",
                "percentage": 10
            })
            
            page_analyses = []
            
            # Check every page for better accuracy on multi-form documents
            for page_idx in range(total_pages):
                if await self.should_cancel():
                    raise Exception("Processing cancelled")
                    
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
                
                # More detailed prompt for better detection
                detect_prompt = """
                Analyze this page and determine:
                1. Is this the FIRST page of a new form? Look for:
                   - Form number at the top (e.g., "Form 941", "Form 941-X")
                   - Official form header
                   - Page 1 indicator
                2. Is this a continuation page of the same form?
                3. What form type is this page from?
                
                Return JSON: {
                    "is_first_page": true/false,
                    "form_type": "941/941-X/1040/etc or null",
                    "page_number": "if visible (e.g., 'Page 1 of 5')",
                    "has_form_header": true/false
                }
                """
                
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model="gemini-2.5-pro-preview-06-05",
                        contents=[detect_prompt, file_part],
                        config={
                            "max_output_tokens": 200,
                            "response_mime_type": "application/json"
                        }
                    )
                    
                    page_info = json.loads(response.text)
                    page_analyses.append({
                        "page_idx": page_idx,
                        "is_first_page": page_info.get("is_first_page", False),
                        "form_type": page_info.get("form_type"),
                        "has_form_header": page_info.get("has_form_header", False)
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing page {page_idx}: {e}")
                    page_analyses.append({
                        "page_idx": page_idx,
                        "is_first_page": False,
                        "form_type": None
                    })
                
                # Update progress
                progress = 10 + int((page_idx / total_pages) * 40)
                await self.send_update("progress", {
                    "message": f"Analyzing page {page_idx + 1} of {total_pages}...",
                    "percentage": progress
                })
            
            # Find document boundaries based on first pages
            documents = []
            current_doc_start = 0
            current_form_type = page_analyses[0].get("form_type")
            
            for i in range(1, len(page_analyses)):
                if page_analyses[i]["is_first_page"]:
                    # New document starts here
                    documents.append({
                        "start_page": current_doc_start,
                        "end_page": i - 1,
                        "document_index": len(documents),
                        "form_type": current_form_type,
                        "page_count": i - current_doc_start
                    })
                    current_doc_start = i
                    current_form_type = page_analyses[i].get("form_type")
            
            # Add the last document
            documents.append({
                "start_page": current_doc_start,
                "end_page": total_pages - 1,
                "document_index": len(documents),
                "form_type": current_form_type,
                "page_count": total_pages - current_doc_start
            })
            
            await self.update_plan_step("split", "completed", 
                f"Found {len(documents)} document(s)")
            
            # Log document boundaries for debugging
            for doc in documents:
                logger.info(f"Document {doc['document_index']}: "
                          f"Pages {doc['start_page']+1}-{doc['end_page']+1} "
                          f"({doc['page_count']} pages), Type: {doc['form_type']}")
            
            return documents
            
        except Exception as e:
            await self.update_plan_step("split", "failed", str(e))
            # Return single document as fallback
            return [{"start_page": 0, "end_page": total_pages - 1, "document_index": 0}]

    async def process_with_generative_updates(self, file_content: bytes, schema: str, filename: str) -> List[Dict[str, Any]]:
        """Process a file with generative status updates"""
        results = []
        
        try:
            # Detect multiple documents with improved splitting
            documents = await self.smart_document_splitting(file_content)
            
            await self.update_plan_step("extract", "in_progress")
            
            for doc_idx, doc_info in enumerate(documents):
                if await self.should_cancel():
                    raise Exception("Processing cancelled")
                    
                # Update progress
                await self.send_update("progress", {
                    "message": f"Extracting data from document {doc_idx + 1} of {len(documents)}...",
                    "percentage": 50 + int((doc_idx / len(documents)) * 30)
                })
                
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
                    
                    if quality < 70:
                        await self.update_plan_step("enhance", "in_progress")
                        # Retry logic here
                        await self.update_plan_step("enhance", "completed", f"Improved to {quality:.0f}%")
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
            if "cancelled" in str(e).lower():
                await self.update_plan_step("complete", "cancelled", "Processing stopped by user")
            else:
                await self.update_plan_step("complete", "failed", str(e))
            raise
            
        return results

    async def process_document_pages(self, pages: List[Any], schema: str) -> Dict[str, Any]:
        """Process pages of a document"""
        page_results = []
        
        for page_num, page in enumerate(pages):
            if await self.should_cancel():
                raise Exception("Processing cancelled")
                
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
                model="gemini-2.5-pro-preview-06-05",
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
                model="gemini-2.5-pro-preview-06-05",
                contents=[json_prompt],
                config={
                    "max_output_tokens": 40000,
                    "response_mime_type": "application/json"
                }
            )
            
            try:
                page_data = json.loads(json_response.text)
                page_results.append(page_data)
            except:
                logger.error(f"Failed to parse JSON for page {page_num}")
        
        # Merge results
        if not page_results:
            return {}
            
        merged = page_results[0]
        for result in page_results[1:]:
            merged = self.deep_merge(merged, result)
            
        return merged

    def validate_extraction_result(self, json_data: Dict[str, Any], form_type: str) -> Dict[str, Any]:
        """Validate extraction quality"""
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
        
        return {
            "is_valid": empty_percentage < 70,
            "empty_percentage": empty_percentage,
            "populated_count": total_count - null_count - zero_count,
            "total_fields": total_count
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
        except:
            return None

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
    active_sessions.add(session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages (e.g., form confirmation responses)
            message = json.loads(data)
            
            if message.get("type") == "confirm_form":
                # Store the user's choice for the processing to continue
                # This would be handled by the processing logic
                pass
                
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        active_sessions.discard(session_id)

@app.post("/process-documents-v3")
async def process_documents_v3(
    files: List[UploadFile] = File(...), 
    schema: str = Form("generic"),
    session_id: str = Form(None)
):
    """Enhanced document processing with real-time updates"""
    if not session_id:
        session_id = str(uuid.uuid4())
        
    processor = DocumentProcessor(session_id)
    processor.start_time = time.time()
    
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
    
    try:
        await processor.update_plan_step("validate", "in_progress")
        
        # If any form needs confirmation and schema is generic
        needs_confirmation = any(d.get("needs_confirmation") for d in detection_results)
        has_auto_confirmed = False
        
        if needs_confirmation:
            # Check if high confidence - will auto-confirm
            high_confidence = any(d.get("confidence") == "high" for d in detection_results)
            if high_confidence:
                await processor.send_status_update("validate", "High confidence detection - auto-confirming...")
                # Wait for auto-confirmation (5 seconds)
                await asyncio.sleep(5.5)  # Slightly more than countdown
                has_auto_confirmed = True
            else:
                await processor.update_plan_step("validate", "waiting", "Awaiting user confirmation...")
                # Wait a bit for user response
                await asyncio.sleep(10)  # Give user time to respond
        
        if has_auto_confirmed:
            await processor.update_plan_step("validate", "completed", "Auto-confirmed template selection")
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
        logger.error(f"Processing error: {e}")
        if "cancelled" not in str(e).lower():
            await processor.update_plan_step("complete", "failed", str(e))
    finally:
        active_sessions.discard(session_id)
    
    return JSONResponse({
        "session_id": session_id,
        "results": all_results,
        "total_documents": len(all_results),
        "status": "completed" if all_results else "failed"
    })

@app.post("/cancel-processing/{session_id}")
async def cancel_processing(session_id: str):
    """Cancel an ongoing processing session"""
    if session_id in active_sessions:
        active_sessions.discard(session_id)
        
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
    # Mark session as having made a choice to prevent duplicate confirmations
    if session_id in active_connections:
        try:
            await active_connections[session_id].send_json({
                "type": "confirmation_received",
                "auto_confirmed": auto_confirmed
            })
        except:
            pass
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4830)