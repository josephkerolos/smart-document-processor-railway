"""
Google Drive Routes and Status Management
Provides APIs for configuration, status tracking, and querying
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os
import aiofiles
from google_drive_config import config_manager, GoogleDriveConfig
from google_drive_integration import GoogleDriveIntegration
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/gdrive", tags=["google_drive"])

# Import database state manager
from db_state_manager import db_state_manager
import asyncio

# Create wrapper for backward compatibility
class ProcessingStatusDBWrapper:
    """Wrapper to provide dict-like interface for processing status DB"""
    def __init__(self):
        self._cache = {}
    
    def __setitem__(self, key, value):
        # Convert dict assignment to DB save
        asyncio.create_task(db_state_manager.save_processing_status(
            key, 
            value.get('metadata', {}).get('batch_id', ''),
            value.get('status', 'processing'),
            value.get('files', []),
            value.get('extractions', {}),
            value.get('metadata', {}).get('file_count', 0),
            value
        ))
        self._cache[key] = value
    
    def __getitem__(self, key):
        return self._cache.get(key, {})
    
    def __contains__(self, key):
        return key in self._cache
    
    def get(self, key, default=None):
        return self._cache.get(key, default)
    
    def keys(self):
        return self._cache.keys()
    
    def values(self):
        return self._cache.values()
    
    def items(self):
        return self._cache.items()

# Use wrapper for backward compatibility
processing_status_db = ProcessingStatusDBWrapper()
folder_mappings_db = {}    # local_path -> gdrive_folder_id

@router.get("/config")
async def get_config():
    """Get current Google Drive configuration"""
    return JSONResponse(content={
        "success": True,
        "config": config_manager.config.to_dict()
    })

@router.post("/config")
async def update_config(updates: Dict[str, Any] = Body(...)):
    """Update Google Drive configuration"""
    try:
        updated_config = config_manager.update_config(updates)
        return JSONResponse(content={
            "success": True,
            "config": updated_config.to_dict(),
            "message": "Configuration updated successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/config/reset")
async def reset_config():
    """Reset Google Drive configuration to defaults"""
    config_manager.config = GoogleDriveConfig()
    config_manager.save_config()
    return JSONResponse(content={
        "success": True,
        "config": config_manager.config.to_dict(),
        "message": "Configuration reset to defaults"
    })

@router.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    """Get processing status for a specific session"""
    if session_id not in processing_status_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return JSONResponse(content={
        "success": True,
        "session_id": session_id,
        "status": processing_status_db[session_id]
    })

@router.post("/status/{session_id}")
async def update_processing_status(
    session_id: str,
    status_update: Dict[str, Any] = Body(...)
):
    """Update processing status for a session"""
    if session_id not in processing_status_db:
        processing_status_db[session_id] = {
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
            "metadata": {}
        }
    
    # Update status
    status = processing_status_db[session_id]
    for key, value in status_update.items():
        if key in status:
            status[key] = value
    
    status["updated_at"] = datetime.now().isoformat()
    
    # Save to file if configured
    if config_manager.config.track_processing_status:
        await save_status_to_file(session_id, status)
    
    return JSONResponse(content={
        "success": True,
        "session_id": session_id,
        "status": status
    })

@router.get("/query")
async def query_processed_documents(
    company_name: Optional[str] = Query(None),
    form_type: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Query processed documents with filters"""
    results = []
    
    for session_id, session_status in processing_status_db.items():
        # Apply filters
        metadata = session_status.get("metadata", {})
        
        if company_name and metadata.get("company_name", "").lower() != company_name.lower():
            continue
        
        if form_type and metadata.get("form_type", "").lower() != form_type.lower():
            continue
        
        if status and session_status.get("status") != status:
            continue
        
        # Date filtering
        created_at = datetime.fromisoformat(session_status.get("created_at", ""))
        if date_from:
            from_date = datetime.fromisoformat(date_from)
            if created_at < from_date:
                continue
        
        if date_to:
            to_date = datetime.fromisoformat(date_to)
            if created_at > to_date:
                continue
        
        results.append({
            "session_id": session_id,
            "status": session_status.get("status"),
            "created_at": session_status.get("created_at"),
            "company_name": metadata.get("company_name"),
            "form_type": metadata.get("form_type"),
            "gdrive_folder_path": session_status.get("gdrive_folder_path"),
            "gdrive_folder_id": session_status.get("gdrive_folder_id"),
            "files_uploaded": len(session_status.get("files_uploaded", []))
        })
    
    # Sort by created_at descending
    results.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply limit
    results = results[:limit]
    
    return JSONResponse(content={
        "success": True,
        "total": len(results),
        "results": results
    })

@router.post("/upload-to-path")
async def upload_to_specific_path(request: Dict[str, Any] = Body(...)):
    """Upload file to a specific Google Drive path"""
    file_path = request.get("file_path")
    gdrive_path = request.get("gdrive_path")  # e.g., "ProcessedDocuments/Company/2024/Q1"
    custom_name = request.get("custom_name")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    try:
        async with GoogleDriveIntegration() as gdrive:
            # Parse the path and create folder structure
            path_parts = gdrive_path.strip('/').split('/')
            current_folder_id = config_manager.config.root_folder_id or "root"
            
            # Create each folder in the path
            for folder_name in path_parts:
                # Search for existing folder
                existing_id = await gdrive.search_folder(folder_name, current_folder_id)
                
                if existing_id:
                    current_folder_id = existing_id
                else:
                    # Create new folder
                    result = await gdrive.create_folder(folder_name, current_folder_id)
                    current_folder_id = result["folder"]["id"]
            
            # Upload file
            result = await gdrive.upload_file(
                file_path,
                current_folder_id,
                custom_name
            )
            
            return JSONResponse(content={
                "success": True,
                "file_id": result["file"]["id"],
                "file_name": result["file"]["name"],
                "folder_path": gdrive_path,
                "folder_id": current_folder_id
            })
            
    except Exception as e:
        logger.error(f"Upload to path failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-from-path")
async def download_from_specific_path(request: Dict[str, Any] = Body(...)):
    """Download file from a specific Google Drive path"""
    gdrive_file_id = request.get("file_id")
    gdrive_path = request.get("gdrive_path")
    local_path = request.get("local_path", "downloads")
    
    if not gdrive_file_id and not gdrive_path:
        raise HTTPException(status_code=400, detail="Either file_id or gdrive_path required")
    
    try:
        async with GoogleDriveIntegration() as gdrive:
            # Implementation would require additional methods in GoogleDriveIntegration
            # For now, return a placeholder
            return JSONResponse(content={
                "success": True,
                "message": "Download functionality to be implemented",
                "file_id": gdrive_file_id,
                "gdrive_path": gdrive_path,
                "local_path": local_path
            })
            
    except Exception as e:
        logger.error(f"Download from path failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/folder-structure")
async def get_folder_structure(
    parent_id: Optional[str] = Query(None),
    max_depth: int = Query(3, ge=1, le=5)
):
    """Get Google Drive folder structure"""
    try:
        async with GoogleDriveIntegration() as gdrive:
            # Get folder structure starting from parent_id or root
            folder_id = parent_id or config_manager.config.root_folder_id or "root"
            
            # This would require implementing a recursive folder listing in GoogleDriveIntegration
            # For now, return configuration-based structure
            structure = {
                "root": folder_id,
                "output_path": config_manager.config.output_folder_path,
                "input_path": config_manager.config.input_folder_path,
                "auto_organize": config_manager.config.auto_organize,
                "folder_template": {
                    "company": config_manager.config.company_folder_template,
                    "form_type": config_manager.config.form_folder_template,
                    "date": config_manager.config.date_folder_template,
                    "batch": config_manager.config.batch_folder_template
                }
            }
            
            return JSONResponse(content={
                "success": True,
                "structure": structure
            })
            
    except Exception as e:
        logger.error(f"Get folder structure failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-custom-structure")
async def create_custom_folder_structure(request: Dict[str, Any] = Body(...)):
    """Create a custom folder structure in Google Drive"""
    structure = request.get("structure", {})  # Nested dict of folders
    parent_id = request.get("parent_id", config_manager.config.root_folder_id or "root")
    
    try:
        async with GoogleDriveIntegration() as gdrive:
            created_folders = await create_nested_folders(gdrive, structure, parent_id)
            
            return JSONResponse(content={
                "success": True,
                "created_folders": created_folders,
                "total_created": len(created_folders)
            })
            
    except Exception as e:
        logger.error(f"Create custom structure failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def save_status_to_file(session_id: str, status: Dict[str, Any]):
    """Save status to a JSON file"""
    status_dir = "processing_status"
    os.makedirs(status_dir, exist_ok=True)
    
    status_file = os.path.join(status_dir, f"{session_id}_status.json")
    async with aiofiles.open(status_file, 'w') as f:
        await f.write(json.dumps(status, indent=2))

async def load_status_from_file(session_id: str) -> Optional[Dict[str, Any]]:
    """Load status from a JSON file"""
    status_file = os.path.join("processing_status", f"{session_id}_status.json")
    if os.path.exists(status_file):
        async with aiofiles.open(status_file, 'r') as f:
            content = await f.read()
            return json.loads(content)
    return None

async def create_nested_folders(gdrive: GoogleDriveIntegration, 
                              structure: Dict[str, Any], 
                              parent_id: str) -> List[Dict[str, str]]:
    """Recursively create nested folder structure"""
    created = []
    
    for folder_name, subfolders in structure.items():
        # Create this folder
        result = await gdrive.create_folder(folder_name, parent_id)
        folder_id = result["folder"]["id"]
        created.append({
            "name": folder_name,
            "id": folder_id,
            "parent_id": parent_id
        })
        
        # Create subfolders if any
        if isinstance(subfolders, dict) and subfolders:
            sub_created = await create_nested_folders(gdrive, subfolders, folder_id)
            created.extend(sub_created)
    
    return created

# Initialize status from saved files on startup
async def load_saved_statuses():
    """Load all saved statuses from files"""
    status_dir = "processing_status"
    if os.path.exists(status_dir):
        for filename in os.listdir(status_dir):
            if filename.endswith("_status.json"):
                session_id = filename.replace("_status.json", "")
                status = await load_status_from_file(session_id)
                if status:
                    processing_status_db[session_id] = status