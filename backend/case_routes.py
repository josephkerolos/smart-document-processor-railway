"""
Case Management Routes
Provides API endpoints for reading and querying case data
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime
from case_reader import CaseReader
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/cases", tags=["cases"])

@router.get("/")
async def get_all_cases(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get all cases with pagination"""
    try:
        reader = CaseReader()
        cases = await reader.load_all_cases()
        
        # Convert to list for pagination
        case_list = [
            {"session_id": session_id, **case_data}
            for session_id, case_data in cases.items()
        ]
        
        # Sort by created_at descending
        case_list.sort(
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )
        
        # Apply pagination
        total = len(case_list)
        paginated = case_list[offset:offset + limit]
        
        return JSONResponse(content={
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "cases": paginated
        })
        
    except Exception as e:
        logger.error(f"Error getting cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_cases_summary():
    """Get summary statistics of all cases"""
    try:
        reader = CaseReader()
        await reader.load_all_cases()
        summary = reader.get_case_summary()
        
        return JSONResponse(content={
            "success": True,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting case summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_cases(
    company_name: Optional[str] = Query(None),
    form_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Search cases with filters"""
    try:
        reader = CaseReader()
        await reader.load_all_cases()
        
        # Parse dates if provided
        date_from_obj = None
        date_to_obj = None
        
        if date_from:
            try:
                date_from_obj = datetime.fromisoformat(date_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
        
        if date_to:
            try:
                date_to_obj = datetime.fromisoformat(date_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")
        
        # Filter cases
        filtered = reader.filter_cases(
            company_name=company_name,
            form_type=form_type,
            status=status,
            date_from=date_from_obj,
            date_to=date_to_obj
        )
        
        # Sort by created_at descending
        filtered.sort(
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )
        
        # Apply limit
        filtered = filtered[:limit]
        
        return JSONResponse(content={
            "success": True,
            "total": len(filtered),
            "filters": {
                "company_name": company_name,
                "form_type": form_type,
                "status": status,
                "date_from": date_from,
                "date_to": date_to
            },
            "cases": filtered
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_case_by_id(session_id: str):
    """Get a specific case by session ID"""
    try:
        reader = CaseReader()
        await reader.load_all_cases()
        case = reader.get_case(session_id)
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "case": case
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/files")
async def get_case_files(session_id: str):
    """Get files associated with a specific case"""
    try:
        reader = CaseReader()
        await reader.load_all_cases()
        case = reader.get_case(session_id)
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        files = case.get("files_uploaded", [])
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "total_files": len(files),
            "files": files,
            "gdrive_folder_id": case.get("gdrive_folder_id"),
            "gdrive_folder_path": case.get("gdrive_folder_path")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting files for case {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_cases():
    """Reload cases from files and memory"""
    try:
        reader = CaseReader()
        cases = await reader.load_all_cases()
        
        return JSONResponse(content={
            "success": True,
            "message": "Cases reloaded successfully",
            "total_cases": len(cases)
        })
        
    except Exception as e:
        logger.error(f"Error reloading cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))