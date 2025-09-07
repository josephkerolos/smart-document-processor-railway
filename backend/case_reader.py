"""
Case Reader Utility
Provides functions to read and access case/session data from the processing system
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CaseReader:
    """Utility class for reading case/session data"""
    
    def __init__(self, status_dir: str = "processing_status"):
        self.status_dir = status_dir
        self.cases = {}
        
    async def load_from_files(self) -> Dict[str, Dict[str, Any]]:
        """Load all cases from saved status files"""
        cases = {}
        
        if os.path.exists(self.status_dir):
            for filename in os.listdir(self.status_dir):
                if filename.endswith("_status.json"):
                    session_id = filename.replace("_status.json", "")
                    filepath = os.path.join(self.status_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            status = json.load(f)
                            cases[session_id] = status
                            logger.info(f"Loaded case {session_id} from file")
                    except Exception as e:
                        logger.error(f"Error loading case {session_id}: {e}")
        
        return cases
    
    def load_from_files_sync(self) -> Dict[str, Dict[str, Any]]:
        """Synchronous version of load_from_files"""
        cases = {}
        
        if os.path.exists(self.status_dir):
            for filename in os.listdir(self.status_dir):
                if filename.endswith("_status.json"):
                    session_id = filename.replace("_status.json", "")
                    filepath = os.path.join(self.status_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            status = json.load(f)
                            cases[session_id] = status
                            logger.info(f"Loaded case {session_id} from file")
                    except Exception as e:
                        logger.error(f"Error loading case {session_id}: {e}")
        
        return cases
    
    async def load_from_memory(self) -> Dict[str, Dict[str, Any]]:
        """Load cases from the in-memory processing_status_db"""
        try:
            # Import here to avoid circular imports
            from google_drive_routes import processing_status_db
            return dict(processing_status_db)
        except ImportError:
            logger.warning("Could not import processing_status_db from google_drive_routes")
            return {}
    
    async def load_all_cases(self) -> Dict[str, Dict[str, Any]]:
        """Load all cases from both files and memory"""
        # Load from files
        file_cases = await self.load_from_files()
        
        # Load from memory
        memory_cases = await self.load_from_memory()
        
        # Merge, with memory taking precedence for duplicates
        all_cases = {**file_cases, **memory_cases}
        
        self.cases = all_cases
        return all_cases
    
    def get_case(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific case by session ID"""
        return self.cases.get(session_id)
    
    def get_all_cases(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded cases"""
        return self.cases
    
    def filter_cases(self, 
                    company_name: Optional[str] = None,
                    form_type: Optional[str] = None,
                    status: Optional[str] = None,
                    date_from: Optional[datetime] = None,
                    date_to: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Filter cases based on criteria"""
        filtered = []
        
        for session_id, case_data in self.cases.items():
            # Extract metadata
            metadata = case_data.get("metadata", {})
            
            # Apply filters
            if company_name and metadata.get("company_name", "").lower() != company_name.lower():
                continue
            
            if form_type and metadata.get("form_type", "").lower() != form_type.lower():
                continue
            
            if status and case_data.get("status") != status:
                continue
            
            # Date filtering
            if case_data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(case_data["created_at"])
                    
                    if date_from and created_at < date_from:
                        continue
                    
                    if date_to and created_at > date_to:
                        continue
                except ValueError:
                    logger.warning(f"Invalid date format for case {session_id}")
            
            # Add session_id to the result
            result = {
                "session_id": session_id,
                **case_data
            }
            filtered.append(result)
        
        return filtered
    
    def get_case_summary(self) -> Dict[str, Any]:
        """Get a summary of all cases"""
        total_cases = len(self.cases)
        
        status_counts = {}
        form_type_counts = {}
        company_counts = {}
        
        for case_data in self.cases.values():
            # Count by status
            status = case_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by form type
            metadata = case_data.get("metadata", {})
            form_type = metadata.get("form_type", "unknown")
            form_type_counts[form_type] = form_type_counts.get(form_type, 0) + 1
            
            # Count by company
            company = metadata.get("company_name", "unknown")
            company_counts[company] = company_counts.get(company, 0) + 1
        
        return {
            "total_cases": total_cases,
            "status_breakdown": status_counts,
            "form_type_breakdown": form_type_counts,
            "company_breakdown": company_counts,
            "last_updated": datetime.now().isoformat()
        }


# Convenience functions for quick access
async def read_all_cases() -> Dict[str, Dict[str, Any]]:
    """Quick function to read all cases"""
    reader = CaseReader()
    return await reader.load_all_cases()


def read_all_cases_sync() -> Dict[str, Dict[str, Any]]:
    """Synchronous version to read all cases"""
    reader = CaseReader()
    # Only load from files in sync mode
    return reader.load_from_files_sync()


async def get_case_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific case by ID"""
    reader = CaseReader()
    await reader.load_all_cases()
    return reader.get_case(session_id)


# Example usage
if __name__ == "__main__":
    # Example of using the case reader
    async def main():
        reader = CaseReader()
        
        # Load all cases
        cases = await reader.load_all_cases()
        print(f"Loaded {len(cases)} cases")
        
        # Get summary
        summary = reader.get_case_summary()
        print(f"Summary: {json.dumps(summary, indent=2)}")
        
        # Filter cases
        filtered = reader.filter_cases(status="completed")
        print(f"Found {len(filtered)} completed cases")
        
        # Get specific case
        if cases:
            first_id = list(cases.keys())[0]
            case = reader.get_case(first_id)
            print(f"First case: {json.dumps(case, indent=2)}")
    
    # Run the example
    asyncio.run(main())