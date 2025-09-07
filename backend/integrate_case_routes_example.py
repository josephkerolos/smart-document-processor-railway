"""
Example: How to integrate case routes into your FastAPI application

Add these lines to your main application file (e.g., main_enhanced_v12.py)
to enable the case reading API endpoints.
"""

# Add this import at the top of your main file
from case_routes import router as case_router

# After creating your FastAPI app instance, add this line:
# app.include_router(case_router)

# Example integration:
"""
# In main_enhanced_v12.py or your main application file:

from fastapi import FastAPI
from case_routes import router as case_router

app = FastAPI()

# Include the case routes
app.include_router(case_router)

# Your other routes and code...
"""

# The following endpoints will be available:
"""
GET  /api/cases/                 - Get all cases with pagination
GET  /api/cases/summary          - Get summary statistics
GET  /api/cases/search           - Search cases with filters
GET  /api/cases/{session_id}     - Get specific case by ID
GET  /api/cases/{session_id}/files - Get files for a specific case
POST /api/cases/reload           - Reload cases from storage

Example usage:

1. Get all cases:
   curl http://localhost:8000/api/cases/

2. Get summary:
   curl http://localhost:8000/api/cases/summary

3. Search cases:
   curl "http://localhost:8000/api/cases/search?status=completed&company_name=Acme"

4. Get specific case:
   curl http://localhost:8000/api/cases/abc123-def456

5. Get case files:
   curl http://localhost:8000/api/cases/abc123-def456/files

6. Reload cases:
   curl -X POST http://localhost:8000/api/cases/reload
"""