import uvicorn
import os
import sys

# Add the parent directory to the Python path so 'backend' module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4830))
    # Use relative import for local execution
    uvicorn.run("backend.main_enhanced_v12:app", host="0.0.0.0", port=port, reload=False)