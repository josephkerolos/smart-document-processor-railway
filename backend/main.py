# Import the latest enhanced version
from main_enhanced_v12 import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4830)