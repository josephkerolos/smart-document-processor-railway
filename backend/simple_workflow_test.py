"""
Simple test of the enhanced workflow without file organizer
"""

import asyncio
from main_enhanced_v12 import app
import uvicorn

if __name__ == "__main__":
    print("Starting simple enhanced workflow test...")
    uvicorn.run(app, host="0.0.0.0", port=4830)