#!/usr/bin/env python3
"""Test schema loading and example generation"""

import json
import os
import sys
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_enhanced_v9 import DocumentProcessor

# Create a dummy processor
processor = DocumentProcessor("test-session")

# Test loading and converting schema
try:
    print("Loading 941-X schema...")
    schema = processor.load_schema("941-X")
    print(f"Schema loaded: {schema is not None}")
    
    if schema:
        print("\nConverting to example...")
        try:
            example = processor.schema_to_example_v2(schema, "941-X")
            print("Example created successfully")
            
            # Try to serialize it
            print("\nSerializing to JSON...")
            json_str = json.dumps(example, indent=2)
            print(f"Serialized successfully, length: {len(json_str)}")
            
        except Exception as e:
            print(f"Error in schema_to_example_v2: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()