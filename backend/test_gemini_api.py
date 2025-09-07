#!/usr/bin/env python3
"""Test Gemini API connectivity and response"""

import os
from dotenv import load_dotenv
from google import genai
import json

# Load environment variables
load_dotenv()

# Test the API key
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:10]}...{api_key[-5:]}")

try:
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Test 1: Simple text generation
    print("\n1. Testing simple text generation...")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=["What is 2+2? Reply with just the number."],
        config={
            "max_output_tokens": 10,
            "temperature": 0.0
        }
    )
    print(f"Response: {response.text if hasattr(response, 'text') else 'No text attribute'}")
    print(f"Response type: {type(response)}")
    print(f"Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
    
    # Test 2: JSON generation
    print("\n2. Testing JSON generation...")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=['Return this exact JSON: {"test": 123, "status": "ok"}'],
        config={
            "max_output_tokens": 100,
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    )
    print(f"Response: {response.text if hasattr(response, 'text') else 'No text attribute'}")
    
    # Test 3: Complex JSON with schema
    print("\n3. Testing complex JSON extraction...")
    test_prompt = """Extract this data and return as JSON:
    Company: ABC Corp
    EIN: 12-3456789
    Amount: $50,000
    
    Return format: {"company": "...", "ein": "...", "amount": ...}"""
    
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[test_prompt],
        config={
            "max_output_tokens": 500,
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    )
    print(f"Response: {response.text if hasattr(response, 'text') else 'No text attribute'}")
    
    # Test 4: Gemini Pro
    print("\n4. Testing Gemini Pro...")
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=['Return this exact JSON: {"status": "pro_test", "value": 999}'],
        config={
            "max_output_tokens": 100,
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    )
    print(f"Response: {response.text if hasattr(response, 'text') else 'No text attribute'}")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")