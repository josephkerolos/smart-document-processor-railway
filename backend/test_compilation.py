#!/usr/bin/env python3
"""
Test script for the enhanced compilation functionality
"""

import json
import logging
from json_fixer import parse_json_safely, extract_json_from_llm_response, advanced_json_reconstruction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test cases with various malformed JSON responses
test_cases = [
    # Case 1: JSON with markdown wrapper
    {
        "name": "Markdown wrapped JSON",
        "input": """Here's the compiled data:
```json
{
    "employerInfo": {
        "ein": "12-3456789",
        "name": "ABC Company LLC"
    },
    "lines": {
        "line1": {
            "corrected": 50000,
            "original": 45000,
            "difference": 5000
        }
    }
}
```
That's the extracted data.""",
        "expected_keys": ["employerInfo", "lines"]
    },
    
    # Case 2: Truncated JSON
    {
        "name": "Truncated JSON",
        "input": """{
    "employerInfo": {
        "ein": "12-3456789",
        "name": "ABC Company LLC"
    },
    "lines": {
        "line1": {
            "corrected": 50000,
            "original": 45000,
            "differenc""",
        "expected_keys": ["employerInfo", "lines"]
    },
    
    # Case 3: JSON with schema mixed in
    {
        "name": "Schema-like response",
        "input": """{
    "employerInfo": {
        "type": "object",
        "properties": {
            "ein": "12-3456789",
            "name": "ABC Company LLC"
        }
    },
    "lines": {
        "line1": {
            "corrected": 50000,
            "original": 45000,
            "difference": 5000
        }
    }
}""",
        "expected_keys": ["employerInfo", "lines"]
    },
    
    # Case 4: Malformed with missing quotes
    {
        "name": "Missing quotes",
        "input": """{
    employerInfo: {
        ein: "12-3456789",
        name: "ABC Company LLC"
    },
    lines: {
        line1: {
            corrected: 50000,
            original: 45000,
            difference: 5000
        }
    }
}""",
        "expected_keys": ["employerInfo", "lines"]
    },
    
    # Case 5: Python-like dict
    {
        "name": "Python dict format",
        "input": """{'employerInfo': {'ein': '12-3456789', 'name': 'ABC Company LLC'}, 'lines': {'line1': {'corrected': 50000, 'original': 45000, 'difference': 5000}}}""",
        "expected_keys": ["employerInfo", "lines"]
    }
]

def test_parsing():
    """Test the various parsing strategies"""
    print("Testing Enhanced JSON Parsing\n" + "=" * 50)
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Input preview: {test_case['input'][:100]}...")
        
        # Test 1: parse_json_safely
        result1 = parse_json_safely(test_case['input'])
        success1 = all(key in result1 for key in test_case['expected_keys']) if not result1.get("_parse_error") else False
        print(f"  parse_json_safely: {'✓' if success1 else '✗'} - {len(result1)} fields")
        
        # Test 2: extract_json_from_llm_response
        result2 = extract_json_from_llm_response(test_case['input'])
        success2 = all(key in result2 for key in test_case['expected_keys']) if result2 else False
        print(f"  extract_json_from_llm_response: {'✓' if success2 else '✗'} - {len(result2) if result2 else 0} fields")
        
        # Test 3: advanced_json_reconstruction
        result3 = advanced_json_reconstruction(test_case['input'])
        success3 = any(key in (result3 or {}) for key in test_case['expected_keys'])
        print(f"  advanced_json_reconstruction: {'✓' if success3 else '✗'} - {len(result3) if result3 else 0} fields")
        
        # Store results
        results.append({
            "test": test_case['name'],
            "parse_json_safely": success1,
            "extract_json_from_llm": success2,
            "advanced_reconstruction": success3,
            "any_success": success1 or success2 or success3
        })
        
        # Show successfully parsed data
        if success1 or success2 or success3:
            successful_result = result1 if success1 else (result2 if success2 else result3)
            print(f"  Successfully parsed data preview: {json.dumps(successful_result, indent=2)[:200]}...")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    total_tests = len(test_cases)
    successful_tests = sum(1 for r in results if r['any_success'])
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    # Detailed breakdown
    print("\nMethod Success Rates:")
    for method in ["parse_json_safely", "extract_json_from_llm", "advanced_reconstruction"]:
        successes = sum(1 for r in results if r[method])
        print(f"  {method}: {successes}/{total_tests} ({successes/total_tests*100:.1f}%)")

if __name__ == "__main__":
    test_parsing()