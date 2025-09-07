"""
JSON Fixer Module
Provides robust JSON fixing capabilities for malformed API responses
"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def fix_json_string(json_str: str) -> str:
    """
    Attempt to fix common JSON formatting issues
    """
    if not json_str:
        return "{}"
    
    # Remove any BOM or invisible characters
    json_str = json_str.encode('utf-8', 'ignore').decode('utf-8-sig').strip()
    
    # Extract JSON if wrapped in markdown code blocks
    if '```json' in json_str:
        json_str = json_str.split('```json')[1].split('```')[0].strip()
    elif '```' in json_str:
        json_str = json_str.split('```')[1].strip()
    
    # Remove control characters
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    
    # Find the actual JSON content
    json_start = json_str.find('{')
    json_end = json_str.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = json_str[json_start:json_end]
    else:
        return "{}"
    
    # Fix common issues
    fixed = json_str
    
    # Fix unquoted keys
    fixed = re.sub(r'(\w+):', r'"\1":', fixed)
    
    # Fix single quotes to double quotes
    fixed = fixed.replace("'", '"')
    
    # Fix missing commas between elements
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)
    fixed = re.sub(r'}\s*"', '},"', fixed)
    fixed = re.sub(r'"\s*{', '",{', fixed)
    fixed = re.sub(r']\s*"', '],"', fixed)
    fixed = re.sub(r'(\d)\s*"', r'\1,"', fixed)
    fixed = re.sub(r'}\s*{', '},{', fixed)
    fixed = re.sub(r']\s*\[', '],[', fixed)
    
    # Fix missing colons
    fixed = re.sub(r'"([^"]+)"\s+\{', r'"\1": {', fixed)
    fixed = re.sub(r'"([^"]+)"\s+\[', r'"\1": [', fixed)
    fixed = re.sub(r'"([^"]+)"\s+"', r'"\1": "', fixed)
    fixed = re.sub(r'"([^"]+)"\s+(\d)', r'"\1": \2', fixed)
    fixed = re.sub(r'"([^"]+)"\s+(true|false|null)', r'"\1": \2', fixed)
    
    # Fix trailing commas
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    # Fix multiple commas
    fixed = re.sub(r',,+', ',', fixed)
    
    # Fix escaped quotes that shouldn't be escaped
    fixed = re.sub(r'\\"', '"', fixed)
    
    # Ensure proper quotes around string values
    # This is tricky - we need to identify unquoted string values
    # Look for patterns like: "key": value where value is not a number, boolean, null, or already quoted
    fixed = re.sub(r':\s*([^",\[\{\d\-][\w\s]*[^",\]\}])\s*([,\]\}])', r': "\1"\2', fixed)
    
    return fixed

def parse_json_safely(json_str: str, page_num: int = None) -> dict:
    """
    Try to parse JSON with multiple fallback strategies
    """
    if not json_str:
        logger.warning(f"Empty JSON string{f' for page {page_num}' if page_num else ''}")
        return {}
    
    # First attempt: direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed{f' for page {page_num}' if page_num else ''}: {e}")
        logger.debug(f"Error position: {e.pos}, looking around: {json_str[max(0, e.pos-50):min(len(json_str), e.pos+50)]}")
    
    # Second attempt: basic fixes
    try:
        fixed = fix_json_string(json_str)
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.warning(f"Fixed JSON parse failed{f' for page {page_num}' if page_num else ''}: {e}")
    
    # Third attempt: handle truncated JSON by trying to close it properly
    try:
        # Check if JSON seems truncated (doesn't end with })
        trimmed = json_str.strip()
        if trimmed and not trimmed.endswith('}'):
            logger.warning("JSON appears truncated, attempting to close structure")
            # Count open braces
            open_braces = trimmed.count('{') - trimmed.count('}')
            open_brackets = trimmed.count('[') - trimmed.count(']')
            
            # Try to close the JSON properly
            if open_brackets > 0:
                trimmed += ']' * open_brackets
            if open_braces > 0:
                trimmed += '}' * open_braces
            
            try:
                return json.loads(trimmed)
            except:
                pass
    except:
        pass
    
    # Fourth attempt: more aggressive fixes
    try:
        # Try to extract just the core data structure
        # Remove everything before first { and after last }
        core = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
        if core:
            return json.loads(core.group())
    except:
        pass
    
    # Fifth attempt: handle nested JSON with better regex
    try:
        # More sophisticated regex to handle nested structures
        import regex  # If available, handles recursive patterns better
        pattern = r'\{(?:[^{}]|(?R))*\}'
        matches = regex.findall(pattern, json_str)
        if matches:
            for match in matches:
                try:
                    result = json.loads(match)
                    if result and len(result) > 0:
                        return result
                except:
                    continue
    except:
        # Fallback if regex module not available
        pass
    
    # Sixth attempt: try to recover partial JSON
    try:
        # Split by common JSON delimiters and try to reconstruct
        lines = json_str.split('\n')
        reconstructed = {}
        current_key = None
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Try to extract key-value pairs
                match = re.match(r'"?(\w+)"?\s*:\s*(.+?)(?:,\s*)?$', line)
                if match:
                    key, value = match.groups()
                    try:
                        # Try to parse the value
                        if value.startswith('"') and value.endswith('"'):
                            reconstructed[key] = value[1:-1]
                        elif value in ['true', 'false', 'null']:
                            reconstructed[key] = json.loads(value)
                        elif value.replace('.', '').replace('-', '').isdigit():
                            reconstructed[key] = json.loads(value)
                    except:
                        reconstructed[key] = value
        
        if reconstructed:
            logger.info(f"Partially reconstructed {len(reconstructed)} fields")
            return reconstructed
    except:
        pass
    
    # Seventh attempt: Use AST literal eval for Python-like structures
    try:
        import ast
        # Try to convert Python-like dict to JSON
        python_dict = ast.literal_eval(json_str)
        if isinstance(python_dict, dict):
            return python_dict
    except:
        pass
    
    # Last resort: return empty structure with error info
    logger.error(f"All JSON parsing attempts failed{f' for page {page_num}' if page_num else ''}")
    return {"_parse_error": "Unable to parse JSON", "_raw_length": len(json_str)}


def advanced_json_reconstruction(text: str) -> Optional[Dict[str, Any]]:
    """
    Advanced JSON reconstruction using pattern matching and heuristics
    """
    try:
        # Pattern to match key-value pairs in various formats
        patterns = [
            # Standard JSON: "key": "value" or "key": value
            r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+\.?\d*|true|false|null|\[.*?\]|\{.*?\})',
            # Unquoted keys: key: "value" or key: value
            r'(\w+)\s*:\s*("(?:[^"\\]|\\.)*"|\d+\.?\d*|true|false|null|\[.*?\]|\{.*?\})',
            # Single quoted: 'key': 'value'
            r"'([^']+)'\s*:\s*('(?:[^'\\]|\\.)*'|\d+\.?\d*|true|false|null|\[.*?\]|\{.*?\})",
        ]
        
        result = {}
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                key = match.group(1)
                value = match.group(2)
                
                # Clean up the key
                key = key.strip().strip('"\'')
                
                # Parse the value
                try:
                    # Handle single quotes by converting to double quotes
                    if value.startswith("'") and value.endswith("'"):
                        value = '"' + value[1:-1].replace('"', '\\"') + '"'
                    
                    # Try to parse as JSON
                    parsed_value = json.loads(value)
                    result[key] = parsed_value
                except:
                    # If parsing fails, store as string
                    result[key] = value.strip('"\'')
        
        return result if result else None
        
    except Exception as e:
        logger.error(f"Advanced reconstruction failed: {e}")
        return None


def extract_json_from_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM responses that may contain explanations or markdown
    """
    # Try multiple extraction strategies
    strategies = [
        # 1. Look for ```json blocks
        lambda t: re.search(r'```json\s*\n([\s\S]*?)\n```', t, re.IGNORECASE),
        # 2. Look for ``` blocks
        lambda t: re.search(r'```\s*\n([\s\S]*?)\n```', t),
        # 3. Look for JSON between curly braces
        lambda t: re.search(r'(\{[\s\S]*\})', t),
        # 4. Look for JSON starting after common phrases
        lambda t: re.search(r'(?:here is|the json|json:|output:)\s*(\{[\s\S]*\})', t, re.IGNORECASE),
    ]
    
    for strategy in strategies:
        match = strategy(response)
        if match:
            try:
                json_str = match.group(1)
                return parse_json_safely(json_str)
            except:
                continue
    
    # If all strategies fail, try the full response
    return parse_json_safely(response)