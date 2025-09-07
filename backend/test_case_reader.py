"""
Test script for the Case Reader utility
Demonstrates how to read and access case data
"""

import asyncio
import json
from case_reader import CaseReader, read_all_cases, read_all_cases_sync, get_case_by_id
from datetime import datetime, timedelta


async def test_case_reader():
    """Test the case reader functionality"""
    
    print("=== Testing Case Reader ===\n")
    
    # Create a case reader instance
    reader = CaseReader()
    
    # 1. Load all cases
    print("1. Loading all cases...")
    cases = await reader.load_all_cases()
    print(f"   Loaded {len(cases)} cases total\n")
    
    # 2. Get case summary
    print("2. Getting case summary...")
    summary = reader.get_case_summary()
    print(f"   Total cases: {summary['total_cases']}")
    print(f"   Status breakdown: {summary['status_breakdown']}")
    print(f"   Form type breakdown: {summary['form_type_breakdown']}")
    print(f"   Company breakdown: {summary['company_breakdown']}\n")
    
    # 3. List first 5 cases
    print("3. First 5 cases:")
    for i, (session_id, case_data) in enumerate(list(cases.items())[:5]):
        metadata = case_data.get("metadata", {})
        print(f"   {i+1}. Session: {session_id}")
        print(f"      Status: {case_data.get('status', 'unknown')}")
        print(f"      Company: {metadata.get('company_name', 'N/A')}")
        print(f"      Form Type: {metadata.get('form_type', 'N/A')}")
        print(f"      Created: {case_data.get('created_at', 'N/A')}")
        print(f"      Files Uploaded: {len(case_data.get('files_uploaded', []))}")
        print()
    
    # 4. Filter cases by status
    print("4. Filtering cases by status 'completed':")
    completed_cases = reader.filter_cases(status="completed")
    print(f"   Found {len(completed_cases)} completed cases\n")
    
    # 5. Filter cases by date (last 7 days)
    print("5. Filtering cases from last 7 days:")
    week_ago = datetime.now() - timedelta(days=7)
    recent_cases = reader.filter_cases(date_from=week_ago)
    print(f"   Found {len(recent_cases)} cases from last 7 days\n")
    
    # 6. Get a specific case (if any exist)
    if cases:
        print("6. Getting a specific case:")
        first_id = list(cases.keys())[0]
        specific_case = reader.get_case(first_id)
        print(f"   Case ID: {first_id}")
        print(f"   Details: {json.dumps(specific_case, indent=4)[:500]}...\n")
    
    # 7. Test convenience functions
    print("7. Testing convenience functions:")
    
    # Test read_all_cases
    all_cases = await read_all_cases()
    print(f"   read_all_cases() returned {len(all_cases)} cases")
    
    # Test get_case_by_id
    if cases:
        first_id = list(cases.keys())[0]
        case = await get_case_by_id(first_id)
        print(f"   get_case_by_id('{first_id}') returned: {case is not None}")
    
    print("\n=== Test Complete ===")


def test_sync_reader():
    """Test synchronous case reading"""
    print("\n=== Testing Synchronous Case Reader ===\n")
    
    # Test synchronous reading (only from files)
    cases = read_all_cases_sync()
    print(f"Synchronous read found {len(cases)} cases from files")
    
    if cases:
        print("\nFirst case from sync read:")
        first_id = list(cases.keys())[0]
        print(f"  ID: {first_id}")
        print(f"  Status: {cases[first_id].get('status', 'unknown')}")


if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_case_reader())
    
    # Run sync tests
    test_sync_reader()