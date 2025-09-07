"""
Quick Integration Example for Smart Document Processor
This example shows how to integrate the API into your existing project
"""

import requests
import time
import json
from typing import List, Dict, Any, Optional

class SmartDocumentClient:
    """Client for interacting with Smart Document Processor API"""
    
    def __init__(self, base_url: str = "http://localhost:4830"):
        self.base_url = base_url
        self.gdrive_api = f"{base_url}/api/gdrive"
        
    def configure_google_drive(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Google Drive settings"""
        response = requests.post(f"{self.gdrive_api}/config", json=config)
        response.raise_for_status()
        return response.json()
    
    def process_single_document(self, file_path: str, 
                              form_type: str = "941-X",
                              custom_gdrive_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single document"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'selected_schema': form_type
            }
            if custom_gdrive_path:
                data['custom_gdrive_path'] = custom_gdrive_path
            
            response = requests.post(f"{self.base_url}/api/process-enhanced", 
                                   files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def process_batch_documents(self, file_paths: List[str], 
                               form_type: str = "941-X") -> Dict[str, Any]:
        """Process multiple documents as a batch"""
        # Initialize batch
        batch_response = requests.post(f"{self.base_url}/api/init-batch", 
                                     json={"file_count": len(file_paths)})
        batch_response.raise_for_status()
        batch_data = batch_response.json()
        batch_id = batch_data["batch_id"]
        
        # Process each file
        session_ids = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'selected_schema': form_type,
                    'batch_id': batch_id
                }
                
                response = requests.post(f"{self.base_url}/api/process-enhanced", 
                                       files=files, data=data)
                response.raise_for_status()
                result = response.json()
                session_ids.append(result["session_id"])
        
        # Wait for all to complete
        all_complete = False
        while not all_complete:
            time.sleep(2)
            all_complete = all(self.check_status(sid)["status"]["status"] == "completed" 
                             for sid in session_ids)
        
        # Finalize batch
        finalize_response = requests.post(f"{self.base_url}/api/finalize-batch",
                                        json={"batch_id": batch_id})
        finalize_response.raise_for_status()
        return finalize_response.json()
    
    def check_status(self, session_id: str) -> Dict[str, Any]:
        """Check processing status"""
        response = requests.get(f"{self.gdrive_api}/status/{session_id}")
        response.raise_for_status()
        return response.json()
    
    def query_documents(self, company_name: Optional[str] = None,
                       form_type: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None,
                       status: Optional[str] = None,
                       limit: int = 50) -> Dict[str, Any]:
        """Query processed documents"""
        params = {
            k: v for k, v in {
                "company_name": company_name,
                "form_type": form_type,
                "date_from": date_from,
                "date_to": date_to,
                "status": status,
                "limit": limit
            }.items() if v is not None
        }
        
        response = requests.get(f"{self.gdrive_api}/query", params=params)
        response.raise_for_status()
        return response.json()
    
    def upload_to_gdrive_path(self, local_file: str, 
                             gdrive_path: str,
                             custom_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload a file to a specific Google Drive path"""
        payload = {
            "file_path": local_file,
            "gdrive_path": gdrive_path
        }
        if custom_name:
            payload["custom_name"] = custom_name
        
        response = requests.post(f"{self.gdrive_api}/upload-to-path", json=payload)
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, session_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for processing to complete with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_status(session_id)
            
            if status["status"]["status"] == "completed":
                return status
            elif status["status"]["status"] == "failed":
                raise Exception(f"Processing failed: {status['status'].get('error', 'Unknown error')}")
            
            # Print progress
            progress = status["status"].get("progress", 0)
            current_step = status["status"].get("current_step", "initializing")
            print(f"Progress: {progress}% - {current_step}")
            
            time.sleep(2)
        
        raise TimeoutError(f"Processing timeout after {timeout} seconds")


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = SmartDocumentClient()
    
    # Example 1: Configure Google Drive with custom paths
    print("=== Configuring Google Drive ===")
    config = client.configure_google_drive({
        "output_folder_path": "TaxDocuments/2024",
        "company_folder_template": "{company}_TaxDocs",
        "create_year_folders": True,
        "create_quarter_folders": True,
        "batch_folder_template": "Batch_{date}_{company}_{quarters}"
    })
    print(f"Configuration updated: {config['message']}")
    
    # Example 2: Process a single document
    print("\n=== Processing Single Document ===")
    result = client.process_single_document(
        file_path="path/to/941x_q1.pdf",
        form_type="941-X"
    )
    print(f"Session ID: {result['session_id']}")
    
    # Wait for completion
    final_status = client.wait_for_completion(result['session_id'])
    print(f"Completed! Google Drive path: {final_status['status']['gdrive_folder_path']}")
    
    # Example 3: Process multiple documents as batch
    print("\n=== Processing Batch Documents ===")
    files = [
        "path/to/941x_q1.pdf",
        "path/to/941x_q2.pdf",
        "path/to/941x_q3.pdf",
        "path/to/941x_q4.pdf"
    ]
    
    batch_result = client.process_batch_documents(files, form_type="941-X")
    print(f"Batch completed! Total documents: {batch_result['total_documents']}")
    print(f"Google Drive link: {batch_result['google_drive_upload']['folder_link']}")
    print(f"All extractions saved to: {batch_result['all_extractions_path']}")
    
    # Example 4: Query processed documents
    print("\n=== Querying Documents ===")
    query_results = client.query_documents(
        company_name="ABC Corp",
        form_type="941X",
        date_from="2024-01-01",
        status="completed"
    )
    
    print(f"Found {query_results['total']} documents:")
    for doc in query_results['results']:
        print(f"- {doc['company_name']} ({doc['created_at']}): {doc['gdrive_folder_path']}")
    
    # Example 5: Upload additional file to specific path
    print("\n=== Uploading to Specific Path ===")
    upload_result = client.upload_to_gdrive_path(
        local_file="path/to/summary_report.pdf",
        gdrive_path="TaxDocuments/2024/Reports",
        custom_name="Q1_Q4_Summary_Report.pdf"
    )
    print(f"File uploaded to: {upload_result['folder_path']}/{upload_result['file_name']}")