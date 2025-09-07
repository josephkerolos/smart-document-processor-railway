"""
Smart Document Processor Client Library
A simple Python client for integrating with the Smart Document Processor API
"""

from typing import List, Dict, Any, Optional, Callable
import requests
import time
import json
import websocket
import threading
from pathlib import Path

class SmartDocumentClient:
    """Client for Smart Document Processor API"""
    
    def __init__(self, base_url: str = "http://localhost:4830", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self._ws_connections = {}
        
    # Configuration Methods
    def get_config(self) -> Dict[str, Any]:
        """Get current Google Drive configuration"""
        response = requests.get(f"{self.base_url}/api/gdrive/config", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def set_config(self, **kwargs) -> Dict[str, Any]:
        """Update Google Drive configuration
        
        Args:
            output_folder_path: Base folder for outputs
            company_folder_template: Template for company folders
            form_folder_template: Template for form type folders
            date_folder_template: Template for date folders
            batch_folder_template: Template for batch folders
            create_year_folders: Create year subfolders
            create_quarter_folders: Create quarter subfolders
            group_by_company: Group by company name
            group_by_form_type: Group by form type
            track_processing_status: Enable status tracking
        """
        response = requests.post(f"{self.base_url}/api/gdrive/config", 
                               json=kwargs, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # Document Processing Methods
    def process_document(self, 
                        file_path: str, 
                        form_type: str = "941-X",
                        expected_value: Optional[float] = None,
                        target_size_mb: Optional[float] = None,
                        wait_for_completion: bool = True,
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process a single document
        
        Args:
            file_path: Path to PDF file
            form_type: Document type (941-X, 941, 1040, etc.)
            expected_value: Expected value for validation
            target_size_mb: Target compression size
            wait_for_completion: Wait for processing to complete
            progress_callback: Function called with progress updates
            
        Returns:
            Processing result with session_id and status
        """
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            data = {'selected_schema': form_type}
            
            if expected_value:
                data['expected_value'] = str(expected_value)
            if target_size_mb:
                data['target_size_mb'] = str(target_size_mb)
            
            response = requests.post(f"{self.base_url}/api/process-enhanced", 
                                   files=files, data=data, headers=self.headers)
            response.raise_for_status()
            result = response.json()
        
        if wait_for_completion:
            return self.wait_for_completion(result['session_id'], progress_callback)
        
        return result
    
    def process_batch(self,
                     file_paths: List[str],
                     form_type: str = "941-X",
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process multiple documents as a batch
        
        Args:
            file_paths: List of PDF file paths
            form_type: Document type
            progress_callback: Function called with progress updates
            
        Returns:
            Batch processing result with combined outputs
        """
        # Initialize batch
        batch_response = requests.post(f"{self.base_url}/api/init-batch", 
                                     json={"file_count": len(file_paths)},
                                     headers=self.headers)
        batch_response.raise_for_status()
        batch_id = batch_response.json()["batch_id"]
        
        # Process each file
        session_ids = []
        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
            
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                data = {
                    'selected_schema': form_type,
                    'batch_id': batch_id
                }
                
                response = requests.post(f"{self.base_url}/api/process-enhanced", 
                                       files=files, data=data, headers=self.headers)
                response.raise_for_status()
                session_ids.append(response.json()["session_id"])
        
        # Wait for all to complete
        if progress_callback:
            progress_callback("Waiting for all files to complete processing...")
            
        for session_id in session_ids:
            self.wait_for_completion(session_id)
        
        # Finalize batch
        if progress_callback:
            progress_callback("Finalizing batch and creating combined outputs...")
            
        finalize_response = requests.post(f"{self.base_url}/api/finalize-batch",
                                        json={"batch_id": batch_id},
                                        headers=self.headers)
        finalize_response.raise_for_status()
        return finalize_response.json()
    
    # Status and Query Methods
    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get processing status for a session"""
        response = requests.get(f"{self.base_url}/api/gdrive/status/{session_id}",
                              headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def query_documents(self, **kwargs) -> Dict[str, Any]:
        """Query processed documents
        
        Args:
            company_name: Filter by company name
            form_type: Filter by form type
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            status: Filter by status
            limit: Maximum results
            
        Returns:
            Query results with matching documents
        """
        response = requests.get(f"{self.base_url}/api/gdrive/query",
                              params=kwargs, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # File Operations
    def upload_to_gdrive(self, 
                        local_file: str,
                        gdrive_path: str,
                        custom_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload file to specific Google Drive path"""
        payload = {
            "file_path": local_file,
            "gdrive_path": gdrive_path
        }
        if custom_name:
            payload["custom_name"] = custom_name
            
        response = requests.post(f"{self.base_url}/api/gdrive/upload-to-path",
                               json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def download_archive(self, session_id: str, archive_name: str, save_path: str):
        """Download an archive file"""
        response = requests.get(
            f"{self.base_url}/api/download-archive/{session_id}/{archive_name}",
            headers=self.headers,
            stream=True
        )
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Helper Methods
    def wait_for_completion(self, 
                           session_id: str, 
                           progress_callback: Optional[Callable] = None,
                           timeout: int = 300) -> Dict[str, Any]:
        """Wait for processing to complete with optional progress updates"""
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            status = self.get_status(session_id)
            current_status = status["status"]["status"]
            current_progress = status["status"].get("progress", 0)
            current_step = status["status"].get("current_step", "initializing")
            
            if current_status == "completed":
                if progress_callback:
                    progress_callback(f"Completed! Files saved to: {status['status'].get('gdrive_folder_path', 'local')}")
                return status
            elif current_status == "failed":
                error = status["status"].get("error", "Unknown error")
                raise Exception(f"Processing failed: {error}")
            
            # Update progress if changed
            if progress_callback and current_progress != last_progress:
                progress_callback(f"Progress: {current_progress}% - {current_step}")
                last_progress = current_progress
            
            time.sleep(2)
        
        raise TimeoutError(f"Processing timeout after {timeout} seconds")
    
    def subscribe_to_updates(self, session_id: str, callback: Callable):
        """Subscribe to real-time WebSocket updates for a session"""
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/enhanced/{session_id}"
        
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
        
        def on_error(ws, error):
            callback({"type": "error", "error": str(error)})
        
        def on_close(ws, close_status_code, close_msg):
            callback({"type": "closed", "code": close_status_code, "message": close_msg})
        
        ws = websocket.WebSocketApp(ws_url,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
        
        # Run in separate thread
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()
        
        self._ws_connections[session_id] = ws
        return ws
    
    def close_updates(self, session_id: str):
        """Close WebSocket connection for a session"""
        if session_id in self._ws_connections:
            self._ws_connections[session_id].close()
            del self._ws_connections[session_id]


# Convenience functions for quick usage
def process_document(file_path: str, **kwargs) -> Dict[str, Any]:
    """Quick function to process a single document"""
    client = SmartDocumentClient()
    return client.process_document(file_path, **kwargs)


def process_batch(file_paths: List[str], **kwargs) -> Dict[str, Any]:
    """Quick function to process multiple documents"""
    client = SmartDocumentClient()
    return client.process_batch(file_paths, **kwargs)


if __name__ == "__main__":
    # Example usage
    client = SmartDocumentClient()
    
    # Configure
    client.set_config(
        output_folder_path="ExampleDocs/2024",
        create_quarter_folders=True
    )
    
    # Process single file
    print("Processing single document...")
    result = client.process_document(
        "example.pdf",
        form_type="941-X",
        progress_callback=print
    )
    print(f"Completed! Google Drive: {result['status']['gdrive_folder_path']}")
    
    # Process batch
    print("\nProcessing batch...")
    batch_result = client.process_batch(
        ["q1.pdf", "q2.pdf", "q3.pdf", "q4.pdf"],
        form_type="941-X",
        progress_callback=print
    )
    print(f"Batch completed! Link: {batch_result['google_drive_upload']['folder_link']}")