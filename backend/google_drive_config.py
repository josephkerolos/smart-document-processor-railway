"""
Google Drive Configuration Management
Handles custom routes, naming templates, and folder structures
"""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
from dataclasses import dataclass, asdict
from enum import Enum

class NamingTemplate(Enum):
    """Predefined naming templates for folder generation"""
    DEFAULT = "{company}_{form_type}_{date}"
    QUARTERLY = "{company}_{form_type}_{year}_Q{quarter}"
    BATCH = "Batch_{date}_{time}_{quarters}"
    CUSTOM = "{custom}"

@dataclass
class GoogleDriveConfig:
    """Configuration for Google Drive integration"""
    # Base paths
    root_folder_id: Optional[str] = None  # Google Drive folder ID for root
    input_folder_path: str = "InputDocuments"  # Where to look for documents to process
    output_folder_path: str = "ProcessedDocuments"  # Where to save processed documents
    
    # Naming configuration
    company_folder_template: str = "{company}"
    form_folder_template: str = "{form_type}"
    date_folder_template: str = "{date}"
    batch_folder_template: str = "Batch_{date}_{time}_{quarters}"
    
    # Auto-organization rules
    auto_organize: bool = True
    create_year_folders: bool = True
    create_quarter_folders: bool = True
    group_by_company: bool = True
    group_by_form_type: bool = True
    
    # File naming patterns
    cleaned_file_suffix: str = "_cleaned"
    compressed_file_suffix: str = "_compressed"
    extraction_file_suffix: str = "_extraction"
    
    # Processing options
    keep_original_structure: bool = False
    create_archive_folder: bool = True
    upload_individual_files: bool = True
    upload_batch_archive: bool = True
    
    # Status tracking
    track_processing_status: bool = True
    status_file_name: str = "processing_status.json"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoogleDriveConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def apply_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Apply variables to a naming template"""
        result = template
        for key, value in variables.items():
            if value is not None:
                # Clean value for folder names
                clean_value = str(value).replace('/', '-').replace('\\', '-')
                clean_value = re.sub(r'[<>:"|?*]', '', clean_value)
                result = result.replace(f"{{{key}}}", clean_value)
        return result

class GoogleDriveConfigManager:
    """Manages Google Drive configurations"""
    
    def __init__(self, config_file: str = "gdrive_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> GoogleDriveConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return GoogleDriveConfig.from_dict(data)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        # Return default config
        return GoogleDriveConfig()
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def update_config(self, updates: Dict[str, Any]) -> GoogleDriveConfig:
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
        return self.config
    
    def generate_folder_path(self, doc_info: Dict[str, Any]) -> str:
        """Generate folder path based on configuration and document info"""
        path_parts = []
        
        # Add output folder
        if self.config.output_folder_path:
            path_parts.append(self.config.output_folder_path)
        
        # Add company folder if grouping by company
        if self.config.group_by_company and doc_info.get('company_name'):
            company_folder = self.config.apply_template(
                self.config.company_folder_template,
                {'company': doc_info['company_name']}
            )
            path_parts.append(company_folder)
        
        # Add form type folder if grouping by form type
        if self.config.group_by_form_type and doc_info.get('form_type'):
            form_folder = self.config.apply_template(
                self.config.form_folder_template,
                {'form_type': doc_info['form_type']}
            )
            path_parts.append(form_folder)
        
        # Add year folder if enabled
        if self.config.create_year_folders and doc_info.get('year'):
            path_parts.append(str(doc_info['year']))
        
        # Add quarter folder if enabled
        if self.config.create_quarter_folders and doc_info.get('quarter'):
            path_parts.append(f"Q{doc_info['quarter']}")
        
        # Add date folder
        date_str = doc_info.get('processing_date', datetime.now().strftime('%Y-%m-%d'))
        date_folder = self.config.apply_template(
            self.config.date_folder_template,
            {'date': date_str}
        )
        path_parts.append(date_folder)
        
        return '/'.join(path_parts)
    
    def generate_batch_folder_name(self, batch_info: Dict[str, Any]) -> str:
        """Generate batch folder name based on template"""
        variables = {
            'date': batch_info.get('date', datetime.now().strftime('%Y-%m-%d')),
            'time': batch_info.get('time', datetime.now().strftime('%H-%M-%S')),
            'quarters': batch_info.get('quarters', ''),
            'company': batch_info.get('company_name', ''),
            'form_type': batch_info.get('form_type', ''),
            'batch_id': batch_info.get('batch_id', '')[:8]
        }
        
        return self.config.apply_template(
            self.config.batch_folder_template,
            variables
        )
    
    def get_status_file_path(self, folder_path: str) -> str:
        """Get the full path for the status file"""
        return os.path.join(folder_path, self.config.status_file_name)

# Global instance
config_manager = GoogleDriveConfigManager()