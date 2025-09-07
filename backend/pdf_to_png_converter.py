"""
PDF to PNG Converter Module
Provides fallback conversion functionality for failed PDF extractions
"""

import io
import os
import logging
import tempfile
from typing import List, Tuple, Optional, Dict, Any
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_bytes = None
from PIL import Image
import requests
import subprocess
import shutil

logger = logging.getLogger(__name__)

class PDFToPNGConverter:
    """Handles PDF to PNG conversion for fallback extraction"""
    
    def __init__(self, use_external_service: bool = False, external_service_url: str = None):
        """
        Initialize converter
        
        Args:
            use_external_service: Whether to use the Node.js service if available
            external_service_url: URL of the external PDF-to-PNG service
        """
        self.use_external_service = use_external_service
        self.external_service_url = external_service_url or "http://localhost:2342"
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        # Check if poppler-utils is installed (required for pdf2image)
        # Only log at debug level to reduce noise
        try:
            # Use stdout and stderr separately instead of capture_output
            result = subprocess.run(['pdftoppm', '-v'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if result.returncode != 0:
                logger.debug("poppler-utils not found. PDF to PNG conversion may use external service.")
        except FileNotFoundError:
            logger.debug("poppler-utils not found. PDF to PNG conversion will use external service if available.")
            if not PDF2IMAGE_AVAILABLE:
                logger.debug("pdf2image module not available. Will rely on external service only.")
    
    def _check_external_service(self) -> bool:
        """Check if external Node.js service is available"""
        if not self.use_external_service:
            return False
            
        try:
            response = requests.get(f"{self.external_service_url}/", timeout=2)
            return response.status_code != 404
        except:
            return False
    
    async def convert_pdf_page_to_png(self, page_content: bytes, page_num: int, 
                                     total_pages: int, quality: str = "high") -> Optional[bytes]:
        """
        Convert a single PDF page to PNG
        
        Args:
            page_content: PDF page content as bytes
            page_num: Page number (0-indexed)
            total_pages: Total number of pages in the document
            quality: Quality setting (standard, high, ultra)
            
        Returns:
            PNG image as bytes or None if conversion fails
        """
        try:
            # First try using external service if available
            if self._check_external_service():
                logger.info(f"Using external service for page {page_num + 1} conversion")
                return await self._convert_using_external_service(
                    page_content, page_num, total_pages, quality
                )
            
            # Fall back to local conversion
            logger.info(f"Using local conversion for page {page_num + 1}")
            return self._convert_using_pdf2image(page_content, page_num, quality)
            
        except Exception as e:
            logger.error(f"Failed to convert page {page_num + 1} to PNG: {e}")
            return None
    
    async def _convert_using_external_service(self, page_content: bytes, 
                                            page_num: int, total_pages: int, 
                                            quality: str) -> Optional[bytes]:
        """Convert using the Node.js PDF-to-PNG service"""
        try:
            # Create temporary file for the page
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(page_content)
                tmp_path = tmp_file.name
            
            # Prepare the request
            files = {
                'pdf': (f'page_{page_num + 1}.pdf', open(tmp_path, 'rb'), 'application/pdf')
            }
            data = {
                'quality': quality
            }
            
            # Send request to external service
            response = requests.post(
                f"{self.external_service_url}/convert",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and result.get('individual'):
                    # Download the PNG
                    png_url = f"{self.external_service_url}{result['individual'][0]['downloadUrl']}"
                    png_response = requests.get(png_url)
                    if png_response.status_code == 200:
                        return png_response.content
            
            logger.error(f"External service failed with status {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"External service error: {e}")
            return None
        finally:
            # Clean up temp file
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    def _convert_using_pdf2image(self, page_content: bytes, page_num: int, 
                                quality: str) -> Optional[bytes]:
        """Convert using local pdf2image library"""
        if not PDF2IMAGE_AVAILABLE or not convert_from_bytes:
            logger.error("pdf2image not available for local conversion")
            return None
            
        try:
            # Set DPI based on quality
            dpi_settings = {
                'standard': 150,
                'high': 300,
                'ultra': 600
            }
            dpi = dpi_settings.get(quality, 300)
            
            # Convert PDF page to PIL Image
            images = convert_from_bytes(
                page_content,
                dpi=dpi,
                fmt='png',
                thread_count=1,
                use_pdftocairo=True  # Better quality
            )
            
            if images:
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                images[0].save(img_buffer, format='PNG', optimize=True, quality=95)
                img_buffer.seek(0)
                return img_buffer.getvalue()
            
            return None
            
        except Exception as e:
            logger.error(f"pdf2image conversion error: {e}")
            return None
    
    async def convert_full_pdf_to_pngs(self, pdf_content: bytes, 
                                      quality: str = "high") -> List[Tuple[int, bytes]]:
        """
        Convert entire PDF to PNG images
        
        Args:
            pdf_content: Full PDF content as bytes
            quality: Quality setting
            
        Returns:
            List of tuples (page_number, png_bytes)
        """
        try:
            # Try external service first for full PDF
            if self._check_external_service():
                return await self._convert_full_pdf_external(pdf_content, quality)
            
            # Fall back to local conversion
            return self._convert_full_pdf_local(pdf_content, quality)
            
        except Exception as e:
            logger.error(f"Failed to convert full PDF: {e}")
            return []
    
    def _convert_full_pdf_local(self, pdf_content: bytes, quality: str) -> List[Tuple[int, bytes]]:
        """Convert full PDF locally"""
        if not PDF2IMAGE_AVAILABLE or not convert_from_bytes:
            logger.error("pdf2image not available for local conversion")
            return []
            
        try:
            dpi_settings = {
                'standard': 150,
                'high': 300,
                'ultra': 600
            }
            dpi = dpi_settings.get(quality, 300)
            
            # Convert all pages
            images = convert_from_bytes(
                pdf_content,
                dpi=dpi,
                fmt='png',
                thread_count=4,
                use_pdftocairo=True
            )
            
            result = []
            for idx, image in enumerate(images):
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG', optimize=True, quality=95)
                img_buffer.seek(0)
                result.append((idx, img_buffer.getvalue()))
            
            return result
            
        except Exception as e:
            logger.error(f"Local full PDF conversion error: {e}")
            return []
    
    async def _convert_full_pdf_external(self, pdf_content: bytes, 
                                       quality: str) -> List[Tuple[int, bytes]]:
        """Convert full PDF using external service"""
        # This would be similar to _convert_using_external_service but for full PDFs
        # For now, we'll use page-by-page conversion
        return []