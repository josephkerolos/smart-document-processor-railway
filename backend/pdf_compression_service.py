"""
PDF Compression Service
Embedded compression functionality that doesn't rely on external File Organizer API
"""

import io
import logging
from typing import Tuple
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import pdf2image
import tempfile
import os

logger = logging.getLogger(__name__)

class PDFCompressionService:
    """Standalone PDF compression service using PyPDF2 and Pillow"""
    
    def __init__(self):
        self.supported_image_formats = ['PNG', 'JPEG', 'JPG']
        
    async def compress_pdf(self, pdf_content: bytes, target_size_mb: float = 2.0) -> bytes:
        """
        Compress PDF to target size using image compression and optimization
        """
        try:
            original_size_mb = len(pdf_content) / (1024 * 1024)
            logger.info(f"Starting PDF compression - Original size: {original_size_mb:.2f}MB, Target: {target_size_mb}MB")
            
            # If already under target size, return original
            if original_size_mb <= target_size_mb:
                logger.info(f"PDF already under target size ({original_size_mb:.2f}MB <= {target_size_mb}MB)")
                return pdf_content
            
            # Calculate compression ratio needed
            compression_ratio = target_size_mb / original_size_mb
            
            # Try different compression strategies
            compressed_content = None
            
            # Strategy 1: Compress images in PDF
            compressed_content = await self._compress_pdf_images(pdf_content, compression_ratio)
            
            compressed_size_mb = len(compressed_content) / (1024 * 1024)
            
            # If still too large, try more aggressive compression
            if compressed_size_mb > target_size_mb:
                logger.info(f"First compression attempt: {compressed_size_mb:.2f}MB, trying more aggressive compression")
                compressed_content = await self._compress_pdf_aggressive(pdf_content, target_size_mb)
                compressed_size_mb = len(compressed_content) / (1024 * 1024)
            
            logger.info(f"Compression complete - Final size: {compressed_size_mb:.2f}MB")
            return compressed_content
            
        except Exception as e:
            logger.error(f"Error compressing PDF: {e}")
            # Return original if compression fails
            return pdf_content
    
    async def _compress_pdf_images(self, pdf_content: bytes, compression_ratio: float) -> bytes:
        """
        Compress PDF by reducing image quality
        """
        try:
            # Read the PDF
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            pdf_writer = PdfWriter()
            
            # Calculate quality based on compression ratio
            # Higher ratio means more compression needed
            quality = max(20, min(95, int(100 * compression_ratio)))
            
            logger.info(f"Compressing PDF images with quality: {quality}")
            
            # Process each page
            for page_num, page in enumerate(pdf_reader.pages):
                # Compress images in page if present
                if '/Resources' in page and '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj['/Subtype'] == '/Image':
                            # Try to compress the image
                            try:
                                # Extract image data
                                size = (obj['/Width'], obj['/Height'])
                                data = obj.get_data()
                                
                                # Convert to PIL Image and compress
                                img = Image.frombytes('RGB', size, data)
                                
                                # Compress image
                                output = io.BytesIO()
                                img.save(output, format='JPEG', quality=quality, optimize=True)
                                compressed_data = output.getvalue()
                                
                                # Update the image object with compressed data
                                obj._data = compressed_data
                                
                            except Exception as e:
                                logger.debug(f"Could not compress image on page {page_num}: {e}")
                
                # Add page to writer
                pdf_writer.add_page(page)
            
            # Add metadata
            pdf_writer.add_metadata(pdf_reader.metadata)
            
            # Write compressed PDF to bytes
            output_buffer = io.BytesIO()
            pdf_writer.write(output_buffer)
            output_buffer.seek(0)
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error in image compression: {e}")
            return pdf_content
    
    async def _compress_pdf_aggressive(self, pdf_content: bytes, target_size_mb: float) -> bytes:
        """
        More aggressive compression using pdf2image conversion
        """
        try:
            logger.info("Attempting aggressive PDF compression via image conversion")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save PDF temporarily
                temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_pdf_path, 'wb') as f:
                    f.write(pdf_content)
                
                # Convert PDF pages to images
                try:
                    images = pdf2image.convert_from_path(
                        temp_pdf_path,
                        dpi=150,  # Reduce DPI for smaller size
                        fmt='JPEG',
                        thread_count=2
                    )
                except Exception as e:
                    logger.warning(f"pdf2image conversion failed: {e}, falling back to original")
                    return pdf_content
                
                # Calculate quality based on target size
                # Estimate: each page ~0.1-0.3 MB at quality 85
                num_pages = len(images)
                estimated_page_size = target_size_mb / num_pages if num_pages > 0 else 0.2
                
                # Adjust quality based on estimated page size
                if estimated_page_size < 0.05:
                    quality = 30
                elif estimated_page_size < 0.1:
                    quality = 50
                elif estimated_page_size < 0.2:
                    quality = 70
                else:
                    quality = 85
                
                logger.info(f"Converting {num_pages} pages with quality {quality}")
                
                # Convert images back to PDF
                pdf_writer = PdfWriter()
                
                for i, image in enumerate(images):
                    # Convert image to PDF page
                    img_buffer = io.BytesIO()
                    
                    # Resize if image is very large
                    width, height = image.size
                    max_dimension = 2000
                    if width > max_dimension or height > max_dimension:
                        ratio = min(max_dimension/width, max_dimension/height)
                        new_size = (int(width * ratio), int(height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Save as JPEG with compression
                    image.save(img_buffer, format='JPEG', quality=quality, optimize=True)
                    img_buffer.seek(0)
                    
                    # Create a new PDF page from the image
                    from PyPDF2 import PageObject
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import letter
                    
                    # Create temporary PDF for this image
                    temp_img_pdf = os.path.join(temp_dir, f"page_{i}.pdf")
                    img_buffer.seek(0)
                    
                    # Save image as PDF page
                    image.save(temp_img_pdf, "PDF", quality=quality, optimize=True)
                    
                    # Read and add to writer
                    temp_reader = PdfReader(temp_img_pdf)
                    if len(temp_reader.pages) > 0:
                        pdf_writer.add_page(temp_reader.pages[0])
                
                # Write final compressed PDF
                output_buffer = io.BytesIO()
                pdf_writer.write(output_buffer)
                output_buffer.seek(0)
                
                compressed_content = output_buffer.getvalue()
                compressed_size_mb = len(compressed_content) / (1024 * 1024)
                
                logger.info(f"Aggressive compression complete: {compressed_size_mb:.2f}MB")
                
                return compressed_content
                
        except Exception as e:
            logger.error(f"Error in aggressive compression: {e}")
            return pdf_content
    
    async def clean_pdf(self, pdf_content: bytes) -> Tuple[bytes, dict]:
        """
        Clean PDF by removing blank pages and optimizing
        """
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_content))
            pdf_writer = PdfWriter()
            
            pages_removed = 0
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                # Check if page has content (simple check)
                text = page.extract_text().strip()
                
                # If page has text or images, keep it
                if text or self._page_has_images(page):
                    pdf_writer.add_page(page)
                else:
                    pages_removed += 1
                    logger.info(f"Removed blank page {page_num + 1}")
            
            # Add metadata
            if pdf_reader.metadata:
                pdf_writer.add_metadata(pdf_reader.metadata)
            
            # Write cleaned PDF
            output_buffer = io.BytesIO()
            pdf_writer.write(output_buffer)
            output_buffer.seek(0)
            
            cleaned_content = output_buffer.getvalue()
            
            return cleaned_content, {
                "pages_removed": pages_removed,
                "original_pages": total_pages,
                "final_pages": total_pages - pages_removed
            }
            
        except Exception as e:
            logger.error(f"Error cleaning PDF: {e}")
            return pdf_content, {"error": str(e)}
    
    def _page_has_images(self, page) -> bool:
        """Check if a PDF page contains images"""
        try:
            if '/Resources' in page and '/XObject' in page['/Resources']:
                xobjects = page['/Resources']['/XObject'].get_object()
                for obj_name in xobjects:
                    obj = xobjects[obj_name]
                    if obj['/Subtype'] == '/Image':
                        return True
        except:
            pass
        return False

# Create singleton instance
pdf_compression_service = PDFCompressionService()