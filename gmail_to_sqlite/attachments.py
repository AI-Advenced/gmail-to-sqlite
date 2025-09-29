"""
Advanced attachment handling system for Gmail to SQLite.

Provides comprehensive attachment extraction, processing, text extraction,
virus scanning, and metadata management capabilities.
"""

import os
import mimetypes
import hashlib
import magic
from typing import Any, Dict, List, Optional, Tuple, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

# Optional imports for text extraction
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import clamd
    CLAMD_AVAILABLE = True
except ImportError:
    CLAMD_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AttachmentMetadata:
    """Metadata for an attachment."""
    filename: str
    size: int
    mime_type: str
    md5_hash: str
    sha256_hash: str
    created_at: datetime
    message_id: str
    content_id: Optional[str] = None
    charset: Optional[str] = None
    is_inline: bool = False
    extracted_text: Optional[str] = None
    virus_scan_result: Optional[str] = None
    thumbnail_path: Optional[str] = None


class AttachmentError(Exception):
    """Raised when attachment operations fail."""
    pass


class VirusDetectedError(AttachmentError):
    """Raised when virus is detected in attachment."""
    pass


class AttachmentProcessor:
    """Advanced attachment processing with text extraction and virus scanning."""
    
    def __init__(self, config: Any):
        self.config = config
        self.download_path = Path(config.attachments.download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize virus scanner if enabled
        self.virus_scanner = None
        if config.attachments.virus_scan_enabled and CLAMD_AVAILABLE:
            try:
                self.virus_scanner = clamd.ClamdUnixSocket()
                # Test connection
                self.virus_scanner.ping()
                logger.info("ClamAV virus scanner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize virus scanner: {e}")
                self.virus_scanner = None
        
        # Initialize magic for MIME type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
            logger.info("libmagic MIME detection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize libmagic: {e}")
            self.magic_mime = None
    
    def extract_attachments(self, message_data: Dict) -> List[AttachmentMetadata]:
        """Extract all attachments from a Gmail message."""
        attachments = []
        
        if not self.config.attachments.enabled:
            return attachments
        
        try:
            payload = message_data.get('payload', {})
            attachments.extend(self._process_payload(payload, message_data['id']))
        except Exception as e:
            logger.error(f"Failed to extract attachments from message {message_data['id']}: {e}")
        
        return attachments
    
    def _process_payload(self, payload: Dict, message_id: str) -> List[AttachmentMetadata]:
        """Process message payload to extract attachments."""
        attachments = []
        
        # Check if this part has an attachment
        if 'body' in payload and 'attachmentId' in payload['body']:
            attachment = self._process_attachment_part(payload, message_id)
            if attachment:
                attachments.append(attachment)
        
        # Recursively process parts
        if 'parts' in payload:
            for part in payload['parts']:
                attachments.extend(self._process_payload(part, message_id))
        
        return attachments
    
    def _process_attachment_part(self, part: Dict, message_id: str) -> Optional[AttachmentMetadata]:
        """Process an individual attachment part."""
        try:
            filename = self._get_attachment_filename(part)
            if not filename:
                return None
            
            # Check file size limits
            size = part.get('body', {}).get('size', 0)
            max_size_bytes = self.config.attachments.max_size_mb * 1024 * 1024
            if size > max_size_bytes:
                logger.warning(f"Attachment {filename} exceeds size limit ({size} > {max_size_bytes})")
                return None
            
            # Check allowed file types
            file_ext = Path(filename).suffix.lower().lstrip('.')
            if file_ext not in self.config.attachments.allowed_types:
                logger.debug(f"Attachment {filename} type not allowed: {file_ext}")
                return None
            
            # Download attachment data
            attachment_id = part['body']['attachmentId']
            attachment_data = self._download_attachment_data(message_id, attachment_id)
            
            if not attachment_data:
                return None
            
            # Generate file paths and hashes
            md5_hash = hashlib.md5(attachment_data).hexdigest()
            sha256_hash = hashlib.sha256(attachment_data).hexdigest()
            
            # Create unique filename to avoid conflicts
            safe_filename = self._sanitize_filename(filename)
            unique_filename = f"{message_id}_{md5_hash[:8]}_{safe_filename}"
            file_path = self.download_path / unique_filename
            
            # Detect MIME type
            mime_type = self._detect_mime_type(attachment_data, filename)
            
            # Virus scan if enabled
            virus_scan_result = self._scan_for_virus(attachment_data)
            if virus_scan_result and virus_scan_result != "OK":
                raise VirusDetectedError(f"Virus detected in {filename}: {virus_scan_result}")
            
            # Save attachment to disk
            with open(file_path, 'wb') as f:
                f.write(attachment_data)
            
            # Extract text if enabled
            extracted_text = None
            if self.config.attachments.extract_text:
                extracted_text = self._extract_text(file_path, mime_type)
            
            # Create metadata
            metadata = AttachmentMetadata(
                filename=filename,
                size=len(attachment_data),
                mime_type=mime_type,
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                created_at=datetime.now(),
                message_id=message_id,
                content_id=part.get('headers', {}).get('Content-ID'),
                is_inline='inline' in part.get('headers', {}).get('Content-Disposition', '').lower(),
                extracted_text=extracted_text,
                virus_scan_result=virus_scan_result
            )
            
            logger.info(f"Processed attachment: {filename} ({len(attachment_data)} bytes)")
            return metadata
            
        except VirusDetectedError:
            raise
        except Exception as e:
            logger.error(f"Failed to process attachment: {e}")
            return None
    
    def _get_attachment_filename(self, part: Dict) -> Optional[str]:
        """Extract filename from attachment part."""
        headers = part.get('headers', [])
        
        # Look for filename in Content-Disposition header
        for header in headers:
            if header['name'].lower() == 'content-disposition':
                content_disp = header['value']
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[1]
                    # Remove quotes and cleanup
                    filename = filename.strip('"\'').split(';')[0]
                    return filename
        
        # Look for name in Content-Type header
        for header in headers:
            if header['name'].lower() == 'content-type':
                content_type = header['value']
                if 'name=' in content_type:
                    filename = content_type.split('name=')[1]
                    filename = filename.strip('"\'').split(';')[0]
                    return filename
        
        # Generate filename from part info
        mime_type = part.get('mimeType', 'application/octet-stream')
        ext = mimetypes.guess_extension(mime_type) or '.bin'
        return f"attachment{ext}"
    
    def _download_attachment_data(self, message_id: str, attachment_id: str) -> Optional[bytes]:
        """Download attachment data from Gmail API."""
        # This would normally use the Gmail API service
        # For now, return None to indicate download needed
        logger.debug(f"Would download attachment {attachment_id} from message {message_id}")
        return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem storage."""
        # Remove or replace dangerous characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Limit length
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:255-len(ext)] + ext
        
        return safe_filename
    
    def _detect_mime_type(self, data: bytes, filename: str) -> str:
        """Detect MIME type of attachment data."""
        # Use libmagic if available
        if self.magic_mime:
            try:
                return self.magic_mime.from_buffer(data)
            except Exception as e:
                logger.debug(f"libmagic MIME detection failed: {e}")
        
        # Fallback to mimetypes module
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    def _scan_for_virus(self, data: bytes) -> Optional[str]:
        """Scan attachment data for viruses using ClamAV."""
        if not self.virus_scanner:
            return None
        
        try:
            result = self.virus_scanner.instream(data)
            status = result['stream'][0]
            if status == 'OK':
                return "OK"
            else:
                logger.warning(f"Virus scan result: {status}")
                return status
        except Exception as e:
            logger.error(f"Virus scan failed: {e}")
            return None
    
    def _extract_text(self, file_path: Path, mime_type: str) -> Optional[str]:
        """Extract text content from attachment based on MIME type."""
        try:
            if mime_type == 'application/pdf' and PDF_AVAILABLE:
                return self._extract_pdf_text(file_path)
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'] and DOCX_AVAILABLE:
                return self._extract_docx_text(file_path)
            elif mime_type.startswith('text/'):
                return self._extract_plain_text(file_path)
            elif mime_type.startswith('image/') and OCR_AVAILABLE:
                return self._extract_image_text(file_path)
            else:
                logger.debug(f"Text extraction not supported for MIME type: {mime_type}")
                return None
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return None
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
        return '\n'.join(text_content)
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        text_content = []
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        return '\n'.join(text_content)
    
    def _extract_plain_text(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        encodings = ['utf-8', 'utf-16', 'ascii', 'latin1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        return ""
    
    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.debug(f"OCR text extraction failed: {e}")
            return ""
    
    def get_attachment_stats(self) -> Dict[str, Any]:
        """Get statistics about processed attachments."""
        attachment_dir = self.download_path
        if not attachment_dir.exists():
            return {"total_files": 0, "total_size": 0}
        
        total_files = 0
        total_size = 0
        file_types = {}
        
        for file_path in attachment_dir.iterdir():
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size
                
                file_ext = file_path.suffix.lower()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "download_path": str(attachment_dir)
        }
    
    def cleanup_old_attachments(self, days_old: int = 90) -> int:
        """Clean up attachments older than specified days."""
        if not self.download_path.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        deleted_count = 0
        
        for file_path in self.download_path.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old attachment: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old attachments")
        return deleted_count


class AttachmentSearcher:
    """Search and filter attachments by various criteria."""
    
    def __init__(self, attachment_processor: AttachmentProcessor):
        self.processor = attachment_processor
    
    def search_by_filename(self, pattern: str, case_sensitive: bool = False) -> List[Path]:
        """Search attachments by filename pattern."""
        results = []
        if not case_sensitive:
            pattern = pattern.lower()
        
        for file_path in self.processor.download_path.iterdir():
            if file_path.is_file():
                filename = file_path.name if case_sensitive else file_path.name.lower()
                if pattern in filename:
                    results.append(file_path)
        
        return results
    
    def search_by_mime_type(self, mime_type: str) -> List[Path]:
        """Search attachments by MIME type."""
        results = []
        
        for file_path in self.processor.download_path.iterdir():
            if file_path.is_file():
                detected_type = self.processor._detect_mime_type(
                    file_path.read_bytes(), file_path.name
                )
                if mime_type in detected_type:
                    results.append(file_path)
        
        return results
    
    def search_by_size_range(self, min_size: int = 0, max_size: int = None) -> List[Path]:
        """Search attachments by size range (in bytes)."""
        results = []
        
        for file_path in self.processor.download_path.iterdir():
            if file_path.is_file():
                size = file_path.stat().st_size
                if size >= min_size and (max_size is None or size <= max_size):
                    results.append(file_path)
        
        return results
    
    def get_duplicate_attachments(self) -> Dict[str, List[Path]]:
        """Find duplicate attachments based on content hash."""
        hash_map = {}
        
        for file_path in self.processor.download_path.iterdir():
            if file_path.is_file():
                try:
                    content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
                    if content_hash not in hash_map:
                        hash_map[content_hash] = []
                    hash_map[content_hash].append(file_path)
                except Exception as e:
                    logger.error(f"Failed to hash {file_path}: {e}")
        
        # Return only duplicates (more than one file with same hash)
        return {h: files for h, files in hash_map.items() if len(files) > 1}