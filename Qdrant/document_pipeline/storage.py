"""
Storage Handler for Document Files
Manages file uploads and persistence
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import mimetypes


class DocumentStorage:
    """
    Handles file storage for uploaded documents
    Organizes files by collection name and assigns UUIDs
    """
    
    def __init__(self, base_upload_dir: str = "./uploads", max_file_size_mb: int = 10):
        """
        Initialize storage handler
        
        Args:
            base_upload_dir: Root directory for uploads
            max_file_size_mb: Maximum file size in MB
        """
        self.base_upload_dir = Path(base_upload_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        self.base_upload_dir.mkdir(parents=True, exist_ok=True)
        print(f"Storage initialized at: {self.base_upload_dir.absolute()}")
    
    def _validate_file(self, file_path: Path, original_filename: str) -> None:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to the file
            original_filename: Original filename for extension check
            
        Raises:
            ValueError: If validation fails
        """
        # Check file exists
        if not file_path.exists():
            raise ValueError("File does not exist")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(f"File size ({file_size / (1024*1024):.2f}MB) exceeds maximum ({max_mb}MB)")
        
        # Check file extension
        if not original_filename.lower().endswith('.docx'):
            raise ValueError("Only .docx files are supported")
        
        # Check file is not empty
        if file_size == 0:
            raise ValueError("File is empty")
    
    def _generate_doc_id(self) -> str:
        """Generate unique document ID"""
        return str(uuid.uuid4())
    
    def save_file(
        self, 
        file_path: str, 
        original_filename: str, 
        collection_name: str = "uncategorized"
    ) -> Dict[str, any]:
        """
        Save uploaded file to storage
        
        Args:
            file_path: Path to temporary uploaded file
            original_filename: Original filename from upload
            collection_name: Collection this document belongs to
            
        Returns:
            Dictionary with document metadata:
            {
                'doc_id': str,
                'stored_path': str,
                'original_filename': str,
                'file_size_bytes': int,
                'collection_name': str,
                'uploaded_at': str (ISO format)
            }
            
        Raises:
            ValueError: If validation fails
        """
        source_path = Path(file_path)
        
        # Validate file
        self._validate_file(source_path, original_filename)
        
        # Generate unique ID
        doc_id = self._generate_doc_id()
        
        # Create collection directory
        collection_dir = self.base_upload_dir / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # Create destination path: uploads/{collection}/{doc_id}.docx
        file_extension = Path(original_filename).suffix
        destination_path = collection_dir / f"{doc_id}{file_extension}"
        
        # Copy file
        shutil.copy2(source_path, destination_path)
        
        # Get file stats
        file_size = destination_path.stat().st_size
        upload_time = datetime.utcnow().isoformat() + "Z"
        
        metadata = {
            'doc_id': doc_id,
            'stored_path': str(destination_path),
            'original_filename': original_filename,
            'file_size_bytes': file_size,
            'collection_name': collection_name,
            'uploaded_at': upload_time
        }
        
        print(f"File saved: {doc_id} ({original_filename})")
        
        return metadata
    
    def get_file_path(self, doc_id: str, collection_name: str) -> Optional[Path]:
        """
        Get path to stored file
        
        Args:
            doc_id: Document ID
            collection_name: Collection name
            
        Returns:
            Path to file if exists, None otherwise
        """
        collection_dir = self.base_upload_dir / collection_name
        
        # Check for .docx extension
        file_path = collection_dir / f"{doc_id}.docx"
        
        if file_path.exists():
            return file_path
        
        return None
    
    def delete_file(self, doc_id: str, collection_name: str) -> bool:
        """
        Delete a stored file
        
        Args:
            doc_id: Document ID
            collection_name: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self.get_file_path(doc_id, collection_name)
        
        if file_path and file_path.exists():
            file_path.unlink()
            print(f"Deleted file: {doc_id}")
            return True
        
        return False
    
    def list_files(self, collection_name: Optional[str] = None) -> list:
        """
        List all stored files
        
        Args:
            collection_name: Optional collection to filter by
            
        Returns:
            List of file info dictionaries
        """
        files = []
        
        if collection_name:
            # List files in specific collection
            collection_dir = self.base_upload_dir / collection_name
            if collection_dir.exists():
                for file_path in collection_dir.glob("*.docx"):
                    files.append(self._get_file_info(file_path, collection_name))
        else:
            # List files in all collections
            for collection_dir in self.base_upload_dir.iterdir():
                if collection_dir.is_dir():
                    for file_path in collection_dir.glob("*.docx"):
                        files.append(self._get_file_info(file_path, collection_dir.name))
        
        return files
    
    def _get_file_info(self, file_path: Path, collection_name: str) -> Dict:
        """Get file information"""
        stat = file_path.stat()
        return {
            'doc_id': file_path.stem,  # Filename without extension
            'filename': file_path.name,
            'collection_name': collection_name,
            'file_size_bytes': stat.st_size,
            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
        }
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage stats
        """
        total_files = 0
        total_size = 0
        collections = {}
        
        for collection_dir in self.base_upload_dir.iterdir():
            if collection_dir.is_dir():
                files = list(collection_dir.glob("*.docx"))
                collection_size = sum(f.stat().st_size for f in files)
                
                collections[collection_dir.name] = {
                    'file_count': len(files),
                    'size_bytes': collection_size
                }
                
                total_files += len(files)
                total_size += collection_size
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'collections': collections
        }