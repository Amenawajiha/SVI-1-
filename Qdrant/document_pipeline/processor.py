"""
Document Processor Module
Handles text extraction, chunking, and embedding generation
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from docx import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from .embedding import E5Embeddings
from .storage import DocumentStorage


class DocumentProcessor:
    """
    Processes documents for RAG pipeline:
    1. Extract text from .docx
    2. Chunk text intelligently
    3. Generate embeddings
    4. Prepare for vector storage
    """
    
    def __init__(
        self,
        embeddings: E5Embeddings,
        storage: DocumentStorage,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            embeddings: E5Embeddings instance
            storage: DocumentStorage instance
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.embeddings = embeddings
        self.storage = storage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Prefer splitting by paragraphs
                "\n",    # Then by lines
                ". ",    # Then by sentences
                " ",     # Then by words
                ""       # Last resort: split anywhere
            ]
        )
        
        print(f"Processor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract all text from a .docx file
        
        Args:
            file_path: Path to .docx file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = Document(file_path)
            
            # Extract text from all paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Join with double newlines to preserve paragraph structure
            text = "\n\n".join(paragraphs)
            
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from {file_path}: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)      # Max 1 space
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using LangChain splitter
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def process_document(
        self,
        file_path: str,
        original_filename: str,
        collection_name: str,
        custom_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to uploaded file
            original_filename: Original filename
            collection_name: Collection to store in
            custom_metadata: Optional additional metadata
            
        Returns:
            Dictionary with processing results:
            {
                'doc_id': str,
                'original_filename': str,
                'collection_name': str,
                'file_size_bytes': int,
                'word_count': int,
                'chunks': List[Dict],  # Contains text, embedding, metadata
                'num_chunks': int,
                'uploaded_at': str,
                'processed_at': str
            }
        """
        print(f"\n{'='*60}")
        print(f"Processing: {original_filename}")
        print(f"{'='*60}")
        
        # Step 1: Save file to storage
        print("[1/5] Saving file to storage...")
        storage_metadata = self.storage.save_file(
            file_path=file_path,
            original_filename=original_filename,
            collection_name=collection_name
        )
        doc_id = storage_metadata['doc_id']
        
        # Step 2: Extract text
        print("[2/5] Extracting text from .docx...")
        raw_text = self.extract_text_from_docx(file_path)
        cleaned_text = self.clean_text(raw_text)
        word_count = self.count_words(cleaned_text)
        print(f"   Extracted {word_count} words")
        
        # Step 3: Chunk text
        print("[3/5] Chunking text...")
        text_chunks = self.chunk_text(cleaned_text)
        print(f"   Created {len(text_chunks)} chunks")
        
        # Step 4: Generate embeddings
        print("[4/5] Generating embeddings...")
        embeddings = self.embeddings.embed_documents(text_chunks)
        print(f"   Generated {len(embeddings)} embeddings")
        
        # Step 5: Prepare chunk data with metadata
        print("[5/5] Preparing chunk metadata...")
        processed_at = datetime.utcnow().isoformat() + "Z"
        
        chunks_data = []
        for idx, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_metadata = {
                'doc_id': doc_id,
                'chunk_index': idx,
                'chunk_id': f"{doc_id}_chunk_{idx}",
                'original_filename': original_filename,
                'collection_name': collection_name,
                'file_size_bytes': storage_metadata['file_size_bytes'],
                'total_chunks': len(text_chunks),
                'uploaded_at': storage_metadata['uploaded_at'],
                'processed_at': processed_at,
                'chunk_word_count': self.count_words(chunk_text)
            }
            
            # Add custom metadata if provided
            if custom_metadata:
                chunk_metadata.update(custom_metadata)
            
            chunks_data.append({
                'text': chunk_text,
                'embedding': embedding,
                'metadata': chunk_metadata
            })
        
        result = {
            'doc_id': doc_id,
            'original_filename': original_filename,
            'collection_name': collection_name,
            'file_size_bytes': storage_metadata['file_size_bytes'],
            'word_count': word_count,
            'chunks': chunks_data,
            'num_chunks': len(text_chunks),
            'uploaded_at': storage_metadata['uploaded_at'],
            'processed_at': processed_at
        }
        
        print(f"Processing complete!")
        print(f"{'='*60}\n")
        
        return result
    
    def get_chunk_statistics(self, chunks_data: List[Dict]) -> Dict:
        """
        Calculate statistics about chunks
        
        Args:
            chunks_data: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks_data:
            return {}
        
        chunk_lengths = [len(chunk['text']) for chunk in chunks_data]
        chunk_word_counts = [self.count_words(chunk['text']) for chunk in chunks_data]
        
        return {
            'total_chunks': len(chunks_data),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_words_per_chunk': sum(chunk_word_counts) / len(chunk_word_counts),
            'min_words_per_chunk': min(chunk_word_counts),
            'max_words_per_chunk': max(chunk_word_counts)
        }