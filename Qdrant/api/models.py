
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    """Response after document upload and processing"""
    status: str = Field(..., description="Status: success or error")
    doc_id: str = Field(..., description="Unique document identifier")
    original_filename: str = Field(..., description="Original filename")
    collection_name: str = Field(..., description="Collection where document is stored")
    file_size_bytes: int = Field(..., description="File size in bytes")
    word_count: int = Field(..., description="Total word count")
    chunks_created: int = Field(..., description="Number of chunks created")
    uploaded_at: str = Field(..., description="Upload timestamp (ISO format)")
    processed_at: str = Field(..., description="Processing completion timestamp (ISO format)")
    message: Optional[str] = Field(None, description="Additional message")


class SearchRequest(BaseModel):
    """Request for semantic search"""
    query: str = Field(..., description="Search query text", min_length=1)
    collection_name: str = Field(..., description="Collection to search in")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=20)
    score_threshold: float = Field(0.3, description="Minimum similarity score", ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")


class SearchResult(BaseModel):
    """Single search result"""
    id: int = Field(..., description="Point ID in Qdrant")
    score: float = Field(..., description="Similarity score (0-1)")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")


class SearchResponse(BaseModel):
    """Response for search requests"""
    status: str = Field(..., description="Status: success or error")
    query: str = Field(..., description="Original search query")
    collection_name: str = Field(..., description="Collection searched")
    results_count: int = Field(..., description="Number of results returned")
    results: List[SearchResult] = Field(..., description="Search results")
    message: Optional[str] = Field(None, description="Additional message")


class CollectionInfo(BaseModel):
    """Information about a collection"""
    name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of points/chunks")
    vectors_count: int = Field(..., description="Number of vectors")
    status: str = Field(..., description="Collection status")


class CollectionsResponse(BaseModel):
    """Response listing all collections"""
    status: str = Field(..., description="Status: success or error")
    total_collections: int = Field(..., description="Total number of collections")
    collections: List[CollectionInfo] = Field(..., description="List of collections")


class DeleteDocumentResponse(BaseModel):
    """Response after document deletion"""
    status: str = Field(..., description="Status: success or error")
    doc_id: str = Field(..., description="Deleted document ID")
    collection_name: str = Field(..., description="Collection name")
    message: str = Field(..., description="Deletion message")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    embeddings_loaded: bool = Field(..., description="Embeddings model status")
    storage_available: bool = Field(..., description="Storage availability")
    message: Optional[str] = Field(None, description="Additional information")


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = Field("error", description="Status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")