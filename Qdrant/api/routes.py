from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from Qdrant.document_pipeline.processor import DocumentProcessor
from Qdrant.document_pipeline.qdrant_manager import QdrantManager
from Qdrant.document_pipeline.storage import DocumentStorage

from .models import (
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    CollectionsResponse,
    CollectionInfo,
    DeleteDocumentResponse,
    HealthResponse,
    ErrorResponse
)

from .dependencies import (
    get_processor,
    get_qdrant_manager,
    get_storage
)

router = APIRouter()

@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code= status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description="Upload a document (DOCX), process it into chunks, generate embeddings, and store in Qdrant"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    collection_name: str = Form(..., description="Target collection name"),
    category: Optional[str] = Form(None, description="Document category"),
    processor: DocumentProcessor = Depends(get_processor),
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager),
    storage: DocumentStorage = Depends(get_storage)
):
    try:
        allowed_extensions = ['.docx']
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code= status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type."
            )
        
        temp_file = storage.temp_dir / file.filename
        storage.temp_dir.mkdir(parents=True, exist_ok=True)

        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = len(content)
        upload_time = datetime.now().isoformat()

        custom_metadata = {}
        if category:
            custom_metadata['category'] = category

        result = processor.process_document(
            file_path=str(temp_file),
            original_filename=file.filename,
            collection_name=collection_name,
            custom_metadata=custom_metadata
        )

        doc_id = result['doc_id']
        chunks_data = result['chunks']
        word_count = result['word_count']
        num_chunks = result["num_chunks"]

        try:
            info = qdrant_manager.get_collection_info(collection_name)
            print(f"DEBUG: Collection '{collection_name}' exists with {info['points_count']} points")
        except Exception as e:
            print(f"DEBUG: Collection '{collection_name}' not found, creating...")
            try:
                qdrant_manager.create_collection(collection_name, recreate=False)
                print(f"DEBUG: Collection '{collection_name}' created successfully")
            except Exception as create_error:
                print(f"DEBUG: Failed to create collection: {create_error}")
                # Check if it was created by another request
                try:
                    qdrant_manager.get_collection_info(collection_name)
                    print(f"DEBUG: Collection exists now (created by another request)")
                except:
                    raise HTTPException(500, f"Cannot create collection: {create_error}")

        added_count = qdrant_manager.add_documents(
            collection_name=collection_name,
            chunks_data=chunks_data
        )
            
        processed_time = datetime.now().isoformat()

        temp_file.unlink(missing_ok=True)

        return DocumentUploadResponse(
            status="success",
            doc_id=doc_id,
            original_filename=file.filename,
            collection_name=collection_name,
            file_size_bytes=file_size,
            word_count=word_count,
            chunks_created=added_count,
            uploaded_at=upload_time,
            processed_at=processed_time,
            message=f"Document processed successfully into {added_count} chunks"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    

@router.post(
    "/documents/search",
    response_model=SearchResponse,
    tags=["Documnets"],
    summary="Search documents",
    description="Semantic search across documents in a collection"
)
async def search_documents(
    request: SearchRequest,
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager)
):
    try: 
        try:
            qdrant_manager.get_collection_info(request.collection_name)
        except:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_name}' not found"
            )
        
        results = qdrant_manager.search(
            collection_name=request.collection_name,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filters=request.filters
        )

        search_results = [
            SearchResult(
                id=result['id'],
                score=result['score'],
                text=result['text'],
                metadata=result['metadata']
            )
            for result in results
        ]

        return SearchResponse(
            status="success",
            query=request.query,
            collection_name=request.collection_name,
            results_count=len(search_results),
            results=search_results,
            message=f"Found {len(search_results)} results"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search error: {str(e)}"
        )
    

@router.get(
    "/collections",
    response_model=CollectionsResponse,
    tags=["Collections"],
    summary="List all collections",
    description="Get list of all Qdrant collections with their statistics"
)
async def list_collections(
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager)
):
    try:
        stats = qdrant_manager.get_stats()
        
        collections = [
            CollectionInfo(
                name=coll['name'],
                points_count=coll['points_count'],
                vectors_count=coll['points_count'],  # Same as points for now
                status="active"
            )
            for coll in stats['collections']
        ]
        
        return CollectionsResponse(
            status="success",
            total_collections=stats['total_collections'],
            collections=collections
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )
    
@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    tags=["Documents"],
    summary="Delete a document",
    description="Delete all chunks of a document from a collection"
)
async def delete_document(
    doc_id: str,
    collection_name: str = Query(..., description="Collection name"),
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager),
    storage: DocumentStorage = Depends(get_storage)
):
    try:
        # Check if collection exists
        try:
            qdrant_manager.get_collection_info(collection_name)
        except:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found"
            )
        
        # Delete from Qdrant
        deleted = qdrant_manager.delete_document(
            collection_name=collection_name,
            doc_id=doc_id
        )
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found in collection '{collection_name}'"
            )

        try:
            storage.delete_document(doc_id)
        except:
            pass  # File deletion is optional
        
        return DeleteDocumentResponse(
            status="success",
            doc_id=doc_id,
            collection_name=collection_name,
            message=f"Document '{doc_id}' deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="Check health status of all components"
)
async def health_check_detailed(
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager),
    storage: DocumentStorage = Depends(get_storage)
):
    try:
        qdrant_connected = False
        try:
            qdrant_manager.get_stats()
            qdrant_connected = True
        except:
            pass

        embeddings_loaded = True

        storage_available = True
        try:
            storage.base_upload_dir.exists()
            storage_available = True
        except:
            pass

        all_healthy = qdrant_connected and embeddings_loaded and storage_available

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            qdrant_connected=qdrant_connected,
            embeddings_loaded=embeddings_loaded,
            storage_available=storage_available,
            message="All systems operational" if all_healthy else "Some components unavailable"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )
    
@router.post(
    "/collections/{collection_name}",
    tags=["Collections"],
    summary="Create a new collection",
    description="Create a new Qdrant collection for storing docs"
)
async def create_collection(
    collection_name: str,
    recreate: bool = Form(False, description="Recreate if exists"),
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager)
):
    try:
        qdrant_manager.create_collection(
            collection_name=collection_name,
            recreate=recreate
        )

        return {
            "status": "success",
            "collection_name": collection_name,
            "message": f"Collection '{collection_name}' created successfully"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating collection: {str(e)}"
        )
    
@router.delete(
    "/collections/{collection_name}",
    tags=["Collections"],
    summary="Delete a collection",
    description="Delete a Qdrant collection and all its documents"
)
async def delete_collection(
    collection_name: str,
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager)
):
    try:
        deleted = qdrant_manager.delete_collection(collection_name)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found"
            )
        
        return {
            "status": "success",
            "collection_name": collection_name,
            "message": f"Collection '{collection_name}' deleted successfully"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}"
        )
    
@router.get(
    "/stats",
    tags=["Statistics"],
    summary="Get overall statistics",
    description="Get stats about all collections and storage"
)
async def get_statistics(
    qdrant_manager: QdrantManager = Depends(get_qdrant_manager),
    storage: DocumentStorage = Depends(get_storage)
):
    try:
        qdrant_stats = qdrant_manager.get_stats()
        storage_stats = storage.get_storage_stats()

        return {
            "status": "success",
            "qdrant": qdrant_stats,
            "storage": storage_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=f"Error getting statistics: {str(e)}"
        )



    
