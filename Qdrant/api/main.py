import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
from datetime import datetime

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from Qdrant.document_pipeline.embedding import E5Embeddings
from Qdrant.document_pipeline.storage import DocumentStorage
from Qdrant.document_pipeline.processor import DocumentProcessor
from Qdrant.document_pipeline.qdrant_manager import QdrantManager

from .routes import router as api_router

class AppState:
    embeddings: Optional[E5Embeddings] = None
    storage: Optional[DocumentStorage] = None
    processor: Optional[DocumentProcessor] = None
    qdrant_manager: Optional[QdrantManager] = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 80)
    print("Initializing Qdrant Document API...")
    print("=" * 80)

    try:

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        chunk_size = int(os.getenv("CHUNK_SIZE", "400"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

        print("\n[1/4] Initializing E5-base-v2 embeddings...")
        app_state.embeddings = E5Embeddings(device=device)
        print(f"   Model loaded: intfloat/e5-base-v2")
        print(f"   Embedding dimension: 768")

        print("\n[2/4] Initializing document storage...")
        app_state.storage = DocumentStorage(base_upload_dir=upload_dir)
        print(f"   Upload directory: {upload_dir}")

        print("\n[3/4] Initializing document processor...")
        app_state.processor = DocumentProcessor(
            embeddings=app_state.embeddings,
            storage=app_state.storage,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"   Chunking: {chunk_size} chars with {chunk_overlap} overlap")
        
        print("\n[4/4] Connecting to Qdrant...")
        app_state.qdrant_manager = QdrantManager(
            host=qdrant_host,
            port=qdrant_port,
            embeddings=app_state.embeddings
        )

        stats = app_state.qdrant_manager.get_stats()
        print(f"   Connected successfully!")
        print(f"   Existing collections: {stats['total_collections']}")

    except Exception as e:
        print(f"\nERROR during initialization: {e}")
        print("Make sure Qdrant is running.")
        raise
    
    yield 

    print("\n" + "=" * 80)
    print("Shutting down Qdrant Document API...")
    print("=" * 80)
    
    # Cleanup if needed
    app_state.embeddings = None
    app_state.storage = None
    app_state.processor = None
    app_state.qdrant_manager = None
    
    print("Shutdown complete")

app = FastAPI(
    title="Document Pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print("Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@app.get("/", tags=['Root'])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Document Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=['Health'])
async def health_check():
    try:
        components_status = {
            "embeddings": app_state.embeddings is not None,
            "storage": app_state.storage is not None,
            "processor": app_state.processor is not None,
            "qdrant_manager": app_state.qdrant_manager is not None
        }

        qdrant_stats = None
        if app_state.qdrant_manager:
            try:
                qdrant_stats = app_state.qdrant_manager.get_stats()
            except Exception as e:
                print(f"Could not get Qdrant stats: {e}")

        all_healthy = all(components_status.values())

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "components": components_status,
            "qdrant": qdrant_stats if qdrant_stats else {"status": "unavailable"},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {e}"
        )
    
app.include_router(
    api_router,
    prefix="/api",
    dependencies=[]
)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print("=" * 80)
    print("Starting Document Pipeline API Server")
    print("=" * 80)
    print(f"\nServer will run on: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Auto-reload: {reload}")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)
    print()

    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\n\nERROR starting server: {e}")
        raise



