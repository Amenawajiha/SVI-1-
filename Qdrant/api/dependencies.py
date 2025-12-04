from fastapi import HTTPException, status

from .main import app_state


def get_embeddings():
    """Dependency to get embeddings instance"""
    if app_state.embeddings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings not initialized"
        )
    return app_state.embeddings


def get_storage():
    """Dependency to get storage instance"""
    if app_state.storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage not initialized"
        )
    return app_state.storage


def get_processor():
    """Dependency to get processor instance"""
    if app_state.processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processor not initialized"
        )
    return app_state.processor


def get_qdrant_manager():
    """Dependency to get Qdrant manager instance"""
    if app_state.qdrant_manager is None:
        print(f"DEBUG: Qdrant manager is None!")
        print(f"DEBUG: Other components:")
        print(f"  - embeddings: {app_state.embeddings is not None}")
        print(f"  - storage: {app_state.storage is not None}")
        print(f"  - processor: {app_state.processor is not None}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant manager not initialized"
        )
    return app_state.qdrant_manager