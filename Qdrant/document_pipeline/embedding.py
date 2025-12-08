"""
E5-base-v2 Embedding Module
Handles text-to-vector conversion with proper prefixes
"""

from typing import List, Protocol
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    """Interface for all embedding providers."""

    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query"""
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for storage"""
        ...


class E5Embeddings:
    """
    Wrapper for E5-base-v2 embedding model
    
    E5 models require specific prefixes:
    - "passage: " for documents/chunks to be stored
    - "query: " for search queries
    
    Implements EmbeddingProvider protocol.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str = "cpu"):
        """
        Initialize E5 embedding model
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        print(f"Loading E5 model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = 768  # Private attribute
        self._max_seq_length = 512
        print(f"Model loaded successfully (dimension: {self._dimension})")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (implements EmbeddingProvider)"""
        return self._dimension
    
    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length"""
        return self._max_seq_length
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents/passages for storage.
        Adds "passage: " prefix to each text.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (768-dimensional)
        """
        if not texts:
            return []
        
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        Adds "query: " prefix.
        
        Args:
            query: Search query string
            
        Returns:
            Single embedding vector (768-dimensional)
        """
        prefixed_query = f"query: {query}"
        
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        
        return embedding.tolist()


def create_embeddings(device: str = "cpu") -> EmbeddingProvider:
    """
    Factory function to create embedding instance.
    
    Args:
        device: 'cpu' or 'cuda'
        
    Returns:
        EmbeddingProvider implementation
    """
    return E5Embeddings(device=device)