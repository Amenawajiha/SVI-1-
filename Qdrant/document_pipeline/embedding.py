"""
E5-base-v2 Embedding Module
Handles text-to-vector conversion with proper prefixes
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class E5Embeddings:
    """
    Wrapper for E5-base-v2 embedding model
    
    E5 models require specific prefixes:
    - "passage: " for documents/chunks to be stored
    - "query: " for search queries
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
        self.dimension = 768  # E5-base-v2 output dimension
        self.max_seq_length = 512  # Maximum token length
        print(f"Model loaded successfully (dimension: {self.dimension})")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents/passages for storage
        Adds "passage: " prefix to each text
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (768-dimensional)
        """
        if not texts:
            return []
        
        # Add passage prefix
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=len(texts) > 10  # Show progress for large batches
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query
        Adds "query: " prefix
        
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
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        return self.dimension
    
    def get_max_length(self) -> int:
        """Return maximum sequence length"""
        return self.max_seq_length


# Convenience function for quick usage
def create_embeddings(device: str = "cpu") -> E5Embeddings:
    """
    Factory function to create E5Embeddings instance
    
    Args:
        device: 'cpu' or 'cuda'
        
    Returns:
        Initialized E5Embeddings instance
    """
    return E5Embeddings(device=device)