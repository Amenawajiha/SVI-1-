"""
Qdrant Manager Module
Handles all Qdrant operations: collection management, storage, and search
"""

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    SearchParams  
)

from .embedding import E5Embeddings


class QdrantManager:
    """
    Manages Qdrant vector database operations
    Handles collection management, document storage, and search
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embeddings: Optional[E5Embeddings] = None
    ):
        """
        Initialize Qdrant manager
        
        Args:
            host: Qdrant host
            port: Qdrant port
            embeddings: E5Embeddings instance for search queries
        """
        self.client = QdrantClient(host=host, port=port)
        self.embeddings = embeddings
        self.vector_size = 768  
        
        print(f"Connected to Qdrant at {host}:{port}")
        
        try:
            collections = self.client.get_collections()
            print(f"Connection verified ({len(collections.collections)} collections)")
        except Exception as e:
            print(f"Connection warning: {e}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create a new Qdrant collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (default: 768 for E5-base-v2)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete existing collection first
            
        Returns:
            True if created, False if already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                if recreate:
                    print(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    print(f"Collection '{collection_name}' already exists")
                    return False
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            print(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except:
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """
        Get information about a collection
        
        Returns:
            Dictionary with collection info or None if not found
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            
            return {
                'name': collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.points_count,  # Same as points_count
                'status': collection_info.status,
                'config': {
                    'params': {
                        'vectors': collection_info.config.params.vectors.dict() if hasattr(collection_info.config.params, 'vectors') else {}
                    }
                }
            }
        except Exception as e:
            print(f"Collection not found or error: {collection_name} - {e}")
            return None
    
    def add_documents(
        self,
        collection_name: str,
        chunks_data: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Add processed document chunks to Qdrant
        
        Args:
            collection_name: Target collection
            chunks_data: List of chunk dictionaries from DocumentProcessor
                        Each should have: 'text', 'embedding', 'metadata'
            batch_size: Number of points to upload at once
            
        Returns:
            Number of points added
        """
        if not chunks_data:
            print("No chunks to add")
            return 0
        
        # Ensure collection exists
        if not self.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}")
            self.create_collection(collection_name)
        
        print(f"Adding {len(chunks_data)} chunks to '{collection_name}'...")
        
        # Prepare points
        points = []
        for idx, chunk_data in enumerate(chunks_data):
            point = PointStruct(
                id=idx,  # Sequential ID for simplicity
                vector=chunk_data['embedding'],
                payload={
                    'text': chunk_data['text'],
                    **chunk_data['metadata']  # Merge metadata
                }
            )
            points.append(point)
        
        # Upload in batches
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            total_added += len(batch)
            print(f"   Uploaded batch: {total_added}/{len(points)}")
        
        print(f"Added {total_added} chunks to '{collection_name}'")
        return total_added
    
    def search(
    self,
    collection_name: str,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Semantic search in collection
        
        Args:
            collection_name: Collection to search
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters
                Example: {'doc_id': 'abc-123'} or {'category': 'travel'}
            
        Returns:
            List of search results with text, metadata, and score
        """
        if not self.embeddings:
            raise ValueError("Embeddings instance required for search")
        
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' not found")
            return []
        
        print(f"Searching in '{collection_name}' for: '{query}'")
        
        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        try:
            from qdrant_client.models import QueryRequest, Query
            
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,  
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            ).points
            
        except (ImportError, AttributeError, TypeError):
            try:
                search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True
                )
            except AttributeError:
                search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )
        
        # Format results
        results = []
        for hit in search_results:
            score = getattr(hit, 'score', None)
            if score is None:
                # Convert distance to score if needed
                distance = getattr(hit, 'distance', 0)
                score = 1 - distance if distance else 0
            
            if score < score_threshold:
                continue
            
            results.append({
                'id': hit.id,
                'score': score,
                'text': hit.payload.get('text', ''),
                'metadata': {
                    k: v for k, v in hit.payload.items() 
                    if k != 'text'  
                }
            })
        
        print(f"Found {len(results)} results")
        return results
    
    def delete_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document
        
        Args:
            collection_name: Collection name
            doc_id: Document ID to delete
            
        Returns:
            Number of points deleted
        """
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' not found")
            return 0
        
        print(f"Deleting chunks for doc_id: {doc_id}")
        
        # Delete by filter
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            )
        )
        
        print(f"Deleted chunks for doc_id: {doc_id}")
        return 1  
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.get_collections().collections
        return [c.name for c in collections]
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Failed to delete collection: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get overall Qdrant statistics"""
        collections = self.client.get_collections().collections
        
        stats = {
            'total_collections': len(collections),
            'collections': []
        }
        
        for collection in collections:
            info = self.get_collection_info(collection.name)
            if info:
                stats['collections'].append(info)
        
        return stats