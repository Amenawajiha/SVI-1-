from typing import List

from src.models import RetrievalResult
from src.vector.chroma_db_client import ChromaDBClient
from src.vector.reranker import Reranker


class VectorRetriever:
    """VectorRetriever is a class that retrieves relevant documents from a vector database."""

    def __init__(
        self,
        client: ChromaDBClient,
        collection_name: str,
        top_k: int = 3,
        reranker: Reranker = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.reranker = reranker

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant documents from a vector database."""
        # Use query_collection
        results = self.client.query_collection(self.collection_name, query, self.top_k)
        return results

    def retrieve_with_reranking(
        self, query: str, initial_k: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents from a vector database with reranking.

        This implements a two-stage retrieval approach:
        1. Fetch initial_k candidates from the vector database
        2. Rerank using the cross-encoder model
        3. Return top-k most relevant results

        Args:
            query: The search query
            initial_k: Number of initial candidates to fetch (should be > top_k)

        Returns:
            Top-k reranked results

        Raises:
            ValueError: If reranker is not configured
        """
        if self.reranker is None:
            raise ValueError(
                "Reranker not configured. Initialize VectorRetriever with a Reranker instance."
            )

        # Fetch initial candidates from vector database
        results = self.client.query_collection(self.collection_name, query, initial_k)

        # Rerank and return top-k
        reranked_results = self.reranker.rerank(query, results)

        return reranked_results
