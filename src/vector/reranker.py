"""Reranker model for document reranking using cross-encoder models."""

from typing import Any, List

from sentence_transformers import CrossEncoder

from src.models import RetrievalResult


class Reranker:
    """Cross-encoder model for reranking retrieved documents.

    This class implements a two-stage retrieval approach:
    1. Retrieve more candidates (e.g., initial_k=10) from vector database
    2. Rerank using a powerful cross-encoder model (e.g., BGE reranker)
    3. Return top-k most relevant results

    Example usage:
        # Without reranking
        retriever = VectorRetriever(client, "flight_info", top_k=3)
        results = retriever.retrieve(query)  # Direct ChromaDB results

        # With reranking (improved accuracy)
        reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3", top_k=3)
        retriever = VectorRetriever(client, "flight_info", reranker=reranker)
        results = retriever.retrieve_with_reranking(query, initial_k=10)
        # Flow: Fetch 10 → Rerank with BGE → Return top 3
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_k: int = 3):
        """Initialize the reranker with a cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model from HuggingFace
            top_k: Number of top results to return after reranking
        """
        self.model_name = model_name
        self.top_k = top_k
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model.

        This is a private method that initializes the model lazily.
        The model is loaded once during initialization.
        """
        self.model = CrossEncoder(self.model_name)

    def calculate_rerank_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calculate reranking scores for query-document pairs.

        Args:
            query: The search query
            documents: List of document contents to score

        Returns:
            List of relevance scores (higher is better)
        """
        if not documents:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores from the model
        scores = self.model.predict(pairs)

        # Convert to list if numpy array
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        return scores

    def rerank(
        self, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank retrieval results based on query relevance.

        This method takes initial retrieval results and reranks them using
        a cross-encoder model, which provides more accurate relevance scores
        than the initial vector similarity scores.

        Args:
            query: The search query
            results: List of initial retrieval results from vector database

        Returns:
            Top-k reranked results with updated relevance scores
        """
        if not results:
            return []

        # Extract document contents for scoring
        documents = [result.content for result in results]

        # Calculate rerank scores
        scores = self.calculate_rerank_scores(query, documents)

        # Combine results with their new scores
        scored_results = list(zip(results, scores))

        # Sort by score in descending order (higher score = more relevant)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update relevance scores and return top-k
        reranked_results = []
        for result, score in scored_results[: self.top_k]:
            # Create a new RetrievalResult with updated relevance score
            reranked_result = RetrievalResult(
                content=result.content,
                metadata=result.metadata,
                relevance_score=float(score),
            )
            reranked_results.append(reranked_result)

        return reranked_results
