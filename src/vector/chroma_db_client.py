from typing import List

import chromadb
from chromadb.api import Collection
from src.models import RetrievalResult
from chromadb.utils import embedding_functions


class ChromaDBClient:
    """ChromaDBClient is a client for the ChromaDB vector database."""

    def __init__(self, db_path: str, embedding_model_name: str = "intfloat/e5-base-v2"):
        self.__db_path = db_path
        self.__embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        )

        self.__client = chromadb.PersistentClient(path=self.__db_path)

    def get_collection(self, name: str) -> Collection:
        """Get a collection from the database."""
        return self.__client.get_collection(
            name=name, embedding_function=self.__embedding_function
        )

    def query_collection(
        self, collection_name: str, user_query: str, top_k: int = 30
    ) -> List[RetrievalResult]:
        """Query a collection in the database."""
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_texts=[user_query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        formatted_results = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            formatted_results.append(
                RetrievalResult(
                    content=doc, metadata=metadata, relevance_score=1 - distance
                )
            )

        return formatted_results


if __name__ == "__main__":
    db = ChromaDBClient(db_path="./chroma_db")
    collection_results = db.query_collection(
        collection_name="test_collection", user_query="test query", top_k=10
    )
    print("Relevant documents:")
    for i, result in enumerate(collection_results, 1):
        print(f"\n{i}. Relevance Score: {result.relevance_score:.4f}")
        print(f"Content: {result.content}")
        print(f"Context: {result.metadata}")
