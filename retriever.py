import chromadb
from chromadb.utils import embedding_functions


def setup_retriever():
    client = chromadb.PersistentClient(path="./chroma_db")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/e5-base-v2"
    )

    collection = client.get_collection(
        name="flight_info",
        embedding_function=embedding_function,
    )

    return collection


def retrieve_docs(query, top_k=3):
    collection = setup_retriever()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    formatted_results = []
    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        formatted_results.append(
            {"content": doc, "metadata": metadata, "relevance_score": 1 - distance}
        )

    return formatted_results


def main():
    test_queries = [
        "What are the cancellation charges?",
        "You guys give pnr number after booking?",
        "How long is this flight itinerary valid?",
    ]

    print("Testing retriever with example queries:\n")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve_docs(query, top_k=2)

        print("\nRelevant Documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Relevance Score: {result['relevance_score']:.4f}")
            print(f"Content: {result['content']}")
            print(f"Context: {result['metadata']}")


if __name__ == "__main__":
    main()
