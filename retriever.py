from chromadb.utils import embedding_functions
import chromadb
import os

def setup_retriever():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "chroma_db")    

    client = chromadb.PersistentClient(path=db_path)
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
   
    collection = client.get_collection(
        name="travel_info",  
        embedding_function=embedding_function
    )
    
    return collection

def retrieve_docs(query, top_k=3):
    collection = setup_retriever()
    results = collection.query(
        query_texts=[query],  
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']  
    )
    
    formatted_results = []
    for doc, metadata, distance in zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    ):
        formatted_results.append({
            'content': doc,
            'metadata': metadata,
            'relevance_score': 1 - distance  
        })
    
    return formatted_results

def main():
    test_queries = [
        # Flight queries
        "What are the cancellation charges for flight?",
        "Can I get a refund?",
        "How long is this flight itinerary valid?",
        # Hotel queries
        "Is the hotel booking valid for embassy or visa submission?",
        "What are the hotel cancellation charges?",
        "Can I get a refund if my visa is rejected?"
    ]
    
    print("Testing retriever with flight and hotel queries:\n")
    print("="*80)
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-"*80)
        results = retrieve_docs(query, top_k=2)
        
        print("\nRelevant Documents:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            data_type = metadata.get('data_type', 'flight')
            print(f"\n{i}. [{data_type.upper()}] Relevance Score: {result['relevance_score']:.4f}")
            print(f"   Section: {metadata.get('section_title', 'N/A')}")
            print(f"   Topic: {metadata.get('topic', 'N/A')}")
            print(f"   Question: {metadata.get('question', 'N/A')}")
            print(f"   Source: {metadata.get('source', 'N/A')}")
            print(f"   Content Preview: {result['content'][:150]}...")
        print("="*80)

if __name__ == "__main__":
    main()