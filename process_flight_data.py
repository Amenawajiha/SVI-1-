import json
import chromadb
from chromadb.utils import embedding_functions

def load_flight_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_qa_pairs(data):
    qa_pairs = []
    doc_ids = []
    metadatas = []
    id_counter = 0
    
    for section in data:
        section_title = section['section_title']
        for subsection in section['subsections']:
            subsection_title = subsection['subsection_title']
            for topic in subsection['topics']:
                topic_name = topic['topic']
                for qa in topic['qna']:
                    # Combine question and answer for better context
                    qa_text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                    qa_pairs.append(qa_text)
                    
                    # Create unique ID
                    doc_ids.append(f"doc_{id_counter}")
                    
                    # Store metadata for context
                    metadata = {
                        "section": section_title,
                        "subsection": subsection_title,
                        "topic": topic_name,
                        "question": qa['question']
                    }
                    metadatas.append(metadata)
                    
                    id_counter += 1
    
    return qa_pairs, doc_ids, metadatas

def setup_chromadb():
    # Initialize persistent client with a specific path
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use sentence-transformers for better embedding quality
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = chroma_client.create_collection(
            name="flight_info",
            embedding_function=embedding_function
        )
    except ValueError:  
        collection = chroma_client.get_collection(
            name="flight_info",
            embedding_function=embedding_function
        )
    
    return collection

def main():
    # Initialize the client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Check if collection exists
    existing_collections = chroma_client.list_collections()
    collection_exists = any(col.name == "flight_info" for col in existing_collections)
    
    if collection_exists:
        user_input = input("Collection 'flight_info' already exists. Do you want to:\n"
                         "1. Delete and recreate\n"
                         "2. Keep existing data\n"
                         "Enter choice (1 or 2): ")
        
        if user_input == "1":
            print("Deleting existing collection...")
            chroma_client.delete_collection("flight_info")
            collection = setup_chromadb()
            
            # Load and add new data
            flight_data = load_flight_data('flight_data.json')
            documents, ids, metadatas = extract_qa_pairs(flight_data)
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully added {len(documents)} new QA pairs to ChromaDB")
        else:
            print("Using existing collection...")
            collection = chroma_client.get_collection(
                name="flight_info",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
    else:
        collection = setup_chromadb()
        # Load and add new data
        flight_data = load_flight_data('flight_data.json')
        documents, ids, metadatas = extract_qa_pairs(flight_data)
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully added {len(documents)} QA pairs to ChromaDB")

    results = collection.query(
        query_texts=["What are the cancellation charges?"],
        n_results=2
    )
    
    print("\nTest Query Results:")
    for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\nResult {idx + 1}:")
        print(f"Document: {doc}")
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main()