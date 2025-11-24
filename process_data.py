import json
import chromadb
from chromadb.utils import embedding_functions

def load_json_data(file_path):
    """Load JSON data from file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_qa_pairs(data, data_type="flight", source_file=""):
    """
    Extract QA pairs from JSON data with improved metadata structure.
    
    Args:
        data: JSON data containing sections, subsections, topics, and QnA
        data_type: Type of data ('flight' or 'hotel')
        source_file: Source filename for tracking
    
    Returns:
        Tuple of (qa_pairs, doc_ids, metadatas)
    """
    qa_pairs = []
    doc_ids = []
    metadatas = []
    id_counter = 0
    
    for section in data:
        # Create stable section identifier
        section_id = section.get("section_id", section.get("section_title", "")).lower().replace(" ", "_")
        section_title = section['section_title']
        
        for subsection in section.get('subsections', []):
            subsection_key = subsection.get('subsection_title', '').lower().replace(" ", "_")
            subsection_title = subsection.get('subsection_title', '')
            
            for topic in subsection.get('topics', []):
                topic_key = topic.get('topic', '').lower().replace(" ", "_")
                topic_name = topic.get('topic', '')
                
                for qa in topic.get('qna', []):
                    question = qa.get('question', '').strip()
                    answer = qa.get('answer', '').strip()
                    
                    # Skip if question or answer is empty
                    if not question or not answer:
                        continue
                    
                    # ✅ FIX: Get paraphrased questions FIRST (before using them)
                    paraphrases = qa.get('paraphrased_questions', []) or qa.get('paraphrases', []) or []
                    
                    # Create content for embedding
                    content = f"Q: {question}\n"
                    if paraphrases:
                        for p in paraphrases:
                            content += f"Also asked as: {p}\n"
                    content += f"A: {answer}"
                    
                    # Create stable document ID
                    doc_id = f"{data_type}.{section_id}.{subsection_key}.{topic_key}.{id_counter}"
                    
                    qa_pairs.append(content)
                    doc_ids.append(doc_id)
                    
                    # Create comprehensive metadata
                    metadata = {
                        "section_id": section_id,
                        "section_title": section_title,
                        "subsection": subsection_title,
                        "topic": topic_name,
                        "question": question,
                        "paraphrases": "; ".join(paraphrases) if paraphrases else "",
                        "data_type": data_type,
                        "source": source_file,
                        "doc_id": doc_id
                    }
                    metadatas.append(metadata)
                    
                    id_counter += 1
    
    return qa_pairs, doc_ids, metadatas

def setup_chromadb(db_path="./chroma_db", collection_name="travel_info"):
    """
    Initialize ChromaDB client and collection.
    
    Args:
        db_path: Path to ChromaDB storage
        collection_name: Name of the collection
    
    Returns:
        ChromaDB collection object
    """
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Use sentence-transformers for better embedding quality
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except ValueError:  
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    return chroma_client, collection

def main():
    # Configuration
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "travel_info"
    
    # Data files to process
    data_files = [
        {
            "path": "data/flight_data.json",
            "type": "flight",
            "name": "flight_data.json"
        },
        {
            "path": "data/hotel_data.json",
            "type": "hotel",
            "name": "hotel_data.json"
        }
    ]
    
    print("="*60)
    print("Travel Information Database Setup")
    print("="*60)
    
    # Initialize the client
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Check if collection exists
    existing_collections = chroma_client.list_collections()
    collection_exists = any(col.name == COLLECTION_NAME for col in existing_collections)
    
    if collection_exists:
        print(f"\n⚠️  Collection '{COLLECTION_NAME}' already exists.")
        user_input = input("Do you want to:\n"
                         "1. Delete and recreate (fresh start)\n"
                         "2. Keep existing data (skip processing)\n"
                         "Enter choice (1 or 2): ")
        
        if user_input == "1":
            print(f"\n🗑️  Deleting existing collection '{COLLECTION_NAME}'...")
            chroma_client.delete_collection(COLLECTION_NAME)
            chroma_client, collection = setup_chromadb(DB_PATH, COLLECTION_NAME)
            print("✅ Collection deleted and recreated.")
        else:
            print("\n✅ Using existing collection. Exiting without changes.")
            return
    else:
        print(f"\n📁 Creating new collection '{COLLECTION_NAME}'...")
        chroma_client, collection = setup_chromadb(DB_PATH, COLLECTION_NAME)
    
    # Process each data file
    total_documents = 0
    
    for file_config in data_files:
        file_path = file_config["path"]
        data_type = file_config["type"]
        file_name = file_config["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_name} ({data_type.upper()} data)")
        print(f"{'='*60}")
        
        try:
            # Load data
            print(f"📖 Loading {file_path}...")
            data = load_json_data(file_path)
            
            # Extract QA pairs
            print(f"⚙️  Extracting QA pairs...")
            documents, ids, metadatas = extract_qa_pairs(data, data_type, file_name)
            
            print(f"✅ Extracted {len(documents)} QA pairs")
            
            # Add to collection
            print(f"💾 Adding to ChromaDB...")
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            total_documents += len(documents)
            print(f"✅ Successfully added {len(documents)} documents from {file_name}")
            
        except FileNotFoundError:
            print(f"❌ Error: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
    
    # Data is automatically persisted in newer ChromaDB versions
    print(f"\n{'='*60}")
    print("💾 Data automatically persisted to disk...")
    print("✅ Database saved successfully!")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total documents added: {total_documents}")
    print(f"Database location: {DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    
    # Test queries
    print(f"\n{'='*60}")
    print("🧪 Running Test Queries")
    print(f"{'='*60}")
    
    test_queries = [
        "What are the flight cancellation charges?",
        "Is the hotel booking valid for visa submission?",
        "What is the refund policy?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        for idx, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0]
        )):
            relevance_score = 1 - distance
            print(f"\n  Result {idx + 1}:")
            print(f"  Data Type: {metadata.get('data_type', 'N/A').upper()}")
            print(f"  Section: {metadata.get('section_title', 'N/A')}")
            print(f"  Topic: {metadata.get('topic', 'N/A')}")
            print(f"  Relevance: {relevance_score:.4f}")
            print(f"  Preview: {doc[:150]}...")
    
    print(f"\n{'='*60}")
    print("✅ Setup Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()