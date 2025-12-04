"""
To test uploading a document - use:
    python cli_testing.py upload --file path/to/document.docx
To test searching documents - use:
    python cli_testing.py search --query "your query"
    sample queries:
    Query: “How should we measure employee performance instead of tracking hours?”
    -> Primary: 7 (metrics shift). Secondary: 2 (mentions trends).
    Reason: directly asks about replacing time-based metrics with outcomes.

    Query: “Best practices for running remote, asynchronous teams”
    -> Primary: 3 (remote-first / async). Secondary: 2.
    Reason: chunk 3 lists async practices (documents, meetings, protocols).

    Query: “How can we support employees to keep learning as their jobs change?”
    -> Primary: 4 (lifelong learning). Secondary: 8 (90-day learning sprints).
    Reason: focuses on learning systems, coaching, microlearning.

    Query: “What guardrails should we add to automation to avoid bias?”
    -> Primary: 5 (ethics / bias / logging / human review). Secondary: 9 (guardrails list).
    Reason: asks about bias prevention and human-in-loop strategies.

    Query: “Which tasks should we automate first?”
    -> Primary: 8 (Appendix step 1 — inventory tasks automatable/collaborative/creative). Secondary: 2.
    Reason: chunk 8 explicitly recommends inventory and prioritization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

from Qdrant.document_pipeline.embedding import E5Embeddings
from Qdrant.document_pipeline.storage import DocumentStorage
from Qdrant.document_pipeline.processor import DocumentProcessor
from Qdrant.document_pipeline.qdrant_manager import QdrantManager


class DocumentPipelineCLI:
    """CLI interface for document pipeline"""
    
    def __init__(self):
        """Initialize all components"""
        print("=" * 80)
        print("Document Pipeline CLI")
        print("=" * 80)
        
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.device = os.getenv("EMBEDDING_DEVICE", "cpu")
        
        print("\nConfiguration:")
        print(f"   Qdrant: {self.qdrant_host}:{self.qdrant_port}")
        print(f"   Upload dir: {self.upload_dir}")
        print(f"   Chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        print(f"   Device: {self.device}")
        
        # Initialize components
        print("\nInitializing components...")
        
        try:
            self.embeddings = E5Embeddings(device=self.device)
            self.storage = DocumentStorage(base_upload_dir=self.upload_dir)
            self.processor = DocumentProcessor(
                embeddings=self.embeddings,
                storage=self.storage,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.qdrant_manager = QdrantManager(
                host=self.qdrant_host,
                port=self.qdrant_port,
                embeddings=self.embeddings
            )
            
            print("All components initialized successfully!")
            
        except Exception as e:
            print(f"\nFailed to initialize components: {e}")
            sys.exit(1)
    
    def upload_document(
        self,
        file_path: str,
        collection_name: str = "cli_documents",
        category: Optional[str] = None
    ):
        """
        Upload and process a document
        
        Args:
            file_path: Path to .docx file
            collection_name: Collection to store in
            category: Optional category metadata
        """
        print("\n" + "=" * 80)
        print("UPLOADING DOCUMENT")
        print("=" * 80)
        
        # Validate file
        doc_path = Path(file_path)
        if not doc_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        if not doc_path.suffix.lower() == '.docx':
            print(f"File must be .docx format")
            return None
        
        print(f"\nFile: {doc_path.name}")
        print(f"Collection: {collection_name}")
        if category:
            print(f"Category: {category}")
        
        # Prepare metadata
        custom_metadata = {}
        if category:
            custom_metadata['category'] = category
        
        # Process document
        try:
            result = self.processor.process_document(
                file_path=str(doc_path),
                original_filename=doc_path.name,
                collection_name=collection_name,
                custom_metadata=custom_metadata
            )
            
            # Ensure collection exists
            if not self.qdrant_manager.collection_exists(collection_name):
                print(f"\nCreating collection: {collection_name}")
                self.qdrant_manager.create_collection(collection_name)
            
            # Store in Qdrant
            print(f"\nStoring in Qdrant...")
            added_count = self.qdrant_manager.add_documents(
                collection_name=collection_name,
                chunks_data=result['chunks']
            )
            
            # Print summary
            print("\n" + "=" * 80)
            print("UPLOAD COMPLETE")
            print("=" * 80)
            print(f"\nSummary:")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   Filename: {result['original_filename']}")
            print(f"   Collection: {result['collection_name']}")
            print(f"   Word count: {result['word_count']}")
            print(f"   Chunks created: {result['num_chunks']}")
            print(f"   Chunks stored: {added_count}")
            print(f"   File size: {result['file_size_bytes'] / 1024:.1f} KB")
            print(f"   Uploaded at: {result['uploaded_at']}")
            print(f"   Processed at: {result['processed_at']}")
            
            # Show chunk details
            print(f"\nChunk Details:")
            stats = self.processor.get_chunk_statistics(result['chunks'])
            print(f"   Average chunk length: {stats['avg_chunk_length']:.0f} chars")
            print(f"   Average words per chunk: {stats['avg_words_per_chunk']:.0f}")
            print(f"   Min/Max length: {stats['min_chunk_length']}/{stats['max_chunk_length']} chars")
            
            # Show first chunk preview
            if result['chunks']:
                first_chunk = result['chunks'][0]
                print(f"\nFirst Chunk Preview:")
                print(f"   {'-' * 76}")
                print(f"   {first_chunk['text'][:200]}...")
                print(f"   {'-' * 76}")
                print(f"   Chunk ID: {first_chunk['metadata']['chunk_id']}")
                print(f"   Embedding dimension: {len(first_chunk['embedding'])}")
            
            return result['doc_id']
            
        except Exception as e:
            print(f"\nUpload failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search_documents(
        self,
        query: str,
        collection_name: str = "cli_documents",
        top_k: int = 5,
        score_threshold: float = 0.3
    ):
        """
        Search for documents
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results
            score_threshold: Minimum score
        """
        print("\n" + "=" * 80)
        print("SEARCHING DOCUMENTS")
        print("=" * 80)
        
        print(f"\nQuery: '{query}'")
        print(f"Collection: {collection_name}")
        print(f"Top-K: {top_k}")
        print(f"Score threshold: {score_threshold}")
        
        # Check if collection exists
        if not self.qdrant_manager.collection_exists(collection_name):
            print(f"\nCollection '{collection_name}' not found")
            print("\nAvailable collections:")
            collections = self.qdrant_manager.list_collections()
            if collections:
                for coll in collections:
                    print(f"   - {coll}")
            else:
                print("   (No collections yet)")
            return
        
        # Search
        try:
            results = self.qdrant_manager.search(
                collection_name=collection_name,
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not results:
                print(f"\nNo results found")
                print(f"\nTips:")
                print(f"   - Try lowering score_threshold (current: {score_threshold})")
                print(f"   - Try different query terms")
                print(f"   - Check if documents were uploaded to this collection")
                return
            
            print(f"\nFound {len(results)} results\n")
            
            # Display results
            for i, result in enumerate(results, 1):
                print("=" * 80)
                print(f"Result #{i}")
                print("=" * 80)
                print(f"\nScore: {result['score']:.4f} (0=worst, 1=best)")
                print(f"Point ID: {result['id']}")
                
                # Metadata
                metadata = result['metadata']
                print(f"\nMetadata:")
                print(f"   Doc ID: {metadata.get('doc_id', 'N/A')}")
                print(f"   Filename: {metadata.get('original_filename', 'N/A')}")
                print(f"   Chunk: {metadata.get('chunk_index', 'N/A')} of {metadata.get('total_chunks', 'N/A')}")
                if 'category' in metadata:
                    print(f"   Category: {metadata['category']}")
                print(f"   Word count: {metadata.get('chunk_word_count', 'N/A')}")
                
                # Text content
                print(f"\nText Content:")
                print(f"   {'-' * 76}")
                text = result['text']
                # Wrap text nicely
                words = text.split()
                lines = []
                current_line = "   "
                for word in words:
                    if len(current_line) + len(word) + 1 <= 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.rstrip())
                        current_line = "   " + word + " "
                if current_line.strip():
                    lines.append(current_line.rstrip())
                print("\n".join(lines))
                print(f"   {'-' * 76}")
                print()
            
            # Summary
            print("=" * 80)
            print("SEARCH SUMMARY")
            print("=" * 80)
            avg_score = sum(r['score'] for r in results) / len(results)
            print(f"\nStatistics:")
            print(f"   Total results: {len(results)}")
            print(f"   Average score: {avg_score:.4f}")
            print(f"   Best score: {results[0]['score']:.4f}")
            print(f"   Worst score: {results[-1]['score']:.4f}")
            
        except Exception as e:
            print(f"\nSearch failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Document Loading Pipeline CLI - Test document upload and retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a document
  python cli_pipeline.py upload --file my_document.docx
  
  # Search documents
  python cli_pipeline.py search --query "visa requirements"

        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload and process a document')
    upload_parser.add_argument('--file', '-f', required=True, help='Path to .docx file')
    upload_parser.add_argument('--collection', '-c', default='cli_documents', help='Collection name (default: cli_documents)')
    upload_parser.add_argument('--category', help='Document category (optional)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('--query', '-q', required=True, help='Search query')
    search_parser.add_argument('--collection', '-c', default='cli_documents', help='Collection name (default: cli_documents)')
    search_parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results (default: 5)')
    search_parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Score threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    try:
        cli = DocumentPipelineCLI()
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'upload':
        cli.upload_document(
            file_path=args.file,
            collection_name=args.collection,
            category=args.category
        )
    
    elif args.command == 'search':
        cli.search_documents(
            query=args.query,
            collection_name=args.collection,
            top_k=args.top_k,
            score_threshold=args.threshold
        )
    
    print()  


if __name__ == "__main__":
    main()