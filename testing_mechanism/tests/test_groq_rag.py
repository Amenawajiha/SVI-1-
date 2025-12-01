"""
LangSmith Evaluation for Groq-based RAG Pipeline
=================================================

Tests your existing Groq RAG pipeline with LangSmith tracing.
Does NOT modify any existing code!
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import existing code
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
load_dotenv()

# Import LangSmith
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

# Import your EXISTING code (unchanged!)
from test_api.groq_utils import generate_answer as groq_generate_answer

# LangSmith client
client = Client()

# Load test dataset
def load_test_dataset():
    """Load test cases from JSON"""
    dataset_path = Path(__file__).parent / "test_dataset.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data["test_cases"]

# Wrap your existing function with LangSmith tracing
@traceable(
    name="groq_rag_pipeline",
    run_type="chain",
    project_name="travel-rag-evaluation"
)
def traced_groq_rag(query: str, top_k: int = 3) -> Dict:
    """
    Wrapper around your existing Groq RAG pipeline with LangSmith tracing.
    This does NOT modify your original code!
    """
    start_time = time.time()
    
    # Call your EXISTING function
    result = groq_generate_answer(query, top_k=top_k)
    
    latency = time.time() - start_time
    
    # Add metadata for LangSmith
    return {
        "answer": result["answer"],
        "source_documents": result.get("source_documents", []),
        "model": result.get("model_used", "unknown"),
        "latency_seconds": latency,
        "num_sources": len(result.get("source_documents", []))
    }

# Evaluation function
def evaluate_answer_quality(run, example) -> Dict:
    """
    Custom evaluator for answer quality.
    FREE - doesn't use LLM, just rule-based checks.
    """
    query = example.inputs["query"]
    expected = example.outputs.get("expected", {})
    actual_answer = run.outputs["answer"]
    
    scores = {}
    
    # Check 1: Answer is not empty
    scores["has_answer"] = 1 if actual_answer and len(actual_answer) > 10 else 0
    
    # Check 2: Answer contains expected keywords (if specified)
    expected_contains = expected.get("expected_answer_contains", [])
    if expected_contains:
        found_keywords = sum(
            1 for keyword in expected_contains 
            if keyword.lower() in actual_answer.lower()
        )
        scores["keyword_match"] = found_keywords / len(expected_contains)
    else:
        scores["keyword_match"] = None
    
    # Check 3: Has source documents
    num_sources = run.outputs.get("num_sources", 0)
    scores["has_sources"] = 1 if num_sources > 0 else 0
    
    # Check 4: Response time
    latency = run.outputs.get("latency_seconds", 0)
    scores["fast_response"] = 1 if latency < 5.0 else 0
    
    # Check 5: Ambiguity handling (if marked as ambiguous)
    if expected.get("ambiguous", False):
        # Should ask for clarification
        clarification_words = ["which", "clarify", "specify", "looking for"]
        has_clarification = any(
            word in actual_answer.lower() 
            for word in clarification_words
        )
        scores["ambiguity_detected"] = 1 if has_clarification else 0
    
    # Overall score (average of non-None scores)
    valid_scores = [v for v in scores.values() if v is not None]
    scores["overall"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    return {
        "key": "answer_quality",
        "score": scores["overall"],
        "comment": f"Scores: {scores}"
    }

# Main test runner
def run_evaluation():
    """Run full evaluation with LangSmith"""
    
    print("=" * 80)
    print("🧪 GROQ RAG PIPELINE EVALUATION WITH LANGSMITH")
    print("=" * 80)
    print()
    
    # Load test cases
    test_cases = load_test_dataset()
    print(f"📊 Loaded {len(test_cases)} test cases")
    print()
    
    # Create dataset in LangSmith (only first time)
    dataset_name = "travel-rag-test-cases"
    
    try:
        # Try to get existing dataset
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✅ Using existing dataset: {dataset_name}")
    except:
        # Create new dataset
        print(f"📝 Creating new dataset: {dataset_name}")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test cases for travel booking RAG assistant"
        )
        
        # Add examples to dataset
        for test_case in test_cases:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"query": test_case["query"]},
                outputs={
                    "expected": {
                        "topics": test_case.get("expected_topics", []),
                        "expected_answer_contains": test_case.get("expected_answer_contains", []),
                        "ambiguous": test_case.get("ambiguous", False)
                    }
                },
                metadata={
                    "test_id": test_case["id"],
                    "category": test_case["category"]
                }
            )
        print(f"✅ Added {len(test_cases)} examples to dataset")
    
    print()
    print("🚀 Running evaluation...")
    print("   This will test all queries and send traces to LangSmith")
    print()
    
    # Run evaluation
    results = evaluate(
        lambda inputs: traced_groq_rag(inputs["query"]),
        data=dataset_name,
        evaluators=[evaluate_answer_quality],
        experiment_prefix="groq-rag",
        metadata={
            "model": "groq-llama-3.3-70b",
            "retriever": "chromadb",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )
    
    print()
    print("=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📊 Results Summary:")
    print(f"   Total tests: {len(test_cases)}")
    print(f"   View detailed results at: https://smith.langchain.com/")
    print(f"   Project: travel-rag-evaluation")
    print()
    print("💡 Tips:")
    print("   - Check the LangSmith dashboard for detailed traces")
    print("   - Each query shows: retrieval → LLM call → response")
    print("   - Evaluations show pass/fail for each test case")
    print()
    
    return results

# Interactive testing mode
def interactive_test():
    """Test individual queries with LangSmith tracing"""
    print("=" * 80)
    print("🔍 INTERACTIVE TESTING MODE (with LangSmith Tracing)")
    print("=" * 80)
    print()
    print("Type your queries to test the RAG pipeline.")
    print("Each query will be traced in LangSmith.")
    print("Type 'quit' to exit.")
    print()
    
    while True:
        query = input("\n📝 Your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        print(f"\n🔄 Processing with Groq RAG pipeline...")
        
        try:
            result = traced_groq_rag(query)
            
            print(f"\n{'='*60}")
            print(f"🤖 Answer:")
            print(f"{'='*60}")
            print(result["answer"])
            print()
            
            print(f"📚 Sources: {result['num_sources']} documents")
            print(f"⏱️  Latency: {result['latency_seconds']:.2f}s")
            print(f"🤖 Model: {result['model']}")
            print()
            print(f"📊 View trace in LangSmith: https://smith.langchain.com/")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Groq RAG pipeline with LangSmith")
    parser.add_argument(
        "--mode",
        choices=["evaluate", "interactive"],
        default="evaluate",
        help="Run full evaluation or interactive testing"
    )
    
    args = parser.parse_args()
    
    # Validate LangSmith config
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ ERROR: LANGCHAIN_API_KEY not set!")
        print("   Set it in your .env file or environment variables")
        sys.exit(1)
    
    if args.mode == "evaluate":
        run_evaluation()
    else:
        interactive_test()