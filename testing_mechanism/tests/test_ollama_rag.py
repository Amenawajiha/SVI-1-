"""
LangSmith Evaluation for Ollama-based RAG Pipeline
===================================================

Tests your existing Ollama RAG pipeline with LangSmith tracing.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict

# Add parent directory
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, traceable
from langsmith.evaluation import evaluate

# Import your EXISTING Ollama code
from langchain_utils import generate_answer as ollama_generate_answer

client = Client()

@traceable(
    name="ollama_rag_pipeline",
    run_type="chain",
    project_name="travel-rag-evaluation"
)
def traced_ollama_rag(query: str, top_k: int = 3) -> Dict:
    """Wrapper around Ollama RAG with tracing"""
    start_time = time.time()
    
    result = ollama_generate_answer(query, top_k=top_k)
    
    latency = time.time() - start_time
    
    return {
        "answer": result["answer"],
        "source_documents": result.get("source_documents", []),
        "model": "ollama-gemma3:1b",
        "latency_seconds": latency,
        "num_sources": len(result.get("source_documents", []))
    }

def evaluate_answer_quality(run, example) -> Dict:
    """Same evaluator as Groq version"""
    actual_answer = run.outputs["answer"]
    expected = example.outputs.get("expected", {})
    
    scores = {}
    scores["has_answer"] = 1 if actual_answer and len(actual_answer) > 10 else 0
    
    expected_contains = expected.get("expected_answer_contains", [])
    if expected_contains:
        found = sum(1 for kw in expected_contains if kw.lower() in actual_answer.lower())
        scores["keyword_match"] = found / len(expected_contains)
    else:
        scores["keyword_match"] = None
    
    scores["has_sources"] = 1 if run.outputs.get("num_sources", 0) > 0 else 0
    scores["fast_response"] = 1 if run.outputs.get("latency_seconds", 0) < 10.0 else 0
    
    valid_scores = [v for v in scores.values() if v is not None]
    scores["overall"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    return {
        "key": "answer_quality",
        "score": scores["overall"],
        "comment": f"Scores: {scores}"
    }

def run_evaluation():
    """Run Ollama evaluation"""
    print("=" * 80)
    print("🧪 OLLAMA RAG PIPELINE EVALUATION WITH LANGSMITH")
    print("=" * 80)
    print()
    
    dataset_name = "travel-rag-test-cases"
    
    print("🚀 Running evaluation with Ollama...")
    
    results = evaluate(
        lambda inputs: traced_ollama_rag(inputs["query"]),
        data=dataset_name,
        evaluators=[evaluate_answer_quality],
        experiment_prefix="ollama-rag",
        metadata={
            "model": "ollama-gemma3:1b",
            "retriever": "chromadb",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )
    
    print()
    print("✅ EVALUATION COMPLETE!")
    print(f"📊 View results: https://smith.langchain.com/")
    print()

if __name__ == "__main__":
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ ERROR: LANGCHAIN_API_KEY not set!")
        sys.exit(1)
    
    run_evaluation()