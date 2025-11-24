# server_groq.py
"""
FastAPI server using Groq API for LLM inference
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import os
import csv
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Import Groq utilities
from groq_utils import generate_answer

app = FastAPI(title="Travel Assistant API (Groq-powered)")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    conversation_id: Optional[str] = None
    text: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    conversation_id: str
    answer: str
    source_documents: List[Dict]
    suggested_buttons: List[Dict] = []
    model_used: str

# Logging setup
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
JSONL_PATH = os.path.join(LOG_DIR, "relevance_logs.jsonl")
CSV_PATH = os.path.join(LOG_DIR, "relevance_logs.csv")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

CSV_FIELDS = [
    "timestamp",
    "conversation_id",
    "user_text",
    "bot_text",
    "llm_latency_ms",
    "model_used",
    "retriever_scores",
    "top_k_sources",
    "embedding_similarity"
]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def append_logs_row(row: dict):
    with open(JSONL_PATH, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        csv_row = {k: (json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k]) for k in CSV_FIELDS}
        writer.writerow(csv_row)

def generate_followups(user_q: str, bot_a: str) -> List[Dict]:
    """Generate contextual follow-up buttons."""
    buttons = []
    
    keywords = {
        "cancel": [
            {"label": "Cancellation Process", "value": "how_to_cancel"},
            {"label": "Refund Policy", "value": "refund_policy"}
        ],
        "book": [
            {"label": "Booking Requirements", "value": "booking_requirements"},
            {"label": "Payment Methods", "value": "payment_methods"}
        ],
        "hotel": [
            {"label": "Hotel Validity", "value": "hotel_validity"},
            {"label": "Documents Needed", "value": "hotel_documents"}
        ],
        "flight": [
            {"label": "Flight Details", "value": "flight_details"},
            {"label": "Ticket Information", "value": "ticket_info"}
        ]
    }
    
    for kw, btns in keywords.items():
        if kw in user_q.lower() or kw in bot_a.lower():
            buttons.extend(btns)
            break
    
    if not buttons:
        buttons = [
            {"label": "Contact Support", "value": "contact_support"},
            {"label": "Ask Another Question", "value": "new_question"}
        ]
    
    return buttons[:3]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "healthy",
        "groq_configured": groq_key_set,
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    }

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Handle user query with Groq API."""
    
    # Generate or use existing conversation ID
    conv_id = req.conversation_id or str(uuid.uuid4())
    
    # Measure latency
    start = time.perf_counter()
    
    # Generate answer using Groq
    result = generate_answer(req.text, top_k=req.top_k or 3)
    
    end = time.perf_counter()
    llm_latency_ms = int((end - start) * 1000)
    
    # Compute similarity scores
    query_emb = embed_model.encode(req.text, convert_to_numpy=True)
    retriever_scores = []
    top_k_sources = []
    
    for doc in result["source_documents"]:
        doc_text = doc.get("content", "")
        doc_emb = embed_model.encode(doc_text, convert_to_numpy=True)
        sim = cosine_sim(query_emb, doc_emb)
        retriever_scores.append(sim)
        
        source_id = f"{doc.get('section', 'N/A')} > {doc.get('topic', 'N/A')}"
        top_k_sources.append(source_id)
    
    embedding_similarity = retriever_scores[0] if retriever_scores else 0.0
    
    # Log interaction
    log_row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "conversation_id": conv_id,
        "user_text": req.text,
        "bot_text": result["answer"],
        "llm_latency_ms": llm_latency_ms,
        "model_used": result.get("model_used", "unknown"),
        "retriever_scores": retriever_scores,
        "top_k_sources": top_k_sources,
        "embedding_similarity": embedding_similarity
    }
    
    try:
        append_logs_row(log_row)
    except Exception as e:
        print("Logging error:", e)
    
    # Generate follow-up suggestions
    suggested = generate_followups(req.text, result["answer"])
    
    return QueryResponse(
        conversation_id=conv_id,
        answer=result["answer"],
        source_documents=result["source_documents"],
        suggested_buttons=suggested,
        model_used=result.get("model_used", "unknown")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)