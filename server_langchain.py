# server_langchain.py
"""
Simple FastAPI server exposing /query that calls langchain_utils.generate_answer(...)
Returns JSON with conversation_id optional (keeps simple)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from langchain_utils import generate_answer
import os
import time
import csv
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ------------------------
# Logging Configuration
# ------------------------
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
JSONL_PATH = os.path.join(LOG_DIR, "relevance_logs.jsonl")
CSV_PATH = os.path.join(LOG_DIR, "relevance_logs.csv")

# Embedding model to compute similarities (same as your retriever's embedder)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# CSV column definitions
CSV_FIELDS = [
    "timestamp",
    "conversation_id",
    "user_text",
    "bot_text",
    "llm_latency_ms",
    "retriever_scores",   # JSON array
    "top_k_sources",      # JSON array
    "embedding_similarity" # top1
]

# Utility: cosine similarity
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def append_logs_row(row: dict):
    """Append log entry to both JSONL and CSV files"""
    # JSONL
    with open(JSONL_PATH, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV: write header if CSV doesn't exist
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        # Convert lists to JSON strings for CSV fields
        csv_row = {
            k: (json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k]) 
            for k in CSV_FIELDS
        }
        writer.writerow(csv_row)

class QueryReq(BaseModel):
    conversation_id: Optional[int] = None
    text: str
    top_k: Optional[int] = 3
    action: Optional[str] = None   # reserved if you want action-based calls later

class QueryResp(BaseModel):
    conversation_id: Optional[int] = None
    answer: str
    source_documents: list
    suggested_buttons: list = []
    prompt: Optional[str] = None

@app.get("/health")
def health_check():
    """Health check endpoint for Streamlit frontend"""
    return {"status": "healthy"}


# ------------------------
# Ambiguity / Clarification helpers (rule-based)
# ------------------------
def is_ambiguous_rule(text: str) -> bool:
    """
    Conservative rule-based ambiguity detection.
    Returns True for short single-word or very short inputs that
    contain domain keywords or otherwise look ambiguous.
    """
    if not text:
        return True
    t = text.strip().lower()
    # domain keywords often ambiguous
    ambiguous_keywords = {
        "flight", "booking", "visa", "itinerary", "hotel", "pnr",
        "book", "reservation", "ticket", "cancel", "refund",
        "room", "stay", "accommodation", "check-in", "checkout"
    }
    tokens = t.split()

    # If it's a single token and short, treat ambiguous ("flight", "visa", "pnr", "hotel")
    if len(tokens) == 1 and len(t) <= 15:
        return True

    # If short (<= 3 tokens) and contains any ambiguous keyword
    if len(tokens) <= 3 and any(k in t for k in ambiguous_keywords):
        return True

    # otherwise, not ambiguous
    return False


def clarification_payload_for(text: str) -> Dict:
    """
    Produce a clarifying prompt and a set of rule-based suggested buttons.
    The returned structure matches what the /query endpoint expects.
    """
    message = (
        "I might need a little more detail. We provide the following services:\n\n"
        "• Flight itinerary for visa purposes (a reservation/PNR you can use for visa applications)\n"
        "• Flight booking for actual travel (confirmed e-ticket)\n"
        "• Hotel reservations for visa applications\n"
        "• Hotel bookings for actual stays\n\n"
        "Which of these would you like help with?"
    )

    suggested_buttons = [
        {"label": "Flight itinerary for visa", "value": "choose_visa_flight", "type": "flow"},
        {"label": "Flight booking for travel", "value": "choose_travel_flight", "type": "flow"},
        {"label": "Hotel reservation for visa", "value": "choose_visa_hotel", "type": "flow"},
        {"label": "Hotel booking", "value": "choose_hotel", "type": "flow"},
        {"label": "Talk to an agent", "value": "connect_agent", "type": "flow"},
    ]

    return {"answer": message, "suggested_buttons": suggested_buttons}


# ------------------------
# Existing logic: suggestions + main endpoint
# ------------------------
def generate_suggestions(question: str, answer: str) -> list:
    """Generate contextual follow-up suggestions (rule-based)."""
    suggestions = []
    q_lower = (question or "").lower()

    # Rule-based suggestions
    if "cancel" in q_lower or "refund" in q_lower:
        suggestions = [
            {"label": "How to request a refund?", "value": "refund_process"},
            {"label": "What documents are needed?", "value": "refund_docs"}
        ]
    elif "reschedule" in q_lower or "change date" in q_lower or "resched" in q_lower:
        suggestions = [
            {"label": "What are the charges?", "value": "reschedule_charges"},
            {"label": "How long does it take?", "value": "reschedule_time"}
        ]
    elif "pnr" in q_lower or "booking" in q_lower or "e-ticket" in q_lower:
        suggestions = [
            {"label": "Tell me about cancellation", "value": "cancellation"},
            {"label": "How to make changes?", "value": "modifications"}
        ]
    elif "visa" in q_lower:
        suggestions = [
            {"label": "What documents do I need?", "value": "visa_docs"},
            {"label": "How long is it valid?", "value": "visa_validity"}
        ]
    else:
        # Generic suggestions
        suggestions = [
            {"label": "Tell me about cancellation policy", "value": "cancellation"},
            {"label": "How do I book?", "value": "booking_process"}
        ]

    return suggestions


@app.post("/query", response_model=QueryResp)
def query(req: QueryReq):
    # Basic safety: ensure text exists
    user_text = (req.text or "").strip()

    # 1) Rule-based ambiguity check — if ambiguous, return clarification immediately
    if is_ambiguous_rule(user_text):
        payload = clarification_payload_for(user_text)
        return QueryResp(
            conversation_id=req.conversation_id,
            answer=payload["answer"],
            source_documents=[],
            suggested_buttons=payload["suggested_buttons"],
            prompt=(None if not os.environ.get("DEBUG_PROMPT") else None)
        )

    # 2) Compute query embedding for similarity tracking
    query_emb = embed_model.encode(user_text, convert_to_numpy=True)
    
    # 3) Run RAG/LLM with timing
    start_time = time.perf_counter()
    out = generate_answer(user_text, top_k=req.top_k or 3)
    end_time = time.perf_counter()
    llm_latency_ms = int((end_time - start_time) * 1000)

    # 4) Compute relevance scores for retrieved documents
    retriever_scores = []
    top_k_sources = []
    
    for doc in out.get("source_documents", []):
        # Build source identifier
        source_id = f"{doc.get('section', 'unknown')} > {doc.get('subsection', 'unknown')} > {doc.get('topic', 'unknown')}"
        top_k_sources.append(source_id)
        
        # Compute embedding similarity
        doc_content = doc.get("content", "")
        if doc_content:
            doc_emb = embed_model.encode(doc_content, convert_to_numpy=True)
            sim = cosine_sim(query_emb, doc_emb)
            retriever_scores.append(sim)
        else:
            retriever_scores.append(0.0)
    
    embedding_similarity = retriever_scores[0] if len(retriever_scores) > 0 else 0.0

    # 5) Generate contextual (rule-based) follow-ups for the UI
    suggested_buttons = generate_suggestions(user_text, out["answer"])

    # 6) Log the interaction
    log_row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "conversation_id": req.conversation_id,
        "user_text": user_text,
        "bot_text": out["answer"],
        "llm_latency_ms": llm_latency_ms,
        "retriever_scores": retriever_scores,
        "top_k_sources": top_k_sources,
        "embedding_similarity": embedding_similarity
    }
    
    try:
        append_logs_row(log_row)
    except Exception as e:
        # Don't break main flow on logging error
        print(f"Logging error: {e}")

    # 7) Return response
    return QueryResp(
        conversation_id=req.conversation_id,
        answer=out["answer"],
        source_documents=out["source_documents"],
        suggested_buttons=suggested_buttons,
        prompt=(out.get("prompt") if os.environ.get("DEBUG_PROMPT") else None)
    )


if __name__ == "__main__":
    uvicorn.run("server_langchain:app", host="0.0.0.0", port=8000, reload=True)
