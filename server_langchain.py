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

app = FastAPI()

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
        "book", "reservation", "ticket", "cancel", "refund"
    }
    tokens = t.split()

    # If it's a single token and short, treat ambiguous ("flight", "visa", "pnr")
    if len(tokens) == 1 and len(t) <= 12:
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
        "I might need a little more detail. We provide two main services:\n\n"
        "• Flight itinerary for visa purposes (a reservation/PNR you can use for visa applications)\n"
        "• Flight booking for actual travel (confirmed e-ticket)\n\n"
        "Which of these would you like help with?"
    )

    suggested_buttons = [
        {"label": "Flight itinerary for visa", "value": "choose_visa", "type": "flow"},
        {"label": "Flight booking for travel", "value": "choose_travel", "type": "flow"},
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

    # 2) Not ambiguous -> run RAG/LLM as before
    out = generate_answer(user_text, top_k=req.top_k or 3)

    # 3) Generate contextual (rule-based) follow-ups for the UI
    suggested_buttons = generate_suggestions(user_text, out["answer"])

    return QueryResp(
        conversation_id=req.conversation_id,
        answer=out["answer"],
        source_documents=out["source_documents"],
        suggested_buttons=suggested_buttons,
        prompt=(out.get("prompt") if os.environ.get("DEBUG_PROMPT") else None)
    )


if __name__ == "__main__":
    uvicorn.run("server_langchain:app", host="0.0.0.0", port=8000, reload=True)
