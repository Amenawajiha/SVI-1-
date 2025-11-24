"""
Enhanced Travel Assistant API with LLM-based Ambiguity Detection
================================================================

Detects ambiguous queries (flight vs hotel, visa vs travel) and asks for clarification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import time
import csv
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
from threading import Lock

# Import utilities
from groq_utils import generate_answer, call_llm_for_json

app = FastAPI(title="Travel Assistant API (Enhanced with Ambiguity Detection)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Configuration
# ------------------------
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
JSONL_PATH = os.path.join(LOG_DIR, "orchestrated_logs.jsonl")
CSV_PATH = os.path.join(LOG_DIR, "orchestrated_logs.csv")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

CSV_FIELDS = [
    "timestamp",
    "conversation_id",
    "user_text",
    "bot_text",
    "llm_latency_ms",
    "retriever_scores",
    "top_k_sources",
    "embedding_similarity",
    "ambiguity_detected",
    "clarification_round"
]

# Conversation state management
conversation_states = {}
conversation_lock = Lock()
CONVERSATION_TTL = 300  # 5 minutes

# ------------------------
# Models
# ------------------------
class QueryReq(BaseModel):
    conversation_id: Optional[str] = None
    text: str
    top_k: Optional[int] = 3
    clarification_response: Optional[str] = None

class QueryResp(BaseModel):
    conversation_id: str
    answer: str
    source_documents: List[Dict] = []
    suggested_buttons: List[Dict] = []
    is_clarification_request: bool = False

# ------------------------
# Utility Functions
# ------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def cleanup_old_conversations():
    """Remove expired conversations"""
    current_time = time.time()
    with conversation_lock:
        expired = [
            conv_id for conv_id, state in conversation_states.items()
            if current_time - state.get("timestamp_unix", 0) > CONVERSATION_TTL
        ]
        for conv_id in expired:
            del conversation_states[conv_id]
        if expired:
            print(f"Cleaned up {len(expired)} expired conversations")

def store_conversation_state(conv_id: str, state: Dict):
    """Store conversation state with timestamp"""
    with conversation_lock:
        state["timestamp_unix"] = time.time()
        conversation_states[conv_id] = state
    cleanup_old_conversations()

def get_conversation_state(conv_id: str) -> Optional[Dict]:
    """Retrieve conversation state"""
    with conversation_lock:
        return conversation_states.get(conv_id)

def append_logs_row(row: dict):
    """Append log entry"""
    try:
        with open(JSONL_PATH, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            csv_row = {
                k: (json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k]) 
                for k in CSV_FIELDS
            }
            writer.writerow(csv_row)
    except Exception as e:
        print(f"Logging error: {e}")

# ------------------------
# LLM-based Ambiguity Detection
# ------------------------
def detect_ambiguity_with_llm(user_query: str, retrieved_docs: List[Dict]) -> Dict:
    """
    Use LLM to detect if query is ambiguous.
    
    Returns:
        {
            "is_ambiguous": bool,
            "clarification_question": str,
            "clarification_options": List[Dict],
            "reasoning": str
        }
    """
    # Build concise context summary
    context_summary = "\n".join([
        f"- {doc.get('data_type', 'general').upper()}: {doc.get('topic', 'N/A')} "
        f"(relevance: {doc.get('relevance_score', 0.0):.2f})"
        for doc in retrieved_docs[:3] if doc
    ])
    
    if not context_summary:
        context_summary = "No relevant documents found"
    
    ambiguity_prompt = f"""Analyze if this travel booking query needs clarification.

OUR SERVICES:
1. FLIGHT/HOTEL for VISA APPLICATION (dummy booking for embassy submission)
2. FLIGHT/HOTEL for ACTUAL TRAVEL (confirmed booking for real trips)

USER QUERY: "{user_query}"

RETRIEVED DOCUMENTS:
{context_summary}

AMBIGUITY RULES:
- Mark as ambiguous ONLY if it's unclear whether user wants VISA service or TRAVEL service
- NOT ambiguous if query mentions: "visa", "embassy", "consulate", "dummy", "actual travel", "confirmed booking"
- Policy/information questions are NOT ambiguous (e.g., "cancellation policy", "how to book")
- If user asks about "flight" or "hotel" without context, it IS ambiguous

EXAMPLES:
Query: "I need a flight booking"
→ AMBIGUOUS (unclear if for visa or travel)

Query: "I need a flight itinerary for my visa application"
→ NOT AMBIGUOUS (clearly for visa)

Query: "What are the cancellation charges?"
→ NOT AMBIGUOUS (information question)

Query: "Can I book a hotel?"
→ AMBIGUOUS (unclear if for visa or travel)

Respond with ONLY this JSON (no markdown, no extra text):
{{
    "is_ambiguous": true,
    "reasoning": "Query doesn't specify if for visa application or actual travel",
    "clarification_question": "I can help you with that! Are you looking for:",
    "clarification_options": [
        {{"label": "✈️ Flight/Hotel for Visa Application (dummy booking)", "value": "visa"}},
        {{"label": "🎫 Flight/Hotel for Actual Travel (confirmed booking)", "value": "travel"}}
    ]
}}

Your response:"""

    try:
        response = call_llm_for_json(ambiguity_prompt)
        
        # Validate response
        if not isinstance(response, dict):
            raise ValueError("Response is not a dictionary")
        
        return {
            "is_ambiguous": response.get("is_ambiguous", False),
            "clarification_question": response.get("clarification_question", "Could you please clarify your question?"),
            "clarification_options": response.get("clarification_options", []),
            "reasoning": response.get("reasoning", "No reasoning provided")
        }
    
    except Exception as e:
        print(f"Ambiguity detection error: {e}")
        # Fallback: not ambiguous
        return {
            "is_ambiguous": False,
            "clarification_question": None,
            "clarification_options": [],
            "reasoning": f"Error during detection: {str(e)}"
        }

def generate_clarified_query(original_query: str, clarification_value: str) -> str:
    """Combine original query with clarification"""
    clarification_map = {
        "visa": "for visa application (dummy booking for embassy)",
        "travel": "for actual travel (confirmed booking)"
    }
    
    clarification_text = clarification_map.get(clarification_value, clarification_value)
    return f"{original_query} - specifically {clarification_text}"

# ------------------------
# Main Endpoints
# ------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_conversations": len(conversation_states),
        "features": ["ambiguity_detection", "conversation_memory"]
    }

@app.post("/query", response_model=QueryResp)
def query(req: QueryReq):
    """Enhanced query endpoint with ambiguity detection"""
    
    try:
        conv_id = req.conversation_id or str(uuid.uuid4())
        user_text = (req.text or "").strip()
        
        # Allow empty text only if clarification is provided
        if not user_text and not req.clarification_response:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        start_time = time.perf_counter()
        
        # ------------------------
        # Case 1: User responded to clarification
        # ------------------------
        if req.clarification_response:
            print(f"Received clarification response: {req.clarification_response}")
            
            state = get_conversation_state(conv_id)
            
            if not state:
                print(f"No conversation state found for {conv_id}")
                print(f"Active conversations: {list(conversation_states.keys())}")
                return QueryResp(
                    conversation_id=conv_id,
                    answer="Session expired or not found. Please ask your question again.",
                    source_documents=[],
                    suggested_buttons=[{"label": "Start Over", "value": "restart"}],
                    is_clarification_request=False
                )
            
            if not state.get("original_query"):
                print(f"State exists but no original_query found")
                return QueryResp(
                    conversation_id=conv_id,
                    answer="Invalid session state. Please ask your question again.",
                    source_documents=[],
                    suggested_buttons=[{"label": "Start Over", "value": "restart"}],
                    is_clarification_request=False
                )
            
            clarification = (req.clarification_response or "").strip()
            if not clarification:
                print(f"Empty clarification received")
                return QueryResp(
                    conversation_id=conv_id,
                    answer=state.get("clarification_question", "Please select an option."),
                    source_documents=[],
                    suggested_buttons=state.get("clarification_options", []),
                    is_clarification_request=True
                )
            
            # Generate clarified query
            original = state["original_query"]
            clarified_query = generate_clarified_query(original, clarification)
            
            print(f"Original: {original}")
            print(f"Clarification: {clarification}")
            print(f"Clarified: {clarified_query}")
            
            # Generate answer with cached context
            try:
                cached_context = state.get("retrieved_context", [])
                print(f"Using {len(cached_context)} cached documents")
                
                out = generate_answer(
                    clarified_query,
                    top_k=req.top_k or 3,
                    context_override=cached_context if cached_context else None
                )
                
                if not out:
                    raise ValueError("generate_answer returned None")
                
                if not out.get("answer"):
                    raise ValueError("No answer in generate_answer output")
                
                print(f"Answer generated: {len(out['answer'])} chars")
                
            except Exception as e:
                print(f"Error in generate_answer: {e}")
                import traceback
                traceback.print_exc()
                
                return QueryResp(
                    conversation_id=conv_id,
                    answer="I encountered an error generating your answer. Please try rephrasing your question.",
                    source_documents=[],
                    suggested_buttons=[
                        {"label": "🔄 Try Again", "value": "retry"},
                        {"label": "📞 Contact Support", "value": "support"}
                    ],
                    is_clarification_request=False
                )
            
            end_time = time.perf_counter()
            llm_latency_ms = int((end_time - start_time) * 1000)
            
            # Clean up state
            with conversation_lock:
                if conv_id in conversation_states:
                    del conversation_states[conv_id]
                    print(f"🧹 Cleaned up conversation state for {conv_id}")
            
            # Log
            log_interaction(
                conv_id,
                f"{original} [clarified: {clarification}]",
                out["answer"],
                llm_latency_ms,
                out.get("source_documents", []),
                clarified_query,
                ambiguity_detected=True,
                clarification_round=2
            )
            
            suggested_buttons = generate_suggestions(clarified_query, out["answer"])
            
            return QueryResp(
                conversation_id=conv_id,
                answer=out["answer"],
                source_documents=out.get("source_documents", []),
                suggested_buttons=suggested_buttons,
                is_clarification_request=False
            )
                
        # ------------------------
        # Case 2: New query - check for ambiguity
        # ------------------------
        
        # Step 1: Retrieve context
        print(f"Processing query: {user_text}")
        retrieval_result = generate_answer(user_text, top_k=req.top_k or 3, retrieval_only=True)
        retrieved_docs = retrieval_result.get("source_documents", [])
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Check ambiguity with LLM
        ambiguity_result = detect_ambiguity_with_llm(user_text, retrieved_docs)
        
        print(f"Ambiguity check: {ambiguity_result['is_ambiguous']} - {ambiguity_result['reasoning']}")
        
        # Step 3: Handle ambiguous query
        if ambiguity_result["is_ambiguous"]:
            store_conversation_state(conv_id, {
                "original_query": user_text,
                "ambiguous": True,
                "clarification_question": ambiguity_result["clarification_question"],
                "clarification_options": ambiguity_result["clarification_options"],
                "retrieved_context": retrieved_docs,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            end_time = time.perf_counter()
            llm_latency_ms = int((end_time - start_time) * 1000)
            
            log_interaction(
                conv_id, user_text, ambiguity_result["clarification_question"],
                llm_latency_ms, retrieved_docs, user_text,
                ambiguity_detected=True, clarification_round=1
            )
            
            clarification_buttons = [
                {"label": opt["label"], "value": opt["value"], "type": "clarification"}
                for opt in ambiguity_result["clarification_options"]
            ]
            
            print(f"Requesting clarification with {len(clarification_buttons)} options")
            
            return QueryResp(
                conversation_id=conv_id,
                answer=ambiguity_result["clarification_question"],
                source_documents=[],
                suggested_buttons=clarification_buttons,
                is_clarification_request=True
            )
        
        # Step 4: Query is clear - generate answer
        print(f"Query is clear, generating answer...")
        out = generate_answer(user_text, top_k=req.top_k or 3)
        
        end_time = time.perf_counter()
        llm_latency_ms = int((end_time - start_time) * 1000)
        
        log_interaction(
            conv_id, user_text, out["answer"], llm_latency_ms,
            out.get("source_documents", []), user_text,
            ambiguity_detected=False, clarification_round=0
        )
        
        suggested_buttons = generate_suggestions(user_text, out["answer"])
        
        return QueryResp(
            conversation_id=conv_id,
            answer=out["answer"],
            source_documents=out.get("source_documents", []),
            suggested_buttons=suggested_buttons,
            is_clarification_request=False
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return QueryResp(
            conversation_id=conv_id if 'conv_id' in locals() else str(uuid.uuid4()),
            answer="I'm experiencing technical difficulties. Please try again or contact support at help@schengenvisaitinerary.com",
            source_documents=[],
            suggested_buttons=[{"label": "Contact Support", "value": "support"}],
            is_clarification_request=False
        )

# ------------------------
# Helper Functions
# ------------------------
def log_interaction(conv_id: str, user_text: str, bot_text: str, 
                   latency_ms: int, source_docs: List[Dict], 
                   query_for_embedding: str,
                   ambiguity_detected: bool = False,
                   clarification_round: int = 0):
    """Log interaction with metrics"""
    
    try:
        query_emb = embed_model.encode(query_for_embedding, convert_to_numpy=True)
        
        retriever_scores = []
        top_k_sources = []
        
        for doc in source_docs:
            section = doc.get('section') or doc.get('data_type', 'unknown')
            topic = doc.get('topic', 'N/A')
            source_id = f"{section} > {topic}"
            top_k_sources.append(source_id)
            
            doc_content = doc.get("content", "")
            if doc_content:
                doc_emb = embed_model.encode(doc_content, convert_to_numpy=True)
                retriever_scores.append(cosine_sim(query_emb, doc_emb))
            else:
                retriever_scores.append(0.0)
        
        log_row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "conversation_id": conv_id,
            "user_text": user_text,
            "bot_text": bot_text[:500],  # Truncate long answers
            "llm_latency_ms": latency_ms,
            "retriever_scores": retriever_scores,
            "top_k_sources": top_k_sources,
            "embedding_similarity": retriever_scores[0] if retriever_scores else 0.0,
            "ambiguity_detected": ambiguity_detected,
            "clarification_round": clarification_round
        }
        
        append_logs_row(log_row)
    
    except Exception as e:
        print(f"Logging error: {e}")

def generate_suggestions(question: str, answer: str) -> List[Dict]:
    """Generate contextual suggestions"""
    q_lower = question.lower()

    if "cancel" in q_lower:
        return [
            {"label": "💰 Refund Process", "value": "refund"},
            {"label": "📄 Required Documents", "value": "cancel_docs"}
        ]
    elif "visa" in q_lower:
        return [
            {"label": "📋 Document Requirements", "value": "visa_docs"},
            {"label": "⏰ Validity Period", "value": "validity"}
        ]
    elif "book" in q_lower:
        return [
            {"label": "💳 Payment Methods", "value": "payment"},
            {"label": "📞 Contact Support", "value": "support"}
        ]
    else:
        return [
            {"label": "❌ Cancellation Policy", "value": "cancel"},
            {"label": "📞 Contact Support", "value": "support"}
        ]

if __name__ == "__main__":
    print("Starting Enhanced Travel Assistant API with Ambiguity Detection")
    print(f"Logs will be saved to: {LOG_DIR}")
    print(f"Server will run on: http://0.0.0.0:8000")
    print(f"API docs: http://0.0.0.0:8000/docs")
    uvicorn.run("server_orchestrated:app", host="0.0.0.0", port=8000, reload=True)