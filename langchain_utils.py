# rag_with_langchain.py
"""
RAG helper that:
- re-uses your existing retriever.retrieve_docs(...)
- calls a local LLM (prefer LangChain Ollama wrapper, fallback to HTTP)
- returns answer + structured source_documents
"""

from retriever import retrieve_docs
import os
import requests
import json
from typing import List, Dict

# Config: update if your Ollama host/port differ
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

# Try to import LangChain's Ollama wrapper. If not present, we will fallback to HTTP.
try:
    from langchain_community.llms import Ollama
    LANGCHAIN_OLLAMA_AVAILABLE = True
except Exception:
    LANGCHAIN_OLLAMA_AVAILABLE = False

def build_prompt(question: str, retrieved: List[Dict]) -> str:
    """
    Builds a prompt using the retrieved results.
    Keeps same instruction style as your original rag_with_gemma.py.
    """
    context_parts = []
    for r in retrieved:
        md = r.get("metadata", {})
        data_type = md.get("data_type", "flight")
        section = md.get("section", "unknown_section")
        subsection = md.get("subsection", "unknown_subsection")
        topic = md.get("topic", "unknown_topic")
        q = md.get("question", "")
        a = r.get("content", "")
        score = r.get("relevance_score", 0.0)
        part = f"""Source [{data_type.upper()}]: {section} > {subsection} > {topic}
Q: {q}
A: {a}
Relevance: {score:.4f}
"""
        context_parts.append(part)

    context = "\n".join(context_parts)

    prompt = f"""
You are an AI assistant for a travel booking service that handles both flight and hotel reservations. Understand the user's question and then answer using the provided relevant information. 

Relevant Information: 
{context} 

Instructions: 
1. Base your answer primarily on the provided information. 
2. If the information doesn't fully answer the question, supplement with general knowledge (concise). 
3. Keep the tone professional and helpful. 
4. Format numeric values clearly. 
5. If multiple sources are relevant, combine them coherently and cite where helpful. 
6. Provide a clear, direct and informational answer but it should be relevant to user question.
7. Pay attention to whether the user is asking about flights or hotels, and answer accordingly.

User Question: 
{question} 

Answer:
"""
    return prompt

def call_llm_via_langchain(prompt: str) -> str:
    """Call local Ollama LLM via LangChain wrapper (if installed)."""
    # LANGCHAIN_OLLAMA_AVAILABLE set at import time
    if not LANGCHAIN_OLLAMA_AVAILABLE:
        raise RuntimeError("LangChain Ollama wrapper not available in environment.")
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    # LangChain LLM objects are callable and return a string
    return llm(prompt)

def call_llm_via_http(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_URL) -> str:
    """Fallback call to Ollama HTTP completions endpoint (same as you used)."""
    url = f"{host}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 512
    }
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP error {r.status_code}: {r.text}")
    data = r.json()
    # Ollama HTTP responses usually have choices[0].text
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0].get("text", "")
    # Some setups differ; try 'result' or 'output'
    return data.get("result") or data.get("output") or json.dumps(data)

def generate_answer(question: str, top_k: int = 3) -> Dict:
    """
    Main helper:
    - retrieves docs via your retriever.retrieve_docs
    - builds prompt
    - calls LLM (langchain Ollama if available, else HTTP)
    - returns dict with answer and source_documents (structured)
    """
    # 1) retrieve
    retrieved = retrieve_docs(question, top_k=top_k)

    # 2) build prompt
    prompt = build_prompt(question, retrieved)

    # 3) call LLM (prefer LangChain wrapper)
    llm_text = None
    try:
        if LANGCHAIN_OLLAMA_AVAILABLE:
            llm_text = call_llm_via_langchain(prompt)
        else:
            llm_text = call_llm_via_http(prompt)
    except Exception as e:
        # fall back to HTTP even if LangChain was available but failed
        try:
            llm_text = call_llm_via_http(prompt)
        except Exception as e2:
            llm_text = f"[LLM ERROR] primary error: {e}; fallback error: {e2}"

    # 4) format source_documents for caller (keep metadata + excerpt + score)
    source_documents = []
    for r in retrieved:
        md = r.get("metadata", {})
        doc = {
            "data_type": md.get("data_type", "flight"),
            "section": md.get("section"),
            "subsection": md.get("subsection"),
            "topic": md.get("topic"),
            "question": md.get("question"),
            "content": r.get("content"),
            "relevance_score": r.get("relevance_score")
        }
        source_documents.append(doc)

    return {
        "answer": llm_text.strip() if isinstance(llm_text, str) else str(llm_text),
        "source_documents": source_documents,
        "prompt": prompt  # optionally included for debugging/dev
    }

# Quick CLI test
if __name__ == "__main__":
    tests = [
        # Flight queries
        "What are the cancellation charges for flight booking?",
        "Will I receive a PNR after booking?",
        "How long is my flight itinerary valid?",
        # Hotel queries
        "Is the hotel booking valid for visa submission?",
        "What are hotel cancellation charges?",
        "Can I modify my hotel reservation dates?"
    ]
    for q in tests:
        print("\n" + "="*80)
        print("QUESTION:", q)
        out = generate_answer(q, top_k=3)
        print("\nANSWER:\n", out["answer"])
        print("\nSOURCES:")
        for s in out["source_documents"]:
            data_type = s.get('data_type', 'flight')
            print(f"- [{data_type.upper()}] {s['section']} > {s['subsection']} > {s['topic']} (score {s['relevance_score']:.3f})")
        print("="*80)
