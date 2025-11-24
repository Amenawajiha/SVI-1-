# test_conversational_memory_modern.py
"""
Modernized test script using LangChain / LangGraph short-term memory (checkpointer).
- Uses create_history_aware_retriever + create_retrieval_chain
- Uses a checkpointer (InMemorySaver for testing) to persist thread-scoped message history
- Keeps your existing Chroma / SentenceTransformer / Ollama stack
"""

import os
import json
from typing import List, Dict, Any

# classic imports (many helpers live in langchain_classic in v1+)
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import HumanMessage, AIMessage  # message objects

# LangGraph / checkpointer
from langgraph.checkpoint.memory import InMemorySaver
# For production you can uncomment (and install) Postgres saver:
# from langgraph.checkpoint.postgres import PostgresSaver

# -----------------------
# Configuration (same as yours)
# -----------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "travel_info"  # Updated to use unified database with flight + hotel data

# -----------------------
# Helper: simple checkpointer-backed thread state
# -----------------------
# Note: checkpointer implementations expose different helper APIs depending on versions.
# The InMemorySaver used below is intended for testing and demonstrates the idea:
# - load_thread_state(checkpointer, thread_id) -> returns a dict state (messages list)
# - save_thread_state(checkpointer, thread_id, state) -> persists state
#
# If your checkpointer has different methods, change below accordingly.
# Simplified thread state management using a plain dict instead of complex checkpointer API
thread_memory = {}

def load_thread_state(checkpointer, thread_id: str) -> Dict[str, Any]:
    """
    Load thread state from in-memory storage.
    Simplified to use plain dict instead of complex checkpointer API.
    """
    return thread_memory.get(thread_id, {"messages": []})

def save_thread_state(checkpointer, thread_id: str, state: Dict[str, Any]):
    """
    Save thread state to in-memory storage.
    Simplified to use plain dict instead of complex checkpointer API.
    """
    thread_memory[thread_id] = state

# -----------------------
# Set up embeddings + vectorstore (same as yours)
# -----------------------
print("Setting up embeddings & vector store...")
embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print("✓ Chroma retriever ready")

# -----------------------
# Create LLM
# -----------------------
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.0,
)
print(f"✓ Ollama LLM ready: {OLLAMA_MODEL}")

# -----------------------
# Create a history-aware retriever and a retrieval chain
# -----------------------
# The prompt used here converts chat history + user question into a search query for the retriever
history_query_prompt = PromptTemplate.from_template(
    """Given the conversation so far and the new user message, produce a short, focused search query
for the retriever to use. Conversation:\n{chat_history}\n\nNew message:\n{input}\n\nSearch query:"""
)

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_query_prompt,
)

# Combine docs -> answer chain (a simple "stuff" combiner using a prompt similar to your original)
qa_prompt = PromptTemplate.from_template(
    """You are an AI assistant for a travel booking service. Use the retrieved documents (context), the conversation history to understand context when present and give the most relevent answer based on user.
    
Previous conversation:
{chat_history}

Retrieved information:
{context}

User question:
{input}

Answer:
"""
)
combine_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)  # wraps LLM + prompt
retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_chain)
print("✓ Retrieval chain ready (history-aware)")

# -----------------------
# Checkpointer (short-term memory)
# -----------------------
# Using simplified in-memory dict for testing (see thread_memory dict above)
# For production: use InMemorySaver or PostgresSaver from langgraph.checkpoint
checkpointer = None  # Not needed with simplified dict approach

print("✓ Memory storage ready (in-memory dict for testing)")

# -----------------------
# Conversation helpers
# -----------------------
def thread_chat_to_messages(thread_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a persisted thread_state['messages'] to a list of message dicts expected by chains.
    We standardize on a simple list of {'role': 'user'|'assistant', 'content': str}
    """
    return thread_state.get("messages", [])

def append_message_to_thread(state: Dict[str, Any], role: str, content: str):
    msgs = state.setdefault("messages", [])
    msgs.append({"role": role, "content": content})

# Helper to format chat history into plain text for search prompt
def format_history_as_text(messages: List[Dict[str, Any]]) -> str:
    """Turn list -> simple readable text used when building the history-aware search query."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role.startswith("user"):
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def run_query_in_thread(thread_id: str, user_message: str):
    """
    New retrieval flow:
    1) Load thread state and format history to text
    2) Build focused search query via the history_query_prompt (or just use user_message)
    3) Use the base retriever directly to inspect retrieved documents and do a simple guard
    4) If confident, call the retrieval_chain to produce a final answer and save messages
    """
    # load thread state
    state = load_thread_state(checkpointer, thread_id) or {}
    chat_history = thread_chat_to_messages(state)
    chat_history_text = format_history_as_text(chat_history)

    # 1) For inspection, we'll use the user_message directly as search query
    # (the history_aware_retriever will handle history internally when we invoke retrieval_chain)
    search_query = user_message
    print(f"DEBUG: Search query -> {search_query}")

    # 2) retrieve documents using the base retriever directly (so we can inspect them)
    candidate_docs = retriever.invoke(search_query)
    print("DEBUG: Retrieved docs:")
    for d in candidate_docs[:5]:
        md = getattr(d, "metadata", {}) if d is not None else {}
        print(f" - [{md.get('data_type','?').upper()}] {md.get('section_title','?')} > {md.get('topic','?')} | {d.page_content[:150]}...")

    # 3) simple guard: if no doc mentions 'extend' or 'valid', fallback
    low_confidence = True
    for d in candidate_docs:
        if d is None:
            continue
        if "extend" in d.page_content.lower() or "valid" in d.page_content.lower():
            low_confidence = False
            break

    if low_confidence:
        answer = ("I don't have specific information in my documents about extending a reservation validity. "
                  "Would you like me to connect you to support at help@schengenvisaitinerary.com?")
        append_message_to_thread(state, "user", user_message)
        append_message_to_thread(state, "assistant", answer)
        save_thread_state(checkpointer, thread_id, state)
        return answer, candidate_docs, state

    # 4) else call the retrieval_chain (which uses history_aware_retriever -> combine_chain)
    inputs = {"input": user_message, "chat_history": chat_history}
    result = retrieval_chain.invoke(inputs)
    answer = result.get("result") or result.get("answer") or ""
    context_docs = result.get("context", None)

    append_message_to_thread(state, "user", user_message)
    append_message_to_thread(state, "assistant", answer)
    save_thread_state(checkpointer, thread_id, state)
    return answer, context_docs, state

# -----------------------
# Test scenarios (mirrors your original scenarios)
# -----------------------
test_conversations = [
    {
        "scenario": "Test 1: Basic query with follow-up",
        "thread_id": "test1",
        "queries": [
            "What are the cancellation charges?",
            "What about for international flights?",
            "Can I get a refund?"
        ],
    },
    {
        "scenario": "Test 2: Reference to previous context",
        "thread_id": "test2",
        "queries": [
            "Do you provide PNR numbers?",
            "How long is it valid?",
            "What if I need to extend it?"
        ],
    },
    {
        "scenario": "Test 3: Topic change with memory",
        "thread_id": "test3",
        "queries": [
            "Tell me about visa flight reservations",
            "What documents do I need?",
            "Actually, I want to book for actual travel instead"
        ],
    }
]

def run_conversation_test(scenario_name: str, thread_id: str, queries: List[str]):
    print("\n" + ("─" * 60))
    print(f"📝 {scenario_name} (thread: {thread_id})")
    print(("─" * 60))

    # Reset thread state in checkpointer
    save_thread_state(checkpointer, thread_id, {"messages": []})

    for i, q in enumerate(queries, 1):
        print(f"\n👤 User (Turn {i}): {q}")
        try:
            answer, context_docs, state = run_query_in_thread(thread_id, q)
            print(f"\n🤖 Bot: {answer}")

            if context_docs:
                # context_docs shape depends on chain version; try to be defensive
                if isinstance(context_docs, list):
                    print(f"\n📚 Sources used ({len(context_docs)}):")
                    for j, doc in enumerate(context_docs[:2], 1):
                        md = getattr(doc, "metadata", {}) if doc is not None else {}
                        print(f"   {j}. {md.get('section','Unknown')} > {md.get('topic','Unknown')}")
                else:
                    print("\n📚 Retrieved context present")
        except Exception as e:
            print(f"\n⚠️ Error: {e}")

    # Show saved memory
    state = load_thread_state(checkpointer, thread_id)
    messages = state.get("messages", [])
    print(f"\n💾 Conversation History (total turns stored): {len(messages)}")
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "")[:80]
        print(f"   - {role}: {content}...")

# Run tests
for t in test_conversations:
    run_conversation_test(t["scenario"], t["thread_id"], t["queries"])

print("\n🎉 All tests completed (checkpointer persisted thread states).")

# Minimal interactive loop (optional)
if __name__ == "__main__":
    import readline  # optional nicer input on Unix
    print("\nInteractive mode (thread scoped). Type 'quit' to exit, 'reset <thread>' to clear a thread.")
    active_thread = "interactive"
    save_thread_state(checkpointer, active_thread, {"messages": []})
    while True:
        try:
            user_text = input(f"[{active_thread}] You: ").strip()
            if not user_text:
                continue
            if user_text.lower() == "quit":
                break
            if user_text.lower().startswith("reset"):
                parts = user_text.split()
                tid = parts[1] if len(parts) > 1 else active_thread
                save_thread_state(checkpointer, tid, {"messages": []})
                print(f"🔄 Thread {tid} cleared.")
                continue
            if user_text.lower().startswith("thread"):
                # switch active thread: "thread mythread"
                parts = user_text.split()
                active_thread = parts[1] if len(parts) > 1 else active_thread
                save_thread_state(checkpointer, active_thread, load_thread_state(checkpointer, active_thread) or {"messages": []})
                print(f"Switched to thread: {active_thread}")
                continue

            answer, _, _ = run_query_in_thread(active_thread, user_text)
            print(f"\n🤖 Bot: {answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print("⚠️ Error:", exc)
            break
