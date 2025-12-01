"""
Refactored Conversational Memory Test Script
=============================================

Uses ONLY ConversationBufferWindowMemory from LangChain for memory management.

This is a hybrid approach: LangChain for memory, custom for everything else.
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to import existing utilities
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
load_dotenv()


from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.schema import HumanMessage, AIMessage


from retriever import retrieve_docs  # Your custom ChromaDB retrieval

# ============================================================================
# CONFIGURATION
# ============================================================================
# LLM Configuration (choose one)
USE_GROQ = True  # Set to False to use Ollama

if USE_GROQ:
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set! Set it in .env or environment")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"✓ Using Groq: {GROQ_MODEL}")
else:
    import requests
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
    print(f"✓ Using Ollama: {OLLAMA_MODEL} at {OLLAMA_URL}")

# Memory Configuration
MEMORY_WINDOW_SIZE = 10  # Keep last 10 messages (5 exchanges)
MEMORY_KEY = "chat_history"

# Retrieval Configuration
TOP_K_DOCS = 3
MIN_RELEVANCE_SCORE = 0.5

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
memory_store: Dict[str, ConversationBufferWindowMemory] = {}

def get_or_create_memory(thread_id: str) -> ConversationBufferWindowMemory:
    """
    Get or create ConversationBufferWindowMemory for a thread.
    
    Args:
        thread_id: Unique thread identifier
    
    Returns:
        ConversationBufferWindowMemory instance
    """
    if thread_id not in memory_store:
        memory_store[thread_id] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key=MEMORY_KEY,
            return_messages=True,  # Return Message objects, not strings
            input_key="input",
            output_key="output"
        )
        print(f"📝 Created new memory for thread: {thread_id}")
    
    return memory_store[thread_id]

def get_conversation_history(thread_id: str) -> List[Dict]:
    """
    Get conversation history as list of message dicts.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        List of {role, content} dicts
    """
    memory = get_or_create_memory(thread_id)
    memory_vars = memory.load_memory_variables({})
    messages = memory_vars.get(MEMORY_KEY, [])
    
    # Convert Message objects to simple dicts
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    
    return history

def clear_thread_memory(thread_id: str):
    """Clear memory for a specific thread."""
    if thread_id in memory_store:
        memory_store[thread_id].clear()
        print(f"🧹 Cleared memory for thread: {thread_id}")

def get_memory_summary(thread_id: str) -> Dict:
    """Get summary of memory state."""
    history = get_conversation_history(thread_id)
    return {
        "thread_id": thread_id,
        "num_messages": len(history),
        "num_exchanges": len(history) // 2,
        "window_full": len(history) >= MEMORY_WINDOW_SIZE
    }

# ============================================================================
# CUSTOM RETRIEVAL LOGIC
# ============================================================================
def build_search_query(history: List[Dict], current_query: str) -> str:
    """
    Build enhanced search query using conversation history.
    
    This is YOUR custom logic to resolve pronouns and add context.
    
    Args:
        history: Conversation history
        current_query: Current user query
    
    Returns:
        Enhanced search query
    """
    query_lower = current_query.lower()
    
    # Check for pronouns that need resolution
    pronouns = ["it", "that", "this", "them", "they", "these", "those"]
    has_pronoun = any(pronoun in query_lower.split() for pronoun in pronouns)
    
    if not has_pronoun or not history:
        # No pronoun or no history - use query as-is
        return current_query
    
    # Get last 2-3 exchanges for context
    recent_history = history[-4:] if len(history) > 4 else history
    
    # Extract topic from last assistant message
    last_topic = None
    for msg in reversed(recent_history):
        if msg["role"] == "assistant":
            content = msg["content"].lower()
            # Simple keyword extraction
            if "pnr" in content:
                last_topic = "PNR"
            elif "flight" in content and "reservation" in content:
                last_topic = "flight reservation"
            elif "hotel" in content and "booking" in content:
                last_topic = "hotel booking"
            elif "cancellation" in content:
                last_topic = "cancellation"
            elif "visa" in content:
                last_topic = "visa application"
            
            if last_topic:
                break
    
    # If we found a topic, enhance the query
    if last_topic:
        # Replace pronouns with topic
        enhanced = current_query
        for pronoun in pronouns:
            enhanced = enhanced.replace(f" {pronoun} ", f" {last_topic} ")
            enhanced = enhanced.replace(f" {pronoun.capitalize()} ", f" {last_topic} ")
        
        print(f"🔍 Enhanced query: '{current_query}' → '{enhanced}'")
        return enhanced
    
    # Fallback: combine with last user query
    if len(recent_history) >= 2:
        last_user_msg = None
        for msg in reversed(recent_history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if last_user_msg:
            combined = f"{last_user_msg} {current_query}"
            print(f"🔍 Combined query: '{combined}'")
            return combined
    
    return current_query

def format_context_from_docs(docs: List[Dict]) -> str:
    """
    Format retrieved documents into context string.
    
    Args:
        docs: List of document dicts from your retriever
    
    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        data_type = doc.get('data_type', 'general')
        section = doc.get('section', 'N/A')
        topic = doc.get('topic', 'N/A')
        content = doc.get('content', '')
        relevance = doc.get('relevance_score', 0.0)
        
        context_parts.append(
            f"Document {i} (Relevance: {relevance:.2f}, Type: {data_type.upper()}):\n"
            f"Section: {section} > {topic}\n"
            f"{content}"
        )
    
    return "\n\n---\n\n".join(context_parts)

def check_retrieval_confidence(docs: List[Dict]) -> bool:
    """
    Check if retrieval results are confident enough.
    
    Args:
        docs: Retrieved documents
    
    Returns:
        True if confident, False otherwise
    """
    if not docs:
        return False
    
    # Check if any document meets minimum relevance threshold
    has_confident_doc = any(
        doc.get('relevance_score', 0.0) >= MIN_RELEVANCE_SCORE 
        for doc in docs
    )
    
    return has_confident_doc

# ============================================================================
# CUSTOM LLM CALLS
# ============================================================================
def call_groq_llm(prompt: str) -> tuple[str, list]:
    """
    Direct call to Groq API.
    
    Args:
        prompt: Full prompt to send
    
    Returns:
        Tuple of (Generated answer, messages sent to API)
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful travel booking assistant. Provide accurate, concise information based on the context and conversation history."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=512,
            top_p=1,
            stream=False
        )
        
        return chat_completion.choices[0].message.content, messages
    
    except Exception as e:
        return f"[Error calling Groq API: {e}]", []

def call_ollama_llm(prompt: str) -> tuple[str, list]:
    """
    Direct call to Ollama API.
    
    Args:
        prompt: Full prompt to send
    
    Returns:
        Tuple of (Generated answer, prompt sent to API)
    """
    try:
        request_data = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=request_data,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[No response from Ollama]"), [request_data]
    
    except Exception as e:
        return f"[Error calling Ollama API: {e}]", []

def call_llm(prompt: str) -> tuple[str, list]:
    """
    Call LLM (Groq or Ollama based on configuration).
    
    Args:
        prompt: Full prompt
    
    Returns:
        Tuple of (Generated answer, API messages/request)
    """
    if USE_GROQ:
        return call_groq_llm(prompt)
    else:
        return call_ollama_llm(prompt)

# ============================================================================
# PROMPT BUILDING
# ============================================================================
def build_prompt(context: str, history: List[Dict], query: str) -> str:
    """
    Build complete prompt for LLM.
    
    Args:
        context: Retrieved document context
        history: Conversation history
        query: Current user query
    
    Returns:
        Complete prompt string
    """
    # Format history
    history_text = ""
    if history:
        history_parts = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts)
    
    prompt = f"""You are a helpful travel booking assistant for flight and hotel reservations.

Previous Conversation:
{history_text if history_text else "None"}

Retrieved Information:
{context}

Current User Question:
{query}

Instructions:
- Answer based on the retrieved information and conversation history
- If the user is referring to something from the conversation (using "it", "that", etc.), use the conversation context
- Be concise, clear, and professional
- If the information is not in the retrieved context, say: "I don't have specific information about that. Please contact support at help@schengenvisaitinerary.com"
- Do not make up information

Answer:"""
    
    return prompt

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
def run_query_in_thread(thread_id: str, user_query: str, verbose: bool = True) -> Dict:
    """
    Main orchestration function - runs a query in a conversational thread.
    
    Args:
        thread_id: Thread identifier
        user_query: User's query
        verbose: Print debug info
    
    Returns:
        Dict with answer, docs, metadata
    """
    start_time = time.time()
    
    # Step 1: Load conversation history
    history = get_conversation_history(thread_id)
    memory = get_or_create_memory(thread_id)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 Processing Query in Thread: {thread_id}")
        print(f"{'='*80}")
        print(f"💬 History: {len(history)} messages")
    
    # Step 2: Build search query (with history context)
    search_query = build_search_query(history, user_query)
    
    if verbose:
        print(f"📝 Original query: {user_query}")
        if search_query != user_query:
            print(f"🔍 Enhanced query: {search_query}")
    
    # Step 3: Retrieve documents (YOUR custom retrieval)
    try:
        retrieved_docs = retrieve_docs(search_query, top_k=TOP_K_DOCS)
        
        if verbose:
            print(f"\n📚 Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                data_type = doc.get('data_type', 'unknown')
                topic = doc.get('topic', 'N/A')
                relevance = doc.get('relevance_score', 0.0)
                print(f"   {i}. [{data_type.upper()}] {topic} (score: {relevance:.2f})")
    
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        retrieved_docs = []
    
    # Step 4: Check confidence
    confident = check_retrieval_confidence(retrieved_docs)
    
    if not confident:
        answer = (
            "I don't have specific information about that in my knowledge base. "
            "Would you like me to connect you with support at help@schengenvisaitinerary.com?"
        )
        
        if verbose:
            print(f"\n⚠️ Low confidence retrieval - using fallback response")
        
        # Save to memory
        memory.save_context(
            {"input": user_query},
            {"output": answer}
        )
        
        latency = time.time() - start_time
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "confident": False,
            "latency_seconds": latency,
            "search_query": search_query,
            "api_messages": []  # No API call made for low confidence
        }
    
    # Step 5: Format context
    context = format_context_from_docs(retrieved_docs)
    
    # Step 6: Build prompt
    prompt = build_prompt(context, history, user_query)
    
    if verbose and os.environ.get("DEBUG_PROMPT"):
        print(f"\n📄 Prompt:\n{prompt}\n")
    
    # Step 7: Call LLM (YOUR direct API call)
    if verbose:
        print(f"\n🤖 Calling LLM...")
    
    answer, api_messages = call_llm(prompt)
    
    if verbose:
        print(f"✅ Answer generated ({len(answer)} chars)")
    
    # Step 8: Save to memory
    memory.save_context(
        {"input": user_query},
        {"output": answer}
    )
    
    latency = time.time() - start_time
    
    if verbose:
        print(f"⏱️  Total latency: {latency:.2f}s")
    
    return {
        "answer": answer,
        "source_documents": retrieved_docs,
        "confident": True,
        "latency_seconds": latency,
        "search_query": search_query,
        "prompt": prompt if os.environ.get("DEBUG_PROMPT") else None,
        "api_messages": api_messages  # Add exact messages sent to API
    }

# ============================================================================
# TEST SCENARIOS
# ============================================================================
test_conversations = [
    {
        "scenario": "Test 1: Basic follow-up (cancellation)",
        "thread_id": "test1",
        "queries": [
            "What are the cancellation charges?",
            "What about for international flights?",
            "Can I get a refund?"
        ],
    },
    {
        "scenario": "Test 2: Pronoun resolution (PNR)",
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
    },
    {
        "scenario": "Test 4: Hotel booking context",
        "thread_id": "test4",
        "queries": [
            "Can I book a hotel for visa application?",
            "How long is it valid for?",
            "What if my embassy appointment is later?"
        ],
    },
    {
        "scenario": "Test 5: Payment and rescheduling",
        "thread_id": "test5",
        "queries": [
            "What payment methods do you accept?",
            "Can I reschedule my booking?",
            "Are there any charges for that?"
        ],
    }
]

# Find the run_conversation_test function and replace the metadata section:

def run_conversation_test(scenario: str, thread_id: str, queries: List[str]):
    """
    Run a multi-turn conversation test.
    
    Args:
        scenario: Test scenario name
        thread_id: Thread ID for this test
        queries: List of queries to run
    """
    print("\n" + "=" * 80)
    print(f"📝 {scenario}")
    print(f"🆔 Thread: {thread_id}")
    print("=" * 80)
    
    # Clear thread memory before test
    clear_thread_memory(thread_id)
    
    for turn, query in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"👤 Turn {turn}: {query}")
        print("─" * 80)
        
        try:
            result = run_query_in_thread(thread_id, query, verbose=False)
            
            print(f"\n🤖 Assistant:")
            print(result["answer"])
            
            print(f"\n📊 Metadata:")
            print(f"   - Confident: {result['confident']}")
            print(f"   - Latency: {result['latency_seconds']:.2f}s")
            print(f"   - Sources: {len(result['source_documents'])} documents")
            
            # NEW: Show exact messages sent to Groq/Ollama API
            if result.get('api_messages'):
                print(f"\n📤 Exact API Messages Sent:")
                print(f"{'─'*60}")
                for i, msg in enumerate(result['api_messages'], 1):
                    if USE_GROQ:
                        # Groq uses messages format
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"   Message {i} - Role: {role}")
                        print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
                        print()
                    else:
                        # Ollama uses prompt format
                        print(f"   Request Data:")
                        print(f"   {json.dumps(msg, indent=6)}")
                print(f"{'─'*60}")
            
            # Show actual documents
            if result['source_documents']:
                print(f"\n📚 Retrieved Documents:")
                for i, doc in enumerate(result['source_documents'], 1):
                    data_type = doc.get('data_type', 'unknown')
                    section = doc.get('section', 'N/A')
                    topic = doc.get('topic', 'N/A')
                    relevance = doc.get('relevance_score', 0.0)
                    content_preview = doc.get('content', '')[:150] + "..." if len(doc.get('content', '')) > 150 else doc.get('content', '')
                    
                    print(f"   {i}. [{data_type.upper()}] {section} > {topic}")
                    print(f"      Relevance: {relevance:.3f}")
                    print(f"      Content: {content_preview}")
            
            if result["search_query"] != query:
                print(f"\n🔍 Enhanced query: {result['search_query']}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Show memory summary
    print(f"\n{'─'*80}")
    summary = get_memory_summary(thread_id)
    print(f"💾 Memory Summary:")
    print(f"   - Total messages: {summary['num_messages']}")
    print(f"   - Exchanges: {summary['num_exchanges']}")
    print(f"   - Window full: {summary['window_full']}")
    
    # Show actual stored messages
    history = get_conversation_history(thread_id)
    print(f"\n📜 Stored Conversation:")
    for i, msg in enumerate(history, 1):
        role = msg["role"].capitalize()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"   {i}. {role}: {content}")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================
def interactive_mode():
    """Interactive testing mode."""
    print("\n" + "=" * 80)
    print("🔍 INTERACTIVE MODE - ConversationBufferWindowMemory Test")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your query to chat")
    print("  - 'thread <name>' - Switch to different thread")
    print("  - 'reset' - Clear current thread memory")
    print("  - 'history' - Show conversation history")
    print("  - 'summary' - Show memory summary")
    print("  - 'quit' - Exit")
    print()
    
    active_thread = "interactive"
    clear_thread_memory(active_thread)
    
    while True:
        try:
            user_input = input(f"\n[{active_thread}] You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() == "quit":
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == "reset":
                clear_thread_memory(active_thread)
                print(f"🔄 Thread '{active_thread}' memory cleared")
                continue
            
            elif user_input.lower() == "history":
                history = get_conversation_history(active_thread)
                print(f"\n📜 Conversation History ({len(history)} messages):")
                for i, msg in enumerate(history, 1):
                    role = msg["role"].capitalize()
                    print(f"   {i}. {role}: {msg['content']}")
                continue
            
            elif user_input.lower() == "summary":
                summary = get_memory_summary(active_thread)
                print(f"\n💾 Memory Summary:")
                print(f"   Thread: {summary['thread_id']}")
                print(f"   Messages: {summary['num_messages']}")
                print(f"   Exchanges: {summary['num_exchanges']}")
                print(f"   Window full: {summary['window_full']}")
                continue
            
            elif user_input.lower().startswith("thread "):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    active_thread = parts[1]
                    print(f"📌 Switched to thread: {active_thread}")
                continue
            
            # Regular query
            result = run_query_in_thread(active_thread, user_input, verbose=False)
            
            print(f"\n🤖 Assistant:")
            print(result["answer"])
            
            # Always show API messages in interactive mode
            if result.get('api_messages'):
                print(f"\n📤 API Messages Sent:")
                for i, msg in enumerate(result['api_messages'], 1):
                    if USE_GROQ:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"   [{role.upper()}]: {content[:150]}..." if len(content) > 150 else f"   [{role.upper()}]: {content}")
                    else:
                        print(f"   Prompt length: {len(msg.get('prompt', ''))} chars")
            
            if os.environ.get("DEBUG"):
                print(f"\n📊 Debug:")
                print(f"   Latency: {result['latency_seconds']:.2f}s")
                print(f"   Sources: {len(result['source_documents'])}")
                print(f"   Confident: {result['confident']}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test conversational memory with custom RAG pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "interactive"],
        default="test",
        help="Run automated tests or interactive mode"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        help="Run specific test scenario (1-5)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🧪 CONVERSATIONAL MEMORY TEST SCRIPT")
    print("=" * 80)
    print(f"\n📝 Configuration:")
    print(f"   LLM: {'Groq' if USE_GROQ else 'Ollama'}")
    print(f"   Model: {GROQ_MODEL if USE_GROQ else OLLAMA_MODEL}")
    print(f"   Memory: ConversationBufferWindowMemory (window={MEMORY_WINDOW_SIZE})")
    print(f"   Retrieval: Custom ChromaDB")
    print(f"   Top-K: {TOP_K_DOCS} documents")
    print(f"   Min Relevance: {MIN_RELEVANCE_SCORE}")
    print()
    
    if args.mode == "interactive":
        interactive_mode()
    else:
        # Run tests
        if args.scenario:
            # Run specific scenario
            scenario_idx = args.scenario - 1
            if 0 <= scenario_idx < len(test_conversations):
                test = test_conversations[scenario_idx]
                run_conversation_test(
                    test["scenario"],
                    test["thread_id"],
                    test["queries"]
                )
            else:
                print(f"❌ Invalid scenario number. Choose 1-{len(test_conversations)}")
        else:
            # Run all scenarios
            for test in test_conversations:
                run_conversation_test(
                    test["scenario"],
                    test["thread_id"],
                    test["queries"]
                )
                time.sleep(1)  # Brief pause between tests
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print(f"\n📊 Total threads in memory: {len(memory_store)}")
        for thread_id in memory_store.keys():
            summary = get_memory_summary(thread_id)
            print(f"   - {thread_id}: {summary['num_messages']} messages")
        print()