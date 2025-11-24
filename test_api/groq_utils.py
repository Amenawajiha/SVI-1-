import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from retriever import retrieve_docs
from typing import List, Dict, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Groq Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

groq_client = Groq(api_key=GROQ_API_KEY)

def build_prompt(question: str, context: str) -> str:
    """Build prompt with context and question."""
    return f"""You are a helpful travel assistant for flight and hotel bookings.

Context Information:
{context}

User Question: {question}

Instructions:
- Answer based ONLY on the context provided above
- Be concise, clear, and professional
- If the context doesn't contain enough information, say: "I don't have specific information about that. Please contact support at help@schengenvisaitinerary.com"
- Do not make up information
- For booking-related queries, always mention contacting support if needed

Answer:"""

def format_context(results: List[Dict]) -> str:
    """Format retrieved documents into context string."""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        data_type = result.get('data_type', 'general')
        relevance = result.get('relevance_score', 0)
        section = result.get('section', 'N/A')
        subsection = result.get('subsection', 'N/A')
        topic = result.get('topic', 'N/A')
        content = result.get('content', '')
        
        context_parts.append(
            f"Document {i} (Relevance: {relevance:.2f}, Type: {data_type})\n"
            f"Section: {section} > {subsection} > {topic}\n"
            f"{content}"
        )
    
    return "\n\n---\n\n".join(context_parts)

def call_groq_api(prompt: str, model: str = GROQ_MODEL, temperature: float = 0.3) -> str:
    """
    Call Groq API with the given prompt.
    
    Args:
        prompt: The prompt to send
        model: Groq model to use
        temperature: Response randomness (0-1)
    
    Returns:
        Generated text response
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful travel booking assistant. Provide accurate, concise information based on the context given."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"[Groq API Error: {e}]"

def call_llm_for_json(prompt: str, model: str = GROQ_MODEL) -> Dict:
    """
    Call Groq API and expect JSON response.
    
    Args:
        prompt: Prompt expecting JSON response
        model: Groq model to use
    
    Returns:
        Parsed JSON dictionary
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds ONLY in valid JSON format. Do not include markdown code blocks or any text outside the JSON object."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
        
        return json.loads(response_text)
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_text}")
        return {
            "is_ambiguous": False,
            "reasoning": "Failed to parse LLM response",
            "clarification_question": None,
            "clarification_options": []
        }
    except Exception as e:
        print(f"LLM call error: {e}")
        return {
            "is_ambiguous": False,
            "reasoning": f"Error: {str(e)}",
            "clarification_question": None,
            "clarification_options": []
        }

def generate_answer(
    question: str, 
    top_k: int = 3, 
    model: str = GROQ_MODEL,
    context_override: Optional[List[Dict]] = None,
    retrieval_only: bool = False
) -> Dict:
    """
    Generate answer using Groq API and retrieved context.
    
    Args:
        question: User's question
        top_k: Number of documents to retrieve
        model: Groq model to use
        context_override: Pre-retrieved documents to use instead of fetching
        retrieval_only: If True, only retrieve docs without generating answer
    
    Returns:
        Dictionary with answer and source documents
    """
    # 1. Retrieve relevant documents (or use override)
    if context_override:
        results = context_override
    else:
        results = retrieve_docs(question, top_k=top_k)
    
    # 2. If retrieval only, return early
    if retrieval_only:
        return {
            "answer": None,
            "source_documents": results,
            "model_used": model,
            "prompt": None
        }
    
    # 3. Format context
    context = format_context(results)
    
    # 4. Build prompt
    prompt = build_prompt(question, context)
    
    # 5. Call Groq API
    answer = call_groq_api(prompt, model=model)
    
    # 6. Return structured response
    return {
        "answer": answer,
        "source_documents": results,
        "model_used": model,
        "prompt": prompt if os.environ.get("DEBUG_PROMPT") else None
    }

def test_groq():
    """Test Groq integration with sample queries."""
    
    print("=" * 80)
    print("Testing Groq API Integration")
    print("=" * 80)
    
    test_queries = [
        "What are the cancellation charges for flights?",
        "Is hotel booking valid for visa application?",
        "Can I reschedule my flight booking?"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: {query}")
        print("-" * 80)
        
        try:
            result = generate_answer(query, top_k=2)
            
            print(f"\nAnswer (using {result['model_used']}):")
            print(result['answer'])
            
            print(f"\nSources ({len(result['source_documents'])} documents):")
            for i, doc in enumerate(result['source_documents'], 1):
                data_type = doc.get('data_type', 'unknown')
                section = doc.get('section', 'N/A')
                topic = doc.get('topic', 'N/A')
                relevance = doc.get('relevance_score', 0.0)
                
                print(f"   {i}. [{data_type.upper()}] {section} > {topic}")
                print(f"      Relevance: {relevance:.2f}")
            
            print()
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_groq()