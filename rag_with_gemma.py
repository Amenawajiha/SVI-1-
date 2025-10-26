from retriever import retrieve_docs
import requests

def format_context(results):
    context_parts = []
    for result in results:
        context_part = f"""
Source: {result['metadata']['section']} > {result['metadata']['subsection']} > {result['metadata']['topic']}
Q: {result['metadata']['question']}
A: {result['content']}
Relevance: {result['relevance_score']:.4f}
"""
        context_parts.append(context_part)
    
    return "\n".join(context_parts)

def query_gemma(question, context):

    prompt = f"""
You are an AI assistant for a flight booking service. Understand user's question. Generate the response based on the information relevent to the user question:

Relevant Information:
{context}

Instructions:
1. Base your answer mostly on the provided information 
2. If the information doesn't fully answer the question, try to give an answer based on what you know
3. Be concise and professional but still informative, the user needs clear guidance
4. Format any numerical values clearly
5. If multiple pieces of information are relevant, combine them coherently and generate a comprehensive answer

User Question: {question}

Answer:"""

    response = requests.post(
        "http://127.0.0.1:11434/v1/completions",
        json={
            "model": "gemma3:1b",
            "prompt": prompt,
            "temperature": 0.7,  
            "top_p": 0.95       
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["text"]
    else:
        return f"Error querying Gemma API: {response.status_code}"

def main():
    test_questions = [
        "What do I require for booking?",
        "How is the weather like in New York in December?",
        "Waht is the earliest I need to get to the airport before my flight?"
    ]
    
    print("Testing RAG Pipeline with Gemma:\n")
    for question in test_questions:
        print("\n" + "="*80)
        print(f"\nQuestion: {question}")
        
        print("\nRetrieving relevant information...")
        results = retrieve_docs(question, top_k=3)
        
        context = format_context(results)
        print("\nRetrieved Context:")
        print(context)
        
        print("\nGenerating response...")
        response = query_gemma(question, context)
        
        print("\nGemma's Response:")
        print(response)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()