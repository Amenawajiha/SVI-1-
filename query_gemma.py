import json
import requests
from pathlib import Path

file_path = Path("C:/Users/shahe/OneDrive/Documents/Desktop/local_llm/company_data.json")
with open(file_path) as f:
    company_data = json.load(f)

context = json.dumps(company_data, indent=2)
user_question = "Can I reschedule my flight after booking, and what are the charges involved?"

prompt = f"""
You are an AI assistant for Schengen Visa Itinerary. Use the following company data to answer questions accurately.

Company Data:
{context}

Provide a clear, concise, and professional answer to the user's question based on the context provided.
If the information is not available, answer appropriately in a helpful manner.
Do not mention that the response is based on company data. Make it sound like it's coming directly from you.

User Question:
{user_question}

Answer:
"""

response = requests.post(
    "http://127.0.0.1:11434/v1/completions",
    json={"model": "gemma3:1b", "prompt": prompt}
)

data = response.json()
full_output = data["choices"][0]["text"]

print(full_output)
