# test_groq.py
"""
Simple test script for Groq API integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key is set
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not set!")
    print("Set it with: $env:GROQ_API_KEY='your-key-here'")
    exit(1)

print("✅ GROQ_API_KEY is set")
print(f"Using model: {os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')}")

# Now import and test
from groq_utils import test_groq

if __name__ == "__main__":
    test_groq()