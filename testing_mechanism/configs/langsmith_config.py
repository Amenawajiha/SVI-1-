"""
LangSmith Configuration for RAG Pipeline Observability
========================================================

This file configures LangSmith for tracing and evaluation.
FREE TIER LIMITS:
- 5,000 traces/month
- 1,000 evaluations/month
- 30 days data retention
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from this directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# LangSmith Configuration
LANGSMITH_CONFIG = {
    "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
    "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    "api_key": os.getenv("LANGCHAIN_API_KEY"),
    "project_name": os.getenv("LANGCHAIN_PROJECT", "travel-rag-evaluation"),
}

# Validate configuration
def validate_config():
    """Validate LangSmith configuration"""
    if not LANGSMITH_CONFIG["api_key"]:
        print("⚠️  WARNING: LANGCHAIN_API_KEY not set!")
        print("   LangSmith tracing will be DISABLED")
        LANGSMITH_CONFIG["tracing_enabled"] = False
        return False
    
    if not LANGSMITH_CONFIG["tracing_enabled"]:
        print("ℹ️  LangSmith tracing is DISABLED")
        print("   Set LANGCHAIN_TRACING_V2=true to enable")
        return False
    
    print("✅ LangSmith configuration valid")
    print(f"   Project: {LANGSMITH_CONFIG['project_name']}")
    print(f"   Endpoint: {LANGSMITH_CONFIG['endpoint']}")
    print(f"   API Key: {LANGSMITH_CONFIG['api_key'][:20]}...")
    return True

# Test connection with a simple trace (doesn't need special permissions)
def test_connection():
    """Test connection to LangSmith with a simple trace"""
    try:
        from langsmith import Client
        from langsmith.run_helpers import traceable
        import time
        
        print("\n🔌 Testing connection to LangSmith...")
        
        # Create client with auto_batch_tracing disabled
        client = Client(
            api_url=LANGSMITH_CONFIG["endpoint"],
            api_key=LANGSMITH_CONFIG["api_key"],
            auto_batch_tracing=False  # ← DISABLE BATCH MODE
        )
        
        # Set environment variable to disable batching globally
        os.environ["LANGCHAIN_AUTO_BATCH_TRACING"] = "false"
        
        print("   ℹ️  Batch mode disabled (better for free tier)")
        
        # Create a simple test trace
        @traceable(
            name="connection_test",
            project_name=LANGSMITH_CONFIG["project_name"],
            client=client
        )
        def test_function():
            time.sleep(0.1)  # Simulate some work
            return {"status": "success", "message": "LangSmith is working!"}
        
        # Execute test function (creates a trace)
        result = test_function()
        
        # Give it a moment to upload
        print("   ⏳ Waiting for trace to upload...")
        time.sleep(2)
        
        print(f"✅ Connected to LangSmith successfully!")
        print(f"   Result: {result}")
        print(f"   Test trace created in project: {LANGSMITH_CONFIG['project_name']}")
        print(f"\n📊 View trace at: https://smith.langchain.com/")
        print(f"   Project: {LANGSMITH_CONFIG['project_name']}")
        print(f"\n💡 Note: Traces may take 5-10 seconds to appear in dashboard")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to LangSmith: {e}")
        print(f"\n🔍 Debugging info:")
        print(f"   - API Key length: {len(LANGSMITH_CONFIG['api_key']) if LANGSMITH_CONFIG['api_key'] else 0}")
        print(f"   - Endpoint: {LANGSMITH_CONFIG['endpoint']}")
        print(f"   - Project: {LANGSMITH_CONFIG['project_name']}")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Check if API key is correct")
        print(f"   2. Verify API key has 'tracing' permissions")
        print(f"   3. Try regenerating API key at: https://smith.langchain.com/settings")
        return False

def get_client():
    """Get a LangSmith client with proper configuration"""
    from langsmith import Client
    
    return Client(
        api_url=LANGSMITH_CONFIG["endpoint"],
        api_key=LANGSMITH_CONFIG["api_key"],
        auto_batch_tracing=False  # Disable batch mode for free tier
    )

if __name__ == "__main__":
    print("=" * 60)
    print("LangSmith Configuration Test")
    print("=" * 60)
    
    if validate_config():
        print()
        test_connection()
    else:
        print("\n⚠️  Configuration validation failed!")
        print("   Please check your .env file")