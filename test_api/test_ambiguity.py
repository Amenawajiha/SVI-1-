"""
Test script for ambiguity detection
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_response(resp: dict):
    """Pretty print response"""
    print(f"\n{'='*80}")
    print(f"Bot Response:")
    print(f"{'='*80}")
    
    # Check if response has an error
    if 'detail' in resp:
        print(f"ERROR: {resp['detail']}")
        return
    
    # Check if response has required fields
    if 'answer' not in resp:
        print(f"ERROR: Invalid response format")
        print(f"Response: {json.dumps(resp, indent=2)}")
        return
    
    print(f"Answer: {resp['answer']}")
    
    if resp.get('is_clarification_request'):
        print(f"\nCLARIFICATION REQUESTED")
        print(f"\nOptions:")
        for btn in resp.get('suggested_buttons', []):
            print(f"  - {btn.get('label', 'N/A')}")
    else:
        print(f"\nSources: {len(resp.get('source_documents', []))} documents")
        for i, doc in enumerate(resp.get('source_documents', [])[:2], 1):
            print(f"  {i}. [{doc.get('data_type', 'N/A')}] {doc.get('topic', 'N/A')}")
    
    if resp.get('suggested_buttons') and not resp.get('is_clarification_request'):
        print(f"\nSuggestions:")
        for btn in resp.get('suggested_buttons', []):
            print(f"  - {btn.get('label', 'N/A')}")

def test_query(text: str, conv_id: str = None, clarification: str = None):
    """Send a query to the server"""
    payload = {
        "text": text,
        "top_k": 3
    }
    
    if conv_id:
        payload["conversation_id"] = conv_id
    
    if clarification:
        payload["clarification_response"] = clarification
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        
        # Check status code
        if response.status_code != 200:
            print(f"HTTP {response.status_code}")
            return {"detail": f"HTTP {response.status_code}: {response.text}"}
        
        return response.json()
    
    except Exception as e:
        print(f"Request error: {e}")
        return {"detail": str(e)}

def print_response(resp: dict):
    """Pretty print response"""
    print(f"\n{'='*80}")
    print(f"Bot Response:")
    print(f"{'='*80}")
    print(f"Answer: {resp['answer']}")
    
    if resp.get('is_clarification_request'):
        print(f"\nCLARIFICATION REQUESTED")
        print(f"\nOptions:")
        for btn in resp.get('suggested_buttons', []):
            print(f"  - {btn['label']}")
    else:
        print(f"\nSources: {len(resp.get('source_documents', []))} documents")
        for i, doc in enumerate(resp.get('source_documents', [])[:2], 1):
            print(f"  {i}. [{doc.get('data_type', 'N/A')}] {doc.get('topic', 'N/A')}")
    
    if resp.get('suggested_buttons') and not resp.get('is_clarification_request'):
        print(f"\nSuggestions:")
        for btn in resp.get('suggested_buttons', []):
            print(f"  - {btn['label']}")

def main():
    """Run ambiguity detection tests"""
    
    print("\n" + "="*80)
    print("TESTING AMBIGUITY DETECTION")
    print("="*80)
    
    # Check health
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"\nServer Status: {health['status']}")
    print(f"🔧 Features: {', '.join(health.get('features', []))}")
    
    # Test 1: Ambiguous query (should ask for clarification)
    print("\n\n" + "="*80)
    print("TEST 1: Ambiguous Query (should request clarification)")
    print("="*80)
    print("Query: 'I need a flight booking'")
    
    resp1 = test_query("I need a flight booking")
    conv_id = resp1['conversation_id']
    print_response(resp1)
    
    if resp1.get('is_clarification_request'):
        print("\nPASS: Server correctly detected ambiguity")
        
        # Follow up with clarification
        print("\n\n" + "="*80)
        print("TEST 1b: Providing Clarification")
        print("="*80)
        print("Clarification: 'visa' (for visa application)")
        
        time.sleep(1)
        resp1b = test_query("", conv_id=conv_id, clarification="visa")
        print_response(resp1b)
        
        if not resp1b.get('is_clarification_request'):
            print("\nPASS: Server generated answer after clarification")
        else:
            print("\nFAIL: Server still asking for clarification")
    else:
        print("\nFAIL: Server did not detect ambiguity")
    
    # Test 2: Clear query (should answer directly)
    print("\n\n" + "="*80)
    print("TEST 2: Clear Query (should answer directly)")
    print("="*80)
    print("Query: 'I need a flight itinerary for my visa application'")
    
    time.sleep(1)
    resp2 = test_query("I need a flight itinerary for my visa application")
    print_response(resp2)
    
    if not resp2.get('is_clarification_request'):
        print("\nPASS: Server correctly answered directly")
    else:
        print("\nFAIL: Server asked for clarification when it shouldn't")
    
    # Test 3: Policy question (should answer directly)
    print("\n\n" + "="*80)
    print("TEST 3: Policy Question (should answer directly)")
    print("="*80)
    print("Query: 'What are the cancellation charges?'")
    
    time.sleep(1)
    resp3 = test_query("What are the cancellation charges?")
    print_response(resp3)
    
    if not resp3.get('is_clarification_request'):
        print("\nPASS: Server correctly answered policy question")
    else:
        print("\nFAIL: Server asked for clarification on policy question")
    
    # Test 4: Another ambiguous query
    print("\n\n" + "="*80)
    print("TEST 4: Ambiguous Hotel Query")
    print("="*80)
    print("Query: 'Can I book a hotel?'")
    
    time.sleep(1)
    resp4 = test_query("Can I book a hotel?")
    print_response(resp4)
    
    if resp4.get('is_clarification_request'):
        print("\nPASS: Server correctly detected ambiguity")
    else:
        print("\nWARNING: Server may have assumed context")
    
    print("\n\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server")
        print("Make sure the server is running:")
        print("  python test_api/server_orchestrated.py")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()