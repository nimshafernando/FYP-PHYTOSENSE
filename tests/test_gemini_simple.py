"""
Test Gemini API with retry logic and rate limit handling
"""

import google.generativeai as genai
from datetime import datetime
import time

def test_gemini_with_retry():
    """Test Gemini API with retry on rate limit"""
    
    print("=" * 70)
    print("GEMINI API TEST WITH RETRY")
    print("=" * 70)
    
    # Configure API
    print("\n🔧 Configuring Gemini API...")
    API_KEY = "AIzaSyDs0R6k0cNn8EWe5nMRHVWR3Q8HTmNpfxw"
    genai.configure(api_key=API_KEY)
    print("   ✅ API Key configured")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("   ✅ Model initialized")
    
    # Simple test prompt
    prompt = """Briefly explain what Quercetin is in 50 words."""
    
    print("\n🧪 Testing API with simple prompt...")
    print("   Waiting 10 seconds to avoid rate limit...")
    time.sleep(10)
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n   Attempt {attempt}/{max_retries}...")
            start = time.time()
            response = model.generate_content(prompt)
            duration = time.time() - start
            
            if hasattr(response, 'text') and response.text:
                print(f"   ✅ SUCCESS in {duration:.2f}s!")
                print(f"\n   Response:\n   {response.text}\n")
                return True
            else:
                print(f"   ⚠️  Empty response")
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error: {type(e).__name__}")
            
            if "429" in error_msg or "RATE_LIMIT" in error_msg:
                print(f"   ⏳ Rate limit hit. Waiting 30 seconds...")
                time.sleep(30)
            elif "quota" in error_msg.lower():
                print(f"   💡 Quota issue detected")
                print(f"   Full error: {error_msg[:200]}...")
                break
            else:
                print(f"   Error: {error_msg[:200]}...")
                break
    
    return False

if __name__ == "__main__":
    print("\n⏰ Current time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    success = test_gemini_with_retry()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 API IS WORKING!")
    else:
        print("❌ API TEST FAILED")
        print("\n💡 Troubleshooting:")
        print("   1. Check if daily quota (1500 req/day) is exhausted")
        print("   2. Wait a few minutes for rate limit reset")
        print("   3. Verify API key is valid and enabled")
        print("   4. Check https://aistudio.google.com/app/apikey")
    print("=" * 70)
