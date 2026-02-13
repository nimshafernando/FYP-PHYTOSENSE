"""
Simple OpenAI API Usage Test
Direct test of OpenAI calls to check dashboard visibility
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def make_test_calls():
    """Make multiple OpenAI API calls to generate visible usage"""
    
    print("🚀 Making Multiple OpenAI API Calls for Dashboard Visibility")
    print("=" * 70)
    
    # Initialize client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    total_tokens = 0
    total_calls = 0
    
    # Test calls - make several to ensure visibility
    test_prompts = [
        "Describe quercetin in 50 words",
        "What are the medicinal properties of turmeric?", 
        "Explain the antioxidant effects of green tea",
        "Describe the chemical structure of aspirin",
        "What makes ginseng medicinally valuable?"
    ]
    
    print(f"📱 API Key: {os.getenv('OPENAI_API_KEY')[:20]}...{os.getenv('OPENAI_API_KEY')[-10:]}")
    print(f"🎯 Making {len(test_prompts)} API calls...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"🔄 Call {i}/{len(test_prompts)}: {prompt[:30]}...")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical expert. Provide concise, scientific answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            # Track usage
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.total_tokens
                total_tokens += tokens
                total_calls += 1
                
                print(f"   ✅ Success! Tokens: {tokens}")
                print(f"   📝 Response: {response.choices[0].message.content[:60]}...")
            else:
                print("   ⚠️ No usage data returned")
            
            # Small delay between calls
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print("\n" + "=" * 70)
    print("📊 FINAL USAGE SUMMARY")
    print("=" * 70)
    print(f"✅ Successful API Calls: {total_calls}")
    print(f"🎯 Total Tokens Used: {total_tokens}")
    print(f"💰 Estimated Cost: ${total_tokens * 0.000015:.6f}")  # gpt-4o-mini pricing
    
    if total_tokens > 0:
        print("\n🎉 SUCCESS! OpenAI API calls are working!")
        print("📈 Your usage SHOULD appear in the dashboard")
        print("\n🕐 Dashboard Update Timeline:")
        print("   • Real-time API calls: ✅ Working")
        print("   • Usage dashboard: 5-15 minutes")
        print("   • Billing page: 1-24 hours")
        
        print("\n🔍 Check Dashboard Now:")
        print("   1. Go to: https://platform.openai.com/usage")
        print("   2. Set date filter to TODAY")
        print("   3. Look for gpt-4o-mini usage")
        print("   4. Check different time periods (hourly/daily)")
        
        print("\n💡 If STILL not showing:")
        print("   • Wait 15 more minutes")
        print("   • Verify you're in the RIGHT OpenAI account")
        print("   • Check if API key belongs to THIS account")
        print("   • Contact OpenAI support if problem persists")
        
    else:
        print("\n❌ No tokens used - API calls failed!")
        print("🔧 Check your API key and network connection")
    
    return total_tokens > 0

if __name__ == "__main__":
    success = make_test_calls()