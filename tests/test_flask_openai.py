"""
Test OpenAI API calls from Flask app functions directly
This will help determine if the issue is with the Flask app or the dashboard
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Add current directory to path to import flask_app
sys.path.append(os.getcwd())

def test_flask_functions():
    """Test the actual OpenAI functions from your Flask app"""
    
    print("🔍 Testing Flask App OpenAI Integration...")
    print("=" * 60)
    
    try:
        # Import your Flask app functions
        from flask_app import generate_ai_description, generate_drug_development_assessment
        
        print("✅ Successfully imported Flask functions")
        
        # Test 1: Generate AI Description
        print("\n🧪 Test 1: Testing generate_ai_description...")
        description = generate_ai_description(
            compound_name="Quercetin",
            smiles="C15H10O7", 
            plant_name="Test Plant",
            fallback_description="Fallback description"
        )
        
        print(f"📝 Description Length: {len(description)} characters")
        print(f"📄 First 100 chars: {description[:100]}...")
        
        # Test 2: Generate Drug Development Assessment  
        print("\n💊 Test 2: Testing generate_drug_development_assessment...")
        assessment = generate_drug_development_assessment(
            compound_name="Quercetin",
            molecular_properties={
                "MolWt": "302.24",
                "LogP": "1.99", 
                "NumHDonors": "5",
                "NumHAcceptors": "7"
            }
        )
        
        print(f"📝 Assessment Length: {len(assessment)} characters")
        print(f"📄 First 100 chars: {assessment[:100]}...")
        
        # Check usage stats
        from flask_app import openai_usage_stats
        print(f"\n📊 Usage Stats After Tests:")
        print(f"   🔄 API Calls: {openai_usage_stats['calls']}")
        print(f"   🎯 Tokens Used: {openai_usage_stats['tokens_used']}")
        print(f"   ❌ Errors: {openai_usage_stats['errors']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Flask functions: {e}")
        return False

def check_dashboard_info():
    """Provide information about OpenAI dashboard visibility"""
    
    print("\n" + "=" * 60)
    print("🎯 DASHBOARD VISIBILITY CHECKLIST")
    print("=" * 60)
    
    print("\n🔑 API Key Verification:")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"   ✅ API Key Found: {api_key[:20]}...{api_key[-10:]}")
        print(f"   🔍 Key Length: {len(api_key)} characters")
    else:
        print("   ❌ No API key found!")
    
    print("\n⏰ Dashboard Update Timing:")
    print("   • Real-time: Immediate usage tracking")
    print("   • Dashboard: Can take 5-15 minutes to update")
    print("   • Billing: Updates hourly/daily")
    
    print("\n🏢 Account Verification:")
    print("   1. Go to https://platform.openai.com/usage")
    print("   2. Check you're logged into the SAME account")
    print("   3. Verify this API key belongs to THIS account")
    print("   4. Check the date filter (today's date)")
    
    print("\n💰 Usage Requirements:")
    print("   • Minimum: $0.01 to show in some views")
    print("   • Our test: ~200 tokens = ~$0.0004")
    print("   • May need more usage to appear")

if __name__ == "__main__":
    print("🚀 Flask App OpenAI Integration Test")
    print("=" * 60)
    
    # Test the functions
    success = test_flask_functions()
    
    if success:
        print("\n✅ All tests completed successfully!")
        check_dashboard_info()
        
        print("\n💡 Next Steps:")
        print("   1. If usage stats show > 0, API calls are working")
        print("   2. Dashboard may take time to update")
        print("   3. Try using the Flask app more to generate more usage")
        print("   4. Check dashboard again in 10-15 minutes")
    else:
        print("\n❌ Tests failed - Flask app may have issues")