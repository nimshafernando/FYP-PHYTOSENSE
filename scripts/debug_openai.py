"""
Debug OpenAI API Integration Test
Tests if OpenAI API calls are working and tokens are being counted properly
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def test_openai_connection():
    """Test basic OpenAI API connection and token usage"""
    try:
        print("🔍 Testing OpenAI API Connection...")
        print(f"📱 API Key Present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
        
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ No API key found in environment variables")
            return False
            
        # Make a simple test call
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, this is a test connection' in exactly 5 words."
                }
            ],
            max_tokens=20
        )
        
        print(f"✅ API Response: {response.choices[0].message.content}")
        
        # Check usage information
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Tokens Used: {response.usage.total_tokens}")
            print(f"📊 Prompt Tokens: {response.usage.prompt_tokens}")
            print(f"📊 Completion Tokens: {response.usage.completion_tokens}")
        else:
            print("⚠️ No usage information in response")
            
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return False

def test_phytochemical_description():
    """Test the phytochemical description function"""
    try:
        print("\n🧪 Testing Phytochemical Description...")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert botanist and phytochemist. Provide scientific yet accessible descriptions of plant compounds and their potential medicinal properties."
                },
                {
                    "role": "user",
                    "content": "Describe the phytochemical quercetin in 100 words, focusing on its medicinal properties and sources."
                }
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        description = response.choices[0].message.content.strip()
        print(f"✅ Description Generated: {len(description)} characters")
        print(f"📄 Content Preview: {description[:100]}...")
        
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Tokens Used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phytochemical Test Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 OpenAI API Debug Test Starting...")
    print("=" * 50)
    
    # Test basic connection
    connection_ok = test_openai_connection()
    
    if connection_ok:
        # Test phytochemical description
        test_phytochemical_description()
        
        print("\n" + "=" * 50)
        print("✅ Debug test completed successfully!")
        print("💡 If your dashboard still shows 0 tokens:")
        print("   - Check if you're logged into the correct OpenAI account")
        print("   - Verify the API key belongs to your account")
        print("   - Dashboard updates may take a few minutes")
    else:
        print("\n❌ Connection failed. Please check your API key.")