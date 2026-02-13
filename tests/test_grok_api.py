import requests

GROK_API_KEY = "your-xai-api-key-here"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-latest"

print("🧪 Testing Grok API...")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROK_API_KEY}"
}

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Say hello and describe what you are in one sentence."
        }
    ],
    "model": GROK_MODEL,
    "stream": False,
    "temperature": 0.7
}

try:
    response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=30)
    
    print(f"\n📊 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"\n✅ Grok Response:\n{message}")
        print("\n🎉 Grok API is working successfully!")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
