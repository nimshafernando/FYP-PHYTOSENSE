from openai import OpenAI

# Using API key directly instead of .env for testing
client = OpenAI(api_key="your-openai-api-key-here")

print("🧪 Testing OpenAI API...")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You explain scientific and machine learning results clearly for academic projects."
            },
            {
                "role": "user",
                "content": "Explain EGFR inhibition in simple academic language."
            }
        ],
        temperature=0.3
    )

    print("✅ Response:")
    print(response.choices[0].message.content)
    
except Exception as e:
    print("❌ Error:", e)
    print("Error type:", type(e).__name__)
