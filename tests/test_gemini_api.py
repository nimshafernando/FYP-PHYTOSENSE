import google.generativeai as genai

# Configure with the API key directly
genai.configure(api_key="AIzaSyAH7iVGDi-6GHqxGT6vMn8fSR02LQ_FfLc")

print(" Testing Gemini API...")

try:
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(
        "Explain EGFR inhibition in simple academic language."
    )
    
    print(" Response:")
    print(response.text)
    
except Exception as e:
    print(" Error:", e)
    print("Error type:", type(e).__name__)
