"""
Test script to verify Gemini API is working and generating descriptions
"""

import google.generativeai as genai
from datetime import datetime

def test_gemini_api():
    """Test Gemini API configuration and description generation"""
    
    print("=" * 70)
    print("GEMINI API TEST")
    print("=" * 70)
    
    # Configure API
    print("\n🔧 Step 1: Configuring Gemini API...")
    try:
        API_KEY = "AIzaSyDs0R6k0cNn8EWe5nMRHVWR3Q8HTmNpfxw"
        genai.configure(api_key=API_KEY)
        print("   ✅ API Key configured")
    except Exception as e:
        print(f"   ❌ API configuration failed: {e}")
        return False
    
    # Try to initialize model
    print("\n🤖 Step 2: Initializing Gemini model...")
    model_names = [
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash',
        'gemini-1.0-pro',
        'gemini-pro'
    ]
    
    model = None
    for model_name in model_names:
        try:
            print(f"   Trying {model_name}...")
            model = genai.GenerativeModel(model_name)
            print(f"   ✅ Successfully initialized {model_name}")
            break
        except Exception as e:
            print(f"   ⚠️  {model_name} failed: {e}")
            continue
    
    if not model:
        print("   ❌ No available models found")
        return False
    
    # Test with a sample compound
    print("\n🧪 Step 3: Testing description generation...")
    print("   Sample compound: Quercetin from Tulsi")
    
    compound_name = "Quercetin"
    smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
    plant_name = "Tulsi (Holy Basil)"
    
    prompt = f"""
    As an expert phytochemist and pharmacologist, provide a detailed and engaging scientific description (200-250 words) of this phytochemical compound:

    **Compound:** {compound_name}
    **SMILES:** {smiles}
    **Source Plant:** {plant_name}

    Structure your response to cover:

    1. **Chemical Classification & Structure:**
       - What class does it belong to? (alkaloid, flavonoid, terpenoid, phenolic compound, etc.)
       - Notable structural features from the SMILES representation
    
    2. **Biological Activities & Pharmacology:**
       - Primary therapeutic effects and biological activities
       - Molecular mechanisms of action (receptor interactions, enzyme inhibition, etc.)
       - Specific cellular pathways affected
    
    3. **Medical Applications:**
       - Traditional medicinal uses in herbal medicine
       - Modern clinical applications and research findings
       - Disease conditions it targets (cancer, inflammation, microbial infections, etc.)
    
    4. **Pharmacokinetics & Safety:**
       - Bioavailability and absorption characteristics
       - Known side effects or contraindications
       - Safety profile and therapeutic index
    
    5. **Research & Clinical Status:**
       - Current research developments
       - Clinical trial status if applicable
       - Future therapeutic potential

    Write in clear, scientific language suitable for medical professionals, researchers, and advanced students. Include specific technical terms but make the content accessible. Focus on evidence-based information.
    """
    
    print(f"\n   📝 Sending request to Gemini API...")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        start_time = datetime.now()
        response = model.generate_content(prompt)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if hasattr(response, 'text') and response.text:
            description = response.text.strip()
            
            print(f"\n   ✅ SUCCESS! Generated description in {duration:.2f} seconds")
            print(f"   Description length: {len(description)} characters")
            print(f"   Word count: {len(description.split())} words")
            
            print("\n" + "=" * 70)
            print("GENERATED DESCRIPTION")
            print("=" * 70)
            print(f"\n{description}\n")
            print("=" * 70)
            
            return True
        else:
            print(f"   ❌ Empty response from API")
            if hasattr(response, 'prompt_feedback'):
                print(f"   Feedback: {response.prompt_feedback}")
            return False
            
    except Exception as e:
        print(f"   ❌ API call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_api()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 TEST PASSED: Gemini API is working correctly!")
        print("Your Flask app will be able to generate AI descriptions.")
    else:
        print("❌ TEST FAILED: Gemini API is not working properly.")
        print("Please check your API key and internet connection.")
    print("=" * 70)
