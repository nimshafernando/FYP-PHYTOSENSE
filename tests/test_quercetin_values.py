#!/usr/bin/env python3
"""
Quick test to process Quercetin and see the debug output
"""

import requests
import json

# Quercetin SMILES from your phytochemical database
quercetin_smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"

print("🧪 Testing QSAR Analysis for Quercetin")
print("=" * 60)

try:
    response = requests.post(
        'http://127.0.0.1:5000/predict_qsar',
        json={
            'smiles': quercetin_smiles,
            'compound_name': 'Quercetin'
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get('success'):
            print("\n✅ QSAR Prediction Successful!")
            print(f"\nTargets: {result.get('targets')}")
            print(f"Predictions: {result.get('prediction')}")
            
            if result.get('interpretations'):
                interp = result['interpretations']
                if 'bioactivity_score' in interp:
                    bio = interp['bioactivity_score']
                    print(f"\n📊 Bioactivity Score:")
                    print(f"   Value: {bio.get('value')}")
                    print(f"   Level: {bio.get('level')}")
                    
                    # Calculate what drug assessment should show
                    bioactivity_value = bio.get('value', 0)
                    ic50 = pow(10, (7 - bioactivity_value)) / 1000
                    inhibition = min(90, max(10, (bioactivity_value / 10) * 100))
                    binding_affinity = bioactivity_value * 2.3 + 0.25
                    
                    print(f"\n🎯 Expected Drug Assessment Values:")
                    print(f"   IC50: {ic50:.1f} μM")
                    print(f"   Inhibition: {inhibition:.1f}%")
                    print(f"   Binding Affinity: -{binding_affinity:.2f} kcal/mol")
        else:
            print(f"\n❌ QSAR Prediction Failed: {result.get('error')}")
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
print("Check Flask terminal for debug output showing values sent to GPT")
