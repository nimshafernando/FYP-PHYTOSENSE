#!/usr/bin/env python3
"""Test dynamic value calculations to ensure they vary by compound."""

def test_dynamic_calculations():
    # Test data representing different compounds with different molecular properties
    test_compounds = [
        {
            'name': 'Gallic acid',
            'mw': 170.12,
            'logp': 0.502,
            'tpsa': 97.99,
            'rotatable_bonds': 1,
            'aromatic_rings': 1,
            'bioactivity': 4.826
        },
        {
            'name': 'Quercetin',
            'mw': 302.238,
            'logp': 1.988,
            'tpsa': 131.36,
            'rotatable_bonds': 1,
            'aromatic_rings': 3,
            'bioactivity': 4.796
        },
        {
            'name': 'Large compound',
            'mw': 650.5,
            'logp': 6.2,
            'tpsa': 180.0,
            'rotatable_bonds': 8,
            'aromatic_rings': 4,
            'bioactivity': 3.2
        }
    ]
    
    print("Testing Dynamic Value Calculations")
    print("=" * 50)
    
    for compound in test_compounds:
        print(f"\n🧪 Testing: {compound['name']}")
        print(f"   MW: {compound['mw']}, LogP: {compound['logp']}, TPSA: {compound['tpsa']}")
        
        # Calculate oral bioavailability (same logic as backend)
        oral_bioavailability = 100
        
        if compound['tpsa'] > 140:
            oral_bioavailability -= 30
        elif compound['tpsa'] > 90:
            oral_bioavailability -= 15
            
        if compound['mw'] > 500:
            oral_bioavailability -= 25
        elif compound['mw'] > 400:
            oral_bioavailability -= 10
            
        if compound['logp'] > 5:
            oral_bioavailability -= 20
        elif compound['logp'] < 0:
            oral_bioavailability -= 15
            
        # Natural compounds bonus
        if 'chavicol' in compound['name'].lower() or 'eugenol' in compound['name'].lower():
            oral_bioavailability += 5
            
        oral_bioavailability = max(15, min(95, oral_bioavailability))
        
        # Calculate drug-likeness score
        score = 2.5  # Base score
        
        if 100 <= compound['mw'] <= 400:
            score += 1.0
        elif compound['mw'] <= 500:
            score += 0.5
        else:
            score -= 0.5
        
        if 0 <= compound['logp'] <= 3:
            score += 1.0
        elif compound['logp'] <= 5:
            score += 0.5
        else:
            score -= 0.5
        
        if compound['rotatable_bonds'] <= 5:
            score += 0.5
        elif compound['rotatable_bonds'] <= 10:
            score += 0.2
        
        if 1 <= compound['aromatic_rings'] <= 3:
            score += 0.5
        
        drug_likeness_score = round(min(5.0, max(0.5, score)), 2)
        
        # Calculate toxicity risk
        risk_score = 0
        if compound['mw'] > 600:
            risk_score += 2
        elif compound['mw'] > 400:
            risk_score += 1
            
        if compound['logp'] > 6:
            risk_score += 2
        elif compound['logp'] > 4:
            risk_score += 1
        
        if risk_score <= 0:
            toxicity_risk = 'Low'
        elif risk_score <= 2:
            toxicity_risk = 'Moderate'
        else:
            toxicity_risk = 'High'
        
        # Calculate ADMET rating
        if drug_likeness_score >= 4.0:
            admet_rating = 'Favorable'
        elif drug_likeness_score >= 2.5:
            admet_rating = 'Moderate'
        else:
            admet_rating = 'Poor'
        
        # Calculate safety profile
        if toxicity_risk == 'Low' and drug_likeness_score >= 3.0:
            safety_profile = 'Generally Safe'
        elif toxicity_risk == 'Low' or drug_likeness_score >= 2.0:
            safety_profile = 'Moderate Safety'
        else:
            safety_profile = 'Requires Caution'
        
        # Display results
        print(f"   📊 Results:")
        print(f"      Oral Bioavailability: {oral_bioavailability}%")
        print(f"      Drug-likeness Score: {drug_likeness_score}/5.0")
        print(f"      Toxicity Risk: {toxicity_risk} Risk")
        print(f"      ADMET Rating: {admet_rating}")
        print(f"      Safety Profile: {safety_profile}")

if __name__ == "__main__":
    test_dynamic_calculations()