#!/usr/bin/env python3
"""
Validate Current QSAR Results

This script takes compounds from your phytochemical mapping and validates
your current QSAR predictions against external databases.
"""

import json
import sys
import os
from qsar_validator import QSARValidator

def load_phytochemical_data():
    """Load compounds from your phytochemical mapping"""
    try:
        with open('data/phytochemical_mapping.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading phytochemical data: {e}")
        return None

def validate_your_qsar_results():
    """Validate your current QSAR model results"""
    
    print("🔬 Validating Your QSAR Results Against External Sources")
    print("=" * 60)
    
    # Load your phytochemical data
    phyto_data = load_phytochemical_data()
    if not phyto_data:
        return
    
    validator = QSARValidator()
    
    # Test with a few key compounds first
    test_compounds = []
    
    # Get compounds that have SMILES data
    for plant, data in list(phyto_data.items())[:5]:  # Test first 5 plants
        for compound in data.get('phytochemicals', []):
            if 'smiles' in compound:
                test_compounds.append({
                    'plant': plant,
                    'name': compound['name'],
                    'smiles': compound['smiles'],
                    'description': compound.get('description', '')
                })
                break  # One compound per plant for testing
    
    print(f"Found {len(test_compounds)} compounds with SMILES for validation\n")
    
    # Run your QSAR model on these compounds and validate
    results = []
    
    for compound in test_compounds:
        print(f"\n{'='*50}")
        print(f"Testing: {compound['name']} from {compound['plant']}")
        print(f"{'='*50}")
        
        # Here you would normally run your QSAR model
        # For now, let's use example predictions (you should replace with actual QSAR predictions)
        example_predictions = {
            'bioactivity_score': 4.5,  # Replace with your model output
            'molecular_weight': 250.0,  # Replace with calculated values
            'logp': 2.0,
            'tpsa': 100.0,
            'hbd': 2,
            'hba': 4,
            'source': 'Your QSAR Model'
        }
        
        # Validate against external sources
        validation_result = validator.validate_compound_properties(
            compound['name'],
            compound['smiles'],
            example_predictions
        )
        
        results.append(validation_result)
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for compound in test_compounds:
        validator.compare_predictions(compound['name'])
    
    # Save comprehensive report
    report_file = validator.generate_validation_report('your_qsar_validation_report.json')
    
    print(f"\n📋 Validation Summary:")
    print(f"✅ Tested {len(test_compounds)} compounds")
    print(f"📄 Full report saved to: {report_file}")
    print(f"\n🔍 External Sources Used:")
    print(f"  • RDKit (Open Source Molecular Descriptors)")
    print(f"  • PubChem (NCBI Database)")
    print(f"  • ChEMBL (Bioactivity Database)")
    print(f"  • SwissADME (Manual Validation)")
    
    return results

def create_validation_checklist():
    """Create a checklist for QSAR validation"""
    checklist = """
# QSAR Validation Checklist

## 📋 Pre-Validation Setup
- [ ] Install required packages: `pip install rdkit requests pandas`
- [ ] Ensure internet connection for database queries
- [ ] Have your SMILES strings ready
- [ ] Document your current QSAR model parameters

## 🔬 Molecular Descriptor Validation
- [ ] Compare Molecular Weight (should match within 1-2%)
- [ ] Validate LogP values (acceptable difference < 0.5)
- [ ] Check TPSA calculations (should be very close)
- [ ] Verify H-bond donor/acceptor counts (should be exact)
- [ ] Confirm aromatic ring counts

## 📊 Bioactivity Validation Sources
- [ ] Search ChEMBL for experimental bioactivity data
- [ ] Check BindingDB for binding affinity data
- [ ] Look up compounds in PubChem BioAssays
- [ ] Search literature (PubMed) for experimental values
- [ ] Compare with other QSAR models if available

## 🧪 Cross-Validation Methods
- [ ] Use SwissADME web tool for independent calculation
- [ ] Try pkCSM for ADMET predictions
- [ ] Use DataWarrior for descriptor calculation
- [ ] Compare with commercial tools (if available)

## 📝 Documentation Requirements
- [ ] Record all external sources used
- [ ] Document differences found
- [ ] Explain any significant discrepancies
- [ ] Note validation dates and versions
- [ ] Keep validation report for peer review

## ⚠️ Red Flags to Watch For
- [ ] Molecular weight differences > 5%
- [ ] LogP differences > 1.0
- [ ] TPSA differences > 20
- [ ] Bioactivity scores with no literature support
- [ ] Unusual descriptor values

## ✅ Publication Requirements
- [ ] External validation for all key compounds
- [ ] Literature citations for experimental data
- [ ] Statistical comparison of predicted vs experimental
- [ ] Clear documentation of model limitations
- [ ] Peer review of validation methodology
"""
    
    with open('qsar_validation_checklist.md', 'w') as f:
        f.write(checklist)
    
    print("📋 Validation checklist created: qsar_validation_checklist.md")

if __name__ == "__main__":
    print("🎯 QSAR Validation for Scientific Publication")
    print("This tool helps ensure your QSAR results are scientifically credible\n")
    
    # Create validation checklist
    create_validation_checklist()
    
    # Run validation
    try:
        results = validate_your_qsar_results()
        
        print(f"\n🎓 Next Steps for Scientific Validation:")
        print(f"1. Review the validation report")
        print(f"2. Follow the checklist in 'qsar_validation_checklist.md'")
        print(f"3. Manually validate key compounds using SwissADME")
        print(f"4. Search literature for experimental bioactivity data")
        print(f"5. Document all validation sources in your research")
        
    except KeyboardInterrupt:
        print(f"\n❌ Validation interrupted by user")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print(f"Make sure you have internet connection and required packages installed")