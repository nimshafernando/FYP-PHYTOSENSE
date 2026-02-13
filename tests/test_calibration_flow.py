#!/usr/bin/env python3
"""
Test to trace the calibration flow and verify value consistency
"""

from reference_ic50_data import calibrate_qsar_prediction
import math

def test_quercetin_flow():
    """Test the complete flow for Quercetin"""
    
    compound_name = "Quercetin"
    
    # Simulate QSAR model prediction (this is what your model predicts)
    original_bioactivity = 4.796074867248535
    
    print("=" * 60)
    print("TESTING QUERCETIN CALIBRATION FLOW")
    print("=" * 60)
    
    print(f"\n1️⃣  ORIGINAL QSAR PREDICTION:")
    print(f"   Bioactivity Score: {original_bioactivity:.2f}")
    
    # Calculate original IC50
    original_ic50 = pow(10, (7 - original_bioactivity)) / 1000
    print(f"   Original IC50: {original_ic50:.1f} μM")
    
    # Apply calibration
    mock_qsar_pred = {'ic50': original_ic50}
    calibration_result = calibrate_qsar_prediction(compound_name, mock_qsar_pred, original_bioactivity)
    
    print(f"\n2️⃣  CALIBRATION APPLIED:")
    print(f"   Calibrated: {calibration_result['calibrated']}")
    print(f"   Calibrated IC50: {calibration_result['ic50']:.1f} μM")
    print(f"   Calibrated Bioactivity: {calibration_result['bioactivity_score']:.2f}")
    print(f"   Classification: {calibration_result['classification']}")
    
    # Now calculate what values we'd get in drug assessment
    calibrated_bioactivity = calibration_result['bioactivity_score']
    
    print(f"\n3️⃣  DRUG ASSESSMENT CALCULATIONS:")
    print(f"   Using Bioactivity: {calibrated_bioactivity:.2f}")
    
    # These are the calculations in generate_drug_development_assessment
    inhibition_percentage = min(90, max(10, (calibrated_bioactivity / 10) * 100))
    binding_affinity = calibrated_bioactivity * 2.3 + 0.25
    recalculated_ic50 = pow(10, (7 - calibrated_bioactivity)) / 1000
    
    print(f"   Inhibition %: {inhibition_percentage:.1f}%")
    print(f"   Binding Affinity: -{binding_affinity:.2f} kcal/mol")
    print(f"   Recalculated IC50: {recalculated_ic50:.1f} μM")
    
    # Check if they match
    print(f"\n4️⃣  VALUE CONSISTENCY CHECK:")
    ic50_match = abs(calibration_result['ic50'] - recalculated_ic50) < 0.1
    print(f"   Calibrated IC50 ({calibration_result['ic50']:.1f}) == Recalculated IC50 ({recalculated_ic50:.1f}): {ic50_match}")
    
    if not ic50_match:
        print(f"\n   ❌ MISMATCH DETECTED!")
        print(f"   Difference: {abs(calibration_result['ic50'] - recalculated_ic50):.1f} μM")
        print(f"\n   💡 EXPLANATION:")
        print(f"   The calibration sets IC50 to {calibration_result['ic50']:.1f} μM")
        print(f"   Then calculates bioactivity from inverse formula: bioactivity = 7 - log10(IC50 * 1000)")
        print(f"   Expected bioactivity: 7 - log10({calibration_result['ic50']:.1f} * 1000) = {7 - math.log10(calibration_result['ic50'] * 1000):.2f}")
        print(f"   Actual bioactivity stored: {calibrated_bioactivity:.2f}")
        print(f"   When we calculate IC50 back from this bioactivity:")
        print(f"   IC50 = pow(10, (7 - {calibrated_bioactivity:.2f})) / 1000 = {recalculated_ic50:.1f} μM")
    else:
        print(f"   ✅ VALUES MATCH PERFECTLY!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_quercetin_flow()
