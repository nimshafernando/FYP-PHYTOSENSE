#!/usr/bin/env python3
"""
Complete trace of Quercetin flow to identify IC50 mismatch
"""

import sys
sys.path.insert(0, '.')

from reference_ic50_data import calibrate_qsar_prediction
import math

def trace_complete_flow():
    """Trace the complete flow for Quercetin from QSAR to Drug Assessment"""
    
    compound_name = "Quercetin"
    
    # Step 1: QSAR Model Prediction
    original_bioactivity = 4.796074867248535  # From your QSAR model
    print("=" * 70)
    print("COMPLETE QUERCETIN FLOW TRACE")
    print("=" * 70)
    
    print(f"\n1️⃣  QSAR MODEL PREDICTION (Raw):")
    print(f"   Bioactivity: {original_bioactivity:.2f}")
    original_ic50 = pow(10, (7 - original_bioactivity)) / 1000
    print(f"   IC50 (uncalibrated): {original_ic50:.1f} μM")
    
    # Step 2: Apply Calibration in Main Route (line 1645-1673)
    print(f"\n2️⃣  CALIBRATION IN MAIN ROUTE:")
    mock_qsar_pred = {'ic50': original_ic50}
    calibration_result = calibrate_qsar_prediction(compound_name, mock_qsar_pred, original_bioactivity)
    
    print(f"   Calibrated: {calibration_result['calibrated']}")
    calibrated_ic50 = calibration_result['ic50']
    calibrated_bioactivity = calibration_result['bioactivity_score']
    print(f"   Calibrated IC50: {calibrated_ic50:.1f} μM")
    print(f"   Calibrated Bioactivity: {calibrated_bioactivity:.2f}")
    
    # Step 3: Update qsar_result["prediction"] with calibrated value
    print(f"\n3️⃣  UPDATE QSAR_RESULT (line 1665):")
    qsar_prediction_array = [calibrated_bioactivity, 6.5, 3.2]  # [bioactivity, drug_likeness, toxicity]
    print(f"   qsar_result['prediction']: {qsar_prediction_array}")
    print(f"   → This is what gets sent to generate_drug_development_assessment()")
    
    # Step 4: Inside generate_drug_development_assessment()
    print(f"\n4️⃣  INSIDE DRUG ASSESSMENT FUNCTION (line 155-175):")
    # Extract bioactivity from qsar_predictions
    bioactivity_value = qsar_prediction_array[0]
    print(f"   Extracted bioactivity_value: {bioactivity_value:.2f}")
    
    # Calculate values for GPT prompt
    inhibition_percentage = min(90, max(10, (bioactivity_value / 10) * 100))
    binding_affinity = bioactivity_value * 2.3 + 0.25
    recalculated_ic50 = pow(10, (7 - bioactivity_value)) / 1000
    
    print(f"   Calculated for GPT Prompt:")
    print(f"      Inhibition: {inhibition_percentage:.1f}%")
    print(f"      Binding Affinity: -{binding_affinity:.2f} kcal/mol")
    print(f"      IC50: {recalculated_ic50:.1f} μM")
    
    # Step 5: Check consistency
    print(f"\n5️⃣  CONSISTENCY CHECK:")
    print(f"   Calibrated IC50 (from step 2): {calibrated_ic50:.1f} μM")
    print(f"   Recalculated IC50 (for GPT): {recalculated_ic50:.1f} μM")
    
    if abs(calibrated_ic50 - recalculated_ic50) < 0.1:
        print(f"   ✅ VALUES MATCH!")
    else:
        print(f"   ❌ MISMATCH DETECTED!")
        print(f"   Difference: {abs(calibrated_ic50 - recalculated_ic50):.1f} μM")
    
    # Step 6: What gets shown in UI
    print(f"\n6️⃣  WHAT UI DISPLAYS:")
    print(f"   QSAR Section (uses calibrated bioactivity {calibrated_bioactivity:.2f}):")
    ui_ic50 = pow(10, (7 - calibrated_bioactivity)) / 1000
    ui_inhibition = min(90, max(10, (calibrated_bioactivity / 10) * 100))
    ui_binding = calibrated_bioactivity * 2.3 + 0.25
    
    print(f"      JavaScript calculates:")
    print(f"      - IC50: {ui_ic50:.1f} μM")
    print(f"      - Inhibition: {ui_inhibition:.1f}%")
    print(f"      - Binding: -{ui_binding:.2f} kcal/mol")
    
    print(f"\n   Drug Assessment Section (from GPT):")
    print(f"      GPT receives in prompt:")
    print(f"      - IC50: {recalculated_ic50:.1f} μM")
    print(f"      - Inhibition: {inhibition_percentage:.1f}%")
    print(f"      - Binding: -{binding_affinity:.2f} kcal/mol")
    
    print(f"\n7️⃣  FINAL VERDICT:")
    if abs(ui_ic50 - recalculated_ic50) < 0.1:
        print(f"   ✅ Both sections should show IC50 = {ui_ic50:.1f} μM")
        print(f"   If GPT text shows different value, GPT is not following instructions")
    else:
        print(f"   ❌ PROBLEM: UI calculates {ui_ic50:.1f} μM but GPT gets {recalculated_ic50:.1f} μM")
        print(f"   This means different bioactivity values are being used!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    trace_complete_flow()
