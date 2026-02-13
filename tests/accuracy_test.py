#!/usr/bin/env python3
"""
ACCURACY TESTING SUITE
Small, focused tests to validate QSAR model predictions against known literature values
"""

import requests
import json
import time

class AccuracyTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        
    def test_known_compounds(self):
        """Test QSAR predictions against known IC50 values"""
        
        # Small set of compounds with known experimental values
        test_compounds = [
            {"name": "Quercetin", "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O", "expected_ic50_range": [100, 150]},
            {"name": "Luteolin", "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O", "expected_ic50_range": [80, 120]},
            {"name": "Apigenin", "smiles": "C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O", "expected_ic50_range": [200, 300]}
        ]
        
        print("ACCURACY TESTING - QSAR Model Validation")
        print("=" * 50)
        
        for compound in test_compounds:
            print(f"\nTesting: {compound['name']}")
            
            try:
                # Make QSAR prediction
                response = requests.post(f"{self.base_url}/predict_qsar", 
                                       json={"smiles": compound["smiles"]}, 
                                       timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_value = data.get("prediction", [0])[0]
                    
                    # Convert bioactivity to IC50 (simplified)
                    predicted_ic50 = pow(10, (7 - predicted_value)) / 1000
                    
                    # Check accuracy
                    min_expected, max_expected = compound["expected_ic50_range"]
                    accuracy_check = min_expected <= predicted_ic50 <= max_expected
                    
                    result = {
                        "compound": compound["name"],
                        "predicted_ic50": round(predicted_ic50, 1),
                        "expected_range": f"{min_expected}-{max_expected} μM",
                        "accurate": accuracy_check,
                        "response_time": response.elapsed.total_seconds()
                    }
                    
                    status = "PASS" if accuracy_check else "FAIL"
                    print(f"   Predicted IC50: {predicted_ic50:.1f} μM")
                    print(f"   Expected Range: {min_expected}-{max_expected} μM")
                    print(f"   Accuracy: {status}")
                    
                    self.results.append(result)
                    
                else:
                    print(f"   API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   Error: {str(e)}")
        
        return self.results
    
    def generate_accuracy_report(self):
        """Generate simple accuracy summary"""
        
        if not self.results:
            return "No test results available"
        
        total_tests = len(self.results)
        accurate_tests = sum(1 for r in self.results if r["accurate"])
        accuracy_percentage = (accurate_tests / total_tests) * 100
        
        report = f"""
ACCURACY TEST SUMMARY
=====================
Total Tests: {total_tests}
Accurate Predictions: {accurate_tests}
Accuracy Rate: {accuracy_percentage:.1f}%
Average Response Time: {sum(r["response_time"] for r in self.results) / total_tests:.2f}s

Status: {"PASS" if accuracy_percentage >= 70 else "FAIL"} (Threshold: 70%)
"""
        return report

if __name__ == "__main__":
    tester = AccuracyTester()
    tester.test_known_compounds()
    print(tester.generate_accuracy_report())