#!/usr/bin/env python3
"""
ENHANCED PERFORMANCE TESTING SUITE
Detailed end-to-end workflow timing for screenshot documentation
"""

import requests
import time
import json
from datetime import datetime

class EnhancedPerformanceTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.workflow_times = {}
        self.detailed_results = []
        
    def timestamp(self):
        """Get current timestamp for logging"""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def test_complete_workflow(self, compound_name="Quercetin", smiles="C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"):
        """Test complete end-to-end PhytoSense workflow with detailed timing"""
        
        print("PHYTOSENSE END-TO-END WORKFLOW PERFORMANCE TEST")
        print("=" * 60)
        print(f"Test Started: {self.timestamp()}")
        print(f"Testing Compound: {compound_name}")
        print(f"SMILES: {smiles}")
        print("-" * 60)
        
        workflow_start = time.time()
        step_times = {}
        
        # Step 1: Homepage Load Test
        print(f"[{self.timestamp()}] STEP 1: Homepage Load Test")
        step_start = time.time()
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            step_1_time = time.time() - step_start
            step_times['homepage_load'] = step_1_time
            print(f"[{self.timestamp()}] Homepage Response: {response.status_code} ({step_1_time:.3f}s)")
        except Exception as e:
            step_1_time = time.time() - step_start
            step_times['homepage_load'] = step_1_time
            print(f"[{self.timestamp()}] Homepage Error: {e} ({step_1_time:.3f}s)")
        
        # Step 2: Molecular Descriptor Calculation
        print(f"[{self.timestamp()}] STEP 2: Molecular Descriptor Calculation")
        step_start = time.time()
        try:
            response = requests.post(f"{self.base_url}/calculate_descriptors", 
                                   json={"smiles": smiles}, timeout=15)
            step_2_time = time.time() - step_start
            step_times['molecular_descriptors'] = step_2_time
            print(f"[{self.timestamp()}] Descriptors Response: {response.status_code} ({step_2_time:.3f}s)")
            
            if response.status_code == 200:
                data = response.json()
                descriptor_count = len(data.get('descriptors', {}))
                print(f"[{self.timestamp()}] Descriptors Calculated: {descriptor_count}")
                
        except Exception as e:
            step_2_time = time.time() - step_start
            step_times['molecular_descriptors'] = step_2_time
            print(f"[{self.timestamp()}] Descriptors Error: {e} ({step_2_time:.3f}s)")
        
        # Step 3: QSAR Prediction (Most Critical)
        print(f"[{self.timestamp()}] STEP 3: QSAR Model Prediction")
        step_start = time.time()
        try:
            response = requests.post(f"{self.base_url}/predict_qsar", 
                                   json={"smiles": smiles}, timeout=20)
            step_3_time = time.time() - step_start
            step_times['qsar_prediction'] = step_3_time
            print(f"[{self.timestamp()}] QSAR Response: {response.status_code} ({step_3_time:.3f}s)")
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('prediction', [])
                targets = data.get('targets', [])
                print(f"[{self.timestamp()}] Predictions Generated: {len(predictions)} targets")
                
                # Extract bioactivity for IC50 calculation
                if predictions and len(predictions) > 0:
                    bioactivity = predictions[0]
                    ic50 = pow(10, (7 - bioactivity)) / 1000
                    print(f"[{self.timestamp()}] Bioactivity Score: {bioactivity:.3f}")
                    print(f"[{self.timestamp()}] Calculated IC50: {ic50:.3f} uM")
                
        except Exception as e:
            step_3_time = time.time() - step_start
            step_times['qsar_prediction'] = step_3_time
            print(f"[{self.timestamp()}] QSAR Error: {e} ({step_3_time:.3f}s)")
        
        # Step 4: 3D Structure Generation
        print(f"[{self.timestamp()}] STEP 4: 3D Structure Generation")
        step_start = time.time()
        try:
            response = requests.post(f"{self.base_url}/api/convert_smiles_to_3d", 
                                   json={"compound_name": compound_name, "smiles": smiles}, 
                                   timeout=15)
            step_4_time = time.time() - step_start
            step_times['3d_structure'] = step_4_time
            print(f"[{self.timestamp()}] 3D Structure Response: {response.status_code} ({step_4_time:.3f}s)")
            
            if response.status_code == 200:
                data = response.json()
                mol_block_size = len(data.get('mol_block_3d', ''))
                print(f"[{self.timestamp()}] 3D Structure Size: {mol_block_size} characters")
                
        except Exception as e:
            step_4_time = time.time() - step_start
            step_times['3d_structure'] = step_4_time
            print(f"[{self.timestamp()}] 3D Structure Error: {e} ({step_4_time:.3f}s)")
        
        # Step 5: AutoDock Vina Simulation
        print(f"[{self.timestamp()}] STEP 5: AutoDock Vina Molecular Docking")
        step_start = time.time()
        try:
            response = requests.post(f"{self.base_url}/api/autodock_vina", 
                                   json={"compound_name": compound_name, "smiles": smiles}, 
                                   timeout=25)
            step_5_time = time.time() - step_start
            step_times['autodock_vina'] = step_5_time
            print(f"[{self.timestamp()}] AutoDock Response: {response.status_code} ({step_5_time:.3f}s)")
            
            if response.status_code == 200:
                data = response.json()
                poses = data.get('poses', [])
                print(f"[{self.timestamp()}] Docking Poses Generated: {len(poses)}")
                
                if poses and len(poses) > 0:
                    best_affinity = poses[0].get('binding_affinity', 'N/A')
                    print(f"[{self.timestamp()}] Best Binding Affinity: {best_affinity} kcal/mol")
                
        except Exception as e:
            step_5_time = time.time() - step_start
            step_times['autodock_vina'] = step_5_time
            print(f"[{self.timestamp()}] AutoDock Error: {e} ({step_5_time:.3f}s)")
        
        # Calculate Total Workflow Time
        total_workflow_time = time.time() - workflow_start
        step_times['total_workflow'] = total_workflow_time
        
        print("-" * 60)
        print("WORKFLOW TIMING SUMMARY")
        print("-" * 60)
        print(f"Step 1 - Homepage Load:        {step_times.get('homepage_load', 0):.3f}s")
        print(f"Step 2 - Molecular Descriptors: {step_times.get('molecular_descriptors', 0):.3f}s")
        print(f"Step 3 - QSAR Prediction:      {step_times.get('qsar_prediction', 0):.3f}s")
        print(f"Step 4 - 3D Structure:         {step_times.get('3d_structure', 0):.3f}s")
        print(f"Step 5 - AutoDock Vina:        {step_times.get('autodock_vina', 0):.3f}s")
        print("-" * 60)
        print(f"TOTAL WORKFLOW TIME:           {total_workflow_time:.3f}s")
        print(f"Test Completed: {self.timestamp()}")
        print("=" * 60)
        
        return step_times
    
    def test_batch_compounds(self):
        """Test multiple compounds for performance consistency"""
        
        test_compounds = [
            {"name": "Quercetin", "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"},
            {"name": "Luteolin", "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"},
            {"name": "Apigenin", "smiles": "C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O"}
        ]
        
        print("BATCH COMPOUND PERFORMANCE TEST")
        print("=" * 50)
        
        batch_results = []
        
        for i, compound in enumerate(test_compounds, 1):
            print(f"\nTEST {i}/3: {compound['name']}")
            print("-" * 30)
            
            step_times = self.test_complete_workflow(compound['name'], compound['smiles'])
            
            batch_results.append({
                'compound': compound['name'],
                'times': step_times
            })
            
            if i < len(test_compounds):
                print("\nWaiting 2 seconds before next test...")
                time.sleep(2)
        
        # Summary of all tests
        print("\nBATCH PERFORMANCE SUMMARY")
        print("=" * 50)
        print("Compound          | QSAR (s) | Total (s) | Status")
        print("-" * 50)
        
        for result in batch_results:
            compound = result['compound']
            qsar_time = result['times'].get('qsar_prediction', 0)
            total_time = result['times'].get('total_workflow', 0)
            status = "PASS" if qsar_time < 5.0 else "SLOW"
            print(f"{compound:16} | {qsar_time:8.3f} | {total_time:9.3f} | {status}")
        
        # Calculate averages
        avg_qsar = sum(r['times'].get('qsar_prediction', 0) for r in batch_results) / len(batch_results)
        avg_total = sum(r['times'].get('total_workflow', 0) for r in batch_results) / len(batch_results)
        
        print("-" * 50)
        print(f"AVERAGES:         | {avg_qsar:8.3f} | {avg_total:9.3f} |")
        print("=" * 50)
        
        return batch_results

if __name__ == "__main__":
    tester = EnhancedPerformanceTester()
    
    # Single workflow test
    print("STARTING ENHANCED PERFORMANCE TESTING")
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\n")
    
    # Run single compound test first
    single_result = tester.test_complete_workflow()
    
    print("\n\n")
    
    # Run batch test
    batch_results = tester.test_batch_compounds()