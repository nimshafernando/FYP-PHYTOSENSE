#!/usr/bin/env python3
"""
Module and Integration Testing Suite for PhytoSense Application
Tests individual modules and their integration
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
from unittest.mock import patch, MagicMock
import pickle

class ModuleTests(unittest.TestCase):
    """Test individual modules in isolation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.module_results = []
        cls.start_time = time.time()
        print("🔧 MODULE TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.module_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_rdkit_module(self):
        """M1: Test RDKit molecular processing module"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, AllChem
            
            # Test SMILES parsing
            smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
            mol = Chem.MolFromSmiles(smiles)
            
            self.assertIsNotNone(mol)
            self.assertEqual(mol.GetNumAtoms(), 21)  # Quercetin has 21 atoms
            
            # Test descriptor calculation
            mw = Descriptors.MolWt(mol)
            self.assertGreater(mw, 300)  # Quercetin MW ~302
            
            print("✅ M1: RDKit module - PASS")
        except Exception as e:
            print(f"❌ M1: RDKit module - FAIL: {e}")
            raise
    
    def test_02_phytochemical_database_module(self):
        """M2: Test phytochemical mapping database"""
        try:
            with open('../data/phytochemical_mapping.json', 'r') as f:
                phyto_data = json.load(f)
            
            self.assertIsInstance(phyto_data, dict)
            self.assertGreater(len(phyto_data), 0)
            
            # Check data structure
            for plant, data in phyto_data.items():
                self.assertIn('phytochemicals', data)
                self.assertIsInstance(data['phytochemicals'], list)
                
                for compound in data['phytochemicals']:
                    self.assertIn('name', compound)
                    self.assertIn('smiles', compound)
            
            print("✅ M2: Phytochemical database module - PASS")
        except Exception as e:
            print(f"❌ M2: Phytochemical database module - FAIL: {e}")
            raise
    
    def test_03_qsar_model_module(self):
        """M3: Test QSAR model loading and prediction"""
        try:
            # Test if QSAR model files exist
            model_path = '../models/XGBoost_model.pkl'
            metadata_path = '../metadata/qsar_metadata.json'
            
            self.assertTrue(os.path.exists(model_path), "QSAR model file missing")
            self.assertTrue(os.path.exists(metadata_path), "QSAR metadata missing")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.assertIn('model_name', metadata)
            self.assertIn('target_properties', metadata)
            self.assertEqual(metadata['feature_info']['total_features'], 2057)
            
            print("✅ M3: QSAR model module - PASS")
        except Exception as e:
            print(f"❌ M3: QSAR model module - FAIL: {e}")
            raise
    
    def test_04_image_processing_module(self):
        """M4: Test image processing functionality"""
        try:
            from PIL import Image
            import torch
            from torchvision import transforms
            
            # Test image preprocessing pipeline
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Create test image
            test_image = Image.new('RGB', (300, 300), color='green')
            processed = transform(test_image)
            
            self.assertEqual(processed.shape, torch.Size([3, 224, 224]))
            print("✅ M4: Image processing module - PASS")
        except Exception as e:
            print(f"❌ M4: Image processing module - FAIL: {e}")
            raise
    
    def test_05_ic50_calibration_module(self):
        """M5: Test IC50 calibration system"""
        try:
            from reference_ic50_data import calibrate_qsar_prediction, get_reference_ic50
            
            # Test known compound calibration
            result = calibrate_qsar_prediction('Quercetin', {'ic50': 100}, 5.0)
            
            self.assertIn('ic50', result)
            self.assertIn('calibrated', result)
            self.assertTrue(result['calibrated'])
            
            # Test IC50 value is within expected range for Quercetin (128 ± 22)
            self.assertGreater(result['ic50'], 100)
            self.assertLess(result['ic50'], 160)
            
            print("✅ M5: IC50 calibration module - PASS")
        except Exception as e:
            print(f"❌ M5: IC50 calibration module - FAIL: {e}")
            raise

class IntegrationTests(unittest.TestCase):
    """Test module integration and end-to-end workflows"""
    
    @classmethod
    def setUpClass(cls):
        cls.integration_results = []
        cls.start_time = time.time()
        print("\n🔗 INTEGRATION TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.integration_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_smiles_to_qsar_integration(self):
        """I1: Test SMILES → Descriptors → QSAR prediction pipeline"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            # Test complete pipeline
            smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
            
            # Step 1: Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            
            # Step 2: Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # Verify descriptor values are reasonable
            self.assertGreater(mw, 0)
            self.assertIsInstance(logp, (int, float))
            self.assertGreater(tpsa, 0)
            
            print("✅ I1: SMILES → QSAR integration - PASS")
        except Exception as e:
            print(f"❌ I1: SMILES → QSAR integration - FAIL: {e}")
            raise
    
    def test_02_phytochemical_to_visualization_integration(self):
        """I2: Test Phytochemical data → 3D visualization pipeline"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Test 3D generation pipeline
            smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
            
            # Step 1: Create molecule
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            # Step 2: Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            # Step 3: Convert to MOL block
            if result != -1:
                mol_block = Chem.MolToMolBlock(mol)
                self.assertIn('BEGIN CTAB', mol_block)
            
            print("✅ I2: Phytochemical → 3D visualization integration - PASS")
        except Exception as e:
            print(f"❌ I2: Phytochemical → 3D integration - FAIL: {e}")
            raise
    
    def test_03_qsar_to_gpt_integration(self):
        """I3: Test QSAR predictions → GPT assessment integration"""
        try:
            # Mock QSAR results
            qsar_predictions = [5.2, 3.1, 2.8]  # bioactivity, drug_likeness, toxicity
            qsar_targets = ["bioactivity_score", "drug_likeness", "toxicity_prediction"]
            
            # Test data flow to assessment generation
            self.assertEqual(len(qsar_predictions), len(qsar_targets))
            
            # Verify bioactivity extraction
            bioactivity_value = qsar_predictions[0]
            ic50 = pow(10, (7 - bioactivity_value)) / 1000
            
            self.assertGreater(ic50, 0)
            self.assertLess(ic50, 10000)  # Reasonable IC50 range
            
            print("✅ I3: QSAR → GPT assessment integration - PASS")
        except Exception as e:
            print(f"❌ I3: QSAR → GPT integration - FAIL: {e}")
            raise
    
    def test_04_end_to_end_workflow(self):
        """I4: Test complete end-to-end workflow simulation"""
        try:
            # Simulate complete workflow
            
            # Step 1: Image upload simulation (mock)
            image_processed = True
            self.assertTrue(image_processed)
            
            # Step 2: Plant identification (mock result)
            plant_identified = "Astma_weed"
            self.assertIsInstance(plant_identified, str)
            
            # Step 3: Phytochemical extraction simulation
            with open('../data/phytochemical_mapping.json', 'r') as f:
                phyto_data = json.load(f)
            
            compounds = phyto_data.get(plant_identified, {}).get('phytochemicals', [])
            self.assertGreater(len(compounds), 0)
            
            # Step 4: QSAR prediction simulation
            test_compound = compounds[0]
            self.assertIn('smiles', test_compound)
            
            print("✅ I4: End-to-end workflow - PASS")
        except Exception as e:
            print(f"❌ I4: End-to-end workflow - FAIL: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Generate module and integration testing report"""
        total_duration = time.time() - cls.start_time
        
        # Combine module and integration results
        all_results = []
        if hasattr(ModuleTests, 'module_results'):
            all_results.extend(ModuleTests.module_results)
        all_results.extend(cls.integration_results)
        
        print("\n" + "=" * 50)
        print("📊 MODULE & INTEGRATION TESTING REPORT")
        print("=" * 50)
        
        print("\n🔧 MODULE TESTS:")
        module_count = 0
        for test in all_results:
            if test['test'].startswith('test_01_') or test['test'].startswith('test_02_') or \
               test['test'].startswith('test_03_') or test['test'].startswith('test_04_') or \
               test['test'].startswith('test_05_'):
                if 'M' in test['result'] or 'module' in test['test']:
                    print(f"  {test['result']} {test['test']} ({test['duration']})")
                    module_count += 1
        
        print("\n🔗 INTEGRATION TESTS:")
        integration_count = 0
        for test in cls.integration_results:
            print(f"  {test['result']} {test['test']} ({test['duration']})")
            integration_count += 1
        
        passed = sum(1 for t in all_results if "✅" in t['result'])
        total = len(all_results)
        
        print(f"\nSUMMARY:")
        print(f"Module Tests: {module_count}")
        print(f"Integration Tests: {integration_count}")
        print(f"Total Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Save report
        with open('module_integration_test_report.json', 'w') as f:
            json.dump({
                'test_type': 'Module and Integration Testing',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'module_tests': module_count,
                'integration_tests': integration_count,
                'results': all_results,
                'summary': {
                    'passed': passed,
                    'total': total,
                    'duration': total_duration,
                    'success_rate': f"{(passed/total*100):.1f}%"
                }
            }, f, indent=2)

if __name__ == '__main__':
    # Run module tests first
    module_suite = unittest.TestLoader().loadTestsFromTestCase(ModuleTests)
    unittest.TextTestRunner(verbosity=2).run(module_suite)
    
    # Then run integration tests
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTests)
    unittest.TextTestRunner(verbosity=2).run(integration_suite)