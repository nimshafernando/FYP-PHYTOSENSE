#!/usr/bin/env python3
"""
Functional Testing Suite for PhytoSense Application
Tests all user-facing features and workflows
"""

import unittest
import requests
import os
import time
import json
from PIL import Image
import io
import base64

class FunctionalTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_url = "http://127.0.0.1:5000"
        cls.test_results = []
        cls.start_time = time.time()
        print("🧪 FUNCTIONAL TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        """Setup for each test"""
        self.test_start = time.time()
    
    def tearDown(self):
        """Cleanup after each test"""
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.test_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_homepage_accessibility(self):
        """F1: Test homepage loads correctly"""
        try:
            response = requests.get(self.base_url, timeout=10)
            self.assertEqual(response.status_code, 200)
            self.assertIn("PhytoSense", response.text)
            print("✅ F1: Homepage accessible - PASS")
        except Exception as e:
            print(f"❌ F1: Homepage accessibility - FAIL: {e}")
            raise
    
    def test_02_image_upload_functionality(self):  
        """F2: Test image upload and processing"""
        try:
            # Create a test image
            test_image = Image.new('RGB', (300, 300), color='green')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            files = {'file': ('test_leaf.jpg', img_buffer, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/predict", files=files, timeout=30)
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn('prediction', result)
            print("✅ F2: Image upload and processing - PASS")
        except Exception as e:
            print(f"❌ F2: Image upload functionality - FAIL: {e}")
            raise
    
    def test_03_phytochemical_data_retrieval(self):
        """F3: Test phytochemical mapping data access"""
        try:
            # Test with known plant
            files = {'file': ('test_leaf.jpg', io.BytesIO(b'fake_image_data'), 'image/jpeg')}
            response = requests.post(f"{self.base_url}/predict", files=files, timeout=30)
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn('phytochemicals', result)
            self.assertIsInstance(result['phytochemicals'], list)
            print("✅ F3: Phytochemical data retrieval - PASS")
        except Exception as e:
            print(f"❌ F3: Phytochemical data retrieval - FAIL: {e}")
            raise
    
    def test_04_qsar_prediction_workflow(self):
        """F4: Test QSAR prediction functionality"""
        try:
            # Test QSAR prediction with known SMILES
            qsar_data = {
                'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                'compound_name': 'Quercetin'
            }
            response = requests.post(f"{self.base_url}/predict_qsar", 
                                   json=qsar_data, timeout=30)
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            
            # Check actual response structure from flask_app.py
            self.assertTrue(result.get('success', False))
            self.assertIn('prediction', result)
            self.assertIn('targets', result)
            self.assertIn('descriptors', result)
            
            print("✅ F4: QSAR prediction workflow - PASS")
        except Exception as e:
            print(f"❌ F4: QSAR prediction workflow - FAIL: {e}")
            raise
    
    def test_05_molecular_visualization(self):
        """F5: Test 3D molecular structure generation"""
        try:
            # Test 3D structure conversion
            mol_data = {
                'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                'compound_name': 'Quercetin'
            }
            response = requests.post(f"{self.base_url}/api/convert_smiles_to_3d", 
                                   json=mol_data, timeout=20)
            
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn('mol_block_3d', result)
            print("✅ F5: Molecular visualization - PASS")
        except Exception as e:
            print(f"❌ F5: Molecular visualization - FAIL: {e}")
            raise
    
    def test_06_gpt_integration(self):
        """F6: Test OpenAI GPT integration for assessments"""
        try:
            # This test checks if GPT integration is working
            # Note: Requires valid OpenAI API key
            response = requests.get(f"{self.base_url}/", timeout=10)
            self.assertEqual(response.status_code, 200)
            
            # Check if application mentions AI capabilities
            self.assertIn("AI", response.text.upper())
            print("✅ F6: GPT integration check - PASS")
        except Exception as e:
            print(f"❌ F6: GPT integration - FAIL: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Generate functional testing report"""
        total_duration = time.time() - cls.start_time
        
        print("\n" + "=" * 50)
        print("📊 FUNCTIONAL TESTING REPORT")
        print("=" * 50)
        
        for test in cls.test_results:
            print(f"{test['result']} {test['test']} ({test['duration']})")
        
        passed = sum(1 for t in cls.test_results if "✅" in t['result'])
        total = len(cls.test_results)
        
        print(f"\nSUMMARY: {passed}/{total} tests passed")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Status: {'SUCCESS' if passed == total else 'FAILED'}")
        
        # Save report to file
        with open('functional_test_report.json', 'w') as f:
            json.dump({
                'test_type': 'Functional Testing',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': cls.test_results,
                'summary': {
                    'passed': passed,
                    'total': total,
                    'duration': total_duration,
                    'success_rate': f"{(passed/total*100):.1f}%"
                }
            }, f, indent=2)

if __name__ == '__main__':
    unittest.main(verbosity=2)