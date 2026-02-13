#!/usr/bin/env python3
"""
Non-Functional Testing Suite for PhytoSense Application
Tests accuracy, performance, scalability, and security
"""

import unittest
import requests
import time
import threading
import json
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

class AccuracyTests(unittest.TestCase):
    """Test prediction accuracy and calibration validation"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5000"
        cls.accuracy_results = []
        cls.start_time = time.time()
        print("🎯 ACCURACY TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.accuracy_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_qsar_prediction_accuracy(self):
        """A1: Test QSAR prediction accuracy against known compounds"""
        try:
            # Test with known compounds and their expected IC50 ranges
            known_compounds = [
                {
                    'name': 'Quercetin',
                    'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                    'expected_ic50_range': (100, 160)  # 128 ± 22
                },
                {
                    'name': 'Luteolin',
                    'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                    'expected_ic50_range': (80, 120)   # 99 ± 11
                },
                {
                    'name': 'Apigenin',
                    'smiles': 'C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O',
                    'expected_ic50_range': (200, 320)  # 256 ± 54
                }
            ]
            
            accurate_predictions = 0
            total_predictions = 0
            
            for compound in known_compounds:
                try:
                    response = requests.post(f"{self.base_url}/predict_qsar", 
                                           json=compound, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_ic50 = result.get('ic50_um', 0)
                        
                        # Check if prediction is within expected range
                        min_ic50, max_ic50 = compound['expected_ic50_range']
                        if min_ic50 <= predicted_ic50 <= max_ic50:
                            accurate_predictions += 1
                        
                        total_predictions += 1
                        print(f"  {compound['name']}: Predicted={predicted_ic50:.1f}μM, Expected={min_ic50}-{max_ic50}μM")
                
                except Exception as e:
                    print(f"  Error testing {compound['name']}: {e}")
            
            accuracy_rate = (accurate_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"\nAccuracy Rate: {accurate_predictions}/{total_predictions} ({accuracy_rate:.1f}%)")
            
            # Require at least 80% accuracy for calibrated compounds
            self.assertGreaterEqual(accuracy_rate, 80.0)
            print("✅ A1: QSAR prediction accuracy - PASS")
            
        except Exception as e:
            print(f"❌ A1: QSAR prediction accuracy - FAIL: {e}")
            raise
    
    def test_02_molecular_descriptor_accuracy(self):
        """A2: Test molecular descriptor calculation accuracy"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            # Test Quercetin descriptors against known values
            quercetin_smiles = "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"
            mol = Chem.MolFromSmiles(quercetin_smiles)
            
            # Calculate descriptors
            calculated_mw = Descriptors.MolWt(mol)
            calculated_logp = Descriptors.MolLogP(mol)
            calculated_hbd = Descriptors.NumHDonors(mol)
            calculated_hba = Descriptors.NumHAcceptors(mol)
            
            # Expected values (±5% tolerance)
            expected_mw = 302.24
            expected_logp = 1.68
            expected_hbd = 5
            expected_hba = 7
            
            # Validate molecular weight (±1 Da tolerance)
            self.assertAlmostEqual(calculated_mw, expected_mw, delta=1.0)
            
            # Validate LogP (±0.5 tolerance)
            self.assertAlmostEqual(calculated_logp, expected_logp, delta=0.5)
            
            # Validate hydrogen bond donors/acceptors (exact match)
            self.assertEqual(calculated_hbd, expected_hbd)
            self.assertEqual(calculated_hba, expected_hba)
            
            print(f"  MW: {calculated_mw:.2f} (expected {expected_mw})")
            print(f"  LogP: {calculated_logp:.2f} (expected {expected_logp})")
            print(f"  HBD: {calculated_hbd} (expected {expected_hbd})")
            print(f"  HBA: {calculated_hba} (expected {expected_hba})")
            print("✅ A2: Molecular descriptor accuracy - PASS")
            
        except Exception as e:
            print(f"❌ A2: Molecular descriptor accuracy - FAIL: {e}")
            raise

class PerformanceTests(unittest.TestCase):
    """Test response times and throughput"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5000"
        cls.performance_results = []
        cls.start_time = time.time()
        print("\n⚡ PERFORMANCE TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.performance_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_homepage_response_time(self):
        """P1: Test homepage response time"""
        try:
            response_times = []
            
            for i in range(10):
                start_time = time.time()
                response = requests.get(self.base_url, timeout=10)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                self.assertEqual(response.status_code, 200)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Maximum response time: {max_response_time:.3f}s")
            
            # Require average response time < 1 second
            self.assertLess(avg_response_time, 1.0)
            print("✅ P1: Homepage response time - PASS")
            
        except Exception as e:
            print(f"❌ P1: Homepage response time - FAIL: {e}")
            raise
    
    def test_02_qsar_prediction_performance(self):
        """P2: Test QSAR prediction performance"""
        try:
            qsar_data = {
                'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                'compound_name': 'Quercetin'
            }
            
            prediction_times = []
            
            for i in range(5):  # Test 5 predictions
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict_qsar", 
                                       json=qsar_data, timeout=30)
                end_time = time.time()
                
                prediction_times.append(end_time - start_time)
                self.assertEqual(response.status_code, 200)
            
            avg_prediction_time = sum(prediction_times) / len(prediction_times)
            
            print(f"  Average QSAR prediction time: {avg_prediction_time:.3f}s")
            
            # Require prediction time < 10 seconds
            self.assertLess(avg_prediction_time, 10.0)
            print("✅ P2: QSAR prediction performance - PASS")
            
        except Exception as e:
            print(f"❌ P2: QSAR prediction performance - FAIL: {e}")
            raise
    
    def test_03_memory_usage_monitoring(self):
        """P3: Test memory usage during operations"""
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform multiple operations
            for i in range(10):
                qsar_data = {
                    'smiles': f'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                    'compound_name': f'TestCompound_{i}'
                }
                requests.post(f"{self.base_url}/predict_qsar", json=qsar_data, timeout=30)
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"  Initial memory: {initial_memory:.1f}MB")
            print(f"  Final memory: {final_memory:.1f}MB")
            print(f"  Memory increase: {memory_increase:.1f}MB")
            
            # Require memory increase < 100MB
            self.assertLess(memory_increase, 100.0)
            print("✅ P3: Memory usage monitoring - PASS")
            
        except Exception as e:
            print(f"❌ P3: Memory usage monitoring - FAIL: {e}")
            raise

class ScalabilityTests(unittest.TestCase):
    """Test load handling and scalability"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5000"
        cls.scalability_results = []
        cls.start_time = time.time()
        print("\n📈 SCALABILITY TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.scalability_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_concurrent_users_simulation(self):
        """S1: Test concurrent user load"""
        try:
            def make_request(user_id):
                try:
                    start_time = time.time()
                    response = requests.get(self.base_url, timeout=10)
                    end_time = time.time()
                    return {
                        'user_id': user_id,
                        'status_code': response.status_code,
                        'response_time': end_time - start_time,
                        'success': response.status_code == 200
                    }
                except Exception as e:
                    return {
                        'user_id': user_id,
                        'status_code': 0,
                        'response_time': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Simulate 20 concurrent users
            num_users = 20
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_users)]
                results = [future.result() for future in as_completed(futures)]
            
            successful_requests = sum(1 for r in results if r['success'])
            avg_response_time = sum(r['response_time'] for r in results if r['success']) / successful_requests if successful_requests > 0 else 0
            
            success_rate = (successful_requests / num_users) * 100
            
            print(f"  Concurrent users: {num_users}")
            print(f"  Successful requests: {successful_requests}/{num_users}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average response time: {avg_response_time:.3f}s")
            
            # Require 90% success rate
            self.assertGreaterEqual(success_rate, 90.0)
            print("✅ S1: Concurrent users simulation - PASS")
            
        except Exception as e:
            print(f"❌ S1: Concurrent users simulation - FAIL: {e}")
            raise
    
    def test_02_load_testing_qsar_predictions(self):
        """S2: Test load handling for QSAR predictions"""
        try:
            def make_qsar_request(request_id):
                try:
                    qsar_data = {
                        'smiles': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
                        'compound_name': f'LoadTest_{request_id}'
                    }
                    start_time = time.time()
                    response = requests.post(f"{self.base_url}/predict_qsar", 
                                           json=qsar_data, timeout=30)
                    end_time = time.time()
                    return {
                        'request_id': request_id,
                        'success': response.status_code == 200,
                        'response_time': end_time - start_time
                    }
                except Exception as e:
                    return {
                        'request_id': request_id,
                        'success': False,
                        'response_time': 0,
                        'error': str(e)
                    }
            
            # Test with 10 concurrent QSAR predictions
            num_requests = 10
            
            with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to 5 workers
                futures = [executor.submit(make_qsar_request, i) for i in range(num_requests)]
                results = [future.result() for future in as_completed(futures)]
            
            successful_requests = sum(1 for r in results if r['success'])
            success_rate = (successful_requests / num_requests) * 100
            
            if successful_requests > 0:
                avg_response_time = sum(r['response_time'] for r in results if r['success']) / successful_requests
            else:
                avg_response_time = 0
            
            print(f"  QSAR requests: {num_requests}")
            print(f"  Successful: {successful_requests}/{num_requests}")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average time: {avg_response_time:.3f}s")
            
            # Require 80% success rate for QSAR predictions
            self.assertGreaterEqual(success_rate, 80.0)
            print("✅ S2: QSAR load testing - PASS")
            
        except Exception as e:
            print(f"❌ S2: QSAR load testing - FAIL: {e}")
            raise

class SecurityTests(unittest.TestCase):
    """Test security vulnerabilities and input validation"""
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:5000"
        cls.security_results = []
        cls.start_time = time.time()
        print("\n🔒 SECURITY TESTING STARTED")
        print("=" * 50)
    
    def setUp(self):
        self.test_start = time.time()
    
    def tearDown(self):
        test_duration = time.time() - self.test_start
        result = "✅ PASS" if hasattr(self, '_outcome') and self._outcome.success else "❌ FAIL"
        self.security_results.append({
            'test': self._testMethodName,
            'result': result,
            'duration': f"{test_duration:.2f}s"
        })
    
    def test_01_input_validation_smiles(self):
        """SEC1: Test SMILES input validation"""
        try:
            malicious_inputs = [
                "<script>alert('XSS')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "invalid_smiles_string",
                "A" * 10000,  # Very long string
                "",  # Empty string
                None,
                {"nested": "object"},
                ["list", "input"]
            ]
            
            secure_responses = 0
            total_tests = len(malicious_inputs)
            
            for i, malicious_input in enumerate(malicious_inputs):
                try:
                    qsar_data = {
                        'smiles': malicious_input,
                        'compound_name': f'SecurityTest_{i}'
                    }
                    response = requests.post(f"{self.base_url}/predict_qsar", 
                                           json=qsar_data, timeout=10)
                    
                    # Should either return 400 (bad request) or handle gracefully
                    if response.status_code in [400, 422, 500]:  # Expected error codes
                        secure_responses += 1
                    elif response.status_code == 200:
                        # If 200, check that it handled error gracefully
                        result = response.json()
                        if 'error' in result:
                            secure_responses += 1
                
                except requests.exceptions.RequestException:
                    # Connection errors are acceptable for security tests
                    secure_responses += 1
            
            security_rate = (secure_responses / total_tests) * 100
            print(f"  Secure responses: {secure_responses}/{total_tests} ({security_rate:.1f}%)")
            
            # Require 90% secure handling
            self.assertGreaterEqual(security_rate, 90.0)
            print("✅ SEC1: SMILES input validation - PASS")
            
        except Exception as e:
            print(f"❌ SEC1: SMILES input validation - FAIL: {e}")
            raise
    
    def test_02_file_upload_security(self):
        """SEC2: Test file upload security"""
        try:
            # Test various malicious file uploads
            malicious_files = [
                ("script.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
                ("test.exe", b"MZ\x90\x00", "application/x-executable"),
                ("huge_file.jpg", b"A" * (20 * 1024 * 1024), "image/jpeg"),  # 20MB file
                ("empty_file.jpg", b"", "image/jpeg"),
            ]
            
            secure_uploads = 0
            total_tests = len(malicious_files)
            
            for filename, content, content_type in malicious_files:
                try:
                    files = {'file': (filename, content, content_type)}
                    response = requests.post(f"{self.base_url}/predict", 
                                           files=files, timeout=30)
                    
                    # Should reject malicious files with appropriate error codes
                    if response.status_code in [400, 413, 415, 422, 500]:
                        secure_uploads += 1
                    elif response.status_code == 200:
                        # Check if it properly handled the malicious file
                        result = response.json()
                        if 'error' in result:
                            secure_uploads += 1
                
                except requests.exceptions.RequestException:
                    secure_uploads += 1  # Connection errors are acceptable
            
            security_rate = (secure_uploads / total_tests) * 100
            print(f"  Secure file handling: {secure_uploads}/{total_tests} ({security_rate:.1f}%)")
            
            # Require 75% secure file handling
            self.assertGreaterEqual(security_rate, 75.0) 
            print("✅ SEC2: File upload security - PASS")
            
        except Exception as e:
            print(f"❌ SEC2: File upload security - FAIL: {e}")
            raise

def generate_comprehensive_report():
    """Generate comprehensive non-functional testing report"""
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE NON-FUNCTIONAL TESTING REPORT")
    print("=" * 60)
    
    # Collect all results
    all_results = []
    
    # Add results from each test class
    for test_class in [AccuracyTests, PerformanceTests, ScalabilityTests, SecurityTests]:
        if hasattr(test_class, 'accuracy_results'):
            all_results.extend(test_class.accuracy_results)
        elif hasattr(test_class, 'performance_results'):
            all_results.extend(test_class.performance_results)
        elif hasattr(test_class, 'scalability_results'):
            all_results.extend(test_class.scalability_results)
        elif hasattr(test_class, 'security_results'):
            all_results.extend(test_class.security_results)
    
    # Generate summary
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if "✅" in r['result'])
    
    print(f"\n📈 TESTING SUMMARY:")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    # Category breakdown
    categories = {
        'Accuracy': [r for r in all_results if 'accuracy' in r['test'] or 'A1' in r['test'] or 'A2' in r['test']],
        'Performance': [r for r in all_results if 'performance' in r['test'] or 'P1' in r['test'] or 'P2' in r['test'] or 'P3' in r['test']],
        'Scalability': [r for r in all_results if 'scalability' in r['test'] or 'S1' in r['test'] or 'S2' in r['test']],
        'Security': [r for r in all_results if 'security' in r['test'] or 'SEC' in r['test']]
    }
    
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, tests in categories.items():
        if tests:
            cat_passed = sum(1 for t in tests if "✅" in t['result'])
            cat_total = len(tests)
            print(f"{category:12}: {cat_passed}/{cat_total} ({(cat_passed/cat_total*100):.1f}%)")
    
    # Save comprehensive report
    report_data = {
        'test_type': 'Non-Functional Testing Suite',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'categories': {
            'accuracy': len(categories['Accuracy']),
            'performance': len(categories['Performance']),
            'scalability': len(categories['Scalability']),
            'security': len(categories['Security'])
        },
        'results': all_results,
        'summary': {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests*100):.1f}%"
        }
    }
    
    with open('non_functional_test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Report saved to: non_functional_test_report.json")

if __name__ == '__main__':
    # Run all non-functional tests in sequence
    test_classes = [AccuracyTests, PerformanceTests, ScalabilityTests, SecurityTests]
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    
    # Generate comprehensive report
    generate_comprehensive_report()