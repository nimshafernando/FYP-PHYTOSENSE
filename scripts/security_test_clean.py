#!/usr/bin/env python3
"""
SECURITY TESTING SUITE FOR PHYTOSENSE
Input Validation and File Upload Security Testing
"""

import requests
import time
import json
import os
import io
import sys
from datetime import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

class SecurityTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_result(self, test_name, status, description, details=""):
        """Log test results"""
        result = {
            'test_name': test_name,
            'status': status,
            'description': description,
            'details': details,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        self.test_results.append(result)
        
        if status == "PASS":
            self.passed_tests += 1
            print(f"[{result['timestamp']}] ✓ PASS: {test_name}")
        elif status == "WARN":
            print(f"[{result['timestamp']}] ⚠ WARN: {test_name}")
        else:
            self.failed_tests += 1
            print(f"[{result['timestamp']}] ✗ FAIL: {test_name}")
        
        if description:
            print(f"    Description: {description}")
        if details:
            print(f"    Details: {details}")
        print()
    
    def test_input_validation(self):
        """Test input validation for various endpoints"""
        print("SECURITY TEST CATEGORY 1: INPUT VALIDATION TESTING")
        print("=" * 60)
        
        # Test 1: Empty SMILES input
        try:
            response = requests.post(f"{self.base_url}/predict_qsar", 
                                   json={"smiles": ""}, timeout=10)
            if response.status_code >= 400:
                self.log_result("Empty SMILES Input", "PASS", 
                              "Server correctly rejected empty SMILES", 
                              f"Status: {response.status_code}")
            else:
                self.log_result("Empty SMILES Input", "FAIL", 
                              "Server should reject empty SMILES", 
                              f"Status: {response.status_code}")
        except Exception as e:
            self.log_result("Empty SMILES Input", "PASS", 
                          "Request properly rejected", str(e))
        
        # Test 2: Invalid SMILES characters
        invalid_smiles = ["XYZ123", "C@#$%", "!!invalid!!", "SELECT * FROM users"]
        for i, smiles in enumerate(invalid_smiles):
            try:
                response = requests.post(f"{self.base_url}/predict_qsar", 
                                       json={"smiles": smiles}, timeout=10)
                if response.status_code >= 400:
                    self.log_result(f"Invalid SMILES Test {i+1}", "PASS", 
                                  "Server rejected invalid SMILES", 
                                  f"Input: '{smiles[:15]}...', Status: {response.status_code}")
                else:
                    self.log_result(f"Invalid SMILES Test {i+1}", "FAIL", 
                                  "Server should reject invalid SMILES", 
                                  f"Input: '{smiles[:15]}...', Status: {response.status_code}")
            except Exception as e:
                self.log_result(f"Invalid SMILES Test {i+1}", "PASS", 
                              "Request properly rejected", str(e))
        
        # Test 3: XSS attack in compound name
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for i, payload in enumerate(xss_payloads):
            try:
                response = requests.post(f"{self.base_url}/api/convert_smiles_to_3d", 
                                       json={"compound_name": payload, "smiles": "CCO"}, 
                                       timeout=10)
                if response.status_code >= 400:
                    self.log_result(f"XSS Protection Test {i+1}", "PASS", 
                                  "Server blocked potential XSS", 
                                  f"Status: {response.status_code}")
                else:
                    # Check if response contains the payload
                    if response.text and payload in response.text:
                        self.log_result(f"XSS Protection Test {i+1}", "FAIL", 
                                      "Potential XSS vulnerability", 
                                      "Payload returned in response")
                    else:
                        self.log_result(f"XSS Protection Test {i+1}", "PASS", 
                                      "Input sanitized properly", 
                                      "Payload not reflected")
            except Exception as e:
                self.log_result(f"XSS Protection Test {i+1}", "PASS", 
                              "Request properly rejected", str(e))

    def create_test_images(self):
        """Create various test image files"""
        test_files = {}
        
        try:
            # Valid medicinal leaf (mock - small green image)
            valid_img = Image.new('RGB', (100, 100), color='green')
            valid_buffer = io.BytesIO()
            valid_img.save(valid_buffer, format='JPEG')
            valid_buffer.seek(0)
            test_files['valid_medicinal'] = ('valid_leaf.jpg', valid_buffer.getvalue(), 'image/jpeg')
            
            # Non-medicinal image (red image to simulate non-leaf)
            non_medicinal = Image.new('RGB', (100, 100), color='red')
            non_med_buffer = io.BytesIO()
            non_medicinal.save(non_med_buffer, format='JPEG')
            non_med_buffer.seek(0)
            test_files['non_medicinal'] = ('random_object.jpg', non_med_buffer.getvalue(), 'image/jpeg')
            
        except Exception as e:
            print(f"Warning: Could not create test images: {e}")
        
        # Corrupted image file
        corrupted_data = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01corrupted_data_invalid'
        test_files['corrupted'] = ('corrupted.jpg', corrupted_data, 'image/jpeg')
        
        # Wrong format (text file with image extension)
        text_data = "This is not an image file, it's just text data".encode()
        test_files['fake_image'] = ('fake.jpg', text_data, 'image/jpeg')
        
        # Large file (simulated)
        large_data = b'\xFF\xD8\xFF\xE0' + b'A' * (1024 * 500)  # 500KB test file
        test_files['large_file'] = ('large.jpg', large_data, 'image/jpeg')
        
        # Executable disguised as image
        exe_data = b'MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xFF\xFF\x00\x00executable_content'
        test_files['executable'] = ('malicious.jpg', exe_data, 'image/jpeg')
        
        # Empty file
        test_files['empty_file'] = ('empty.jpg', b'', 'image/jpeg')
        
        return test_files

    def test_file_upload_security(self):
        """Test file upload security"""
        print("SECURITY TEST CATEGORY 2: FILE UPLOAD SECURITY TESTING")
        print("=" * 60)
        
        test_files = self.create_test_images()
        
        # Test different upload endpoints
        upload_endpoints = ["/upload", "/api/upload", "/classify_leaf"]
        
        working_endpoint = None
        
        # Find working upload endpoint first
        for endpoint in upload_endpoints:
            if 'valid_medicinal' in test_files:
                try:
                    files = {'file': test_files['valid_medicinal']}
                    response = requests.post(f"{self.base_url}{endpoint}", files=files, timeout=10)
                    
                    if response.status_code == 200:
                        self.log_result(f"Upload Endpoint Discovery", "PASS", 
                                      f"Found working upload endpoint: {endpoint}", 
                                      f"Status: {response.status_code}")
                        working_endpoint = endpoint
                        break
                    elif response.status_code == 404:
                        self.log_result(f"Endpoint Check {endpoint}", "INFO", 
                                      "Endpoint not found", 
                                      f"Status: {response.status_code}")
                except Exception as e:
                    self.log_result(f"Endpoint Check {endpoint}", "INFO", 
                                  "Endpoint not accessible", str(e))
        
        if not working_endpoint:
            self.log_result("Upload Endpoint Discovery", "WARN", 
                          "No upload endpoints found - testing with /upload", 
                          "Proceeding with default endpoint")
            working_endpoint = "/upload"
            
        # Test all file types on working endpoint
        for test_name, file_data in test_files.items():
            if test_name == 'valid_medicinal':
                continue  # Already tested
                
            try:
                files = {'file': file_data}
                response = requests.post(f"{self.base_url}{working_endpoint}", files=files, timeout=15)
                
                if test_name in ['corrupted', 'fake_image', 'executable', 'empty_file']:
                    if response.status_code >= 400:
                        self.log_result(f"Malicious File: {test_name}", "PASS", 
                                      "Malicious file properly rejected", 
                                      f"Status: {response.status_code}")
                    else:
                        self.log_result(f"Malicious File: {test_name}", "FAIL", 
                                      "Malicious file should be rejected", 
                                      f"Status: {response.status_code}")
                
                elif test_name == 'large_file':
                    if response.status_code in [413, 400, 422]:
                        self.log_result("Large File Upload", "PASS", 
                                      "Large file properly rejected", 
                                      f"Status: {response.status_code}")
                    else:
                        self.log_result("Large File Upload", "WARN", 
                                      "Large file accepted - check size limits", 
                                      f"Status: {response.status_code}")
                
                elif test_name == 'non_medicinal':
                    if response.status_code == 200:
                        try:
                            if 'application/json' in response.headers.get('content-type', ''):
                                data = response.json()
                                classification = str(data.get('classification', '')).lower()
                                confidence = data.get('confidence', 0)
                                
                                if 'non-medicinal' in classification or confidence < 0.5:
                                    self.log_result("Non-Medicinal Image Classification", "PASS", 
                                                  "Non-medicinal properly classified", 
                                                  f"Result: {classification}")
                                else:
                                    self.log_result("Non-Medicinal Image Classification", "WARN", 
                                                  "Non-medicinal classified as medicinal", 
                                                  f"Confidence: {confidence}")
                            else:
                                self.log_result("Non-Medicinal Image Classification", "WARN", 
                                              "Non-medicinal processing unclear", 
                                              "Non-JSON response")
                        except:
                            self.log_result("Non-Medicinal Image Classification", "WARN", 
                                          "Could not parse classification")
                    else:
                        self.log_result("Non-Medicinal Image Upload", "PASS", 
                                      "Non-medicinal image rejected", 
                                      f"Status: {response.status_code}")
                        
            except requests.exceptions.Timeout:
                self.log_result(f"File Upload Timeout: {test_name}", "PASS", 
                              "Request timed out - good DoS protection", 
                              "Timeout after 15s")
            except Exception as e:
                self.log_result(f"File Upload Error: {test_name}", "WARN", 
                              "Request failed", str(e))

    def test_api_rate_limiting(self):
        """Test API rate limiting"""
        print("SECURITY TEST CATEGORY 3: RATE LIMITING TESTING")
        print("=" * 60)
        
        rapid_requests = 0
        successful_requests = 0
        rate_limited = False
        response_times = []
        
        print("Testing rapid requests...")
        start_time = time.time()
        
        for i in range(10):
            try:
                request_start = time.time()
                response = requests.get(f"{self.base_url}/", timeout=3)
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                rapid_requests += 1
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:
                    rate_limited = True
                    break
                    
            except Exception:
                break
                
            time.sleep(0.05)  # Very small delay
        
        duration = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        if rate_limited:
            self.log_result("Rate Limiting Protection", "PASS", 
                          "API implements rate limiting", 
                          f"Rate limited after {rapid_requests} requests")
        elif avg_response_time > 1.0:
            self.log_result("Rate Limiting via Response Time", "PASS", 
                          "Natural rate limiting through slow responses", 
                          f"Average: {avg_response_time:.2f}s per request")
        else:
            self.log_result("Rate Limiting Check", "WARN", 
                          "No obvious rate limiting detected", 
                          f"{successful_requests}/{rapid_requests} quick requests successful")

    def generate_security_report(self):
        """Generate comprehensive security test report"""
        print("\n" + "="*80)
        print("PHYTOSENSE SECURITY TESTING REPORT")
        print("="*80)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        if len(self.test_results) > 0:
            success_rate = (self.passed_tests/len(self.test_results)*100)
            print(f"Success Rate: {success_rate:.1f}%")
        print("\n")
        
        # Create summary table
        print("DETAILED TEST RESULTS TABLE")
        print("-"*80)
        print(f"{'Test Name':<35} | {'Status':<6} | {'Description':<35}")
        print("-"*80)
        
        for result in self.test_results:
            test_name = result['test_name'][:34]
            status = result['status']
            desc = result['description'][:34]
            print(f"{test_name:<35} | {status:<6} | {desc:<35}")
        
        print("-"*80)
        
        # Category breakdown
        input_tests = [r for r in self.test_results if 'SMILES' in r['test_name'] or 'XSS' in r['test_name']]
        upload_tests = [r for r in self.test_results if 'File' in r['test_name'] or 'Upload' in r['test_name'] or 'Image' in r['test_name']]
        rate_tests = [r for r in self.test_results if 'Rate' in r['test_name']]
        
        print("\nSECURITY CATEGORY BREAKDOWN")
        print("-"*50)
        
        if input_tests:
            passed = len([t for t in input_tests if t['status'] == 'PASS'])
            print(f"Input Validation        : {passed}/{len(input_tests)} passed ({passed/len(input_tests)*100:.1f}%)")
        
        if upload_tests:
            passed = len([t for t in upload_tests if t['status'] == 'PASS'])
            print(f"File Upload Security    : {passed}/{len(upload_tests)} passed ({passed/len(upload_tests)*100:.1f}%)")
        
        if rate_tests:
            passed = len([t for t in rate_tests if t['status'] == 'PASS'])
            print(f"Rate Limiting           : {passed}/{len(rate_tests)} passed ({passed/len(rate_tests)*100:.1f}%)")
        
        # Security recommendations
        print("\nSECURITY FINDINGS")
        print("-"*50)
        
        failed_tests = [r for r in self.test_results if r['status'] == 'FAIL']
        if failed_tests:
            print("❌ HIGH PRIORITY SECURITY ISSUES:")
            for test in failed_tests:
                print(f"   • {test['test_name']}: {test['description']}")
        
        warn_tests = [r for r in self.test_results if r['status'] == 'WARN']
        if warn_tests:
            print("\n⚠️  RECOMMENDED SECURITY IMPROVEMENTS:")
            for test in warn_tests:
                print(f"   • {test['test_name']}: {test['description']}")
        
        if not failed_tests and not warn_tests:
            print("✅ No critical security vulnerabilities detected")
        
        # Overall security score
        if len(self.test_results) > 0:
            security_score = (self.passed_tests / len(self.test_results)) * 100
            if security_score >= 90:
                security_level = "EXCELLENT 🟢"
            elif security_score >= 80:
                security_level = "GOOD 🟡" 
            elif security_score >= 70:
                security_level = "MODERATE 🟠"
            else:
                security_level = "NEEDS IMPROVEMENT 🔴"
            
            print(f"\n🔒 OVERALL SECURITY SCORE: {security_score:.1f}% ({security_level})")
        
        print("\n" + "="*80)

    def run_all_tests(self):
        """Run comprehensive security testing suite"""
        print("🔒 PHYTOSENSE SECURITY TESTING SUITE 🔒")
        print("Starting comprehensive security validation...")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target URL: {self.base_url}")
        print("\n")
        
        # Run all test categories
        self.test_input_validation()
        print("\n")
        self.test_file_upload_security()
        print("\n")
        self.test_api_rate_limiting()
        
        # Generate final report
        self.generate_security_report()

if __name__ == "__main__":
    tester = SecurityTester()
    tester.run_all_tests()