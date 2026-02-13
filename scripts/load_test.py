#!/usr/bin/env python3
"""
LOAD TESTING SUITE FOR PHYTOSENSE
Tests concurrent request handling and performance under load
"""

import requests
import time
import json
import threading
import concurrent.futures
from datetime import datetime
import statistics

class LoadTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        self.concurrent_results = []
        self.failed_requests = 0
        self.successful_requests = 0
        self.total_response_time = 0
        
    def single_request_test(self, endpoint, payload=None, request_id=None):
        """Test a single request with timing"""
        start_time = time.time()
        try:
            if payload:
                response = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=30)
            else:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=30)
                
            response_time = time.time() - start_time
            
            result = {
                'request_id': request_id,
                'endpoint': endpoint,
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code < 400,
                'response_size': len(response.content) if response.content else 0,
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            if result['success']:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                
            self.total_response_time += response_time
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            result = {
                'request_id': request_id,
                'endpoint': endpoint,
                'status_code': 0,
                'response_time': response_time,
                'success': False,
                'response_size': 0,
                'error': str(e),
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            self.failed_requests += 1
            self.total_response_time += response_time
            return result

    def test_concurrent_requests(self, num_threads=10, requests_per_thread=5):
        """Test concurrent request handling"""
        print("LOAD TESTING - CONCURRENT REQUEST HANDLING")
        print("=" * 60)
        print(f"Configuration: {num_threads} threads, {requests_per_thread} requests per thread")
        print(f"Total Requests: {num_threads * requests_per_thread}")
        print(f"Test Started: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        # Reset counters
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0
        
        # Test scenarios with different endpoints
        test_scenarios = [
            {"endpoint": "/", "payload": None, "name": "Homepage Load"},
            {"endpoint": "/predict_qsar", "payload": {"smiles": "CCO"}, "name": "QSAR Prediction"},
            {"endpoint": "/calculate_descriptors", "payload": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}, "name": "Descriptor Calculation"},
            {"endpoint": "/api/convert_smiles_to_3d", "payload": {"compound_name": "TestCompound", "smiles": "CCO"}, "name": "3D Conversion"},
            {"endpoint": "/api/autodock_vina", "payload": {"compound_name": "TestCompound", "smiles": "CCO"}, "name": "AutoDock Vina"}
        ]
        
        overall_start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            request_id = 1
            
            # Submit all requests
            for thread_id in range(num_threads):
                for req_num in range(requests_per_thread):
                    # Cycle through test scenarios
                    scenario = test_scenarios[req_num % len(test_scenarios)]
                    
                    future = executor.submit(
                        self.single_request_test,
                        scenario["endpoint"],
                        scenario["payload"],
                        request_id
                    )
                    futures.append({
                        'future': future,
                        'scenario': scenario["name"],
                        'thread_id': thread_id + 1,
                        'request_id': request_id
                    })
                    request_id += 1
            
            # Collect results as they complete
            print("Request Processing Status:")
            completed_count = 0
            
            for future_info in concurrent.futures.as_completed([f['future'] for f in futures]):
                result = future_info.result()
                self.concurrent_results.append(result)
                completed_count += 1
                
                # Find the corresponding scenario info
                scenario_info = next((f for f in futures if f['future'] == future_info), None)
                if scenario_info:
                    status = "SUCCESS" if result['success'] else "FAILED"
                    print(f"[{result['timestamp']}] Request {result['request_id']:2d} | Thread {scenario_info['thread_id']:2d} | {scenario_info['scenario']:<20} | {status} ({result['response_time']:.3f}s)")
        
        overall_duration = time.time() - overall_start_time
        
        # Calculate statistics
        response_times = [r['response_time'] for r in self.concurrent_results]
        successful_times = [r['response_time'] for r in self.concurrent_results if r['success']]
        
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS SUMMARY")
        print("=" * 60)
        
        # Overall Statistics
        total_requests = len(self.concurrent_results)
        success_rate = (self.successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        print(f"Total Requests Sent:        {total_requests}")
        print(f"Successful Requests:        {self.successful_requests}")
        print(f"Failed Requests:            {self.failed_requests}")
        print(f"Success Rate:               {success_rate:.1f}%")
        print(f"Total Test Duration:        {overall_duration:.3f}s")
        print(f"Requests per Second:        {total_requests/overall_duration:.2f}")
        
        # Response Time Statistics
        if response_times:
            print(f"\nResponse Time Analysis:")
            print(f"Average Response Time:      {statistics.mean(response_times):.3f}s")
            print(f"Median Response Time:       {statistics.median(response_times):.3f}s")
            print(f"Fastest Response:           {min(response_times):.3f}s")
            print(f"Slowest Response:           {max(response_times):.3f}s")
            print(f"Response Time Std Dev:      {statistics.stdev(response_times):.3f}s" if len(response_times) > 1 else "Response Time Std Dev:      N/A")
        
        # Successful requests only
        if successful_times:
            print(f"\nSuccessful Requests Only:")
            print(f"Average Success Time:       {statistics.mean(successful_times):.3f}s")
            print(f"Median Success Time:        {statistics.median(successful_times):.3f}s")
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'duration': overall_duration,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'requests_per_second': total_requests/overall_duration if overall_duration > 0 else 0
        }

    def test_endpoint_performance(self):
        """Test individual endpoint performance under load"""
        print("\n" + "=" * 60)
        print("ENDPOINT PERFORMANCE TESTING")
        print("=" * 60)
        
        endpoints = [
            {"path": "/", "method": "GET", "payload": None, "name": "Homepage"},
            {"path": "/predict_qsar", "method": "POST", "payload": {"smiles": "CCO"}, "name": "QSAR Prediction"},
            {"path": "/calculate_descriptors", "method": "POST", "payload": {"smiles": "CCO"}, "name": "Descriptors"},
            {"path": "/api/convert_smiles_to_3d", "method": "POST", "payload": {"compound_name": "Test", "smiles": "CCO"}, "name": "3D Conversion"}
        ]
        
        endpoint_results = {}
        
        for endpoint in endpoints:
            print(f"\nTesting {endpoint['name']} ({endpoint['path']}):")
            print("-" * 40)
            
            endpoint_times = []
            successful_count = 0
            
            # Send 10 requests to each endpoint
            for i in range(1, 11):
                result = self.single_request_test(endpoint['path'], endpoint['payload'], f"EP{i}")
                endpoint_times.append(result['response_time'])
                
                status = "SUCCESS" if result['success'] else f"FAILED ({result.get('error', 'HTTP ' + str(result['status_code']))})"
                print(f"Request {i:2d}: {result['response_time']:.3f}s - {status}")
                
                if result['success']:
                    successful_count += 1
                
                time.sleep(0.1)  # Small delay between requests
            
            # Calculate endpoint statistics
            if endpoint_times:
                avg_time = statistics.mean(endpoint_times)
                endpoint_success_rate = (successful_count / len(endpoint_times)) * 100
                
                print(f"\nEndpoint Summary:")
                print(f"Success Rate: {endpoint_success_rate:.1f}% ({successful_count}/10)")
                print(f"Average Time: {avg_time:.3f}s")
                print(f"Fastest:      {min(endpoint_times):.3f}s")
                print(f"Slowest:      {max(endpoint_times):.3f}s")
                
                endpoint_results[endpoint['name']] = {
                    'success_rate': endpoint_success_rate,
                    'avg_time': avg_time,
                    'min_time': min(endpoint_times),
                    'max_time': max(endpoint_times)
                }
        
        return endpoint_results

    def test_sustained_load(self, duration_minutes=2):
        """Test sustained load over time"""
        print("\n" + "=" * 60)
        print(f"SUSTAINED LOAD TEST - {duration_minutes} MINUTES")
        print("=" * 60)
        
        end_time = time.time() + (duration_minutes * 60)
        request_count = 0
        sustained_results = []
        
        print(f"Test Duration: {duration_minutes} minute(s)")
        print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
        print("\nOngoing Results:")
        
        while time.time() < end_time:
            request_count += 1
            result = self.single_request_test("/predict_qsar", {"smiles": "CCO"}, f"SUS{request_count}")
            sustained_results.append(result)
            
            # Print every 10th request
            if request_count % 10 == 0:
                recent_times = [r['response_time'] for r in sustained_results[-10:]]
                recent_successes = sum(1 for r in sustained_results[-10:] if r['success'])
                avg_recent = statistics.mean(recent_times)
                
                elapsed_time = time.time() - (end_time - duration_minutes * 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Request {request_count:3d} | Last 10 avg: {avg_recent:.3f}s | Success: {recent_successes}/10 | Elapsed: {elapsed_time:.1f}s")
            
            time.sleep(0.5)  # Half second between requests
        
        # Calculate sustained load results
        total_sustained = len(sustained_results)
        sustained_successes = sum(1 for r in sustained_results if r['success'])
        sustained_success_rate = (sustained_successes / total_sustained * 100) if total_sustained > 0 else 0
        
        if sustained_results:
            sustained_times = [r['response_time'] for r in sustained_results]
            avg_sustained_time = statistics.mean(sustained_times)
        
        print(f"\nSustained Load Results:")
        print(f"Total Requests: {total_sustained}")
        print(f"Success Rate: {sustained_success_rate:.1f}%")
        print(f"Average Response Time: {avg_sustained_time:.3f}s")
        print(f"Requests per Minute: {total_sustained/duration_minutes:.1f}")
        
        return {
            'total_requests': total_sustained,
            'success_rate': sustained_success_rate,
            'avg_response_time': avg_sustained_time,
            'requests_per_minute': total_sustained/duration_minutes
        }

    def generate_load_test_report(self, concurrent_results, endpoint_results, sustained_results):
        """Generate comprehensive load test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LOAD TESTING REPORT")
        print("="*80)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target URL: {self.base_url}")
        print("\nTEST RESULTS SUMMARY:")
        print("-"*50)
        
        # Concurrent Load Results
        print(f"Concurrent Load Test:")
        print(f"  - Total Requests: {concurrent_results['total_requests']}")
        print(f"  - Success Rate: {concurrent_results['success_rate']:.1f}%")
        print(f"  - Average Response Time: {concurrent_results['avg_response_time']:.3f}s")
        print(f"  - Throughput: {concurrent_results['requests_per_second']:.2f} req/sec")
        
        # Endpoint Performance
        print(f"\nEndpoint Performance:")
        for endpoint, stats in endpoint_results.items():
            print(f"  - {endpoint}: {stats['success_rate']:.1f}% success, {stats['avg_time']:.3f}s avg")
        
        # Sustained Load
        print(f"\nSustained Load Test:")
        print(f"  - Success Rate: {sustained_results['success_rate']:.1f}%")
        print(f"  - Average Response Time: {sustained_results['avg_response_time']:.3f}s")
        print(f"  - Sustained Throughput: {sustained_results['requests_per_minute']:.1f} req/min")
        
        # Performance Assessment
        overall_score = (
            concurrent_results['success_rate'] * 0.4 +
            min([stats['success_rate'] for stats in endpoint_results.values()]) * 0.3 +
            sustained_results['success_rate'] * 0.3
        )
        
        print(f"\nPERFORMANCE ASSESSMENT:")
        print(f"Overall Load Handling Score: {overall_score:.1f}%")
        
        if overall_score >= 90:
            performance_level = "EXCELLENT"
        elif overall_score >= 80:
            performance_level = "GOOD"
        elif overall_score >= 70:
            performance_level = "MODERATE"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        print(f"Performance Level: {performance_level}")
        print("="*80)

    def run_all_load_tests(self):
        """Run comprehensive load testing suite"""
        print("PHYTOSENSE LOAD TESTING SUITE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Testing application performance under various load conditions...")
        print("\n")
        
        # Test 1: Concurrent requests
        concurrent_results = self.test_concurrent_requests(num_threads=8, requests_per_thread=5)
        
        # Test 2: Individual endpoint performance
        endpoint_results = self.test_endpoint_performance()
        
        # Test 3: Sustained load
        sustained_results = self.test_sustained_load(duration_minutes=1)
        
        # Generate final report
        self.generate_load_test_report(concurrent_results, endpoint_results, sustained_results)

if __name__ == "__main__":
    print("Initializing Load Testing Suite...")
    tester = LoadTester()
    tester.run_all_load_tests()