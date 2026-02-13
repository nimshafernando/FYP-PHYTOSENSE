#!/usr/bin/env python3
"""
COMPREHENSIVE TEST RUNNER WITH ENHANCED PERFORMANCE TRACKING
Runs all testing suites with detailed timing for documentation
"""

import time
import subprocess
import sys
from datetime import datetime

class TestRunner:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        self.step_times = {}
        
    def log_step(self, step_name, start_time):
        """Log step timing"""
        duration = time.time() - start_time
        self.step_times[step_name] = duration
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {step_name}: {duration:.3f}s")
        return duration
        
    def run_accuracy_tests(self):
        """Run accuracy testing suite with timing"""
        print("\nACCURACY TESTING SUITE")
        print("-" * 40)
        step_start = time.time()
        
        try:
            from accuracy_test import AccuracyTester
            tester = AccuracyTester()
            tester.test_known_compounds()
            accuracy_report = tester.generate_accuracy_report()
            
            # Extract key metrics
            accurate_count = sum(1 for r in tester.results if r["accurate"])
            total_count = len(tester.results)
            accuracy_rate = (accurate_count / total_count * 100) if total_count > 0 else 0
            
            self.results['accuracy'] = {
                'status': 'PASS' if accuracy_rate >= 70 else 'FAIL',
                'score': f"{accuracy_rate:.1f}%",
                'details': f"{accurate_count}/{total_count} accurate predictions"
            }
            
            duration = self.log_step("Accuracy Testing", step_start)
            
        except Exception as e:
            duration = self.log_step("Accuracy Testing", step_start)
            self.results['accuracy'] = {
                'status': 'ERROR',
                'score': '0%',
                'details': f"Error: {str(e)}"
            }
        
        return duration
    
    def run_enhanced_performance_tests(self):
        """Run enhanced performance testing with detailed workflow timing"""
        print("\nENHANCED PERFORMANCE TESTING SUITE")
        print("-" * 45)
        step_start = time.time()
        
        try:
            from enhanced_performance_test import EnhancedPerformanceTester
            tester = EnhancedPerformanceTester()
            
            # Run single workflow test
            workflow_times = tester.test_complete_workflow()
            
            # Extract key metrics
            qsar_time = workflow_times.get('qsar_prediction', 0)
            total_time = workflow_times.get('total_workflow', 0)
            
            self.results['performance'] = {
                'status': 'PASS' if qsar_time < 5.0 else 'SLOW',
                'score': f"{total_time:.2f}s",
                'details': f"QSAR: {qsar_time:.2f}s, Workflow: {total_time:.2f}s"
            }
            
            duration = self.log_step("Enhanced Performance Testing", step_start)
            
        except Exception as e:
            duration = self.log_step("Enhanced Performance Testing", step_start)
            self.results['performance'] = {
                'status': 'ERROR',
                'score': 'N/A',
                'details': f"Error: {str(e)}"
            }
        
        return duration
    
    def run_load_tests(self):
        """Run load/scalability testing suite with timing"""
        print("\nLOAD & SCALABILITY TESTING SUITE")
        print("-" * 40)
        step_start = time.time()
        
        try:
            from load_test import LoadTester
            tester = LoadTester()
            scalability_results = tester.scalability_stress_test()
            
            if scalability_results:
                final_result = scalability_results[-1]
                success_rate = final_result['success_rate']
                avg_response = final_result['avg_response_time']
                
                self.results['load'] = {
                    'status': 'STABLE' if success_rate >= 90 else 'DEGRADED',
                    'score': f"{success_rate:.1f}%",
                    'details': f"Avg response: {avg_response:.2f}s"
                }
            else:
                raise Exception("No scalability results")
                
            duration = self.log_step("Load & Scalability Testing", step_start)
            
        except Exception as e:
            duration = self.log_step("Load & Scalability Testing", step_start)
            self.results['load'] = {
                'status': 'ERROR',
                'score': '0%',
                'details': f"Error: {str(e)}"
            }
        
        return duration
    
    def run_security_tests(self):
        """Run security testing suite with timing"""
        print("\nSECURITY TESTING SUITE")
        print("-" * 30)
        step_start = time.time()
        
        try:
            from security_test import SecurityTester
            tester = SecurityTester()
            tester.test_input_validation()
            rate_test = tester.test_rate_limiting()
            info_test = tester.test_error_information_disclosure()
            
            secure_count = sum(1 for r in tester.results if r["security_passed"])
            total_count = len(tester.results)
            security_score = (secure_count / total_count * 100) if total_count > 0 else 0
            
            self.results['security'] = {
                'status': 'SECURE' if security_score >= 80 else 'NEEDS REVIEW',
                'score': f"{security_score:.1f}%",
                'details': f"{secure_count}/{total_count} tests passed"
            }
            
            duration = self.log_step("Security Testing", step_start)
            
        except Exception as e:
            duration = self.log_step("Security Testing", step_start)
            self.results['security'] = {
                'status': 'ERROR',
                'score': '0%', 
                'details': f"Error: {str(e)}"
            }
        
        return duration
    
    def generate_comprehensive_report(self):
        """Generate detailed test report with timing information"""
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        report = f"""
PHYTOSENSE COMPREHENSIVE TEST EXECUTION REPORT
===============================================
Test Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Test End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Execution Duration: {total_duration:.3f} seconds

INDIVIDUAL TEST SUITE EXECUTION TIMES:
---------------------------------------
Accuracy Testing:          {self.step_times.get('Accuracy Testing', 0):.3f}s
Enhanced Performance Test:  {self.step_times.get('Enhanced Performance Testing', 0):.3f}s
Load & Scalability Test:    {self.step_times.get('Load & Scalability Testing', 0):.3f}s
Security Testing:           {self.step_times.get('Security Testing', 0):.3f}s

TEST RESULTS SUMMARY:
=====================
Category          | Status      | Score     | Details
------------------|-------------|-----------|---------------------------
Accuracy Testing  | {self.results.get('accuracy', {}).get('status', 'N/A'):<11} | {self.results.get('accuracy', {}).get('score', 'N/A'):<9} | {self.results.get('accuracy', {}).get('details', 'N/A')}
Performance Test  | {self.results.get('performance', {}).get('status', 'N/A'):<11} | {self.results.get('performance', {}).get('score', 'N/A'):<9} | {self.results.get('performance', {}).get('details', 'N/A')}
Load Testing      | {self.results.get('load', {}).get('status', 'N/A'):<11} | {self.results.get('load', {}).get('score', 'N/A'):<9} | {self.results.get('load', {}).get('details', 'N/A')}
Security Testing  | {self.results.get('security', {}).get('status', 'N/A'):<11} | {self.results.get('security', {}).get('score', 'N/A'):<9} | {self.results.get('security', {}).get('details', 'N/A')}

PERFORMANCE BENCHMARKS:
=======================
"""
        
        # Calculate overall status
        statuses = [r.get('status', '') for r in self.results.values()]
        error_count = sum(1 for status in statuses if 'ERROR' in status)
        fail_count = sum(1 for status in statuses if 'FAIL' in status)
        
        if error_count > 0:
            overall = "SYSTEM ERRORS"
        elif fail_count > 0:
            overall = "NEEDS IMPROVEMENT"
        else:
            overall = "ALL TESTS PASSED"
        
        report += f"Overall Test Status: {overall}\n"
        report += f"Test Framework Version: PhytoSense Complete Test Suite v1.0\n"
        report += f"Environment: Local Development Server\n"
        report += f"Certification Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 65 + "\n"
        
        return report

if __name__ == "__main__":
    print("PHYTOSENSE COMPREHENSIVE TESTING FRAMEWORK")
    print("=" * 50)
    print(f"Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    runner = TestRunner()
    
    # Run all test suites with detailed timing
    accuracy_time = runner.run_accuracy_tests()
    performance_time = runner.run_enhanced_performance_tests()
    load_time = runner.run_load_tests()
    security_time = runner.run_security_tests()
    
    # Generate and display final report
    final_report = runner.generate_comprehensive_report()
    print("\n")
    print(final_report)
    
    # Save report to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'comprehensive_test_report_{timestamp}.txt'
    
    with open(report_filename, 'w') as f:
        f.write(final_report)
    
    print(f"REPORT SAVED: {report_filename}")
    print("Testing framework execution completed.")