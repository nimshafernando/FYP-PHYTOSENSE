#!/usr/bin/env python3
"""
Performance monitoring middleware for Flask app
"""

import time
import functools
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, app=None):
        self.app = app
        self.response_times = []
        
    def init_app(self, app):
        self.app = app
        app.before_request(self._start_timer)
        app.after_request(self._log_response_time)
        
    def _start_timer(self):
        from flask import g
        g.start_time = time.time()
        
    def _log_response_time(self, response):
        from flask import g, request
        
        if hasattr(g, 'start_time'):
            response_time = (time.time() - g.start_time) * 1000  # Convert to ms
            endpoint = request.endpoint or request.path
            method = request.method
            status_code = response.status_code
            
            # Log to console
            print(f"[PERF] {method} {endpoint} - {response_time:.2f}ms - {status_code}")
            
            # Store for analysis
            self.response_times.append({
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'method': method,
                'response_time': response_time,
                'status_code': status_code
            })
            
            # Add response time header
            response.headers['X-Response-Time'] = f"{response_time:.2f}ms"
            
        return response
        
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.response_times:
            return {}
            
        times = [r['response_time'] for r in self.response_times]
        return {
            'total_requests': len(times),
            'average_response_time': sum(times) / len(times),
            'min_response_time': min(times),
            'max_response_time': max(times),
            'recent_requests': self.response_times[-10:]  # Last 10 requests
        }

def add_performance_monitoring(app):
    """Add performance monitoring to Flask app"""
    monitor = PerformanceMonitor()
    monitor.init_app(app)
    return monitor