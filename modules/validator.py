"""
Advanced Proxy Validation Module
Provides queue-based and async validation with progress tracking
"""

import queue
import threading
import asyncio
import time
import requests
import httpx
from typing import List, Optional, Callable, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class ValidationResult:
    """Result of proxy validation"""
    proxy: Any  # Will be ProxyInfo from main app
    success: bool
    latency: Optional[float] = None
    endpoint_tested: str = ""
    error_message: Optional[str] = None
    headers_leaked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    exit_ip: Optional[str] = None

class QueueBasedValidator:
    """Queue-based proxy validator with thread pool"""
    
    def __init__(self, num_threads: int = 20, timeout: int = 10, no_apis: bool = False):
        self.num_threads = num_threads
        self.timeout = timeout
        self.no_apis = no_apis
        self.proxy_queue = queue.Queue()
        self.valid_proxies = []
        self.invalid_proxies = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.progress_count = 0
        self.total_count = 0
        
    def get_endpoints(self) -> List[tuple]:
        """Get validation endpoints based on mode"""
        endpoints = [
            ("https://www.wikipedia.org/robots.txt", "text"),
            ("https://www.cloudflare.com/robots.txt", "text"),
            ("https://www.google.com/robots.txt", "text"),
        ]
        
        if not self.no_apis:
            endpoints.extend([
                ("https://api.ipify.org?format=json", "json"),
                ("https://httpbin.org/ip", "json"),
                ("http://ipinfo.io/json", "json"),
            ])
        
        return endpoints
        
    def validate_proxies(
        self, 
        proxies: List[Any], 
        max_valid: int = None,
        progress_callback: Callable = None
    ) -> List[Any]:
        """Validate proxies using thread pool"""
        
        # Reset state
        self.valid_proxies = []
        self.invalid_proxies = []
        self.stop_flag.clear()
        self.progress_count = 0
        self.total_count = len(proxies)
        
        # Fill queue
        for proxy in proxies:
            self.proxy_queue.put(proxy)
        
        # Start worker threads
        threads = []
        for i in range(min(self.num_threads, len(proxies))):
            t = threading.Thread(
                target=self._worker,
                args=(max_valid, progress_callback),
                name=f"Worker-{i}"
            )
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for completion
        self.proxy_queue.join()
        self.stop_flag.set()
        
        # Wait for threads to finish
        for t in threads:
            t.join(timeout=1)
        
        return self.valid_proxies
    
    def _worker(self, max_valid: int, progress_callback: Callable):
        """Worker thread for validation"""
        while not self.stop_flag.is_set():
            try:
                proxy = self.proxy_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Check if we've reached the limit
            with self.lock:
                if max_valid and len(self.valid_proxies) >= max_valid:
                    self.proxy_queue.task_done()
                    self.stop_flag.set()
                    break
                self.progress_count += 1
            
            # Validate proxy
            validation_result = self._validate_single(proxy)
            
            with self.lock:
                if validation_result.success:
                    if not max_valid or len(self.valid_proxies) < max_valid:
                        self.valid_proxies.append(proxy)
                        if progress_callback:
                            try:
                                progress_callback(proxy, True, self.progress_count, self.total_count)
                            except:
                                pass
                else:
                    self.invalid_proxies.append(proxy)
                    if progress_callback:
                        try:
                            progress_callback(proxy, False, self.progress_count, self.total_count)
                        except:
                            pass
            
            self.proxy_queue.task_done()
    
    def _validate_single(self, proxy) -> ValidationResult:
        """Validate a single proxy"""
        proxy_url = proxy.as_url()
        endpoints = self.get_endpoints()
        
        for endpoint, response_type in endpoints:
            try:
                start_time = time.perf_counter()
                
                # Make request through proxy
                res = requests.get(
                    endpoint,
                    proxies={"http": proxy_url, "https": proxy_url},
                    timeout=self.timeout,
                    verify=False,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                
                if res.status_code in [200, 301, 302, 403, 404]:
                    # Calculate latency
                    latency = (time.perf_counter() - start_time) * 1000
                    proxy.latency = latency
                    proxy.is_valid = True
                    proxy.last_tested = datetime.now()
                    
                    # Extract additional info
                    if response_type == "json":
                        try:
                            data = res.json()
                            if "ipinfo.io" in endpoint:
                                proxy.country = data.get("country")
                                proxy.city = data.get("city")
                                proxy.exit_ip = data.get("ip")
                                loc = data.get("loc", "").split(",")
                                if len(loc) == 2:
                                    proxy.lat = float(loc[0])
                                    proxy.lon = float(loc[1])
                            elif "ipify" in endpoint:
                                proxy.exit_ip = data.get("ip")
                            elif "httpbin" in endpoint:
                                proxy.exit_ip = data.get("origin", "").split(",")[0].strip()
                        except:
                            pass
                    
                    # Check for leaked headers
                    leaked = []
                    for header in ["X-Forwarded-For", "X-Real-IP", "Via", "X-Originating-IP"]:
                        if header in res.headers:
                            leaked.append(header)
                    
                    # Determine anonymity level
                    proxy.anonymity_level = "transparent" if leaked else "anonymous"
                    
                    return ValidationResult(
                        proxy=proxy,
                        success=True,
                        latency=latency,
                        endpoint_tested=endpoint,
                        headers_leaked=leaked,
                        exit_ip=getattr(proxy, 'exit_ip', None)
                    )
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                continue
            except Exception:
                continue
        
        return ValidationResult(
            proxy=proxy,
            success=False,
            error_message="All validation endpoints failed"
        )

class AsyncQueueValidator:
    """Async queue-based validator for better performance"""
    
    def __init__(self, max_concurrent: int = 30, timeout: int = 10, no_apis: bool = False):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.no_apis = no_apis
        self.semaphore = None
    
    def get_endpoints(self) -> List[tuple]:
        """Get validation endpoints based on mode"""
        endpoints = [
            ("https://www.wikipedia.org/robots.txt", "text"),
            ("https://www.cloudflare.com/robots.txt", "text"),
            ("https://www.google.com/robots.txt", "text"),
        ]
        
        if not self.no_apis:
            endpoints.extend([
                ("https://api.ipify.org?format=json", "json"),
                ("https://httpbin.org/ip", "json"),
            ])
        
        return endpoints
        
    async def validate_batch(
        self,
        proxies: List[Any],
        max_valid: int = None,
        progress_callback: Callable = None
    ) -> List[Any]:
        """Async batch validation with concurrency control"""
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        valid_proxies = []
        invalid_proxies = []
        tasks = []
        progress_count = 0
        total_count = len(proxies)
        stop_flag = False
        
        async def validate_with_limit(proxy, index):
            nonlocal progress_count, stop_flag, valid_proxies
            
            if stop_flag:
                return None
                
            async with self.semaphore:
                if max_valid and len(valid_proxies) >= max_valid:
                    stop_flag = True
                    r
