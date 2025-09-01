"""
Complete Enhanced Proxy Validation Module for ProxyStream
Fully functional with all dependencies included
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
from urllib.parse import urlsplit
import json

# ---------------------------------------------------
# Data Models
# ---------------------------------------------------

@dataclass
class ProxyInfo:
    """Complete proxy information model"""
    host: str
    port: int
    protocol: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    latency: Optional[float] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    is_valid: bool = False
    last_tested: Optional[datetime] = None
    exit_ip: Optional[str] = None
    anonymity_level: str = "unknown"
    success_rate: float = 0.0
    test_count: int = 0
    tags: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash((self.host, self.port, self.protocol))

    def as_url(self) -> str:
        """Convert proxy to URL format"""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "username": self.username,
            "password": self.password,
            "latency": self.latency,
            "country": self.country,
            "country_code": self.country_code,
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "is_valid": self.is_valid,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "anonymity_level": self.anonymity_level,
            "success_rate": self.success_rate,
            "exit_ip": self.exit_ip,
            "url": self.as_url()
        }

@dataclass
class ValidationResult:
    """Result of proxy validation"""
    proxy: ProxyInfo
    success: bool
    latency: Optional[float] = None
    endpoint_tested: str = ""
    error_message: Optional[str] = None
    headers_leaked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    exit_ip: Optional[str] = None

# ---------------------------------------------------
# Queue-Based Threading Validator
# ---------------------------------------------------

class QueueBasedValidator:
    """Queue-based proxy validator with thread pool"""
    
    def __init__(self, num_threads: int = 20, timeout: int = 10):
        self.num_threads = num_threads
        self.timeout = timeout
        self.proxy_queue = queue.Queue()
        self.valid_proxies = []
        self.invalid_proxies = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.progress_count = 0
        self.total_count = 0
        
    def validate_proxies(
        self, 
        proxies: List[ProxyInfo], 
        max_valid: int = None,
        progress_callback: Callable = None
    ) -> List[ProxyInfo]:
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
        for i in range(self.num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(max_valid, progress_callback),
                name=f"Worker-{i}"
            )
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for completion or stop flag
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
                            progress_callback(proxy, True, self.progress_count, self.total_count)
                else:
                    self.invalid_proxies.append(proxy)
                    if progress_callback:
                        progress_callback(proxy, False, self.progress_count, self.total_count)
            
            self.proxy_queue.task_done()
    
    def _validate_single(self, proxy: ProxyInfo) -> ValidationResult:
        """Validate a single proxy"""
        proxy_url = proxy.as_url()
        
        # Multiple validation endpoints for reliability
        endpoints = [
            ("http://ipinfo.io/json", "json"),
            ("https://api.ipify.org?format=json", "json"),
            ("https://httpbin.org/ip", "json"),
            ("https://www.cloudflare.com/cdn-cgi/trace", "text"),
            ("https://www.google.com/robots.txt", "text")
        ]
        
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
                
                if res.status_code in [200, 301, 302]:
                    # Calculate latency
                    latency = (time.perf_counter() - start_time) * 1000
                    proxy.latency = latency
                    proxy.is_valid = True
                    proxy.last_tested = datetime.now()
                    
                    # Try to extract additional info
                    if response_type == "json" and "ipinfo.io" in endpoint:
                        try:
                            data = res.json()
                            proxy.country = data.get("country")
                            proxy.city = data.get("city")
                            proxy.exit_ip = data.get("ip")
                            loc = data.get("loc", "").split(",")
                            if len(loc) == 2:
                                proxy.lat = float(loc[0])
                                proxy.lon = float(loc[1])
                        except:
                            pass
                    elif response_type == "json" and "ipify" in endpoint:
                        try:
                            data = res.json()
                            proxy.exit_ip = data.get("ip")
                        except:
                            pass
                    
                    # Check for leaked headers
                    leaked = []
                    for header in ["X-Forwarded-For", "X-Real-IP", "Via", "X-Originating-IP"]:
                        if header in res.headers:
                            leaked.append(header)
                    
                    # Determine anonymity level
                    if leaked:
                        proxy.anonymity_level = "transparent"
                    else:
                        proxy.anonymity_level = "anonymous"
                    
                    return ValidationResult(
                        proxy=proxy,
                        success=True,
                        latency=latency,
                        endpoint_tested=endpoint,
                        headers_leaked=leaked,
                        exit_ip=proxy.exit_ip
                    )
                    
            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                continue
        
        # All endpoints failed
        return ValidationResult(
            proxy=proxy,
            success=False,
            error_message="All validation endpoints failed"
        )

# ---------------------------------------------------
# Async Validator
# ---------------------------------------------------

class AsyncQueueValidator:
    """Async queue-based validator for better performance"""
    
    def __init__(self, max_concurrent: int = 30, timeout: int = 10):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = None
        
    async def validate_batch(
        self,
        proxies: List[ProxyInfo],
        max_valid: int = None,
        progress_callback: Callable = None
    ) -> List[ProxyInfo]:
        """Async batch validation with concurrency control"""
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        valid_proxies = []
        invalid_proxies = []
        tasks = []
        progress_count = 0
        total_count = len(proxies)
        
        async def validate_with_limit(proxy, index):
            nonlocal progress_count
            async with self.semaphore:
                result = await self._validate_async(proxy)
                progress_count += 1
                
                if result.success:
                    valid_proxies.append(proxy)
                    if progress_callback:
                        await progress_callback(proxy, True, progress_count, total_count)
                else:
                    invalid_proxies.append(proxy)
                    if progress_callback:
                        await progress_callback(proxy, False, progress_count, total_count)
                
                return result
        
        # Create tasks for all proxies
        for i, proxy in enumerate(proxies):
            if max_valid and len(valid_proxies) >= max_valid:
                break
            task = asyncio.create_task(validate_with_limit(proxy, i))
            tasks.append(task)
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ValidationResult) and r.success]
        
        return valid_proxies[:max_valid] if max_valid else valid_proxies
    
    async def _validate_async(self, proxy: ProxyInfo) -> ValidationResult:
        """Async validation of single proxy"""
        proxy_url = proxy.as_url()
        proxy_dict = {"http://": proxy_url, "https://": proxy_url}
        
        # Test endpoints
        test_endpoints = [
            ("https://api.ipify.org?format=json", "json"),
            ("https://httpbin.org/ip", "json"),
            ("https://www.cloudflare.com/robots.txt", "text"),
            ("https://www.google.com/robots.txt", "text")
        ]
        
        for url, response_type in test_endpoints:
            try:
                start = time.perf_counter()
                
                async with httpx.AsyncClient(
                    proxies=proxy_dict,
                    timeout=httpx.Timeout(self.timeout),
                    verify=False,
                    follow_redirects=True
                ) as client:
                    response = await client.get(
                        url,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                    )
                    
                    if response.status_code in [200, 301, 302]:
                        # Calculate latency
                        latency = (time.perf_counter() - start) * 1000
                        proxy.latency = latency
                        proxy.is_valid = True
                        proxy.last_tested = datetime.now()
                        
                        # Extract IP if available
                        if response_type == "json" and "ipify" in url:
                            try:
                                data = response.json()
                                proxy.exit_ip = data.get("ip")
                            except:
                                pass
                        elif response_type == "json" and "httpbin" in url:
                            try:
                                data = response.json()
                                proxy.exit_ip = data.get("origin", "").split(",")[0].strip()
                            except:
                                pass
                        
                        # Check headers for anonymity
                        leaked = []
                        for header in ["X-Forwarded-For", "X-Real-IP", "Via"]:
                            if header.lower() in response.headers:
                                leaked.append(header)
                        
                        proxy.anonymity_level = "transparent" if leaked else "anonymous"
                        
                        return ValidationResult(
                            proxy=proxy,
                            success=True,
                            latency=latency,
                            endpoint_tested=url,
                            headers_leaked=leaked,
                            exit_ip=proxy.exit_ip
                        )
                        
            except (httpx.TimeoutException, httpx.ConnectError):
                continue
            except Exception as e:
                continue
        
        return ValidationResult(
            proxy=proxy,
            success=False,
            error_message="All endpoints failed"
        )

# ---------------------------------------------------
# File-based Proxy Loader
# ---------------------------------------------------

class ProxyFileLoader:
    """Load proxies from various file formats"""
    
    @staticmethod
    def load_from_file(filepath: str, protocol: str = "http") -> List[ProxyInfo]:
        """Load proxies from a text file"""
        proxies = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse different formats
                proxy = ProxyFileLoader._parse_proxy_line(line, protocol)
                if proxy:
                    proxies.append(proxy)
        
        except Exception as e:
            print(f"Error loading file: {e}")
        
        return proxies
    
    @staticmethod
    def load_from_string(content: str, protocol: str = "http") -> List[ProxyInfo]:
        """Load proxies from string content"""
        proxies = []
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            proxy = ProxyFileLoader._parse_proxy_line(line, protocol)
            if proxy:
                proxies.append(proxy)
        
        return proxies
    
    @staticmethod
    def _parse_proxy_line(line: str, default_protocol: str) -> Optional[ProxyInfo]:
        """Parse a single proxy line"""
        
        # Remove any whitespace
        line = line.strip()
        
        # Handle different formats
        if "://" not in line:
            # Assume IP:PORT format
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    # Validate IP format
                    ip_parts = parts[0].split(".")
                    if len(ip_parts) == 4 and all(0 <= int(p) <= 255 for p in ip_parts):
                        return ProxyInfo(
                            host=parts[0],
                            port=int(parts[1]),
                            protocol=default_protocol
                        )
                except (ValueError, AttributeError):
                    pass
            elif len(parts) == 4:
                # Might be IP:PORT:USER:PASS format
                try:
                    return ProxyInfo(
                        host=parts[0],
                        port=int(parts[1]),
                        protocol=default_protocol,
                        username=parts[2],
                        password=parts[3]
                    )
                except ValueError:
                    pass
        else:
            # Parse as URL
            try:
                u = urlsplit(line)
                if u.hostname and u.port:
                    return ProxyInfo(
                        host=u.hostname,
                        port=u.port,
                        protocol=u.scheme or default_protocol,
                        username=u.username,
                        password=u.password
                    )
            except:
                pass
        
        return None
    
    @staticmethod
    def save_to_file(proxies: List[ProxyInfo], filepath: str, format: str = "url"):
        """Save proxies to file in specified format"""
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "url":
                for proxy in proxies:
                    f.write(f"{proxy.as_url()}\n")
            elif format == "ip_port":
                for proxy in proxies:
                    f.write(f"{proxy.host}:{proxy.port}\n")
            elif format == "json":
                data = [proxy.to_dict() for proxy in proxies]
                json.dump(data, f, indent=2, default=str)

# ---------------------------------------------------
# Performance Benchmark
# ---------------------------------------------------

class ValidationBenchmark:
    """Benchmark different validation methods"""
    
    @staticmethod
    async def compare_methods(proxies: List[ProxyInfo], sample_size: int = 20) -> Dict[str, Any]:
        """Compare validation methods performance"""
        
        results = {}
        sample = proxies[:min(sample_size, len(proxies))]
        
        # Method 1: Queue-based threading
        print("Testing Queue-based threading...")
        start = time.time()
        validator = QueueBasedValidator(num_threads=10, timeout=5)
        valid_queue = validator.validate_proxies(sample)
        results['Queue Threading'] = {
            'time': time.time() - start,
            'valid': len(valid_queue),
            'proxies_per_second': len(sample) / (time.time() - start)
        }
        
        # Method 2: Async
        print("Testing Async validation...")
        start = time.time()
        async_validator = AsyncQueueValidator(max_concurrent=20, timeout=5)
        valid_async = await async_validator.validate_batch(sample)
        results['Async'] = {
            'time': time.time() - start,
            'valid': len(valid_async),
            'proxies_per_second': len(sample) / (time.time() - start)
        }
        
        return results

# ---------------------------------------------------
# Streamlit Integration Helper
# ---------------------------------------------------

def create_streamlit_ui():
    """Create Streamlit UI components for proxy validation"""
    import streamlit as st
    
    # Initialize session state
    if 'file_proxies' not in st.session_state:
        st.session_state.file_proxies = []
    if 'validated_proxies' not in st.session_state:
        st.session_state.validated_proxies = []
    
    with st.sidebar:
        st.subheader("🔧 Advanced Proxy Validator")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload proxy list",
            type=['txt', 'csv'],
            help="Supported formats: IP:PORT, http://IP:PORT, or full proxy URLs"
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            loader = ProxyFileLoader()
            st.session_state.file_proxies = loader.load_from_string(content)
            st.success(f"Loaded {len(st.session_state.file_proxies)} proxies")
        
        # Manual input
        with st.expander("Manual Input"):
            proxy_text = st.text_area(
                "Enter proxies (one per line):",
                height=100,
                placeholder="192.168.1.1:8080\nhttp://proxy.example.com:3128"
            )
            
            if st.button("Load Manual Proxies"):
                loader = ProxyFileLoader()
                manual_proxies = loader.load_from_string(proxy_text)
                st.session_state.file_proxies.extend(manual_proxies)
                st.success(f"Added {len(manual_proxies)} proxies")
        
        # Validation settings
        st.subheader("Validation Settings")
        
        validation_method = st.radio(
            "Method:",
            ["Queue Threading (Stable)", "Async (Faster)"]
        )
        
        num_concurrent = st.slider(
            "Concurrent validations:",
            min_value=5,
            max_value=50,
            value=20
        )
        
        timeout = st.slider(
            "Timeout (seconds):",
            min_value=3,
            max_value=30,
            value=10
        )
        
        max_valid = st.number_input(
            "Stop after finding:",
            min_value=0,
            max_value=1000,
            value=0,
            help="0 = validate all"
        )
        
        # Validate button
        if st.button("🚀 Start Validation", type="primary", disabled=len(st.session_state.file_proxies) == 0):
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_placeholder = st.empty()
            
            start_time = time.time()
            
            if validation_method.startswith("Queue"):
                # Threading validation
                validator = QueueBasedValidator(
                    num_threads=num_concurrent,
                    timeout=timeout
                )
                
                def progress_callback(proxy, is_valid, current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status = "✅" if is_valid else "❌"
                    status_text.text(f"{status} {proxy.host}:{proxy.port}")
                    
                    # Update stats
                    elapsed = time.time() - start_time
                    rate = current / elapsed if elapsed > 0 else 0
                    stats_placeholder.text(f"Speed: {rate:.1f} proxies/sec")
                
                valid = validator.validate_proxies(
                    st.session_state.file_proxies,
                    max_valid=max_valid if max_valid > 0 else None,
                    progress_callback=progress_callback
                )
            else:
                # Async validation
                validator = AsyncQueueValidator(
                    max_concurrent=num_concurrent,
                    timeout=timeout
                )
                
                async def async_callback(proxy, is_valid, current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status = "✅" if is_valid else "❌"
                    status_text.text(f"{status} {proxy.host}:{proxy.port}")
                    
                    elapsed = time.time() - start_time
                    rate = current / elapsed if elapsed > 0 else 0
                    stats_placeholder.text(f"Speed: {rate:.1f} proxies/sec")
                
                # Run async validation
                valid = asyncio.run(
                    validator.validate_batch(
                        st.session_state.file_proxies,
                        max_valid=max_valid if max_valid > 0 else None,
                        progress_callback=async_callback
                    )
                )
            
            # Final stats
            elapsed = time.time() - start_time
            st.success(f"✅ Found {len(valid)} working proxies in {elapsed:.1f} seconds")
            st.session_state.validated_proxies = valid
            
            # Save results
            if valid:
                # Create download buttons
                loader = ProxyFileLoader()
                
                # URL format
                with open("valid_proxies.txt", "w") as f:
                    for proxy in valid:
                        f.write(f"{proxy.as_url()}\n")
                
                with open("valid_proxies.txt", "rb") as f:
                    st.download_button(
                        "📥 Download Valid Proxies (URL)",
                        f.read(),
                        "valid_proxies.txt",
                        "text/plain"
                    )
                
                # JSON format
                json_data = json.dumps([p.to_dict() for p in valid], indent=2, default=str)
                st.download_button(
                    "📥 Download Valid Proxies (JSON)",
                    json_data,
                    "valid_proxies.json",
                    "application/json"
                )

# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

def example_usage():
    """Example of how to use the validators"""
    
    # Create test proxies
    test_proxies = [
        ProxyInfo(host="192.168.1.1", port=8080),
        ProxyInfo(host="10.0.0.1", port=3128),
        ProxyInfo(host="proxy.example.com", port=8888),
    ]
    
    print("=" * 50)
    print("PROXY VALIDATOR EXAMPLE")
    print("=" * 50)
    
    # 1. Queue-based validation
    print("\n1. Queue-based Threading Validation:")
    print("-" * 30)
    
    validator = QueueBasedValidator(num_threads=5, timeout=5)
    
    def progress_handler(proxy, is_valid, current, total):
        status = "✓" if is_valid else "✗"
        print(f"  [{current}/{total}] {status} {proxy.host}:{proxy.port}")
    
    valid_proxies = validator.validate_proxies(
        test_proxies,
        progress_callback=progress_handler
    )
    
    print(f"\nFound {len(valid_proxies)} valid proxies")
    
    # 2. Async validation
    print("\n2. Async Validation:")
    print("-" * 30)
    
    async def run_async_validation():
        async_validator = AsyncQueueValidator(max_concurrent=10, timeout=5)
        
        async def async_progress(proxy, is_valid, current, total):
            status = "✓" if is_valid else "✗"
            print(f"  [{current}/{total}] {status} {proxy.host}:{proxy.port}")
        
        return await async_validator.validate_batch(
            test_proxies,
            progress_callback=async_progress
        )
    
    valid_async = asyncio.run(run_async_validation())
    print(f"\nFound {len(valid_async)} valid proxies")
    
    # 3. File loading
    print("\n3. File Loading Example:")
    print("-" * 30)
    
    # Create a sample file
    sample_content = """
    192.168.1.1:8080
    http://proxy.example.com:3128
    socks5://10.0.0.1:1080
    # This is a comment
    user:pass@proxy.com:8888
    """
    
    loader = ProxyFileLoader()
    file_proxies = loader.load_from_string(sample_content)
    print(f"Loaded {len(file_proxies)} proxies from sample content")
    for proxy in file_proxies:
        print(f"  - {proxy.as_url()}")

if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Uncomment to run benchmark
    # proxies = [ProxyInfo(host=f"192.168.1.{i}", port=8080) for i in range(20)]
    # results = asyncio.run(ValidationBenchmark.compare_methods(proxies))
    # print("\nBenchmark Results:")
    # for method, stats in results.items():
    #     print(f"{method}: {stats['proxies_per_second']:.2f} proxies/sec")
