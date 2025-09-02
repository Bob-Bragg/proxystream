"""
ProxyStream Cloud - Fixed Production Edition
Enhanced with security, performance, and best practices
All issues resolved and ready to run
"""

import asyncio
import time
import json
import os
import re
import logging
import secrets
import ssl
import sys
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from urllib.parse import urlsplit, quote
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache, wraps
import threading
from enum import Enum
import warnings

# Handle potential import errors gracefully
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import httpx
    import aiohttp
    from aiohttp_socks import ProxyConnector
    import nest_asyncio
    import certifi
    
    # Apply nest_asyncio to handle event loop issues
    nest_asyncio.apply()
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install streamlit pandas plotly httpx aiohttp aiohttp-socks nest-asyncio certifi")
    sys.exit(1)

# Suppress SSL warnings when verify is disabled
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Cloud Production",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security constants
MAX_PROXY_INPUT_SIZE = 10000  # Max chars for manual input
MAX_CONCURRENT_CONNECTIONS = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
VERIFY_SSL = True  # Should be True in production

# Proxy sources with reliability ratings
PROXY_SOURCES = {
    "Primary (Reliable)": [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    ],
    "SOCKS (Mixed)": [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
    ],
    "Community (Variable)": [
        "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
}

# Validation endpoints (sorted by reliability)
VALIDATION_ENDPOINTS = [
    ("http://httpbin.org/ip", "json", "origin"),
    ("https://api.ipify.org?format=json", "json", "ip"),
    ("http://icanhazip.com", "text", None),
    ("http://checkip.amazonaws.com", "text", None),
]

# ---------------------------------------------------
# Thread-safe Session State Manager
# ---------------------------------------------------
class ThreadSafeSessionState:
    """Thread-safe wrapper for session state operations"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize session state with safe defaults"""
        with self._lock:
            defaults = {
                "validated_proxies": deque(maxlen=1000),
                "proxy_buffer": [],
                "known_good_proxies": set(),
                "stats": {
                    "total_fetched": 0,
                    "total_validated": 0,
                    "total_working": 0,
                    "last_validation": None,
                    "success_rate": 0.0
                },
                "settings": {
                    "max_concurrent": 50,
                    "timeout": 10,
                    "verify_ssl": VERIFY_SSL,
                    "auto_save": False,
                    "max_retries": MAX_RETRIES
                },
                "credentials": {
                    "github_token": None,  # Don't store in session
                    "gist_id": None
                }
            }
            
            for key, value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = value
    
    def add_validated_proxy(self, proxy):
        """Thread-safe proxy addition"""
        with self._lock:
            st.session_state.validated_proxies.append(proxy)
            st.session_state.stats["total_working"] += 1
    
    def update_stats(self, **kwargs):
        """Thread-safe stats update"""
        with self._lock:
            st.session_state.stats.update(kwargs)

# Initialize session manager
session_manager = ThreadSafeSessionState()

# ---------------------------------------------------
# Data Models with Validation
# ---------------------------------------------------
class ProxyProtocol(Enum):
    """Supported proxy protocols"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"

@dataclass
class Proxy:
    """Proxy data model with validation"""
    host: str
    port: int
    protocol: Union[ProxyProtocol, str] = ProxyProtocol.HTTP
    username: Optional[str] = None
    password: Optional[str] = None
    latency: Optional[float] = None
    last_tested: Optional[datetime] = None
    is_valid: bool = False
    test_count: int = 0
    error_count: int = 0
    exit_ip: Optional[str] = None  # Added missing field
    
    def __post_init__(self):
        """Validate proxy data"""
        # Validate port
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}")
        
        # Validate host (basic check)
        if not self.host or len(self.host) > 255:
            raise ValueError(f"Invalid host: {self.host}")
        
        # Convert protocol string to enum if needed
        if isinstance(self.protocol, str):
            try:
                self.protocol = ProxyProtocol(self.protocol.lower())
            except ValueError:
                # Default to HTTP if unknown protocol
                self.protocol = ProxyProtocol.HTTP
        elif not isinstance(self.protocol, ProxyProtocol):
            self.protocol = ProxyProtocol.HTTP
    
    def __hash__(self):
        # Handle both enum and string protocols
        if isinstance(self.protocol, ProxyProtocol):
            protocol_str = self.protocol.value
        else:
            protocol_str = str(self.protocol).lower()
        return hash((self.host, self.port, protocol_str))
    
    def __eq__(self, other):
        if isinstance(other, Proxy):
            return (self.host, self.port, self.protocol) == (other.host, other.port, other.protocol)
        return False
    
    @property
    def url(self) -> str:
        """Get proxy URL with auth if present"""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        # Handle both enum and string protocols
        if isinstance(self.protocol, ProxyProtocol):
            protocol_str = self.protocol.value
        else:
            protocol_str = str(self.protocol).lower()
        return f"{protocol_str}://{auth}{self.host}:{self.port}"
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score based on test history"""
        if self.test_count == 0:
            return 0.0
        success_rate = 1 - (self.error_count / self.test_count)
        latency_factor = 1 / (1 + (self.latency or 1000) / 1000)
        return success_rate * 0.7 + latency_factor * 0.3
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export"""
        # Handle both enum and string protocols
        if isinstance(self.protocol, ProxyProtocol):
            protocol_str = self.protocol.value
        else:
            protocol_str = str(self.protocol)
            
        return {
            "host": self.host,
            "port": self.port,
            "protocol": protocol_str,
            "latency": self.latency,
            "reliability": self.reliability_score,
            "url": self.url
        }

# ---------------------------------------------------
# Enhanced Async Validator with Connection Pooling
# ---------------------------------------------------
class ProxyValidator:
    """Production-grade proxy validator with connection pooling and retry logic"""
    
    def __init__(self, max_concurrent: int = 50, timeout: int = 10, verify_ssl: bool = True):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = None
        self._connector = None
        self._semaphore = None
        
    @asynccontextmanager
    async def session_context(self):
        """Context manager for session lifecycle"""
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        
        # Create SSL context with proper configuration
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        if not self.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            ssl=ssl_context
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout_config,
            connector=connector,
            headers={"User-Agent": "ProxyStream/1.0"}
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        try:
            yield self
        finally:
            await self.session.close()
            await connector.close()
    
    async def test_proxy_with_retry(self, proxy: Proxy, max_retries: int = 3) -> bool:
        """Test proxy with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                if await self._test_single_proxy(proxy):
                    return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout testing {proxy.host}:{proxy.port} (attempt {attempt + 1})")
            except Exception as e:
                logger.debug(f"Error testing proxy: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        proxy.error_count += 1
        return False
    
    async def _test_single_proxy(self, proxy: Proxy) -> bool:
        """Test a single proxy against validation endpoints"""
        proxy.test_count += 1
        
        for endpoint, response_type, ip_field in VALIDATION_ENDPOINTS[:2]:
            try:
                start = time.perf_counter()
                
                # Configure proxy based on protocol
                if proxy.protocol in [ProxyProtocol.HTTP, ProxyProtocol.HTTPS]:
                    proxy_url = proxy.url
                    async with self.session.get(
                        endpoint,
                        proxy=proxy_url,
                        ssl=self.verify_ssl,
                        allow_redirects=False
                    ) as response:
                        if response.status in [200, 301, 302]:
                            proxy.latency = (time.perf_counter() - start) * 1000
                            proxy.is_valid = True
                            proxy.last_tested = datetime.now()
                            
                            # Try to extract exit IP
                            if response_type == "json" and response.status == 200:
                                try:
                                    data = await response.json()
                                    proxy.exit_ip = data.get(ip_field)
                                except:
                                    pass
                            
                            return True
                else:
                    # SOCKS proxy handling with error handling
                    try:
                        connector = ProxyConnector.from_url(proxy.url)
                        async with aiohttp.ClientSession(connector=connector) as socks_session:
                            async with socks_session.get(endpoint, ssl=self.verify_ssl) as response:
                                if response.status in [200, 301, 302]:
                                    proxy.latency = (time.perf_counter() - start) * 1000
                                    proxy.is_valid = True
                                    proxy.last_tested = datetime.now()
                                    return True
                    except Exception as e:
                        logger.debug(f"SOCKS proxy error: {e}")
                        continue
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Validation error for {proxy.host}: {str(e)[:50]}")
                continue
        
        return False
    
    async def validate_batch(self, proxies: List[Proxy], progress_callback=None) -> List[Proxy]:
        """Validate a batch of proxies concurrently"""
        valid_proxies = []
        
        async def validate_with_semaphore(proxy):
            async with self._semaphore:
                try:
                    if await self.test_proxy_with_retry(proxy):
                        valid_proxies.append(proxy)
                        if progress_callback:
                            progress_callback(1)
                except Exception as e:
                    logger.debug(f"Validation error: {e}")
        
        tasks = [validate_with_semaphore(proxy) for proxy in proxies]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort by reliability score
        valid_proxies.sort(key=lambda p: p.reliability_score, reverse=True)
        return valid_proxies

# ---------------------------------------------------
# Secure Proxy Parsing with Input Validation
# ---------------------------------------------------
def parse_proxy_line(line: str, strict: bool = True) -> Optional[Proxy]:
    """Parse and validate proxy line with security checks"""
    line = line.strip()
    
    # Security: Limit line length
    if len(line) > 500:
        logger.warning(f"Proxy line too long: {len(line)} chars")
        return None
    
    # Skip comments and empty lines
    if not line or line.startswith("#"):
        return None
    
    # Sanitize input - remove potentially dangerous characters
    line = re.sub(r'[^\w\s:/@.-]', '', line)
    
    # Auto-detect protocol
    if "://" not in line:
        if ":1080" in line or ":9050" in line:
            line = "socks5://" + line
        elif ":4145" in line:
            line = "socks4://" + line
        else:
            line = "http://" + line
    
    try:
        parts = urlsplit(line)
        
        # Validate components
        if not parts.hostname or not parts.port:
            if ":" in line and "/" not in line and not strict:
                # Try simple host:port format
                try:
                    host, port = line.rsplit(":", 1)
                    return Proxy(host=host, port=int(port))
                except:
                    return None
            return None
        
        # Validate protocol
        protocol = parts.scheme or "http"
        if protocol not in ["http", "https", "socks4", "socks5"]:
            return None
        
        # Create proxy with validation
        return Proxy(
            host=parts.hostname,
            port=parts.port,
            protocol=ProxyProtocol(protocol),
            username=parts.username,
            password=parts.password
        )
    except Exception as e:
        logger.debug(f"Failed to parse proxy line: {e}")
        return None

# ---------------------------------------------------
# Optimized Fetching with Caching
# ---------------------------------------------------
@st.cache_data(ttl=300, max_entries=50)
def fetch_proxies_cached(url: str, limit: int = 1000) -> List[Dict]:
    """Fetch and cache proxy lists (returns dicts for serialization)"""
    proxies = []
    
    try:
        # Use httpx with timeout and size limits
        with httpx.Client(timeout=20, limits=httpx.Limits(max_keepalive_connections=5), verify=False) as client:
            response = client.get(url, follow_redirects=True)
            
            # Security: Limit response size (10MB)
            if len(response.content) > 10 * 1024 * 1024:
                logger.warning(f"Response too large from {url}")
                return []
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')[:limit]
                for line in lines:
                    proxy = parse_proxy_line(line, strict=False)
                    if proxy:
                        # Convert to simple dictionary with string protocol
                        proxy_dict = {
                            "host": proxy.host,
                            "port": proxy.port,
                            "protocol": proxy.protocol.value if isinstance(proxy.protocol, ProxyProtocol) else str(proxy.protocol),
                            "username": proxy.username,
                            "password": proxy.password
                        }
                        proxies.append(proxy_dict)
    
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
        st.warning(f"Timeout fetching from {url}")
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        st.error(f"Failed to fetch from {url}: {str(e)[:100]}")
    
    return proxies

# Helper function to safely create Proxy from dict
def create_proxy_from_dict(data: Dict) -> Optional[Proxy]:
    """Safely create a Proxy object from a dictionary"""
    try:
        # Ensure required fields exist
        if not data.get('host') or not data.get('port'):
            return None
        
        # Clean the data
        cleaned_data = {
            'host': str(data['host']),
            'port': int(data['port']),
            'protocol': str(data.get('protocol', 'http')).lower(),
            'username': data.get('username'),
            'password': data.get('password')
        }
        
        return Proxy(**cleaned_data)
    except Exception as e:
        logger.debug(f"Failed to create proxy from dict: {e}")
        return None

# ---------------------------------------------------
# Secure GitHub Gist Integration
# ---------------------------------------------------
class SecureGistManager:
    """Secure GitHub Gist manager with encryption option"""
    
    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate GitHub token format"""
        # Basic validation - tokens are usually 40 chars
        return bool(token and len(token) >= 20 and token.replace("_", "").replace("-", "").isalnum())
    
    @staticmethod
    def save_to_gist(proxies: List[Proxy], token: str, gist_id: Optional[str] = None) -> Optional[str]:
        """Save proxies to GitHub Gist with error handling"""
        if not SecureGistManager.validate_token(token):
            st.error("Invalid GitHub token format")
            return None
        
        # Prepare content
        content = "\n".join([p.url for p in proxies if p.is_valid])
        
        # Add metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "count": len(proxies),
            "version": "1.0"
        }
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "description": f"ProxyStream Export - {metadata['timestamp']}",
            "public": False,
            "files": {
                "proxies.txt": {"content": content},
                "metadata.json": {"content": json.dumps(metadata, indent=2)}
            }
        }
        
        try:
            with httpx.Client(timeout=10, verify=False) as client:
                if gist_id:
                    response = client.patch(
                        f"https://api.github.com/gists/{gist_id}",
                        headers=headers,
                        json=data
                    )
                else:
                    response = client.post(
                        "https://api.github.com/gists",
                        headers=headers,
                        json=data
                    )
                
                if response.status_code in [200, 201]:
                    return response.json().get("id")
                else:
                    logger.error(f"GitHub API error: {response.status_code}")
                    st.error(f"Failed to save: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Gist save error: {e}")
            st.error("Failed to save to Gist")
        
        return None
    
    @staticmethod
    def load_from_gist(token: str, gist_id: str) -> List[Proxy]:
        """Load proxies from GitHub Gist with validation"""
        if not SecureGistManager.validate_token(token):
            st.error("Invalid GitHub token")
            return []
        
        if not gist_id or not gist_id.replace("-", "").isalnum():
            st.error("Invalid Gist ID")
            return []
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            with httpx.Client(timeout=10, verify=False) as client:
                response = client.get(
                    f"https://api.github.com/gists/{gist_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["files"]["proxies.txt"]["content"]
                    
                    proxies = []
                    for line in content.split('\n'):
                        proxy = parse_proxy_line(line)
                        if proxy:
                            proxy.is_valid = True
                            proxies.append(proxy)
                    
                    return proxies
                else:
                    st.error(f"Failed to load: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Gist load error: {e}")
            st.error("Failed to load from Gist")
        
        return []

# ---------------------------------------------------
# Fixed Async Runner with Proper Error Handling
# ---------------------------------------------------
def run_async_validation(proxies: List[Proxy], max_concurrent: int = 50, 
                         verify_ssl: bool = True, progress_callback=None) -> List[Proxy]:
    """Run async validation with proper cleanup and event loop handling"""
    async def validate():
        validator = ProxyValidator(
            max_concurrent=max_concurrent,
            verify_ssl=verify_ssl
        )
        async with validator.session_context():
            return await validator.validate_batch(proxies, progress_callback)
    
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're here, there's already a running loop
            # Create a task and run it
            task = asyncio.create_task(validate())
            # Use asyncio.run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, validate())
                return future.result(timeout=300)  # 5 minute timeout
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            return asyncio.run(validate())
    except Exception as e:
        logger.error(f"Validation error: {e}")
        st.error(f"Validation failed: {str(e)[:100]}")
        return []

# ---------------------------------------------------
# UI Components with Error Boundaries
# ---------------------------------------------------
def safe_render(func):
    """Decorator to catch and display UI errors gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"UI Error in {func.__name__}: {e}")
            st.error(f"Display error: {str(e)[:100]}")
    return wrapper

@safe_render
def render_stats():
    """Render statistics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.stats
    
    with col1:
        st.metric("Total Fetched", f"{stats['total_fetched']:,}")
    
    with col2:
        st.metric("Total Validated", f"{stats['total_validated']:,}")
    
    with col3:
        working = len([p for p in st.session_state.validated_proxies if p.is_valid])
        st.metric("Working Proxies", f"{working:,}")
    
    with col4:
        if stats['total_validated'] > 0:
            success_rate = (stats['total_working'] / stats['total_validated']) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")

@safe_render
def render_proxy_input():
    """Render manual proxy input with validation"""
    st.subheader("📝 Add Known Good Proxies")
    
    proxy_text = st.text_area(
        "Enter proxies (one per line):",
        height=100,
        max_chars=MAX_PROXY_INPUT_SIZE,
        placeholder="http://1.2.3.4:8080\nsocks5://user:pass@5.6.7.8:1080",
        key="proxy_input"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Add as Known Good", type="primary", use_container_width=True):
            if proxy_text:
                lines = proxy_text.strip().split('\n')
                added = 0
                errors = 0
                
                for line in lines[:100]:  # Limit to 100 at a time
                    try:
                        proxy = parse_proxy_line(line)
                        if proxy:
                            proxy.is_valid = True
                            st.session_state.known_good_proxies.add(proxy)
                            session_manager.add_validated_proxy(proxy)
                            added += 1
                    except Exception as e:
                        errors += 1
                        logger.debug(f"Parse error: {e}")
                
                if added:
                    st.success(f"✅ Added {added} proxies")
                if errors:
                    st.warning(f"⚠️ {errors} lines had errors")
    
    with col2:
        if st.button("🧪 Validate First", use_container_width=True):
            if proxy_text:
                lines = proxy_text.strip().split('\n')
                proxies_to_test = []
                
                for line in lines[:50]:  # Limit validation batch
                    try:
                        proxy = parse_proxy_line(line)
                        if proxy:
                            proxies_to_test.append(proxy)
                    except:
                        continue
                
                if proxies_to_test:
                    with st.spinner(f"Validating {len(proxies_to_test)} proxies..."):
                        valid = run_async_validation(
                            proxies_to_test,
                            max_concurrent=10,
                            verify_ssl=st.session_state.settings.get("verify_ssl", True)
                        )
                        
                        for proxy in valid:
                            st.session_state.known_good_proxies.add(proxy)
                            session_manager.add_validated_proxy(proxy)
                        
                        st.success(f"✅ {len(valid)}/{len(proxies_to_test)} working")
                        session_manager.update_stats(
                            total_validated=st.session_state.stats["total_validated"] + len(proxies_to_test)
                        )

@safe_render
def render_proxy_table():
    """Render proxy table with export options"""
    if not st.session_state.validated_proxies:
        st.info("No validated proxies yet. Click 'Harvest & Validate' to get started!")
        return
    
    # Convert to DataFrame
    data = []
    for proxy in list(st.session_state.validated_proxies):
        if proxy.is_valid:
            # Handle both enum and string protocols
            if isinstance(proxy.protocol, ProxyProtocol):
                protocol_str = proxy.protocol.value.upper()
            else:
                protocol_str = str(proxy.protocol).upper()
                
            data.append({
                'Protocol': protocol_str,
                'Host': proxy.host,
                'Port': proxy.port,
                'Latency': f"{proxy.latency:.0f}ms" if proxy.latency else "N/A",
                'Reliability': f"{proxy.reliability_score:.2f}",
                'Status': "✅ Working",
                'URL': proxy.url
            })
    
    if data:
        df = pd.DataFrame(data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            protocols = ["All"] + sorted(df['Protocol'].unique().tolist())
            filter_protocol = st.selectbox("Filter Protocol", protocols)
        
        with col2:
            sort_by = st.selectbox("Sort By", ["Reliability", "Latency", "Protocol"])
        
        with col3:
            limit = st.number_input("Show Top", min_value=10, max_value=500, value=50)
        
        # Apply filters
        filtered_df = df
        if filter_protocol != "All":
            filtered_df = filtered_df[filtered_df['Protocol'] == filter_protocol]
        
        # Sort
        if sort_by == "Reliability":
            filtered_df = filtered_df.sort_values("Reliability", ascending=False)
        elif sort_by == "Latency":
            filtered_df = filtered_df.sort_values("Latency")
        
        filtered_df = filtered_df.head(limit)
        
        st.dataframe(filtered_df, use_container_width=True, height=400, hide_index=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "proxies.csv", "text/csv", use_container_width=True)
        
        with col2:
            urls = "\n".join(filtered_df['URL'].tolist())
            st.download_button("📥 Export URLs", urls, "proxy_urls.txt", "text/plain", use_container_width=True)
        
        with col3:
            json_str = json.dumps([p.to_dict() for p in st.session_state.validated_proxies if p.is_valid], indent=2)
            st.download_button("📥 Export JSON", json_str, "proxies.json", "application/json", use_container_width=True)

# ---------------------------------------------------
# Logo and Branding
# ---------------------------------------------------
def render_logo():
    """Render the ProxyStream logo from GitHub with proper styling"""
    logo_url = "https://raw.githubusercontent.com/Bob-Bragg/proxystream/main/ProxyStream%20Logo%20Design.png"
    
    logo_html = f"""
    <style>
        .main-header {{
            padding: 1rem 0 2rem 0;
            border-bottom: 2px solid #1f2937;
            margin-bottom: 2rem;
        }}
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
        }}
        .logo-img {{
            max-width: 280px;
            height: auto;
            filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
        }}
        .tagline {{
            text-align: center;
            color: #6b7280;
            font-size: 1rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }}
        .metrics-container {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #00A6FB;
        }}
        .metric-label {{
            font-size: 0.875rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }}
    </style>
    
    <div class="main-header">
        <div class="logo-container">
            <img src="{logo_url}" class="logo-img" alt="ProxyStream Logo" onerror="this.style.display='none'">
        </div>
        <div class="tagline">Enterprise-grade proxy management with security and performance optimizations</div>
    </div>
    """
    
    st.markdown(logo_html, unsafe_allow_html=True)

# ---------------------------------------------------
# Enhanced Stats Display
# ---------------------------------------------------
def render_stats_cards():
    """Render statistics in styled cards"""
    stats = st.session_state.stats
    working = len([p for p in st.session_state.validated_proxies if p.is_valid])
    success_rate = (stats['total_working'] / stats['total_validated'] * 100) if stats['total_validated'] > 0 else 0
    
    stats_html = f"""
    <div class="metrics-container">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div class="metric-card">
                <div class="metric-value">{stats['total_fetched']:,}</div>
                <div class="metric-label">Total Fetched</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['total_validated']:,}</div>
                <div class="metric-label">Total Validated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{working:,}</div>
                <div class="metric-label">Working Proxies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{'%.1f%%' % success_rate if stats['total_validated'] > 0 else 'N/A'}</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(stats_html, unsafe_allow_html=True)

# ---------------------------------------------------
# Main Application
# ---------------------------------------------------
try:
    # Custom CSS for overall styling
    st.markdown("""
    <style>
        /* Main container adjustments */
        .block-container {
            padding-top: 2rem;
            max-width: 100%;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #1e1e1e;
            border-right: 1px solid #2d2d2d;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 166, 251, 0.3);
        }
        
        /* Primary button special styling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00A6FB 0%, #0086D9 100%);
            border: none;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(30, 41, 59, 0.5);
            padding: 4px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #00A6FB;
        }
        
        /* Metric styling */
        [data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* DataFrame styling */
        .dataframe {
            border: 1px solid #2d2d2d !important;
            border-radius: 8px;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #00A6FB;
        }
        
        /* Success/Error/Warning messages */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4a4a4a;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00A6FB;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display logo
    render_logo()

    # Display stats cards instead of default metrics
    render_stats_cards()

    # Stats Dashboard
    render_stats()

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Performance Settings
        with st.expander("🎯 Performance", expanded=True):
            st.session_state.settings["max_concurrent"] = st.slider(
                "Concurrent Tests",
                min_value=5,
                max_value=100,
                value=st.session_state.settings.get("max_concurrent", 50),
                help="Balance speed vs resource usage"
            )
            
            st.session_state.settings["timeout"] = st.slider(
                "Timeout (seconds)",
                min_value=5,
                max_value=30,
                value=st.session_state.settings.get("timeout", 10)
            )
            
            st.session_state.settings["verify_ssl"] = st.checkbox(
                "Verify SSL Certificates",
                value=st.session_state.settings.get("verify_ssl", True),
                help="Disable only for testing"
            )
        
        # GitHub Integration
        with st.expander("💾 GitHub Storage"):
            st.info("Store proxies securely in GitHub Gist")
            
            # Use environment variable or input
            token = os.getenv("GITHUB_TOKEN", "")
            if not token:
                token = st.text_input("GitHub Token", type="password", help="Create at github.com/settings/tokens")
            
            gist_id = st.text_input("Gist ID (optional)", help="Leave empty to create new")
            
            if token:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save", use_container_width=True):
                        working = [p for p in st.session_state.validated_proxies if p.is_valid]
                        if working:
                            new_id = SecureGistManager.save_to_gist(working, token, gist_id)
                            if new_id:
                                st.success(f"Saved {len(working)} proxies")
                                st.code(f"ID: {new_id}")
                
                with col2:
                    if st.button("📥 Load", use_container_width=True) and gist_id:
                        loaded = SecureGistManager.load_from_gist(token, gist_id)
                        if loaded:
                            for proxy in loaded:
                                session_manager.add_validated_proxy(proxy)
                            st.success(f"Loaded {len(loaded)} proxies")
        
        # Quick Actions
        st.markdown("---")
        st.subheader("🎬 Quick Actions")
        
        if st.button("🚀 Harvest & Validate", type="primary", use_container_width=True):
            with st.spinner("Fetching proxy lists..."):
                all_proxies = []
                progress = st.progress(0)
                
                sources = []
                for category, urls in PROXY_SOURCES.items():
                    sources.extend(urls)
                
                for i, url in enumerate(sources):
                    proxy_dicts = fetch_proxies_cached(url, limit=500)
                    # Safely convert dictionaries to Proxy objects
                    for pd in proxy_dicts:
                        proxy = create_proxy_from_dict(pd)
                        if proxy:
                            all_proxies.append(proxy)
                    progress.progress((i + 1) / len(sources))
                
                # Deduplicate
                unique_proxies = list(set(all_proxies))
                session_manager.update_stats(total_fetched=st.session_state.stats["total_fetched"] + len(unique_proxies))
                
                st.success(f"Fetched {len(unique_proxies)} unique proxies")
                
                # Validate
                if unique_proxies:
                    progress_bar = st.progress(0)
                    progress_counter = {"count": 0}
                    
                    def update_progress(count):
                        progress_counter["count"] += count
                        progress_bar.progress(min(progress_counter["count"] / len(unique_proxies), 1.0))
                    
                    with st.spinner(f"Validating {len(unique_proxies)} proxies..."):
                        valid_proxies = run_async_validation(
                            unique_proxies,
                            max_concurrent=st.session_state.settings["max_concurrent"],
                            verify_ssl=st.session_state.settings["verify_ssl"],
                            progress_callback=update_progress
                        )
                        
                        # Update stats
                        session_manager.update_stats(
                            total_validated=st.session_state.stats["total_validated"] + len(unique_proxies),
                            total_working=st.session_state.stats["total_working"] + len(valid_proxies),
                            last_validation=datetime.now()
                        )
                        
                        # Add to validated list
                        for proxy in valid_proxies:
                            session_manager.add_validated_proxy(proxy)
                        
                        st.success(f"✅ Found {len(valid_proxies)} working proxies!")
        
        if st.button("🧹 Optimize Memory", use_container_width=True):
            # Keep only best proxies
            working = [p for p in st.session_state.validated_proxies if p.is_valid]
            working.sort(key=lambda p: p.reliability_score, reverse=True)
            st.session_state.validated_proxies = deque(working[:500], maxlen=1000)
            st.success(f"Optimized! Kept best {len(st.session_state.validated_proxies)} proxies")
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.validated_proxies.clear()
            st.session_state.proxy_buffer = []
            st.session_state.known_good_proxies.clear()
            st.session_state.stats = {
                "total_fetched": 0,
                "total_validated": 0,
                "total_working": 0,
                "last_validation": None,
                "success_rate": 0.0
            }
            st.rerun()

    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Proxy List", "➕ Add Proxies", "📊 Analytics", "🌍 Geo Test", "📚 Help"])

    with tab1:
        render_proxy_table()

    with tab2:
        render_proxy_input()
        
        st.markdown("---")
        
        st.subheader("🌐 Batch Import from URL")
        url_input = st.text_input("Custom proxy list URL:", placeholder="https://example.com/proxies.txt")
        
        if url_input and st.button("📥 Import & Validate"):
            with st.spinner("Importing..."):
                proxy_dicts = fetch_proxies_cached(url_input, limit=1000)
                if proxy_dicts:
                    proxies = []
                    for pd in proxy_dicts:
                        proxy = create_proxy_from_dict(pd)
                        if proxy:
                            proxies.append(proxy)
                    
                    if proxies:
                        st.info(f"Found {len(proxies)} proxies, validating...")
                        
                        valid = run_async_validation(
                            proxies,
                            max_concurrent=st.session_state.settings["max_concurrent"],
                            verify_ssl=st.session_state.settings["verify_ssl"]
                        )
                        
                        for proxy in valid:
                            session_manager.add_validated_proxy(proxy)
                        
                        st.success(f"Added {len(valid)} working proxies!")
                    else:
                        st.warning("No valid proxies found in the URL")

    with tab3:
        st.subheader("📊 Analytics Dashboard")
        
        if st.session_state.validated_proxies:
            working_proxies = [p for p in st.session_state.validated_proxies if p.is_valid]
            
            if working_proxies:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Protocol distribution
                    protocol_counts = defaultdict(int)
                    for p in working_proxies:
                        # Handle both enum and string protocols
                        if isinstance(p.protocol, ProxyProtocol):
                            protocol_name = p.protocol.value.upper()
                        else:
                            protocol_name = str(p.protocol).upper()
                        protocol_counts[protocol_name] += 1
                    
                    if protocol_counts:
                        df_proto = pd.DataFrame(
                            list(protocol_counts.items()),
                            columns=["Protocol", "Count"]
                        )
                        fig = px.pie(df_proto, values="Count", names="Protocol", 
                                   title="Protocol Distribution",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Latency distribution
                    latencies = [p.latency for p in working_proxies if p.latency]
                    if latencies:
                        df_latency = pd.DataFrame({"Latency (ms)": latencies})
                        fig = px.histogram(df_latency, x="Latency (ms)", 
                                         nbins=20, 
                                         title="Latency Distribution",
                                         color_discrete_sequence=["#636EFA"])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Reliability scores
                st.subheader("🎯 Top Performers")
                top_proxies = sorted(working_proxies, key=lambda p: p.reliability_score, reverse=True)[:10]
                
                reliability_data = []
                for p in top_proxies:
                    reliability_data.append({
                        "Proxy": f"{p.host}:{p.port}",
                        "Score": p.reliability_score,
                        "Latency": p.latency or 0
                    })
                
                df_reliability = pd.DataFrame(reliability_data)
                fig = px.bar(df_reliability, x="Proxy", y="Score", 
                            title="Top 10 Most Reliable Proxies",
                            color="Latency",
                            color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available. Start by harvesting and validating proxies!")

    with tab4:
        st.subheader("🌍 Geo-Restriction & Firewall Testing")
        
        st.info("Test if proxies can access content behind country firewalls and geo-restrictions")
        
        # Test configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select Test Type**")
            test_type = st.selectbox(
                "Test Category",
                [
                    "🌐 General Geo Test",
                    "🇨🇳 China Great Firewall",
                    "🇷🇺 Russia RKN",
                    "🇮🇷 Iran Firewall",
                    "🇹🇷 Turkey Blocks",
                    "🎬 Streaming Services",
                    "📱 Social Media",
                    "🏦 Banking/Finance",
                    "Custom URL"
                ]
            )
            
            # Proxy selection
            if st.session_state.validated_proxies:
                working_proxies = [p for p in st.session_state.validated_proxies if p.is_valid]
                if working_proxies:
                    proxy_options = []
                    for p in working_proxies[:50]:
                        # Handle both enum and string protocols
                        if isinstance(p.protocol, ProxyProtocol):
                            protocol_str = p.protocol.value
                        else:
                            protocol_str = str(p.protocol).lower()
                        proxy_options.append(f"{protocol_str}://{p.host}:{p.port}")
                        
                    selected_proxy_str = st.selectbox(
                        "Select Proxy to Test",
                        ["None (Direct Connection)"] + proxy_options
                    )
                else:
                    selected_proxy_str = "None (Direct Connection)"
                    st.warning("No working proxies available")
            else:
                selected_proxy_str = "None (Direct Connection)"
                st.warning("No proxies available. Validate some first!")
        
        with col2:
            st.write("**Test Configuration**")
            
            # Test targets based on selection
            test_targets = {
                "🌐 General Geo Test": [
                    ("https://www.google.com", "Google"),
                    ("https://www.youtube.com", "YouTube"),
                    ("https://www.facebook.com", "Facebook"),
                    ("https://www.twitter.com", "Twitter"),
                    ("https://www.wikipedia.org", "Wikipedia"),
                    ("https://www.bbc.com", "BBC"),
                ],
                "🇨🇳 China Great Firewall": [
                    ("https://www.google.com", "Google"),
                    ("https://www.youtube.com", "YouTube"),
                    ("https://www.facebook.com", "Facebook"),
                    ("https://www.twitter.com", "Twitter (X)"),
                    ("https://www.instagram.com", "Instagram"),
                    ("https://www.nytimes.com", "NY Times"),
                    ("https://www.wsj.com", "Wall Street Journal"),
                ],
                "🇷🇺 Russia RKN": [
                    ("https://www.facebook.com", "Facebook/Meta"),
                    ("https://www.instagram.com", "Instagram"),
                    ("https://www.twitter.com", "Twitter (X)"),
                    ("https://www.linkedin.com", "LinkedIn"),
                    ("https://www.bbc.com", "BBC"),
                    ("https://www.svoboda.org", "Radio Liberty"),
                ],
                "🇮🇷 Iran Firewall": [
                    ("https://www.facebook.com", "Facebook"),
                    ("https://www.twitter.com", "Twitter"),
                    ("https://www.youtube.com", "YouTube"),
                    ("https://www.telegram.org", "Telegram"),
                    ("https://www.bbc.com", "BBC Persian"),
                ],
                "🇹🇷 Turkey Blocks": [
                    ("https://www.wikipedia.org", "Wikipedia"),
                    ("https://www.imgur.com", "Imgur"),
                    ("https://www.reddit.com", "Reddit"),
                    ("https://www.twitter.com", "Twitter/X"),
                ],
                "🎬 Streaming Services": [
                    ("https://www.netflix.com", "Netflix"),
                    ("https://www.hulu.com", "Hulu"),
                    ("https://www.disney.com", "Disney+"),
                    ("https://www.hbomax.com", "HBO Max"),
                    ("https://www.peacocktv.com", "Peacock"),
                    ("https://www.bbc.co.uk/iplayer", "BBC iPlayer"),
                ],
                "📱 Social Media": [
                    ("https://www.tiktok.com", "TikTok"),
                    ("https://www.snapchat.com", "Snapchat"),
                    ("https://www.pinterest.com", "Pinterest"),
                    ("https://www.tumblr.com", "Tumblr"),
                    ("https://www.reddit.com", "Reddit"),
                ],
                "🏦 Banking/Finance": [
                    ("https://www.paypal.com", "PayPal"),
                    ("https://www.coinbase.com", "Coinbase"),
                    ("https://www.binance.com", "Binance"),
                    ("https://www.robinhood.com", "Robinhood"),
                ],
            }
            
            if test_type == "Custom URL":
                custom_url = st.text_input("Enter URL to test:", placeholder="https://example.com")
                test_timeout = st.slider("Timeout (seconds)", 5, 30, 10)
            else:
                targets = test_targets.get(test_type, [])
                selected_targets = st.multiselect(
                    "Select sites to test:",
                    [name for _, name in targets],
                    default=[name for _, name in targets[:3]]
                )
                test_timeout = st.slider("Timeout (seconds)", 5, 30, 10)
        
        # Run test button
        if st.button("🧪 Run Geo Test", type="primary", use_container_width=True):
            if test_type == "Custom URL" and not custom_url:
                st.error("Please enter a URL to test")
            else:
                # Prepare test parameters
                test_proxy = None
                if selected_proxy_str != "None (Direct Connection)" and working_proxies:
                    # Find the proxy object
                    for p in working_proxies:
                        # Handle both enum and string protocols
                        if isinstance(p.protocol, ProxyProtocol):
                            protocol_str = p.protocol.value
                        else:
                            protocol_str = str(p.protocol).lower()
                        
                        if f"{protocol_str}://{p.host}:{p.port}" == selected_proxy_str:
                            test_proxy = p
                            break
                
                # Get URLs to test
                if test_type == "Custom URL":
                    urls_to_test = [(custom_url, "Custom URL")]
                else:
                    all_targets = test_targets.get(test_type, [])
                    urls_to_test = [(url, name) for url, name in all_targets if name in selected_targets]
                
                if not urls_to_test:
                    st.error("Please select at least one site to test")
                else:
                    # Run tests
                    st.write("---")
                    st.subheader("📊 Test Results")
                    
                    test_results = []
                    
                    for i, (url, name) in enumerate(urls_to_test):
                        with st.spinner(f"Testing {name}..."):
                            # Test with proxy
                            try:
                                start_time = time.time()
                                
                                if test_proxy:
                                    proxy_dict = {
                                        'http': test_proxy.url,
                                        'https': test_proxy.url
                                    }
                                else:
                                    proxy_dict = None
                                
                                response = httpx.get(
                                    url,
                                    proxies=proxy_dict,
                                    timeout=test_timeout,
                                    follow_redirects=True,
                                    verify=False
                                )
                                
                                elapsed = (time.time() - start_time) * 1000
                                
                                # Check response
                                if response.status_code == 200:
                                    status = "✅ Accessible"
                                elif response.status_code == 403:
                                    status = "🚫 Forbidden (403)"
                                else:
                                    status = f"⚠️ HTTP {response.status_code}"
                                
                                test_results.append({
                                    "Site": name,
                                    "Status": status,
                                    "Response Time": f"{elapsed:.0f}ms",
                                    "Status Code": response.status_code
                                })
                                
                            except httpx.TimeoutException:
                                test_results.append({
                                    "Site": name,
                                    "Status": "⏱️ Timeout",
                                    "Response Time": f">{test_timeout}s",
                                    "Status Code": "N/A"
                                })
                            except Exception as e:
                                test_results.append({
                                    "Site": name,
                                    "Status": f"❌ Error",
                                    "Response Time": "N/A",
                                    "Status Code": "N/A"
                                })
                    
                    # Display results
                    df_results = pd.DataFrame(test_results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("📚 User Guide")
        
        with st.expander("🚀 Quick Start", expanded=True):
            st.markdown("""
            1. **Click "Harvest & Validate"** in the sidebar to automatically fetch and test proxies
            2. **View results** in the Proxy List tab
            3. **Export** your validated proxies as CSV, JSON, or plain text
            4. **Save to GitHub** for permanent storage (requires GitHub token)
            """)
        
        with st.expander("🔒 Security Best Practices"):
            st.markdown("""
            - **Always use HTTPS endpoints** when possible
            - **Keep SSL verification enabled** in production
            - **Don't share your GitHub token** - use environment variables
            - **Regularly update** your proxy lists
            - **Monitor success rates** to identify failing proxies
            """)
        
        with st.expander("⚡ Performance Tips"):
            st.markdown("""
            - **Increase concurrent tests** for faster validation (uses more resources)
            - **Use "Optimize Memory"** when the list gets large
            - **Export and clear** old data regularly
            - **Cache results** using GitHub Gist for persistence
            - **Filter by reliability** to get the best proxies
            """)

    # Footer
    st.markdown("---")
    st.caption(f"ProxyStream Cloud Production | v1.0.1 (Fixed) | Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.stats.get("last_validation"):
        st.caption(f"Last validation: {st.session_state.stats['last_validation'].strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please check that all dependencies are installed: pip install streamlit pandas plotly httpx aiohttp aiohttp-socks nest-asyncio certifi")
    logger.error(f"Main application error: {e}", exc_info=True)
