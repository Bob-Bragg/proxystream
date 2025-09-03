"""
ProxyStream Ultimate - Complete Merged Edition
Combines harvest/validate from Cloud edition with advanced chaining from Advanced edition
All features included: harvesting, validation, chaining, geo-testing, visualization
"""

import asyncio
import ssl
import time
import json
import sqlite3
import base64
import os
import re
import logging
import secrets
import sys
import threading
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from urllib.parse import urlsplit, quote
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache, wraps
from enum import Enum
import ipaddress
import statistics
import random
import socket
import pathlib

# Handle imports with error checking
try:
    import streamlit as st
    import streamlit.components.v1 as components
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import httpx
    import aiohttp
    from aiohttp_socks import ProxyConnector, ChainProxyConnector
    import nest_asyncio
    import certifi
    import numpy as np
    
    # Apply nest_asyncio to handle event loop issues
    nest_asyncio.apply()
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install streamlit pandas plotly httpx aiohttp aiohttp-socks nest-asyncio certifi numpy")
    sys.exit(1)

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Ultimate",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------
MAX_PROXY_INPUT_SIZE = 10000
MAX_CONCURRENT_CONNECTIONS = 100
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
VERIFY_SSL = True
DB_PATH = "proxystream_ultimate.db"

# Enhanced proxy sources (combining both versions)
PROXY_SOURCES = {
    "Primary (Reliable)": [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
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

# Validation endpoints
VALIDATION_ENDPOINTS = [
    ("http://httpbin.org/ip", "json", "origin"),
    ("https://api.ipify.org?format=json", "json", "ip"),
    ("http://icanhazip.com", "text", None),
    ("http://checkip.amazonaws.com", "text", None),
]

# Geo APIs for location data
GEO_APIS = [
    {"url": "http://ip-api.com/json/{}", "has_asn": True},
    {"url": "https://ipapi.co/{}/json/", "has_asn": True},
    {"url": "https://ipwhois.app/json/{}", "has_asn": True},
]

# ---------------------------------------------------
# Database Initialization
# ---------------------------------------------------
@st.cache_resource
def init_database():
    """Initialize SQLite database with proper tables and indexes"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA mmap_size=268435456;
    """)
    
    # Create proxies table with geo data
    cur.execute("""
        CREATE TABLE IF NOT EXISTS proxies (
            host TEXT,
            port INTEGER,
            protocol TEXT DEFAULT 'http',
            username TEXT,
            password TEXT,
            latency REAL,
            last_tested TIMESTAMP,
            country TEXT,
            country_code TEXT,
            city TEXT,
            region TEXT,
            lat REAL,
            lon REAL,
            asn TEXT,
            org TEXT,
            isp TEXT,
            is_valid BOOLEAN DEFAULT 0,
            reliability_score REAL DEFAULT 0,
            test_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            PRIMARY KEY (host, port, protocol)
        )
    """)
    
    # Create chain tests table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chain_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT,
            tested_at TIMESTAMP,
            exit_ip TEXT,
            total_latency REAL,
            hop_count INTEGER,
            hop_timings_json TEXT,
            stats_json TEXT,
            anonymity_json TEXT,
            success BOOLEAN
        )
    """)
    
    # Create indexes
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_proxies_last_tested ON proxies(last_tested);
        CREATE INDEX IF NOT EXISTS idx_proxies_country_asn ON proxies(country_code, asn);
        CREATE INDEX IF NOT EXISTS idx_proxies_latency ON proxies(latency);
        CREATE INDEX IF NOT EXISTS idx_proxies_valid ON proxies(is_valid);
        CREATE INDEX IF NOT EXISTS idx_chain_tests_date ON chain_tests(tested_at);
    """)
    
    conn.commit()
    conn.close()

init_database()

# ---------------------------------------------------
# Session State Management
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
                # From Cloud edition
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
                # From Advanced edition
                "proxy_chain": [],
                "client_location": None,
                "chain_test_result": None,
                "proxies_raw": [],
                "proxies_validated_db": []
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

session_manager = ThreadSafeSessionState()

# ---------------------------------------------------
# Data Models
# ---------------------------------------------------
class ProxyProtocol(Enum):
    """Supported proxy protocols"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"

@dataclass
class ProxyInfo:
    """Enhanced proxy data model combining both versions"""
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
    exit_ip: Optional[str] = None
    # Geo data
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    asn: Optional[str] = None
    org: Optional[str] = None
    isp: Optional[str] = None
    
    def __post_init__(self):
        """Validate proxy data"""
        # Validate port
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}")
        
        # Validate host
        if not self.host or len(self.host) > 255:
            raise ValueError(f"Invalid host: {self.host}")
        
        # Convert protocol string to enum if needed
        if isinstance(self.protocol, str):
            try:
                self.protocol = ProxyProtocol(self.protocol.lower())
            except ValueError:
                self.protocol = ProxyProtocol.HTTP
        elif not isinstance(self.protocol, ProxyProtocol):
            self.protocol = ProxyProtocol.HTTP
    
    def __hash__(self):
        protocol_str = self.protocol.value if isinstance(self.protocol, ProxyProtocol) else str(self.protocol).lower()
        return hash((self.host, self.port, protocol_str))
    
    def __eq__(self, other):
        if isinstance(other, ProxyInfo):
            return (self.host, self.port, self.protocol) == (other.host, other.port, other.protocol)
        return False
    
    @property
    def url(self) -> str:
        """Get proxy URL with auth if present"""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        protocol_str = self.protocol.value if isinstance(self.protocol, ProxyProtocol) else str(self.protocol).lower()
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
        protocol_str = self.protocol.value if isinstance(self.protocol, ProxyProtocol) else str(self.protocol)
        return {
            "host": self.host,
            "port": self.port,
            "protocol": protocol_str,
            "latency": self.latency,
            "reliability": self.reliability_score,
            "url": self.url,
            "country": self.country,
            "city": self.city,
            "asn": self.asn,
            "org": self.org
        }

# ---------------------------------------------------
# Geo Location Functions
# ---------------------------------------------------
async def get_proxy_geo(ip: str) -> Dict[str, Any]:
    """Get geo data for an IP address"""
    out = {}
    async with httpx.AsyncClient(timeout=10, verify=False) as client:
        for api in GEO_APIS:
            try:
                r = await client.get(api["url"].format(ip))
                if r.status_code == 200:
                    d = r.json()
                    out['country'] = d.get('country') or d.get('country_name')
                    out['country_code'] = d.get('country_code') or d.get('countryCode')
                    out['city'] = d.get('city')
                    out['region'] = d.get('region') or d.get('regionName')
                    out['lat'] = d.get('latitude') or d.get('lat')
                    out['lon'] = d.get('longitude') or d.get('lon')
                    out['asn'] = d.get('asn') or d.get('as')
                    out['org'] = d.get('org')
                    out['isp'] = d.get('isp') or d.get('org')
                    break
            except Exception:
                continue
    return out

# ---------------------------------------------------
# Proxy Parsing
# ---------------------------------------------------
def parse_proxy_line(line: str, strict: bool = True) -> Optional[ProxyInfo]:
    """Parse and validate proxy line with security checks"""
    line = line.strip()
    
    # Security: Limit line length
    if len(line) > 500:
        logger.warning(f"Proxy line too long: {len(line)} chars")
        return None
    
    # Skip comments and empty lines
    if not line or line.startswith("#"):
        return None
    
    # Sanitize input
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
                    return ProxyInfo(host=host, port=int(port))
                except:
                    return None
            return None
        
        # Validate protocol
        protocol = parts.scheme or "http"
        if protocol not in ["http", "https", "socks4", "socks5"]:
            return None
        
        # Create proxy with validation
        return ProxyInfo(
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
# Proxy Fetching and Validation
# ---------------------------------------------------
@st.cache_data(ttl=300, max_entries=50)
def fetch_proxies_cached(url: str, limit: int = 1000) -> List[Dict]:
    """Fetch and cache proxy lists"""
    proxies = []
    
    try:
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
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
    
    return proxies

def create_proxy_from_dict(data: Dict) -> Optional[ProxyInfo]:
    """Safely create a ProxyInfo object from a dictionary"""
    try:
        if not data.get('host') or not data.get('port'):
            return None
        
        cleaned_data = {
            'host': str(data['host']),
            'port': int(data['port']),
            'protocol': str(data.get('protocol', 'http')).lower(),
            'username': data.get('username'),
            'password': data.get('password')
        }
        
        return ProxyInfo(**cleaned_data)
    except Exception as e:
        logger.debug(f"Failed to create proxy from dict: {e}")
        return None

# ---------------------------------------------------
# Enhanced Proxy Validator
# ---------------------------------------------------
class ProxyValidator:
    """Production-grade proxy validator with connection pooling"""
    
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
        
        # Create SSL context
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
    
    async def test_proxy_with_retry(self, proxy: ProxyInfo, max_retries: int = 3) -> bool:
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
                await asyncio.sleep(2 ** attempt)
        
        proxy.error_count += 1
        return False
    
    async def _test_single_proxy(self, proxy: ProxyInfo) -> bool:
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
                            
                            # Get geo data
                            if proxy.exit_ip:
                                geo = await get_proxy_geo(proxy.exit_ip)
                                proxy.country = geo.get('country')
                                proxy.country_code = geo.get('country_code')
                                proxy.city = geo.get('city')
                                proxy.region = geo.get('region')
                                proxy.lat = geo.get('lat')
                                proxy.lon = geo.get('lon')
                                proxy.asn = geo.get('asn')
                                proxy.org = geo.get('org')
                                proxy.isp = geo.get('isp')
                            
                            return True
                else:
                    # SOCKS proxy handling
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
    
    async def validate_batch(self, proxies: List[ProxyInfo], progress_callback=None) -> List[ProxyInfo]:
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
        
        # Save to database
        save_proxies_to_db(valid_proxies)
        
        return valid_proxies

# ---------------------------------------------------
# Database Operations
# ---------------------------------------------------
def save_proxies_to_db(proxies: List[ProxyInfo]):
    """Save validated proxies to database"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("BEGIN")
    
    for p in proxies:
        cur.execute("""
            INSERT OR REPLACE INTO proxies
            (host, port, protocol, username, password, latency, last_tested,
             country, country_code, city, region, lat, lon, asn, org, isp,
             is_valid, reliability_score, test_count, error_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.host, p.port, 
            p.protocol.value if isinstance(p.protocol, ProxyProtocol) else str(p.protocol),
            p.username, p.password, p.latency, p.last_tested,
            p.country, p.country_code, p.city, p.region, p.lat, p.lon, 
            p.asn, p.org, p.isp, p.is_valid, p.reliability_score,
            p.test_count, p.error_count
        ))
    
    conn.commit()
    conn.close()

def load_proxies_from_db(filters: Dict[str, Any] = None) -> List[ProxyInfo]:
    """Load proxies from database with optional filters"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    q = """
        SELECT host, port, protocol, username, password, latency, last_tested,
               country, country_code, city, region, lat, lon, asn, org, isp,
               is_valid, reliability_score, test_count, error_count
        FROM proxies
        WHERE is_valid = 1
    """
    params = []
    
    if filters:
        if filters.get('country'):
            q += " AND (country LIKE ? OR country_code = ?)"
            params += [f"%{filters['country']}%", filters['country']]
        if filters.get('asn'):
            q += " AND asn LIKE ?"
            params.append(f"%{filters['asn']}%")
        if filters.get('protocol'):
            q += " AND protocol = ?"
            params.append(filters['protocol'])
    
    q += " ORDER BY reliability_score DESC, latency ASC LIMIT 500"
    cur.execute(q, params)
    rows = cur.fetchall()
    conn.close()
    
    proxies = []
    for r in rows:
        p = ProxyInfo(
            host=r[0], port=r[1], protocol=r[2], username=r[3], password=r[4],
            latency=r[5], last_tested=r[6], country=r[7], country_code=r[8],
            city=r[9], region=r[10], lat=r[11], lon=r[12], asn=r[13],
            org=r[14], isp=r[15], is_valid=r[16], test_count=r[18],
            error_count=r[19]
        )
        proxies.append(p)
    
    return proxies

# ---------------------------------------------------
# Proxy Chain Testing
# ---------------------------------------------------
async def test_proxy_chain(chain: List[ProxyInfo], samples: int = 3) -> Dict[str, Any]:
    """Test a multi-hop proxy chain"""
    if not chain:
        return {"success": False, "error": "Empty chain"}
    
    if len(chain) < 2:
        return {"success": False, "error": "Chain needs at least 2 proxies"}
    
    hop_timings = []
    total_start = time.perf_counter()
    
    try:
        # Test chain using aiohttp-socks ChainProxyConnector
        urls = [p.url for p in chain]
        connector = ChainProxyConnector.from_urls(urls)
        timeout_cfg = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
            # Test the chain
            async with session.get("https://api.ipify.org?format=json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    exit_ip = data.get("ip")
                    
                    total_latency = (time.perf_counter() - total_start) * 1000
                    
                    # Get exit IP geo data
                    exit_geo = await get_proxy_geo(exit_ip) if exit_ip else {}
                    
                    # Save to database
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO chain_tests 
                        (chain_id, tested_at, exit_ip, total_latency, hop_count, success)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        "-".join([f"{p.host}:{p.port}" for p in chain]),
                        datetime.now(),
                        exit_ip,
                        total_latency,
                        len(chain),
                        True
                    ))
                    conn.commit()
                    conn.close()
                    
                    return {
                        "success": True,
                        "exit_ip": exit_ip,
                        "exit_geo": exit_geo,
                        "total_latency": total_latency,
                        "hop_count": len(chain)
                    }
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
                    
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}

# ---------------------------------------------------
# Async Runner
# ---------------------------------------------------
def run_async_validation(proxies: List[ProxyInfo], max_concurrent: int = 50, 
                         verify_ssl: bool = True, progress_callback=None) -> List[ProxyInfo]:
    """Run async validation with proper cleanup"""
    async def validate():
        validator = ProxyValidator(
            max_concurrent=max_concurrent,
            verify_ssl=verify_ssl
        )
        async with validator.session_context():
            return await validator.validate_batch(proxies, progress_callback)
    
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, validate())
                return future.result(timeout=300)
        except RuntimeError:
            return asyncio.run(validate())
    except Exception as e:
        logger.error(f"Validation error: {e}")
        st.error(f"Validation failed: {str(e)[:100]}")
        return []

# ---------------------------------------------------
# UI Components
# ---------------------------------------------------
def render_logo():
    """Render the ProxyStream logo"""
    st.markdown("""
    <style>
        .main-header {
            padding: 1rem 0 2rem 0;
            margin-bottom: 2rem;
        }
        .logo-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        .tagline {
            text-align: center;
            color: #6b7280;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
    </style>
    
    <div class="main-header">
        <div class="logo-title">🚀 ProxyStream Ultimate</div>
        <div class="tagline">Complete proxy management with harvesting, validation, and multi-hop chaining</div>
    </div>
    """, unsafe_allow_html=True)

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

# ---------------------------------------------------
# Main Application
# ---------------------------------------------------
render_logo()
render_stats()

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Harvest & Validate Section
    st.subheader("🚀 Harvest & Validate")
    
    if st.button("📥 Harvest Proxies", type="primary", use_container_width=True):
        with st.spinner("Fetching proxy lists..."):
            all_proxies = []
            progress = st.progress(0)
            
            sources = []
            for category, urls in PROXY_SOURCES.items():
                sources.extend(urls)
            
            for i, url in enumerate(sources):
                proxy_dicts = fetch_proxies_cached(url, limit=500)
                for proxy_dict in proxy_dicts:
                    proxy = create_proxy_from_dict(proxy_dict)
                    if proxy:
                        all_proxies.append(proxy)
                progress.progress((i + 1) / len(sources))
            
            # Deduplicate
            unique_proxies = list(set(all_proxies))
            st.session_state.proxies_raw = unique_proxies
            session_manager.update_stats(total_fetched=len(unique_proxies))
            
            st.success(f"✅ Fetched {len(unique_proxies)} unique proxies")
    
    if st.session_state.proxies_raw:
        validate_count = st.slider("Proxies to validate:", 10, 200, 50)
        
        if st.button("✅ Validate Proxies", use_container_width=True):
            progress_bar = st.progress(0)
            progress_counter = {"count": 0}
            
            def update_progress(count):
                progress_counter["count"] += count
                progress_bar.progress(min(progress_counter["count"] / validate_count, 1.0))
            
            with st.spinner(f"Validating {validate_count} proxies..."):
                sample = st.session_state.proxies_raw[:validate_count]
                valid_proxies = run_async_validation(
                    sample,
                    max_concurrent=st.session_state.settings["max_concurrent"],
                    verify_ssl=st.session_state.settings["verify_ssl"],
                    progress_callback=update_progress
                )
                
                # Update stats
                session_manager.update_stats(
                    total_validated=st.session_state.stats["total_validated"] + len(sample),
                    total_working=st.session_state.stats["total_working"] + len(valid_proxies),
                    last_validation=datetime.now()
                )
                
                # Add to validated list
                for proxy in valid_proxies:
                    session_manager.add_validated_proxy(proxy)
                
                st.success(f"✅ Found {len(valid_proxies)} working proxies!")
    
    # Load from Database
    st.markdown("---")
    st.subheader("💾 Database")
    
    if st.button("📂 Load from DB", use_container_width=True):
        proxies = load_proxies_from_db()
        st.session_state.proxies_validated_db = proxies
        st.success(f"Loaded {len(proxies)} proxies from database")
    
    # Chain Builder
    st.markdown("---")
    st.subheader("🔗 Chain Builder")
    
    all_validated = list(st.session_state.validated_proxies) + st.session_state.proxies_validated_db
    
    if all_validated:
        countries = {}
        for proxy in all_validated:
            key = proxy.country or "Unknown"
            countries.setdefault(key, []).append(proxy)
        
        if countries:
            selected_country = st.selectbox("Select Country", list(countries.keys()))
            
            if selected_country:
                country_proxies = countries[selected_country][:30]
                proxy_display = [
                    f"{p.protocol.value if isinstance(p.protocol, ProxyProtocol) else p.protocol}://{p.host}:{p.port} | {p.latency:.0f}ms"
                    for p in country_proxies
                ]
                selected_proxy_str = st.selectbox("Select Proxy", proxy_display)
                
                if st.button("➕ Add to Chain"):
                    idx = proxy_display.index(selected_proxy_str)
                    proxy = country_proxies[idx]
                    if len(st.session_state.proxy_chain) < 5:
                        st.session_state.proxy_chain.append(proxy)
                        st.success(f"Added hop {len(st.session_state.proxy_chain)}")
                    else:
                        st.error("Maximum 5 hops")
    
    # Current Chain
    if st.session_state.proxy_chain:
        st.write(f"**Current Chain ({len(st.session_state.proxy_chain)} hops):**")
        for i, p in enumerate(st.session_state.proxy_chain):
            protocol_str = p.protocol.value if isinstance(p.protocol, ProxyProtocol) else str(p.protocol)
            st.write(f"{i+1}. {protocol_str}://{p.host}:{p.port} ({p.country_code or 'N/A'})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test Chain", use_container_width=True):
                if len(st.session_state.proxy_chain) >= 2:
                    with st.spinner("Testing chain..."):
                        try:
                            result = asyncio.run(test_proxy_chain(st.session_state.proxy_chain))
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(test_proxy_chain(st.session_state.proxy_chain))
                        
                        st.session_state.chain_test_result = result
                        if result["success"]:
                            st.success("✅ Chain works!")
                            st.metric("Exit IP", result.get('exit_ip', 'N/A'))
                            st.metric("Latency", f"{result.get('total_latency', 0):.0f}ms")
                        else:
                            st.error(f"❌ {result['error']}")
                else:
                    st.error("Need at least 2 proxies in chain")
        
        with col2:
            if st.button("🗑️ Clear Chain", use_container_width=True):
                st.session_state.proxy_chain = []
                st.session_state.chain_test_result = None
                st.rerun()
    
    # Settings
    st.markdown("---")
    with st.expander("⚙️ Settings"):
        st.session_state.settings["max_concurrent"] = st.slider(
            "Concurrent Tests",
            min_value=5,
            max_value=100,
            value=st.session_state.settings.get("max_concurrent", 50)
        )
        
        st.session_state.settings["timeout"] = st.slider(
            "Timeout (seconds)",
            min_value=5,
            max_value=30,
            value=st.session_state.settings.get("timeout", 10)
        )
        
        st.session_state.settings["verify_ssl"] = st.checkbox(
            "Verify SSL",
            value=st.session_state.settings.get("verify_ssl", True)
        )

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Proxy List",
    "🗺️ Chain Visualization", 
    "📊 Analytics",
    "🌐 Browse via Chain"
])

with tab1:
    st.subheader("📋 Validated Proxies")
    
    all_validated = list(st.session_state.validated_proxies) + st.session_state.proxies_validated_db
    
    if all_validated:
        data = []
        for proxy in all_validated:
            if proxy.is_valid:
                protocol_str = proxy.protocol.value if isinstance(proxy.protocol, ProxyProtocol) else str(proxy.protocol)
                data.append({
                    'Protocol': protocol_str.upper(),
                    'Host': proxy.host,
                    'Port': proxy.port,
                    'Country': proxy.country or 'Unknown',
                    'City': proxy.city or 'Unknown',
                    'ASN': proxy.asn or 'N/A',
                    'ISP': proxy.isp or proxy.org or 'N/A',
                    'Latency': f"{proxy.latency:.0f}ms" if proxy.latency else "N/A",
                    'Reliability': f"{proxy.reliability_score:.2f}",
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
                countries = ["All"] + sorted(df['Country'].unique().tolist())
                filter_country = st.selectbox("Filter Country", countries)
            
            with col3:
                limit = st.number_input("Show Top", min_value=10, max_value=500, value=50)
            
            # Apply filters
            filtered_df = df
            if filter_protocol != "All":
                filtered_df = filtered_df[filtered_df['Protocol'] == filter_protocol]
            if filter_country != "All":
                filtered_df = filtered_df[filtered_df['Country'] == filter_country]
            
            filtered_df = filtered_df.head(limit)
            
            st.dataframe(filtered_df, use_container_width=True, height=400, hide_index=True)
            
            # Export
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button("📥 Export CSV", csv, "proxies.csv", "text/csv", use_container_width=True)
            
            with col2:
                urls = "\n".join(filtered_df['URL'].tolist())
                st.download_button("📥 Export URLs", urls, "proxy_urls.txt", "text/plain", use_container_width=True)
    else:
        st.info("No validated proxies yet. Click 'Harvest & Validate' to get started!")

with tab2:
    st.subheader("🗺️ Proxy Chain Visualization")
    
    if st.session_state.proxy_chain and len(st.session_state.proxy_chain) >= 2:
        fig = go.Figure()
        
        # Add proxy locations
        lats, lons, texts = [], [], []
        
        for i, p in enumerate(st.session_state.proxy_chain):
            if p.lat and p.lon:
                lats.append(p.lat)
                lons.append(p.lon)
                texts.append(f"Hop {i+1}: {p.city or 'Unknown'}, {p.country or 'Unknown'}")
        
        if lats and lons:
            # Add markers
            fig.add_trace(go.Scattermapbox(
                mode='markers+text',
                lon=lons,
                lat=lats,
                marker={'size': 15, 'color': 'red'},
                text=texts,
                textposition="top center"
            ))
            
            # Add lines between hops
            for i in range(len(lats) - 1):
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=[lons[i], lons[i+1]],
                    lat=[lats[i], lats[i+1]],
                    line=dict(width=3, color='blue'),
                    showlegend=False
                ))
            
            # Exit point if tested
            if st.session_state.chain_test_result and st.session_state.chain_test_result.get("success"):
                exit_geo = st.session_state.chain_test_result.get("exit_geo", {})
                if exit_geo.get('lat') and exit_geo.get('lon'):
                    fig.add_trace(go.Scattermapbox(
                        mode='markers',
                        lon=[exit_geo['lon']],
                        lat=[exit_geo['lat']],
                        marker={'size': 20, 'color': 'green', 'symbol': 'star'},
                        text=[f"Exit: {st.session_state.chain_test_result.get('exit_ip')}"],
                        textposition="top center"
                    ))
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lon=lons[0], lat=lats[0]),
                    zoom=2
                ),
                showlegend=False,
                height=600,
                margin=dict(r=0, t=0, l=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No location data available for chain proxies")
    else:
        st.info("Build a chain with at least 2 proxies to visualize")

with tab3:
    st.subheader("📊 Analytics Dashboard")
    
    all_validated = list(st.session_state.validated_proxies) + st.session_state.proxies_validated_db
    
    if all_validated:
        col1, col2 = st.columns(2)
        
        with col1:
            # Protocol distribution
            protocols = [p.protocol.value if isinstance(p.protocol, ProxyProtocol) else str(p.protocol) 
                        for p in all_validated]
            if protocols:
                protocol_counts = pd.Series(protocols).value_counts()
                fig = px.pie(values=protocol_counts.values, names=protocol_counts.index,
                           title="Protocol Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Country distribution
            countries = [p.country for p in all_validated if p.country]
            if countries:
                country_counts = pd.Series(countries).value_counts().head(10)
                fig = px.bar(x=country_counts.values, y=country_counts.index,
                           orientation='h', title="Top 10 Countries")
                st.plotly_chart(fig, use_container_width=True)
        
        # Latency distribution
        latencies = [p.latency for p in all_validated if p.latency]
        if latencies:
            fig = px.histogram(x=latencies, nbins=30, title="Latency Distribution (ms)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available. Start by harvesting and validating proxies!")

with tab4:
    st.subheader("🌐 Browse via Proxy Chain")
    
    if st.session_state.proxy_chain and len(st.session_state.proxy_chain) >= 2:
        url = st.text_input("Enter URL to fetch:", placeholder="https://example.com")
        
        if st.button("🌐 Fetch via Chain", use_container_width=True):
            if url:
                with st.spinner("Fetching through proxy chain..."):
                    try:
                        # Simple fetch through chain
                        urls = [p.url for p in st.session_state.proxy_chain]
                        connector = ChainProxyConnector.from_urls(urls)
                        timeout_cfg = aiohttp.ClientTimeout(total=30)
                        
                        async def fetch():
                            async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
                                async with session.get(url) as resp:
                                    return {
                                        "status": resp.status,
                                        "text": await resp.text(),
                                        "headers": dict(resp.headers)
                                    }
                        
                        try:
                            result = asyncio.run(fetch())
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(fetch())
                        
                        st.success(f"✅ Status: {result['status']}")
                        
                        with st.expander("Response Headers"):
                            st.json(result['headers'])
                        
                        with st.expander("Response Body (first 2000 chars)"):
                            st.text(result['text'][:2000])
                    
                    except Exception as e:
                        st.error(f"Failed: {str(e)[:200]}")
            else:
                st.warning("Please enter a URL")
    else:
        st.info("Build a chain with at least 2 proxies first")

# Footer
st.markdown("---")
st.caption(f"ProxyStream Ultimate | Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
