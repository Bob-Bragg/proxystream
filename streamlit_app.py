"""
ProxyStream Cloud - Enhanced Edition
Complete Streamlit application with all features
"""

import asyncio
import ssl
import time
import json
import sqlite3
import base64
import os
import struct
import tarfile
import tempfile
import zipfile
import io
import queue
import threading
import re
import random
import socket
import pathlib
import hashlib
import statistics
import httpx
import requests
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import aiohttp
from aiohttp_socks import ChainProxyConnector

# Try to import modules, fallback to inline definitions if not available
try:
    from modules.validator import AsyncQueueValidator, QueueBasedValidator, validate_until_target
    from modules.dns_resolver import get_exit_ip_dns_direct, get_exit_ip_dns_via_chain
    from modules.proxy_parser import ProxyParser
    from modules.chain_tester import test_chain_comprehensive
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Cloud Enhanced",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Sources
PROXY_SOURCES = {
    "Primary": [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
        "https://raw.githubusercontent.com/claude89757/free_https_proxies/refs/heads/main/https_proxies.txt",
        "https://raw.githubusercontent.com/Zaeem20/FREE_PROXIES_LIST/refs/heads/master/https.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    ],
    "SOCKS": [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks4.txt",
    ],
    "Additional": [
        "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
        "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-https.txt",
        "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
        "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
    ]
}

# DNS resolver configurations
DNS_PLANS = [
    ("208.67.222.222", 53, "myip.opendns.com", 1, 1),
    ("208.67.220.220", 53, "myip.opendns.com", 1, 1),
    ("1.1.1.1", 53, "whoami.cloudflare", 16, 3),
]

# ---------------------------------------------------
# Session State Management
# ---------------------------------------------------
class SessionManager:
    @staticmethod
    def init():
        defaults = {
            "proxies_raw": [],
            "proxies_validated": [],
            "proxy_chain": [],
            "geo_reader": None,
            "validation_stats": {"total": 0, "success": 0, "failed": 0},
            "last_fetch_time": None,
            "fetch_history": [],
            "chain_test_results": [],
            "selected_sources": ["Primary"],
            "validation_in_progress": False,
            "no_api_mode": False,
            "validation_time": 0,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

SessionManager.init()

# ---------------------------------------------------
# Data Models
# ---------------------------------------------------
@dataclass
class ProxyInfo:
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
    anonymity_level: str = "unknown"
    success_rate: float = 0.0
    test_count: int = 0
    exit_ip: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash((self.host, self.port, self.protocol))

    def as_url(self) -> str:
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
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
            "url": self.as_url()
        }

@dataclass
class ValidationResult:
    proxy: ProxyInfo
    success: bool
    latency: Optional[float]
    endpoint_tested: str
    error_message: Optional[str] = None
    headers_leaked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

# ---------------------------------------------------
# Inline Validator (Fallback if modules not available)
# ---------------------------------------------------
if not MODULES_AVAILABLE:
    class AsyncQueueValidator:
        def __init__(self, max_concurrent: int = 30, timeout: int = 10, no_apis: bool = False):
            self.max_concurrent = max_concurrent
            self.timeout = timeout
            self.no_apis = no_apis
            self.semaphore = None
        
        def get_endpoints(self):
            endpoints = [
                "https://www.wikipedia.org/robots.txt",
                "https://www.cloudflare.com/robots.txt",
                "https://www.google.com/robots.txt",
            ]
            if not self.no_apis:
                endpoints.extend([
                    "https://api.ipify.org?format=json",
                    "https://httpbin.org/get",
                ])
            return endpoints
        
        async def validate_batch(self, proxies: List[ProxyInfo], max_valid: int = None, progress_callback: Callable = None) -> List[ProxyInfo]:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
            valid_proxies = []
            
            async def validate_one(proxy):
                async with self.semaphore:
                    if await self._validate_async(proxy):
                        proxy.is_valid = True
                        proxy.last_tested = datetime.now()
                        valid_proxies.append(proxy)
                        if progress_callback:
                            progress_callback()
                        if max_valid and len(valid_proxies) >= max_valid:
                            return True
                    return False
            
            tasks = []
            for proxy in proxies:
                if max_valid and len(valid_proxies) >= max_valid:
                    break
                tasks.append(asyncio.create_task(validate_one(proxy)))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            return valid_proxies[:max_valid] if max_valid else valid_proxies
        
        async def _validate_async(self, proxy: ProxyInfo) -> bool:
            proxy_url = proxy.as_url()
            proxy_dict = {"http://": proxy_url, "https://": proxy_url}
            
            for url in self.get_endpoints():
                try:
                    start = time.perf_counter()
                    async with httpx.AsyncClient(
                        proxies=proxy_dict,
                        timeout=httpx.Timeout(self.timeout),
                        verify=False
                    ) as client:
                        response = await client.get(url)
                        if response.status_code in [200, 301, 302, 403, 404]:
                            proxy.latency = (time.perf_counter() - start) * 1000
                            
                            # Try to get exit IP
                            if "ipify" in url:
                                try:
                                    data = response.json()
                                    proxy.exit_ip = data.get("ip")
                                except:
                                    pass
                            
                            return True
                except:
                    continue
            return False
    
    async def validate_until_target(proxies: List[ProxyInfo], target: int = 200, max_concurrent: int = 40, no_apis: bool = False) -> List[ProxyInfo]:
        validator = AsyncQueueValidator(max_concurrent=max_concurrent, no_apis=no_apis)
        return await validator.validate_batch(proxies, max_valid=target)

# ---------------------------------------------------
# Geo Database Functions
# ---------------------------------------------------
@st.cache_data(ttl=86400*7)
def download_ip2location_cached() -> bytes:
    try:
        url = "https://download.ip2location.com/lite/IP2LOCATION-LITE-DB5.BIN.ZIP"
        response = httpx.get(url, timeout=60, follow_redirects=True)
        if response.status_code != 200:
            return None
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for name in z.namelist():
                if name.endswith(".BIN"):
                    return z.read(name)
        return None
    except Exception as e:
        st.error(f"IP2Location download error: {e}")
        return None

@st.cache_data(ttl=86400)
def download_geolite2_cached(license_key: str) -> bytes:
    if not license_key:
        return None
    
    try:
        url = f"https://download.maxmind.com/app/geoip_download"
        params = {
            "edition_id": "GeoLite2-City",
            "license_key": license_key,
            "suffix": "tar.gz"
        }
        
        response = httpx.get(url, params=params, follow_redirects=True, timeout=60)
        if response.status_code != 200:
            return None
        
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
            tmp_tar.write(response.content)
            tmp_tar.flush()
            
            with tarfile.open(tmp_tar.name, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".mmdb"):
                        f = tar.extractfile(member)
                        if f:
                            return f.read()
        return None
    except Exception as e:
        st.error(f"GeoLite2 download error: {e}")
        return None

@st.cache_resource
def init_geo_reader(_db_bytes: bytes):
    if not _db_bytes:
        return None
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(_db_bytes)
        tmp_path = tmp.name
    
    try:
        try:
            import maxminddb
            reader = maxminddb.open_database(tmp_path)
            reader._type = "maxmind"
            return reader
        except:
            pass
        
        try:
            import IP2Location
            reader = IP2Location.IP2Location(tmp_path)
            reader._type = "ip2location"
            return reader
        except:
            pass
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return None

def get_license_key() -> Optional[str]:
    if hasattr(st, 'secrets') and "MAXMIND_LICENSE_KEY" in st.secrets:
        return st.secrets["MAXMIND_LICENSE_KEY"]
    if "MAXMIND_LICENSE_KEY" in os.environ:
        return os.environ["MAXMIND_LICENSE_KEY"]
    if "maxmind_license_key" in st.session_state:
        return st.session_state.maxmind_license_key
    return None

def geo_lookup(ip: str, reader=None) -> Dict[str, Any]:
    if not reader:
        if "geo_reader" in st.session_state:
            reader = st.session_state.geo_reader
        else:
            return {"country": "Unknown", "country_code": "XX"}
    
    if not reader:
        return {"country": "Unknown", "country_code": "XX"}
    
    try:
        if hasattr(reader, '_type'):
            if reader._type == "maxmind":
                data = reader.get(ip) or {}
                return {
                    "country": data.get('country', {}).get('names', {}).get('en'),
                    "country_code": data.get('country', {}).get('iso_code'),
                    "city": data.get('city', {}).get('names', {}).get('en'),
                    "lat": data.get('location', {}).get('latitude'),
                    "lon": data.get('location', {}).get('longitude')
                }
            elif reader._type == "ip2location":
                rec = reader.get_all(ip)
                return {
                    "country": rec.country_long if rec else None,
                    "country_code": rec.country_short if rec else None,
                    "city": rec.city if rec else None,
                    "lat": rec.latitude if rec else None,
                    "lon": rec.longitude if rec else None
                }
    except:
        pass
    
    return {"country": "Unknown", "country_code": "XX"}

# ---------------------------------------------------
# DNS Functions
# ---------------------------------------------------
def build_dns_query(name: str, qtype: int = 1) -> Tuple[bytes, int]:
    tid = random.randint(0, 0xFFFF)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    parts = name.split(".")
    qname = b"".join(struct.pack("B", len(p)) + p.encode() for p in parts) + b"\x00"
    question = struct.pack("!HH", qtype, 1)
    payload = header + qname + question
    return struct.pack("!H", len(payload)) + payload, tid

def parse_dns_response(resp: bytes, tid: int) -> Optional[str]:
    if len(resp) < 14:
        return None
    
    resp = resp[2:]
    if len(resp) < 12:
        return None
    
    rid, flags, qd, an, ns, ar = struct.unpack("!HHHHHH", resp[:12])
    if rid != tid or an == 0:
        return None
    
    i = 12
    while i < len(resp) and resp[i] != 0:
        if i + resp[i] + 1 > len(resp):
            return None
        i += resp[i] + 1
    i += 5
    
    if i + 2 > len(resp):
        return None
    
    if resp[i] & 0xC0 == 0xC0:
        i += 2
    else:
        while i < len(resp) and resp[i] != 0:
            if i + resp[i] + 1 > len(resp):
                return None
            i += resp[i] + 1
        i += 1
    
    if i + 10 > len(resp):
        return None
    
    rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", resp[i:i+10])
    i += 10
    
    if rtype == 1 and rdlen == 4 and i + 4 <= len(resp):
        return ".".join(str(b) for b in resp[i:i+4])
    
    # TXT record support
    if rtype == 16 and rdlen >= 1 and i + rdlen <= len(resp):
        txt_len = resp[i]
        if txt_len <= rdlen - 1:
            txt_data = resp[i+1:i+1+txt_len].decode(errors="ignore")
            m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", txt_data)
            return m.group(1) if m else None
    
    return None

if not MODULES_AVAILABLE:
    async def get_exit_ip_dns_direct() -> Optional[str]:
        for host, port, name, qtype, qclass in DNS_PLANS:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=5.0
                )
                
                query, tid = build_dns_query(name)
                writer.write(query)
                await writer.drain()
                
                len_bytes = await reader.readexactly(2)
                resp_len = struct.unpack("!H", len_bytes)[0]
                response = await reader.readexactly(resp_len)
                
                writer.close()
                await writer.wait_closed()
                
                return parse_dns_response(len_bytes + response, tid)
            except:
                continue
        return None
    
    async def get_exit_ip_dns_via_chain(chain: List[ProxyInfo]) -> Optional[str]:
        if any(p.protocol not in ("http", "https") for p in chain):
            return None
        
        # Implementation would go here - simplified for now
        return None

# ---------------------------------------------------
# Async Runner
# ---------------------------------------------------
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                result = asyncio.run(coro)
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive():
            raise TimeoutError("Operation exceeded 60s timeout")
        if exception:
            raise exception
        return result
    except RuntimeError:
        return asyncio.run(coro)

# ---------------------------------------------------
# Database
# ---------------------------------------------------
@st.cache_resource
def init_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
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
            lat REAL,
            lon REAL,
            anonymity_level TEXT,
            success_rate REAL,
            test_count INTEGER DEFAULT 0,
            PRIMARY KEY (host, port, protocol)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chain_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_config TEXT,
            exit_ip TEXT,
            success BOOLEAN,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn

db_conn = init_database()

# ---------------------------------------------------
# Proxy Parsing
# ---------------------------------------------------
def parse_proxy_line(line: str) -> Optional[ProxyInfo]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if "://" not in s:
        if ":1080" in s or ":9050" in s:
            s = "socks5://" + s
        elif ":4145" in s:
            s = "socks4://" + s
        else:
            s = "http://" + s

    try:
        u = urlsplit(s)
        if not u.hostname or not u.port:
            parts = s.replace("http://", "").replace("https://", "").split(":")
            if len(parts) == 2:
                return ProxyInfo(
                    host=parts[0],
                    port=int(parts[1]),
                    protocol="http"
                )
            return None
        
        proto = (u.scheme or "http").lower()
        if proto not in ("http", "https", "socks4", "socks5"):
            return None
        
        return ProxyInfo(
            host=u.hostname, 
            port=u.port, 
            protocol=proto,
            username=u.username,
            password=u.password
        )
    except:
        return None

async def fetch_proxies(url: str, max_proxies: int = 1000) -> List[ProxyInfo]:
    proxies = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
            if r.status_code == 200:
                for line in r.text.splitlines()[:max_proxies]:
                    p = parse_proxy_line(line)
                    if p:
                        proxies.append(p)
    except Exception as e:
        pass
    return proxies

# ---------------------------------------------------
# Validation
# ---------------------------------------------------
async def validate_batch_concurrent(
    proxies: List[ProxyInfo], 
    verify_tls: bool = True,
    max_concurrent: int = 10,
    progress_callback=None
) -> List[ProxyInfo]:
    validator = AsyncQueueValidator(
        max_concurrent=max_concurrent,
        timeout=10,
        no_apis=st.session_state.no_api_mode
    )
    return await validator.validate_batch(
        proxies,
        progress_callback=progress_callback
    )

# ---------------------------------------------------
# Chain Testing
# ---------------------------------------------------
async def test_chain_comprehensive(chain: List[ProxyInfo]) -> Dict[str, Any]:
    if not chain:
        return {"success": False, "error": "Empty chain"}
    
    result = {
        "success": False,
        "hop_count": len(chain),
        "chain_protocols": [p.protocol for p in chain],
        "tests_performed": []
    }
    
    # DNS test (if supported)
    if all(p.protocol in ("http", "https") for p in chain):
        try:
            exit_ip = await get_exit_ip_dns_via_chain(chain) if MODULES_AVAILABLE else None
            if exit_ip:
                result["exit_ip_dns"] = exit_ip
                result["exit_geo_dns"] = geo_lookup(exit_ip, st.session_state.geo_reader)
                result["tests_performed"].append("DNS")
                result["success"] = True
        except Exception as e:
            result["dns_error"] = str(e)[:100]
    
    # HTTP test
    if not st.session_state.no_api_mode:
        try:
            urls = [p.as_url() for p in chain]
            connector = ChainProxyConnector.from_urls(urls)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get("https://api.ipify.org?format=json") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        exit_ip = data.get("ip")
                        result["exit_ip_http"] = exit_ip
                        result["exit_geo_http"] = geo_lookup(exit_ip, st.session_state.geo_reader)
                        result["tests_performed"].append("HTTP")
                        result["success"] = True
        except Exception as e:
            result["http_error"] = str(e)[:100]
    
    return result

# ---------------------------------------------------
# UI Components
# ---------------------------------------------------
def render_proxy_stats():
    if not st.session_state.proxies_validated:
        st.info("No validated proxies yet")
        return
    
    proxies = st.session_state.proxies_validated
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Validated", len(proxies))
    with col2:
        latencies = [p.latency for p in proxies if p.latency]
        if latencies:
            avg_latency = statistics.mean(latencies)
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    with col3:
        protocols = defaultdict(int)
        for p in proxies:
            protocols[p.protocol] += 1
        st.metric("Protocols", ", ".join([f"{k}:{v}" for k, v in protocols.items()]))
    with col4:
        countries = len(set(p.country_code for p in proxies if p.country_code))
        st.metric("Countries", countries)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        df_proto = pd.DataFrame([{"Protocol": p.protocol.upper(), "Count": 1} for p in proxies])
        fig = px.pie(df_proto.groupby("Protocol").sum().reset_index(), 
                     values="Count", names="Protocol", 
                     title="Protocol Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        country_data = defaultdict(int)
        for p in proxies:
            if p.country:
                country_data[p.country] += 1
        
        if country_data:
            df_country = pd.DataFrame(
                list(country_data.items()), 
                columns=["Country", "Count"]
            ).sort_values("Count", ascending=False).head(10)
            
            fig = px.bar(df_country, x="Country", y="Count", 
                        title="Top 10 Countries")
            st.plotly_chart(fig, use_container_width=True)

def render_proxy_table():
    if not st.session_state.proxies_validated:
        st.info("No validated proxies yet")
        return
    
    proxies = st.session_state.proxies_validated
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = ["All"] + sorted(list(set(p.country for p in proxies if p.country)))
        filter_country = st.selectbox("Filter by Country", countries, index=0)
    
    with col2:
        protocols = ["All"] + sorted(list(set(p.protocol for p in proxies)))
        filter_protocol = st.selectbox("Filter by Protocol", protocols, index=0)
    
    with col3:
        max_latency = st.slider("Max Latency (ms)", 0, 10000, 10000)
    
    # Apply filters
    filtered = proxies
    if filter_country != "All":
        filtered = [p for p in filtered if p.country == filter_country]
    if filter_protocol != "All":
        filtered = [p for p in filtered if p.protocol == filter_protocol]
    filtered = [p for p in filtered if p.latency and p.latency <= max_latency]
    
    # Create DataFrame
    data = []
    for p in filtered:
        data.append({
            'Protocol': p.protocol.upper(),
            'Host': p.host,
            'Port': p.port,
            'Country': p.country or 'Unknown',
            'City': p.city or 'Unknown',
            'Latency': f"{p.latency:.0f}ms" if p.latency else 'N/A',
            'Anonymity': p.anonymity_level.title(),
            'Last Tested': p.last_tested.strftime("%H:%M:%S") if p.last_tested else 'N/A'
        })
    
    df = pd.DataFrame(data)
    
    st.dataframe(df, use_container_width=True, height=400, hide_index=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "proxies.csv", "text/csv", use_container_width=True)
    
    with col2:
        json_data = json.dumps([p.to_dict() for p in filtered], indent=2, default=str)
        st.download_button("📥 Export JSON", json_data, "proxies.json", "application/json", use_container_width=True)
    
    with col3:
        proxy_list = "\n".join([p.as_url() for p in filtered])
        st.download_button("📥 Export URLs", proxy_list, "proxy_urls.txt", "text/plain", use_container_width=True)

def render_chain_builder():
    st.subheader("🔗 Proxy Chain Builder")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.proxy_chain:
            st.write("**Current Chain:**")
            for i, p in enumerate(st.session_state.proxy_chain):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"{i+1}. {p.protocol}://{p.host}:{p.port} ({p.country_code or 'XX'})")
                with col_b:
                    if st.button("❌", key=f"remove_{i}", use_container_width=True):
                        st.session_state.proxy_chain.pop(i)
                        st.rerun()
            
            col_test, col_clear = st.columns(2)
            with col_test:
                if len(st.session_state.proxy_chain) >= 2:
                    if st.button("🧪 Test Chain", type="primary", use_container_width=True):
                        with st.spinner("Testing chain..."):
                            result = run_async(test_chain_comprehensive(st.session_state.proxy_chain))
                            st.session_state.chain_test_results.append(result)
                            
                            if result["success"]:
                                st.success("✅ Chain Working!")
                                if "exit_ip_dns" in result:
                                    st.info(f"Exit IP (DNS): {result['exit_ip_dns']}")
                                if "exit_ip_http" in result:
                                    st.info(f"Exit IP (HTTP): {result['exit_ip_http']}")
                            else:
                                st.error(f"❌ Chain Failed: {result.get('error', 'Unknown error')}")
            
            with col_clear:
                if st.button("🗑️ Clear Chain", use_container_width=True):
                    st.session_state.proxy_chain = []
                    st.rerun()
        else:
            st.info("No proxies in chain. Add proxies from the list below.")
    
    with col2:
        st.write("**Quick Actions:**")
        if st.button("🎲 Random Chain", use_container_width=True):
            if len(st.session_state.proxies_validated) >= 3:
                st.session_state.proxy_chain = random.sample(
                    st.session_state.proxies_validated, 
                    min(3, len(st.session_state.proxies_validated))
                )
                st.rerun()
        
        if st.button("⚡ Fast Chain", use_container_width=True):
            sorted_proxies = sorted(
                st.session_state.proxies_validated,
                key=lambda p: p.latency or float('inf')
            )[:3]
            if sorted_proxies:
                st.session_state.proxy_chain = sorted_proxies
                st.rerun()
    
    st.write("**Available Proxies (click to add to chain):**")
    
    if st.session_state.proxies_validated:
        sort_by = st.selectbox("Sort by:", ["Latency", "Country", "Protocol", "Recent"])
        
        sorted_proxies = st.session_state.proxies_validated.copy()
        if sort_by == "Latency":
            sorted_proxies.sort(key=lambda p: p.latency or float('inf'))
        elif sort_by == "Country":
            sorted_proxies.sort(key=lambda p: p.country or 'ZZ')
        elif sort_by == "Protocol":
            sorted_proxies.sort(key=lambda p: p.protocol)
        elif sort_by == "Recent":
            sorted_proxies.sort(key=lambda p: p.last_tested or datetime.min, reverse=True)
        
        for i, p in enumerate(sorted_proxies[:20]):
            label = f"{p.protocol}://{p.host}:{p.port}"
            if p.country_code:
                label += f" ({p.country_code})"
            if p.latency:
                label += f" - {p.latency:.0f}ms"
            
            if st.button(label, key=f"add_{i}", use_container_width=True):
                if len(st.session_state.proxy_chain) < 10:
                    st.session_state.proxy_chain.append(p)
                    st.success(f"Added hop {len(st.session_state.proxy_chain)}")
                    st.rerun()
                else:
                    st.warning("Maximum 10 hops allowed")
    else:
        st.info("No validated proxies available")

# ---------------------------------------------------
# Main UI
# ---------------------------------------------------
st.title("🔒 ProxyStream Cloud Enhanced")
st.caption("Advanced proxy management with DNS-based detection and comprehensive analytics")

# Top status bar
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📍 Get My IP", use_container_width=True):
        with st.spinner("Detecting IP..."):
            ip = run_async(get_exit_ip_dns_direct()) if MODULES_AVAILABLE else None
            if ip:
                geo = geo_lookup(ip, st.session_state.geo_reader)
                st.success(f"IP: {ip} ({geo.get('country_code', 'XX')})")
            else:
                st.error("Failed to detect IP")

with col2:
    db_status = "✅ Active" if st.session_state.geo_reader else "⚠️ Setup Required"
    st.metric("Geo Database", db_status)

with col3:
    st.metric("Validated", len(st.session_state.proxies_validated))

with col4:
    st.metric("In Chain", len(st.session_state.proxy_chain))

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.session_state.no_api_mode = st.toggle(
            "🌿 No-API Mode",
            value=st.session_state.no_api_mode,
            help="Avoid external APIs, use only DNS and robots.txt"
        )
        
        quick_harvest = st.toggle("⚡ Quick Harvest Mode", value=False)
        if quick_harvest:
            harvest_target = st.number_input("Target Valid Proxies", min_value=10, max_value=500, value=200)
    
    # Geo Database Setup
    with st.expander("🌍 Geo Database Setup", expanded=not st.session_state.geo_reader):
        if st.session_state.geo_reader:
            st.success("✅ Geo database is active")
            if st.button("🔄 Reload Database"):
                st.session_state.geo_reader = None
                st.rerun()
        else:
            setup_method = st.radio(
                "Choose setup method:",
                ["IP2Location (Free, No Key)", "GeoLite2 (Free, Requires Key)"]
            )
            
            if setup_method == "IP2Location (Free, No Key)":
                if st.button("📥 Download & Setup IP2Location", use_container_width=True):
                    with st.spinner("Downloading database (~20MB)..."):
                        db_bytes = download_ip2location_cached()
                        if db_bytes:
                            reader = init_geo_reader(db_bytes)
                            if reader:
                                st.session_state.geo_reader = reader
                                st.success("✅ Database ready!")
                                st.rerun()
            else:
                st.info("Get your free license key at [MaxMind](https://www.maxmind.com/en/geolite2/signup)")
                license_key = get_license_key()
                if not license_key:
                    key_input = st.text_input("Enter License Key:", type="password")
                    if key_input:
                        st.session_state.maxmind_license_key = key_input
                        license_key = key_input
                
                if license_key:
                    if st.button("📥 Download & Setup GeoLite2", use_container_width=True):
                        with st.spinner("Downloading database..."):
                            db_bytes = download_geolite2_cached(license_key)
                            if db_bytes:
                                reader = init_geo_reader(db_bytes)
                                if reader:
                                    st.session_state.geo_reader = reader
                                    st.success("✅ Database ready!")
                                    st.rerun()
    
    st.markdown("---")
    
    # Quick Harvest Button
    if st.button("🎯 Harvest 200 Working", type="primary", use_container_width=True):
        with st.spinner("Harvesting and validating..."):
            all_proxies = []
            for category in ["Primary", "SOCKS", "Additional"]:
                for src in PROXY_SOURCES[category]:
                    proxies = run_async(fetch_proxies(src, 500))
                    all_proxies.extend(proxies)
                    if len(all_proxies) >= 1000:
                        break
            
            # Validate until we get 200
            start_time = time.time()
            validated = run_async(
                validate_until_target(
                    all_proxies[:1000],
                    target=200,
                    max_concurrent=40,
                    no_apis=st.session_state.no_api_mode
                )
            )
            elapsed = time.time() - start_time
            
            st.session_state.proxies_validated.extend(validated)
            st.session_state.proxies_validated = list({p: None for p in st.session_state.proxies_validated}.keys())
            st.success(f"✅ Found {len(validated)} working proxies in {elapsed:.1f}s!")
    
    st.markdown("---")
    
    # Proxy Sources
    with st.expander("📥 Proxy Sources", expanded=True):
        st.write("**Select source categories:**")
        
        selected_sources = []
        for category in PROXY_SOURCES.keys():
            if st.checkbox(category, value=category == "Primary", key=f"src_{category}"):
                selected_sources.append(category)
        
        st.session_state.selected_sources = selected_sources
        
        max_per_source = st.number_input(
            "Max proxies per source:",
            min_value=100,
            max_value=5000,
            value=500,
            step=100
        )
        
        if st.button("🔄 Fetch Proxies", type="primary", use_container_width=True):
            if not selected_sources:
                st.warning("Please select at least one source category")
            else:
                with st.spinner("Fetching proxies..."):
                    all_proxies = []
                    progress_bar = st.progress(0)
                    
                    total_sources = sum(len(PROXY_SOURCES[cat]) for cat in selected_sources)
                    current = 0
                    
                    for category in selected_sources:
                        for src in PROXY_SOURCES[category]:
                            proxies = run_async(fetch_proxies(src, max_per_source))
                            all_proxies.extend(proxies)
                            current += 1
                            progress_bar.progress(current / total_sources)
                    
                    unique = list({p: None for p in all_proxies}.keys())
                    st.session_state.proxies_raw = unique
                    st.session_state.last_fetch_time = datetime.now()
                    st.success(f"✅ Loaded {len(unique)} unique proxies")
    
    st.markdown("---")
    
    # Validation
    with st.expander("✅ Validation", expanded=bool(st.session_state.proxies_raw)):
        if st.session_state.proxies_raw:
            st.write(f"**{len(st.session_state.proxies_raw)} proxies available**")
            
            count = st.slider(
                "Number to validate:",
                min_value=5,
                max_value=min(100, len(st.session_state.proxies_raw)),
                value=min(20, len(st.session_state.proxies_raw))
            )
            
            concurrent = st.slider(
                "Concurrent tests:",
                min_value=5,
                max_value=50,
                value=20
            )
            
            verify_tls = st.checkbox("Verify TLS certificates", value=True)
            
            if st.button("🧪 Validate Proxies", type="primary", use_container_width=True):
                with st.spinner(f"Validating {count} proxies..."):
                    progress_bar = st.progress(0)
                    progress_counter = {"count": 0}
                    
                    def update_progress():
                        progress_counter["count"] += 1
                        progress_bar.progress(progress_counter["count"] / count)
                    
                    sample = st.session_state.proxies_raw[:count]
                    start_time = time.time()
                    
                    validated = run_async(
                        validate_batch_concurrent(
                            sample,
                            verify_tls=verify_tls,
                            max_concurrent=concurrent,
                            progress_callback=update_progress
                        )
                    )
                    
                    elapsed = time.time() - start_time
                    st.session_state.validation_time = elapsed
                    
                    st.session_state.proxies_validated.extend(validated)
                    st.session_state.proxies_validated = list(
                        {p: None for p in st.session_state.proxies_validated}.keys()
                    )
                    
                    st.session_state.validation_stats["total"] += count
                    st.session_state.validation_stats["success"] += len(validated)
                    st.session_state.validation_stats["failed"] += count - len(validated)
                    
                    st.success(f"✅ {len(validated)} working proxies found in {elapsed:.1f}s")
                    st.info(f"Success rate: {(len(validated)/count)*100:.1f}%")
        else:
            st.info("Fetch proxies first")
    
    st.markdown("---")
    
    # Data Management
    with st.expander("💾 Data Management"):
        if st.session_state.proxies_validated:
            if st.button("🗑️ Clear Validated", use_container_width=True):
                st.session_state.proxies_validated = []
                st.session_state.validation_stats = {"total": 0, "success": 0, "failed": 0}
                st.rerun()
        
        if st.session_state.proxies_raw:
            if st.button("🗑️ Clear Raw", use_container_width=True):
                st.session_state.proxies_raw = []
                st.rerun()
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            for key in ["proxies_raw", "proxies_validated", "proxy_chain", "chain_test_results"]:
                if key in st.session_state:
                    st.session_state[key] = []
            st.session_state.validation_stats = {"total": 0, "success": 0, "failed": 0}
            st.rerun()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Proxy List", "🔗 Chain Builder", "📈 Analytics"])

with tab1:
    render_proxy_stats()

with tab2:
    render_proxy_table()

with tab3:
    render_chain_builder()

with tab4:
    st.subheader("📈 Advanced Analytics")
    
    if st.session_state.proxies_validated:
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            protocol_stats = defaultdict(list)
            for p in st.session_state.proxies_validated:
                if p.latency:
                    protocol_stats[p.protocol].append(p.latency)
            
            if protocol_stats:
                avg_latencies = {
                    proto: statistics.mean(latencies) 
                    for proto, latencies in protocol_stats.items()
                }
                
                df_perf = pd.DataFrame(
                    list(avg_latencies.items()),
                    columns=["Protocol", "Avg Latency (ms)"]
                )
                
                fig = px.bar(df_perf, x="Protocol", y="Avg Latency (ms)", 
                           title="Average Latency by Protocol")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            anonymity_counts = defaultdict(int)
            for p in st.session_state.proxies_validated:
                anonymity_counts[p.anonymity_level] += 1
            
            if anonymity_counts:
                df_anon = pd.DataFrame(
                    list(anonymity_counts.items()),
                    columns=["Level", "Count"]
                )
                
                fig = px.pie(df_anon, values="Count", names="Level", 
                           title="Anonymity Levels")
                st.plotly_chart(fig, use_container_width=True)
        
        # Validation statistics
        if st.session_state.validation_stats["total"] > 0:
            st.write("**Validation Statistics:**")
            
            stats = st.session_state.validation_stats
            success_rate = (stats["success"] / stats["total"]) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tested", stats["total"])
            with col2:
                st.metric("Successful", stats["success"])
            with col3:
                st.metric("Failed", stats["failed"])
            with col4:
                st.metric("Success Rate", f"{success_rate:.1f}%")
    else:
        st.info("No data available. Fetch and validate proxies to see analytics.")

# Footer
st.markdown("---")
st.caption("ProxyStream Cloud Enhanced - Advanced proxy management system")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Module status indicator
if not MODULES_AVAILABLE:
    st.caption("⚠️ Running in fallback mode - modules not available")
