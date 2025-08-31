"""
ProxyStream Enhanced - DNS over TCP & Offline-First Architecture
No dependency on HTTP "what's my IP" APIs
"""

import asyncio
import ssl
import time
import json
import sqlite3
import base64
import os
import struct
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import ipaddress
import re
import statistics
import random
import socket
import pathlib
import hashlib
import threading

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import numpy as np
import aiohttp
from aiohttp_socks import ChainProxyConnector

# Optional deps
try:
    import redis
except Exception:
    redis = None
try:
    import pyasn
except Exception:
    pyasn = None
try:
    import maxminddb
except Exception:
    maxminddb = None

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Enhanced",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

# Validation endpoints - normal websites, not APIs
VALIDATION_ENDPOINTS = [
    "https://www.wikipedia.org/robots.txt",
    "https://www.cloudflare.com/robots.txt",
    "https://www.github.com/robots.txt",
    "https://www.google.com/robots.txt",
    "https://www.microsoft.com/robots.txt"
]

# DNS resolvers for exit IP detection
DNS_RESOLVERS = [
    ("208.67.222.222", 53),  # OpenDNS
    ("208.67.220.220", 53),  # OpenDNS backup
    ("1.1.1.1", 53),         # Cloudflare
    ("8.8.8.8", 53),         # Google
]

DB_PATH = "proxystream_enhanced.db"

# Offline database paths
GEOLITE2_PATH = os.getenv("GEOLITE2_DB", "GeoLite2-City.mmdb")
PYASN_DB_PATH = os.getenv("PYASN_DB", "ipasn.dat")

# Initialize offline databases
pyasn_obj = pyasn.pyasn(PYASN_DB_PATH) if (pyasn and pathlib.Path(PYASN_DB_PATH).exists()) else None
geo_reader = maxminddb.open_database(GEOLITE2_PATH) if (maxminddb and pathlib.Path(GEOLITE2_PATH).exists()) else None

# Redis (optional)
REDIS_URL = os.getenv("PROXYSTREAM_REDIS_URL", "")
redis_client = redis.Redis.from_url(REDIS_URL) if (redis and REDIS_URL) else None

# ---------------------------------------------------
# DNS over TCP Module
# ---------------------------------------------------
def build_dns_query(name: str, qtype: int = 1, qclass: int = 1) -> Tuple[bytes, int]:
    """Build DNS query packet with TCP framing"""
    tid = random.randint(0, 0xFFFF)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    parts = name.split(".")
    qname = b"".join(struct.pack("B", len(p)) + p.encode() for p in parts) + b"\x00"
    question = struct.pack("!HH", qtype, qclass)
    payload = header + qname + question
    return struct.pack("!H", len(payload)) + payload, tid

def parse_dns_response(resp: bytes, tid: int) -> Optional[str]:
    """Parse DNS A record response"""
    if len(resp) < 2:
        return None
    resp = resp[2:]  # Skip TCP length prefix
    
    if len(resp) < 12:
        return None
    
    rid, flags, qd, an, ns, ar = struct.unpack("!HHHHHH", resp[:12])
    if rid != tid or an == 0:
        return None
    
    # Skip question section
    i = 12
    while i < len(resp) and resp[i] != 0:
        i += resp[i] + 1
    i += 5  # Skip null + QTYPE + QCLASS
    
    if i + 2 > len(resp):
        return None
    
    # Handle compression
    if resp[i] & 0xC0 == 0xC0:
        i += 2
    else:
        while i < len(resp) and resp[i] != 0:
            i += resp[i] + 1
        i += 1
    
    if i + 10 > len(resp):
        return None
    
    rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", resp[i:i+10])
    i += 10
    
    if rtype == 1 and rdlen == 4:  # A record
        return ".".join(str(b) for b in resp[i:i+4])
    
    return None

async def dns_query_direct(resolver: str, port: int, query_name: str, timeout: float = 5.0) -> Optional[str]:
    """Direct DNS query over TCP"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(resolver, port),
            timeout=timeout
        )
        
        query, tid = build_dns_query(query_name)
        writer.write(query)
        await writer.drain()
        
        len_bytes = await asyncio.wait_for(reader.readexactly(2), timeout=timeout)
        resp_len = struct.unpack("!H", len_bytes)[0]
        response = await asyncio.wait_for(reader.readexactly(resp_len), timeout=timeout)
        
        writer.close()
        await writer.wait_closed()
        
        return parse_dns_response(len_bytes + response, tid)
    except Exception:
        return None

async def get_exit_ip_dns_via_chain(chain: List['ProxyInfo']) -> Optional[str]:
    """Get exit IP using DNS over TCP through proxy chain"""
    from aiohttp_socks import ChainProxyConnector
    
    for resolver_host, resolver_port in DNS_RESOLVERS:
        try:
            # Use the chain to connect to DNS resolver
            urls = [p.as_url() for p in chain]
            connector = ChainProxyConnector.from_urls(urls)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                # Create raw TCP connection through chain
                conn = await connector._wrap_create_connection(
                    lambda: asyncio.Protocol(),
                    resolver_host,
                    resolver_port
                )
                
                if not conn:
                    continue
                    
                transport, protocol = conn
                reader = asyncio.StreamReader()
                writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())
                
                # Send DNS query
                query, tid = build_dns_query("myip.opendns.com")
                writer.write(query)
                await writer.drain()
                
                # Read response
                len_bytes = await reader.readexactly(2)
                resp_len = struct.unpack("!H", len_bytes)[0]
                response = await reader.readexactly(resp_len)
                
                writer.close()
                
                ip = parse_dns_response(len_bytes + response, tid)
                if ip:
                    return ip
                    
        except Exception:
            continue
    
    return None

async def get_client_ip_dns() -> Optional[str]:
    """Get client's public IP via DNS (no HTTP)"""
    for resolver, port in DNS_RESOLVERS:
        ip = await dns_query_direct(resolver, port, "myip.opendns.com")
        if ip:
            return ip
    return None

# ---------------------------------------------------
# Offline Geo/ASN Lookup
# ---------------------------------------------------
def offline_geo_lookup(ip: str) -> Dict[str, Any]:
    """Get geo data from local MaxMind database"""
    out = {}
    
    # Try pyasn first for ASN data
    if pyasn_obj and _is_ip(ip):
        try:
            asn, prefix = pyasn_obj.lookup(ip)
            if asn:
                out['asn'] = f"AS{asn}"
                out['prefix'] = prefix
        except Exception:
            pass
    
    # Try MaxMind for geo data
    if geo_reader:
        try:
            data = geo_reader.get(ip) or {}
            country = data.get('country', {})
            city = data.get('city', {})
            location = data.get('location', {})
            
            out['country_code'] = country.get('iso_code')
            out['country'] = country.get('names', {}).get('en')
            out['city'] = city.get('names', {}).get('en')
            
            subdivisions = data.get('subdivisions', [])
            if subdivisions:
                out['region'] = subdivisions[0].get('names', {}).get('en')
            
            out['lat'] = location.get('latitude')
            out['lon'] = location.get('longitude')
            out['accuracy'] = location.get('accuracy_radius')
            
        except Exception:
            pass
    
    return out

def _is_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except Exception:
        return False

# ---------------------------------------------------
# Safe async runner for Streamlit
# ---------------------------------------------------
def run_async(coro):
    """Safe async runner that works with Streamlit's event loop"""
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
        thread.join()
        
        if exception:
            raise exception
        return result
    except RuntimeError:
        return asyncio.run(coro)

# ---------------------------------------------------
# Cache
# ---------------------------------------------------
class TTLCache:
    def __init__(self, ttl_seconds: int = 300, maxsize: int = 4096):
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        if len(self._store) >= self.maxsize:
            self._store.pop(next(iter(self._store)))
        self._store[key] = (time.time(), value)

exit_ip_cache = TTLCache(ttl_seconds=300)
geo_cache = TTLCache(ttl_seconds=3600)

# ---------------------------------------------------
# Database
# ---------------------------------------------------
@st.cache_resource
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
    """)
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
            PRIMARY KEY (host, port, protocol)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chain_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tested_at TIMESTAMP,
            exit_ip TEXT,
            total_latency REAL,
            hop_count INTEGER,
            anonymity_json TEXT
        )
    """)
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_proxies_last_tested ON proxies(last_tested);
        CREATE INDEX IF NOT EXISTS idx_proxies_latency ON proxies(latency);
    """)
    conn.commit()
    conn.close()

init_database()

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "proxies_raw" not in st.session_state: st.session_state.proxies_raw = []
if "proxies_validated" not in st.session_state: st.session_state.proxies_validated = []
if "proxy_chain" not in st.session_state: st.session_state.proxy_chain = []
if "client_location" not in st.session_state: st.session_state.client_location = None
if "chain_test_result" not in st.session_state: st.session_state.chain_test_result = None

# ---------------------------------------------------
# Models
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
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    asn: Optional[str] = None
    org: Optional[str] = None
    isp: Optional[str] = None
    is_valid: bool = False
    last_tested: Optional[datetime] = None

    def __hash__(self):
        return hash((self.host, self.port, self.protocol))

    def as_url(self) -> str:
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

# ---------------------------------------------------
# Rate Limiter
# ---------------------------------------------------
class RateLimiter:
    def __init__(self, rate: int, per_seconds: int):
        self.rate = rate
        self.per = per_seconds
        self.buckets: Dict[str, Tuple[int, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        tokens, ts = self.buckets.get(key, (0, now))
        if now - ts > self.per:
            tokens, ts = 0, now
        tokens += 1
        self.buckets[key] = (tokens, ts)
        return tokens <= self.rate

rate_limiter = RateLimiter(rate=30, per_seconds=60)

# ---------------------------------------------------
# Proxy Validation (No HTTP APIs!)
# ---------------------------------------------------
async def validate_proxy_no_api(proxy: ProxyInfo, timeout: int = 15, verify: bool = False) -> Tuple[bool, float, Optional[str]]:
    """Validate proxy by fetching normal websites, not APIs"""
    if not rate_limiter.allow(f"validate:{proxy.host}:{proxy.port}"):
        return False, 0.0, None

    proxy_url = proxy.as_url()
    
    # Try multiple normal endpoints
    for endpoint in VALIDATION_ENDPOINTS:
        try:
            start = time.perf_counter_ns()
            async with httpx.AsyncClient(
                proxies=proxy_url,
                timeout=timeout,
                follow_redirects=True,
                verify=verify
            ) as client:
                r = await client.get(endpoint)
                
                # Accept various status codes that indicate the proxy works
                if r.status_code in [200, 301, 302, 403, 404]:
                    latency_ms = (time.perf_counter_ns() - start) / 1_000_000
                    
                    # For exit IP, we'll use DNS later in the chain test
                    # For now, just return the proxy as working
                    return True, latency_ms, proxy.host
                    
        except Exception:
            continue
    
    return False, 0.0, None

# ---------------------------------------------------
# Proxy Parsing (supports hostnames)
# ---------------------------------------------------
def parse_proxy_line(line: str) -> Optional[ProxyInfo]:
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    proto = "http"
    rest = line
    user = pw = None

    if "://" in line:
        proto, rest = line.split("://", 1)

    if "@" in rest:
        auth, addr = rest.split("@", 1)
        if ":" in auth:
            user, pw = auth.split(":", 1)
        rest = addr

    # Parse host:port (works for IPs and hostnames)
    if ":" in rest:
        host, port_str = rest.rsplit(":", 1)
        try:
            port = int(port_str)
            if 1 <= port <= 65535:
                return ProxyInfo(host=host, port=port, protocol=proto.lower(), 
                               username=user, password=pw)
        except ValueError:
            pass
    
    return None

async def fetch_proxies(source_url: str, limit: int = 1000) -> List[ProxyInfo]:
    proxies: List[ProxyInfo] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(source_url)
            if r.status_code == 200:
                for line in r.text.strip().splitlines()[:limit]:
                    pi = parse_proxy_line(line)
                    if pi:
                        proxies.append(pi)
    except Exception as e:
        st.warning(f"Failed to fetch {source_url}: {str(e)[:100]}")
    return proxies

# ---------------------------------------------------
# Batch Validation with Offline Geo
# ---------------------------------------------------
async def validate_batch(proxies: List[ProxyInfo], max_concurrent: int = 5, verify_tls: bool = False) -> List[ProxyInfo]:
    """Validate proxies without using HTTP APIs"""
    validated: List[ProxyInfo] = []
    sem = asyncio.Semaphore(max_concurrent)

    async def validate_one(proxy: ProxyInfo):
        async with sem:
            try:
                ok, lat_ms, _ = await validate_proxy_no_api(proxy, timeout=20, verify=verify_tls)
                if ok:
                    proxy.is_valid = True
                    proxy.latency = lat_ms
                    proxy.last_tested = datetime.now()
                    
                    # Get geo from offline database or cache
                    cache_key = f"geo:{proxy.host}"
                    geo = geo_cache.get(cache_key)
                    if not geo:
                        geo = offline_geo_lookup(proxy.host)
                        geo_cache.set(cache_key, geo)
                    
                    proxy.country = geo.get('country')
                    proxy.country_code = geo.get('country_code')
                    proxy.city = geo.get('city')
                    proxy.region = geo.get('region')
                    proxy.lat = geo.get('lat')
                    proxy.lon = geo.get('lon')
                    proxy.asn = geo.get('asn')
                    
                    return proxy
            except Exception:
                pass
            return None

    # Process in small batches
    for i in range(0, len(proxies), 5):
        batch = proxies[i:i+5]
        results = await asyncio.gather(*(validate_one(p) for p in batch), return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception):
                validated.append(r)
        await asyncio.sleep(0.5)  # Brief pause between batches

    # Save to database
    if validated:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for p in validated:
            cur.execute("""
                INSERT OR REPLACE INTO proxies
                (host, port, protocol, username, password, latency, last_tested,
                 country, country_code, city, region, lat, lon, asn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.host, p.port, p.protocol, p.username, p.password, p.latency, p.last_tested,
                p.country, p.country_code, p.city, p.region, p.lat, p.lon, p.asn
            ))
        conn.commit()
        conn.close()
    
    return validated

def load_proxies_from_db(filters: Dict[str, Any] = None) -> List[ProxyInfo]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q = "SELECT * FROM proxies WHERE last_tested > datetime('now', '-7 days')"
    params = []
    
    if filters:
        if filters.get('country'):
            q += " AND (country LIKE ? OR country_code = ?)"
            params += [f"%{filters['country']}%", filters['country']]
        if filters.get('asn'):
            q += " AND asn LIKE ?"
            params.append(f"%{filters['asn']}%")
    
    q += " ORDER BY latency ASC LIMIT 500"
    cur.execute(q, params)
    rows = cur.fetchall()
    conn.close()
    
    proxies = []
    for r in rows:
        p = ProxyInfo(
            host=r[0], port=r[1], protocol=r[2],
            username=r[3], password=r[4], latency=r[5],
            last_tested=r[6], country=r[7], country_code=r[8],
            city=r[9], region=r[10], lat=r[11], lon=r[12],
            asn=r[13], is_valid=True
        )
        proxies.append(p)
    
    return proxies

# ---------------------------------------------------
# Chain Testing with DNS Exit IP
# ---------------------------------------------------
async def test_chain_dns(chain: List[ProxyInfo]) -> Dict[str, Any]:
    """Test proxy chain using DNS for exit IP detection"""
    if not chain:
        return {"success": False, "error": "Empty chain"}
    
    try:
        # Get exit IP via DNS through the chain
        exit_ip = await get_exit_ip_dns_via_chain(chain)
        if not exit_ip:
            return {"success": False, "error": "Could not determine exit IP via DNS"}
        
        # Get geo for exit IP
        exit_geo = offline_geo_lookup(exit_ip) if exit_ip else {}
        
        # Test anonymity by fetching a normal page
        urls = [p.as_url() for p in chain]
        connector = ChainProxyConnector.from_urls(urls)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get("https://www.wikipedia.org/robots.txt") as resp:
                # Check for proxy headers
                via = resp.headers.get("Via")
                xff = resp.headers.get("X-Forwarded-For")
                
                anonymity = "elite"
                if via or xff:
                    anonymity = "anonymous"
                if chain[0].host in str(resp.headers):
                    anonymity = "transparent"
        
        # Save test result
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chain_tests (tested_at, exit_ip, hop_count, anonymity_json)
            VALUES (?, ?, ?, ?)
        """, (datetime.utcnow(), exit_ip, len(chain), json.dumps({"level": anonymity})))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "exit_ip": exit_ip,
            "exit_geo": exit_geo,
            "hop_count": len(chain),
            "anonymity": anonymity
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.markdown("""
<style>
    .main { padding-top: 0; }
    h1 { color: #2563eb; }
    .stTabs [aria-selected="true"] { background: #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔒 ProxyStream Enhanced - Offline-First Architecture")

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📍 Get My IP (DNS)", use_container_width=True):
        with st.spinner("Getting IP via DNS..."):
            client_ip = run_async(get_client_ip_dns())
            if client_ip:
                client_geo = offline_geo_lookup(client_ip)
                st.session_state.client_location = {
                    "ip": client_ip,
                    "city": client_geo.get("city", "Unknown"),
                    "country_code": client_geo.get("country_code", "")
                }
                st.success(f"Your IP: {client_ip}")
                st.rerun()

if st.session_state.client_location:
    loc = st.session_state.client_location
    with col2: st.metric("Location", f"{loc.get('city')}, {loc.get('country_code')}")
    with col3: st.metric("Your IP", loc.get('ip'))
    with col4: 
        db_status = "✅" if geo_reader else "❌"
        st.metric("Offline DB", db_status)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Database status
    st.subheader("📊 Database Status")
    if geo_reader:
        st.success("✅ GeoLite2 database loaded")
    else:
        st.warning("⚠️ GeoLite2 not found - download from MaxMind")
    
    if pyasn_obj:
        st.success("✅ pyasn database loaded")
    else:
        st.info("ℹ️ pyasn not found - optional")
    
    # Settings
    st.subheader("⚙️ Settings")
    verify_tls = st.checkbox("Verify TLS", value=False)
    use_dns = st.checkbox("Use DNS for exit IP", value=True, help="More reliable than HTTP APIs")
    
    # Load proxies
    st.subheader("📥 Load Proxies")
    if st.button("🔄 Fetch from Sources", use_container_width=True):
        with st.spinner("Fetching..."):
            all_proxies = []
            for src in PROXY_SOURCES:
                proxies = run_async(fetch_proxies(src))
                all_proxies.extend(proxies)
            unique = list({p: None for p in all_proxies}.keys())
            st.session_state.proxies_raw = unique
            st.success(f"Loaded {len(unique)} proxies")
    
    # Validate
    if st.session_state.proxies_raw:
        st.subheader("✅ Validate")
        count = st.slider("Validate count:", 5, 50, 10)
        if st.button("🧪 Validate", use_container_width=True):
            with st.spinner(f"Validating {count} proxies..."):
                sample = st.session_state.proxies_raw[:count]
                validated = run_async(validate_batch(sample, verify_tls=verify_tls))
                st.session_state.proxies_validated.extend(validated)
                st.success(f"✅ {len(validated)} working")
    
    # Search
    st.subheader("🔍 Search Database")
    filters = {
        'country': st.text_input("Country"),
        'asn': st.text_input("ASN")
    }
    if st.button("🔍 Search", use_container_width=True):
        filtered = {k: v for k, v in filters.items() if v}
        results = load_proxies_from_db(filtered)
        st.session_state.proxies_validated = results
        st.success(f"Found {len(results)} proxies")
    
    # Chain builder
    st.subheader("🔗 Chain Builder")
    if st.session_state.proxies_validated:
        proxy_options = [
            f"{p.host}:{p.port} ({p.country_code or 'XX'}) {p.latency:.0f}ms" if p.latency 
            else f"{p.host}:{p.port}"
            for p in st.session_state.proxies_validated[:50]
        ]
        selected = st.selectbox("Select proxy:", proxy_options)
        
        if st.button("➕ Add to Chain"):
            idx = proxy_options.index(selected)
            proxy = st.session_state.proxies_validated[idx]
            if len(st.session_state.proxy_chain) < 5:
                st.session_state.proxy_chain.append(proxy)
                st.success(f"Added hop {len(st.session_state.proxy_chain)}")
    
    # Current chain
    if st.session_state.proxy_chain:
        st.write(f"**Chain ({len(st.session_state.proxy_chain)} hops):**")
        for i, p in enumerate(st.session_state.proxy_chain):
            st.write(f"{i+1}. {p.host}:{p.port}")
        
        if len(st.session_state.proxy_chain) >= 2:
            if st.button("🧪 Test Chain (DNS)", use_container_width=True):
                with st.spinner("Testing via DNS..."):
                    result = run_async(test_chain_dns(st.session_state.proxy_chain))
                    st.session_state.chain_test_result = result
                    if result["success"]:
                        st.success(f"✅ Exit IP: {result['exit_ip']}")
                        st.info(f"Anonymity: {result['anonymity']}")
                    else:
                        st.error(result['error'])
        
        if st.button("🗑️ Clear Chain", use_container_width=True):
            st.session_state.proxy_chain = []
            st.rerun()

# Main area - Proxy list
st.subheader("📊 Validated Proxies")
if st.session_state.proxies_validated:
    data = [{
        'Host': p.host,
        'Port': p.port,
        'Protocol': p.protocol,
        'Country': p.country or 'Unknown',
        'City': p.city or 'Unknown',
        'ASN': p.asn or 'N/A',
        'Latency': f"{p.latency:.0f}ms" if p.latency else 'N/A'
    } for p in st.session_state.proxies_validated]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=400)
else:
    st.info("No validated proxies. Load and validate from sidebar.")

st.markdown("---")
st.caption("ProxyStream Enhanced - DNS over TCP exit detection, offline geo databases, no HTTP API dependencies")
