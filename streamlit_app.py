"""
ProxyStream Complete - Streamlit Cloud Ready
All functions included, uses cached geo database for ephemeral storage
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
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import numpy as np
import aiohttp
from aiohttp_socks import ChainProxyConnector

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Cloud",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sources
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

VALIDATION_ENDPOINTS = [
    "https://www.wikipedia.org/robots.txt",
    "https://www.cloudflare.com/robots.txt",
    "https://www.google.com/robots.txt",
]

DNS_RESOLVERS = [
    ("208.67.222.222", 53),  # OpenDNS
    ("208.67.220.220", 53),  # OpenDNS backup
    ("1.1.1.1", 53),         # Cloudflare
    ("8.8.8.8", 53),         # Google
]

# ---------------------------------------------------
# Streamlit Cloud Geo Database Setup
# ---------------------------------------------------

@st.cache_data(ttl=86400*7)  # Cache for 7 days
def download_ip2location_cached() -> bytes:
    """Download IP2Location LITE database and cache it"""
    try:
        url = "https://download.ip2location.com/lite/IP2LOCATION-LITE-DB5.BIN.ZIP"
        
        response = httpx.get(url, timeout=60, follow_redirects=True)
        if response.status_code != 200:
            return None
        
        # Extract BIN file from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for name in z.namelist():
                if name.endswith(".BIN"):
                    return z.read(name)
        
        return None
    except Exception as e:
        st.error(f"IP2Location download error: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def download_geolite2_cached(license_key: str) -> bytes:
    """Download GeoLite2 and cache it"""
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
            st.error(f"GeoLite2 download failed: {response.status_code}")
            return None
        
        # Extract .mmdb file from tar.gz
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
    """Initialize geo reader from cached bytes"""
    if not _db_bytes:
        return None
    
    # Write to temp file (required by both libraries)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(_db_bytes)
        tmp_path = tmp.name
    
    try:
        # Try MaxMind format first
        try:
            import maxminddb
            reader = maxminddb.open_database(tmp_path)
            # Mark as MaxMind type
            reader._type = "maxmind"
            return reader
        except:
            pass
        
        # Try IP2Location format
        try:
            import IP2Location
            reader = IP2Location.IP2Location(tmp_path)
            # Mark as IP2Location type
            reader._type = "ip2location"
            return reader
        except:
            pass
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return None

def get_license_key() -> Optional[str]:
    """Get MaxMind license key from various sources"""
    # Check Streamlit secrets (for deployment)
    if hasattr(st, 'secrets') and "MAXMIND_LICENSE_KEY" in st.secrets:
        return st.secrets["MAXMIND_LICENSE_KEY"]
    
    # Check environment variable
    if "MAXMIND_LICENSE_KEY" in os.environ:
        return os.environ["MAXMIND_LICENSE_KEY"]
    
    # Check session state
    if "maxmind_license_key" in st.session_state:
        return st.session_state.maxmind_license_key
    
    return None

def geo_lookup(ip: str, reader=None) -> Dict[str, Any]:
    """Universal geo lookup that works with any reader type"""
    if not reader:
        if "geo_reader" in st.session_state:
            reader = st.session_state.geo_reader
        else:
            return {"country": "Unknown", "country_code": "XX"}
    
    if not reader:
        return {"country": "Unknown", "country_code": "XX"}
    
    try:
        # Check reader type
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
        
        # Fallback: try both formats
        if hasattr(reader, 'get'):  # MaxMind
            data = reader.get(ip) or {}
            return {
                "country": data.get('country', {}).get('names', {}).get('en'),
                "country_code": data.get('country', {}).get('iso_code'),
                "city": data.get('city', {}).get('names', {}).get('en'),
                "lat": data.get('location', {}).get('latitude'),
                "lon": data.get('location', {}).get('longitude')
            }
        elif hasattr(reader, 'get_all'):  # IP2Location
            rec = reader.get_all(ip)
            return {
                "country": rec.country_long if rec else None,
                "country_code": rec.country_short if rec else None,
                "city": rec.city if rec else None,
                "lat": rec.latitude if rec else None,
                "lon": rec.longitude if rec else None
            }
    except Exception as e:
        pass
    
    return {"country": "Unknown", "country_code": "XX"}

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
    lat: Optional[float] = None
    lon: Optional[float] = None
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
# Manual CONNECT tunnel for raw TCP
# ---------------------------------------------------
async def _connect_chain_http_connect(
    chain: List[ProxyInfo], 
    dest_host: str, 
    dest_port: int
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Build a raw TCP tunnel through HTTP/HTTPS proxies only."""
    if any(p.protocol not in ("http", "https") for p in chain):
        raise RuntimeError("DNS over TCP requires HTTP/HTTPS hops only (no SOCKS)")

    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    
    try:
        # Connect to first hop
        r, w = await asyncio.open_connection(chain[0].host, chain[0].port)
        reader, writer = r, w

        for i, hop in enumerate(chain):
            # Determine next destination
            if i == len(chain) - 1:
                next_host, next_port = dest_host, dest_port
            else:
                next_host, next_port = chain[i + 1].host, chain[i + 1].port

            # Build CONNECT request
            req = f"CONNECT {next_host}:{next_port} HTTP/1.1\r\n"
            req += f"Host: {next_host}:{next_port}\r\n"
            req += "Connection: keep-alive\r\n"
            
            # Add auth if needed
            if hop.username and hop.password:
                token = base64.b64encode(f"{hop.username}:{hop.password}".encode()).decode()
                req += f"Proxy-Authorization: Basic {token}\r\n"
            req += "\r\n"

            writer.write(req.encode("ascii"))
            await writer.drain()

            # Read status line
            status_line = await reader.readuntil(b"\r\n")
            if b"200" not in status_line and b"HTTP/1.1 200" not in status_line:
                raise RuntimeError(f"CONNECT failed at hop {i+1}: {status_line.decode(errors='ignore').strip()}")

            # Drain headers
            while True:
                line = await reader.readuntil(b"\r\n")
                if line in (b"\r\n", b"\n", b""):
                    break

        return reader, writer
        
    except Exception as e:
        if writer:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        raise

# ---------------------------------------------------
# DNS over TCP Module
# ---------------------------------------------------
def build_dns_query(name: str, qtype: int = 1) -> Tuple[bytes, int]:
    """Build DNS query packet with TCP framing"""
    tid = random.randint(0, 0xFFFF)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    parts = name.split(".")
    qname = b"".join(struct.pack("B", len(p)) + p.encode() for p in parts) + b"\x00"
    question = struct.pack("!HH", qtype, 1)  # IN class
    payload = header + qname + question
    return struct.pack("!H", len(payload)) + payload, tid

def parse_dns_response(resp: bytes, tid: int) -> Optional[str]:
    """Parse DNS A record response with proper boundary checking"""
    if len(resp) < 14:
        return None
    
    # Skip TCP length prefix
    resp = resp[2:]
    
    # Parse header
    if len(resp) < 12:
        return None
    
    rid, flags, qd, an, ns, ar = struct.unpack("!HHHHHH", resp[:12])
    if rid != tid or an == 0:
        return None
    
    # Skip question section
    i = 12
    while i < len(resp) and resp[i] != 0:
        if i + resp[i] + 1 > len(resp):
            return None
        i += resp[i] + 1
    i += 5  # Skip null + QTYPE + QCLASS
    
    if i + 2 > len(resp):
        return None
    
    # Handle compression in answer
    if resp[i] & 0xC0 == 0xC0:
        i += 2
    else:
        while i < len(resp) and resp[i] != 0:
            if i + resp[i] + 1 > len(resp):
                return None
            i += resp[i] + 1
        i += 1
    
    # Parse RR fields
    if i + 10 > len(resp):
        return None
    
    rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", resp[i:i+10])
    i += 10
    
    # Extract IP address with boundary check
    if rtype == 1 and rdlen == 4 and i + 4 <= len(resp):
        return ".".join(str(b) for b in resp[i:i+4])
    
    return None

async def get_exit_ip_dns_direct() -> Optional[str]:
    """Get client IP via DNS (no proxy)"""
    for resolver_host, resolver_port in DNS_RESOLVERS:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(resolver_host, resolver_port),
                timeout=5.0
            )
            
            query, tid = build_dns_query("myip.opendns.com")
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
    """Get exit IP using DNS over TCP through HTTP/HTTPS chain"""
    # Check chain compatibility
    if any(p.protocol not in ("http", "https") for p in chain):
        return None  # Silently fail for SOCKS chains
    
    for resolver_host, resolver_port in DNS_RESOLVERS:
        try:
            # Use manual CONNECT tunnel
            rd, wr = await _connect_chain_http_connect(chain, resolver_host, resolver_port)
            try:
                query, tid = build_dns_query("myip.opendns.com")
                wr.write(query)
                await wr.drain()
                
                len_bytes = await rd.readexactly(2)
                resp_len = struct.unpack("!H", len_bytes)[0]
                payload = await rd.readexactly(resp_len)
                
                ip = parse_dns_response(len_bytes + payload, tid)
                if ip:
                    return ip
            finally:
                wr.close()
                try:
                    await wr.wait_closed()
                except:
                    pass
        except Exception:
            continue
    
    return None

# ---------------------------------------------------
# Safe async runner for Streamlit
# ---------------------------------------------------
def run_async(coro):
    """Safe async execution in Streamlit context"""
    try:
        loop = asyncio.get_running_loop()
        # Use thread to avoid event loop conflicts
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
        thread.join(timeout=30)
        
        if exception:
            raise exception
        return result
    except RuntimeError:
        return asyncio.run(coro)

# ---------------------------------------------------
# Database (In-Memory for Streamlit Cloud)
# ---------------------------------------------------
@st.cache_resource
def init_database():
    """Initialize in-memory SQLite database"""
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
            PRIMARY KEY (host, port, protocol)
        )
    """)
    conn.commit()
    return conn

# Get shared database connection
db_conn = init_database()

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "proxies_raw" not in st.session_state: 
    st.session_state.proxies_raw = []
if "proxies_validated" not in st.session_state: 
    st.session_state.proxies_validated = []
if "proxy_chain" not in st.session_state: 
    st.session_state.proxy_chain = []
if "geo_reader" not in st.session_state:
    st.session_state.geo_reader = None

# ---------------------------------------------------
# Proxy Parsing
# ---------------------------------------------------
def parse_proxy_line(line: str) -> Optional[ProxyInfo]:
    """Parse proxy with proper URL parsing"""
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    # Add scheme if missing
    if "://" not in s:
        s = "http://" + s

    try:
        u = urlsplit(s)
        if not u.hostname or not u.port:
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
    except Exception:
        return None

async def fetch_proxies(url: str) -> List[ProxyInfo]:
    """Fetch and parse proxy list"""
    proxies = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
            if r.status_code == 200:
                for line in r.text.splitlines()[:500]:
                    p = parse_proxy_line(line)
                    if p:
                        proxies.append(p)
    except:
        pass
    return proxies

# ---------------------------------------------------
# Proxy Validation
# ---------------------------------------------------
async def validate_proxy(proxy: ProxyInfo, timeout: int = 15, verify: bool = True) -> Tuple[bool, float]:
    """Validate proxy with proper httpx syntax"""
    proxy_url = proxy.as_url()
    
    # Build proper proxy dict for httpx
    proxy_dict = {"http://": proxy_url, "https://": proxy_url}
    
    for endpoint in VALIDATION_ENDPOINTS:
        try:
            start = time.perf_counter_ns()
            async with httpx.AsyncClient(
                proxies=proxy_dict,
                timeout=httpx.Timeout(timeout, connect=timeout),
                follow_redirects=True,
                verify=verify
            ) as client:
                r = await client.get(endpoint)
                
                # Check status and content size
                text = r.text[:8192]  # Cap for sanity
                if r.status_code in [200, 301, 302, 403, 404] and 0 < len(text) < 200_000:
                    latency = (time.perf_counter_ns() - start) / 1_000_000
                    return True, latency
        except:
            continue
    
    return False, 0

async def validate_batch(proxies: List[ProxyInfo], verify_tls: bool = True) -> List[ProxyInfo]:
    """Validate proxy batch"""
    validated = []
    
    for proxy in proxies[:20]:  # Reasonable limit
        try:
            ok, latency = await validate_proxy(proxy, verify=verify_tls)
            if ok:
                proxy.is_valid = True
                proxy.latency = latency
                proxy.last_tested = datetime.now()
                
                # Geo lookup
                geo = geo_lookup(proxy.host, st.session_state.geo_reader)
                proxy.country = geo.get('country')
                proxy.country_code = geo.get('country_code')
                proxy.city = geo.get('city')
                proxy.lat = geo.get('lat')
                proxy.lon = geo.get('lon')
                
                validated.append(proxy)
        except:
            pass
    
    # Save to in-memory database
    if validated:
        cur = db_conn.cursor()
        for p in validated:
            cur.execute("""
                INSERT OR REPLACE INTO proxies
                (host, port, protocol, username, password, latency, last_tested,
                 country, country_code, city, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.host, p.port, p.protocol, p.username, p.password,
                p.latency, p.last_tested, p.country, p.country_code,
                p.city, p.lat, p.lon
            ))
        db_conn.commit()
    
    return validated

# ---------------------------------------------------
# Chain Testing
# ---------------------------------------------------
async def test_chain_dns(chain: List[ProxyInfo]) -> Dict[str, Any]:
    """Test proxy chain with DNS exit IP detection"""
    if not chain:
        return {"success": False, "error": "Empty chain"}
    
    # Check if chain supports DNS over TCP
    if any(p.protocol not in ("http", "https") for p in chain):
        # For SOCKS chains, fall back to HTTP-based testing
        return await test_chain_http(chain)
    
    try:
        # Get exit IP via DNS
        exit_ip = await get_exit_ip_dns_via_chain(chain)
        if not exit_ip:
            return {"success": False, "error": "Could not determine exit IP"}
        
        # Get geo for exit
        exit_geo = geo_lookup(exit_ip, st.session_state.geo_reader) if exit_ip else {}
        
        # Test anonymity
        urls = [p.as_url() for p in chain]
        connector = ChainProxyConnector.from_urls(urls)
        
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get("https://www.wikipedia.org/robots.txt") as resp:
                via = resp.headers.get("Via")
                xff = resp.headers.get("X-Forwarded-For")
                
                anonymity = "elite"
                if via or xff:
                    anonymity = "anonymous"
                if chain[0].host in str(resp.headers):
                    anonymity = "transparent"
        
        return {
            "success": True,
            "exit_ip": exit_ip,
            "exit_geo": exit_geo,
            "hop_count": len(chain),
            "anonymity": anonymity
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}

async def test_chain_http(chain: List[ProxyInfo]) -> Dict[str, Any]:
    """Fallback chain test using HTTP (for SOCKS chains)"""
    try:
        urls = [p.as_url() for p in chain]
        connector = ChainProxyConnector.from_urls(urls)
        
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30)) as session:
            # Try to get exit IP from HTTP service
            async with session.get("https://api.ipify.org?format=json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    exit_ip = data.get("ip")
                    
                    return {
                        "success": True,
                        "exit_ip": exit_ip,
                        "exit_geo": geo_lookup(exit_ip, st.session_state.geo_reader) if exit_ip else {},
                        "hop_count": len(chain),
                        "method": "HTTP (SOCKS chain)"
                    }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}
    
    return {"success": False, "error": "Chain test failed"}

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🔒 ProxyStream Cloud Edition")
st.caption("DNS-based exit detection, automatic geo database setup")

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📍 Get My IP (DNS)", use_container_width=True):
        with st.spinner("Getting IP via DNS..."):
            ip = run_async(get_exit_ip_dns_direct())
            if ip:
                geo = geo_lookup(ip, st.session_state.geo_reader)
                st.success(f"IP: {ip} ({geo.get('country_code', 'XX')})")

with col2:
    db_status = "✅ Ready" if st.session_state.geo_reader else "⚠️ Not Setup"
    st.metric("Geo Database", db_status)

with col3:
    st.metric("Validated", len(st.session_state.proxies_validated))

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Geo Database Setup
    st.subheader("🌍 Geo Database Setup")
    
    if st.session_state.geo_reader:
        st.success("✅ Geo database active")
    else:
        setup_method = st.radio(
            "Choose method:",
            ["IP2Location (Free)", "GeoLite2 (License Key)"]
        )
        
        if setup_method == "IP2Location (Free)":
            if st.button("📥 Setup IP2Location", use_container_width=True):
                with st.spinner("Downloading (~20MB)..."):
                    db_bytes = download_ip2location_cached()
                    if db_bytes:
                        reader = init_geo_reader(db_bytes)
                        if reader:
                            st.session_state.geo_reader = reader
                            st.success("✅ Ready!")
                            st.rerun()
        else:
            license_key = get_license_key()
            
            if not license_key:
                st.info("Get free key at [MaxMind](https://www.maxmind.com/en/geolite2/signup)")
                
                # For deployment: add to secrets
                with st.expander("Streamlit Cloud Setup"):
                    st.code("""
# .streamlit/secrets.toml
MAXMIND_LICENSE_KEY = "your_key"
                    """)
                
                # For testing
                key_input = st.text_input("License Key:", type="password")
                if key_input:
                    st.session_state.maxmind_license_key = key_input
                    license_key = key_input
            
            if license_key:
                if st.button("📥 Setup GeoLite2", use_container_width=True):
                    with st.spinner("Downloading..."):
                        db_bytes = download_geolite2_cached(license_key)
                        if db_bytes:
                            reader = init_geo_reader(db_bytes)
                            if reader:
                                st.session_state.geo_reader = reader
                                st.success("✅ Ready!")
                                st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    verify_tls = st.checkbox("Verify TLS", value=True, help="Default ON for security")
    
    st.markdown("---")
    
    # Load proxies
    st.subheader("📥 Load Proxies")
    if st.button("🔄 Fetch from Sources", use_container_width=True):
        with st.spinner("Fetching..."):
            all_proxies = []
            for src in PROXY_SOURCES:
                proxies = run_async(fetch_proxies(src))
                all_proxies.extend(proxies)
            
            # Deduplicate
            unique = list({p: None for p in all_proxies}.keys())
            st.session_state.proxies_raw = unique
            st.success(f"Loaded {len(unique)} unique proxies")
    
    # Validate
    if st.session_state.proxies_raw:
        st.subheader("✅ Validate")
        count = st.slider("Count:", 5, 50, 10)
        if st.button("🧪 Validate", use_container_width=True):
            with st.spinner(f"Validating {count}..."):
                sample = st.session_state.proxies_raw[:count]
                validated = run_async(validate_batch(sample, verify_tls=verify_tls))
                st.session_state.proxies_validated.extend(validated)
                # Deduplicate
                st.session_state.proxies_validated = list({p: None for p in st.session_state.proxies_validated}.keys())
                st.success(f"✅ {len(validated)} working")
    
    # Clear
    if st.session_state.proxies_validated:
        if st.button("🗑️ Clear Validated", use_container_width=True):
            st.session_state.proxies_validated = []
            st.rerun()
    
    st.markdown("---")
    
    # Chain builder
    st.subheader("🔗 Chain Builder")
    if st.session_state.proxies_validated:
        for i, p in enumerate(st.session_state.proxies_validated[:10]):
            label = f"{p.protocol}://{p.host}:{p.port}"
            if p.country_code:
                label += f" ({p.country_code})"
            if p.latency:
                label += f" {p.latency:.0f}ms"
            
            if st.button(label, key=f"p_{i}", use_container_width=True):
                if len(st.session_state.proxy_chain) < 5:
                    st.session_state.proxy_chain.append(p)
                    st.success(f"Added hop {len(st.session_state.proxy_chain)}")
    
    # Current chain
    if st.session_state.proxy_chain:
        st.write("**Current Chain:**")
        for i, p in enumerate(st.session_state.proxy_chain):
            st.write(f"{i+1}. {p.protocol}://{p.host}:{p.port}")
        
        # Test
        if len(st.session_state.proxy_chain) >= 2:
            if st.button("🧪 Test Chain", use_container_width=True):
                with st.spinner("Testing..."):
                    result = run_async(test_chain_dns(st.session_state.proxy_chain))
                    if result["success"]:
                        st.success(f"✅ Exit: {result['exit_ip']}")
                        if result.get('anonymity'):
                            st.info(f"{result['anonymity']}")
                    else:
                        st.error(result['error'])
        
        if st.button("🗑️ Clear Chain", use_container_width=True):
            st.session_state.proxy_chain = []
            st.rerun()

# Main area
st.subheader("📊 Validated Proxies")

if st.session_state.proxies_validated:
    data = []
    for p in st.session_state.proxies_validated:
        data.append({
            'Protocol': p.protocol.upper(),
            'Host': p.host,
            'Port': p.port,
            'Country': p.country or 'Unknown',
            'City': p.city or 'Unknown',
            'Latency': f"{p.latency:.0f}ms" if p.latency else 'N/A'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button("📥 Export CSV", csv, "proxies.csv", "text/csv")
else:
    st.info("No validated proxies yet")
    
    if not st.session_state.geo_reader:
        st.warning("⚠️ Setup geo database first (in sidebar)")

st.markdown("---")
st.caption("ProxyStream Cloud - Complete implementation with all functions included")
