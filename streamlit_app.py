"""
ProxyStream Production-Ready Version
All critical issues fixed, properly architected
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
import shutil
import zipfile
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
    page_title="ProxyStream Production",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
GEOLITE2_DIR = Path("geolite2_data")
GEOLITE2_DB_PATH = GEOLITE2_DIR / "GeoLite2-City.mmdb"
IP2LOCATION_DIR = Path("ip2location_data")
IP2LOCATION_DB = IP2LOCATION_DIR / "IP2LOCATION-LITE-DB5.BIN"
DB_PATH = "proxystream.db"

# Sources
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
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
# Manual CONNECT tunnel for raw TCP (FIXED)
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
# DNS over TCP Module (FIXED)
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
    """Get exit IP using DNS over TCP through HTTP/HTTPS chain (FIXED)"""
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
        except Exception as e:
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
# Geo Database Auto-Setup
# ---------------------------------------------------
async def download_ip2location() -> bool:
    """Download IP2Location LITE (no registration required)"""
    try:
        IP2LOCATION_DIR.mkdir(exist_ok=True)
        
        # Check if recent
        if IP2LOCATION_DB.exists():
            age = datetime.now() - datetime.fromtimestamp(IP2LOCATION_DB.stat().st_mtime)
            if age < timedelta(days=30):
                return True
        
        url = "https://download.ip2location.com/lite/IP2LOCATION-LITE-DB5.BIN.ZIP"
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return False
            
            zip_file = IP2LOCATION_DIR / "db.zip"
            with open(zip_file, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(IP2LOCATION_DIR)
            
            zip_file.unlink()
            return True
    except:
        return False

@st.cache_resource
def init_geo_database():
    """Initialize geo database with auto-download"""
    # Try MaxMind first
    if GEOLITE2_DB_PATH.exists():
        try:
            import maxminddb
            return maxminddb.open_database(str(GEOLITE2_DB_PATH))
        except:
            pass
    
    # Try IP2Location
    if IP2LOCATION_DB.exists():
        try:
            import IP2Location
            return IP2Location.IP2Location(str(IP2LOCATION_DB))
        except:
            pass
    
    # Auto-download IP2Location
    with st.spinner("Setting up geo database (one-time, ~20MB)..."):
        if run_async(download_ip2location()):
            try:
                import IP2Location
                return IP2Location.IP2Location(str(IP2LOCATION_DB))
            except:
                pass
    
    return None

def geo_lookup(ip: str, db=None) -> Dict[str, Any]:
    """Universal geo lookup"""
    if not db:
        db = init_geo_database()
    
    if not db:
        return {"country": "Unknown", "country_code": "XX"}
    
    try:
        # MaxMind format
        if hasattr(db, 'get'):
            data = db.get(ip) or {}
            return {
                "country": data.get('country', {}).get('names', {}).get('en'),
                "country_code": data.get('country', {}).get('iso_code'),
                "city": data.get('city', {}).get('names', {}).get('en'),
                "lat": data.get('location', {}).get('latitude'),
                "lon": data.get('location', {}).get('longitude')
            }
        # IP2Location format
        elif hasattr(db, 'get_all'):
            rec = db.get_all(ip)
            return {
                "country": rec.country_long,
                "country_code": rec.country_short,
                "city": rec.city,
                "lat": rec.latitude,
                "lon": rec.longitude
            }
    except:
        pass
    
    return {"country": "Unknown", "country_code": "XX"}

# ---------------------------------------------------
# Database
# ---------------------------------------------------
@st.cache_resource
def init_database():
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

init_database()

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "proxies_raw" not in st.session_state: st.session_state.proxies_raw = []
if "proxies_validated" not in st.session_state: st.session_state.proxies_validated = []
if "proxy_chain" not in st.session_state: st.session_state.proxy_chain = []
if "geo_db" not in st.session_state: st.session_state.geo_db = init_geo_database()

# ---------------------------------------------------
# Proxy Parsing (FIXED - handles IPv6 and auth)
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
                for line in r.text.splitlines()[:200]:
                    p = parse_proxy_line(line)
                    if p:
                        proxies.append(p)
    except:
        pass
    return proxies

# ---------------------------------------------------
# Proxy Validation (FIXED)
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
                geo = geo_lookup(proxy.host, st.session_state.geo_db)
                proxy.country = geo.get('country')
                proxy.country_code = geo.get('country_code')
                proxy.city = geo.get('city')
                proxy.lat = geo.get('lat')
                proxy.lon = geo.get('lon')
                
                validated.append(proxy)
        except:
            pass
    
    # Save to database
    if validated:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
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
        conn.commit()
        conn.close()
    
    return validated

# ---------------------------------------------------
# Chain Testing (FIXED)
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
        exit_geo = geo_lookup(exit_ip, st.session_state.geo_db) if exit_ip else {}
        
        # Test anonymity with ChainProxyConnector for mixed chains
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
                        "exit_geo": geo_lookup(exit_ip, st.session_state.geo_db) if exit_ip else {},
                        "hop_count": len(chain),
                        "method": "HTTP (SOCKS chain)"
                    }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}
    
    return {"success": False, "error": "Chain test failed"}

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🔒 ProxyStream Production")

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📍 Get My IP (DNS)", use_container_width=True):
        with st.spinner("Getting IP via DNS..."):
            ip = run_async(get_exit_ip_dns_direct())
            if ip:
                geo = geo_lookup(ip, st.session_state.geo_db)
                st.success(f"IP: {ip} ({geo.get('country_code', 'XX')})")

with col2:
    db_status = "✅ Ready" if st.session_state.geo_db else "⚠️ No DB"
    st.metric("Geo Database", db_status)

with col3:
    st.metric("Validated", len(st.session_state.proxies_validated))

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Settings (TLS verify defaults to TRUE for security)
    st.subheader("⚙️ Settings")
    verify_tls = st.checkbox("Verify TLS certificates", value=True, help="Default ON for security")
    
    # Database status
    st.subheader("📊 Database")
    if st.session_state.geo_db:
        if GEOLITE2_DB_PATH.exists():
            st.success("✅ Using GeoLite2")
        elif IP2LOCATION_DB.exists():
            st.success("✅ Using IP2Location")
    else:
        if st.button("📥 Setup Database", use_container_width=True):
            st.session_state.geo_db = init_geo_database()
            st.rerun()
    
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
        count = st.slider("Count to validate:", 5, 50, 10)
        if st.button("🧪 Validate Proxies", use_container_width=True):
            with st.spinner(f"Validating {count} proxies..."):
                sample = st.session_state.proxies_raw[:count]
                validated = run_async(validate_batch(sample, verify_tls=verify_tls))
                st.session_state.proxies_validated.extend(validated)
                # Deduplicate validated list
                st.session_state.proxies_validated = list({p: None for p in st.session_state.proxies_validated}.keys())
                st.success(f"✅ {len(validated)} working")
    
    # Clear validated list
    if st.session_state.proxies_validated:
        if st.button("🗑️ Clear Validated List", use_container_width=True):
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
            
            if st.button(label, key=f"proxy_{i}", use_container_width=True):
                if len(st.session_state.proxy_chain) < 5:
                    st.session_state.proxy_chain.append(p)
                    st.success(f"Added hop {len(st.session_state.proxy_chain)}")
                else:
                    st.error("Max 5 hops")
    
    # Current chain
    if st.session_state.proxy_chain:
        st.write("**Current Chain:**")
        for i, p in enumerate(st.session_state.proxy_chain):
            st.write(f"{i+1}. {p.protocol}://{p.host}:{p.port}")
        
        # Test chain
        if len(st.session_state.proxy_chain) >= 2:
            if st.button("🧪 Test Chain", use_container_width=True):
                with st.spinner("Testing chain..."):
                    result = run_async(test_chain_dns(st.session_state.proxy_chain))
                    if result["success"]:
                        st.success(f"✅ Exit: {result['exit_ip']}")
                        if result.get('anonymity'):
                            st.info(f"Anonymity: {result['anonymity']}")
                        if result.get('method'):
                            st.caption(f"Method: {result['method']}")
                    else:
                        st.error(result['error'])
        
        if st.button("🗑️ Clear Chain", use_container_width=True):
            st.session_state.proxy_chain = []
            st.rerun()

# Main area - Proxy List
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
    st.download_button(
        "📥 Export CSV",
        csv,
        "proxies.csv",
        "text/csv",
        use_container_width=True
    )
else:
    st.info("No validated proxies yet. Load and validate from sidebar.")
    
    if not st.session_state.geo_db:
        st.warning("⚠️ Setup geo database first (click Setup Database in sidebar)")

st.markdown("---")
st.caption("ProxyStream Production - Robust DNS over TCP, proper error handling, secure defaults")
