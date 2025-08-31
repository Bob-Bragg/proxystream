"""
ProxyStream Complete - Auto-Setup Version
Automatically downloads and configures geo databases
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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import numpy as np
import aiohttp
from aiohttp_socks import ChainProxyConnector

# ---------------------------------------------------
# Auto-Setup Configuration
# ---------------------------------------------------
GEOLITE2_DIR = Path("geolite2_data")
GEOLITE2_DB_PATH = GEOLITE2_DIR / "GeoLite2-City.mmdb"
GEOLITE2_ASN_PATH = GEOLITE2_DIR / "GeoLite2-ASN.mmdb"
IP2LOCATION_DIR = Path("ip2location_data")
IP2LOCATION_DB = IP2LOCATION_DIR / "IP2LOCATION-LITE-DB5.BIN"

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Auto-Setup",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Auto-Download Functions
# ---------------------------------------------------
async def download_geolite2(license_key: str) -> bool:
    """Download GeoLite2 database from MaxMind"""
    try:
        GEOLITE2_DIR.mkdir(exist_ok=True)
        
        # Download City database
        city_url = f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City&license_key={license_key}&suffix=tar.gz"
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            response = await client.get(city_url)
            if response.status_code != 200:
                return False
            
            # Extract
            temp_file = GEOLITE2_DIR / "city.tar.gz"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            with tarfile.open(temp_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".mmdb"):
                        tar.extract(member)
                        extracted = Path(member.name)
                        shutil.move(str(extracted), str(GEOLITE2_DB_PATH))
                        # Clean up
                        if Path(member.name.split('/')[0]).exists():
                            shutil.rmtree(member.name.split('/')[0])
                        break
            
            temp_file.unlink()
            return True
    except:
        return False

async def download_ip2location() -> bool:
    """Download IP2Location LITE (no registration required)"""
    try:
        IP2LOCATION_DIR.mkdir(exist_ok=True)
        
        # Check if already exists and is recent
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
    """Initialize geo database - auto-download if needed"""
    # Try GeoLite2 first
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
    
    # Auto-download IP2Location (no registration required)
    with st.spinner("Setting up geo database (one-time download)..."):
        if asyncio.run(download_ip2location()):
            try:
                import IP2Location
                return IP2Location.IP2Location(str(IP2LOCATION_DB))
            except:
                pass
    
    return None

def geo_lookup(ip: str, db=None) -> Dict[str, Any]:
    """Universal geo lookup using available database"""
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
                "region": (data.get('subdivisions', [{}])[0].get('names', {}).get('en') 
                          if data.get('subdivisions') else None),
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
                "region": rec.region,
                "lat": rec.latitude,
                "lon": rec.longitude
            }
    except:
        pass
    
    return {"country": "Unknown", "country_code": "XX"}

# ---------------------------------------------------
# Constants
# ---------------------------------------------------
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
]

VALIDATION_ENDPOINTS = [
    "https://www.wikipedia.org/robots.txt",
    "https://www.cloudflare.com/robots.txt",
]

DNS_RESOLVERS = [
    ("208.67.222.222", 53),  # OpenDNS
    ("1.1.1.1", 53),         # Cloudflare
]

DB_PATH = "proxystream.db"

# ---------------------------------------------------
# DNS over TCP Module
# ---------------------------------------------------
def build_dns_query(name: str, qtype: int = 1) -> Tuple[bytes, int]:
    """Build DNS query packet"""
    tid = random.randint(0, 0xFFFF)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    parts = name.split(".")
    qname = b"".join(struct.pack("B", len(p)) + p.encode() for p in parts) + b"\x00"
    question = struct.pack("!HH", qtype, 1)
    payload = header + qname + question
    return struct.pack("!H", len(payload)) + payload, tid

def parse_dns_response(resp: bytes, tid: int) -> Optional[str]:
    """Parse DNS A record response"""
    if len(resp) < 14:
        return None
    resp = resp[2:]  # Skip TCP length
    
    rid = struct.unpack("!H", resp[:2])[0]
    if rid != tid:
        return None
    
    # Skip to answer (simplified parser)
    i = 12
    while i < len(resp) and resp[i] != 0:
        i += resp[i] + 1
    i += 5  # Skip null + type + class
    
    if i + 16 > len(resp):
        return None
    
    # Skip name, parse RR
    if resp[i] & 0xC0:
        i += 2
    i += 10  # Skip type, class, TTL, length
    
    # Get IP (4 bytes)
    if i + 4 <= len(resp):
        return ".".join(str(b) for b in resp[i:i+4])
    
    return None

async def get_exit_ip_dns() -> Optional[str]:
    """Get public IP via DNS"""
    for resolver, port in DNS_RESOLVERS:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(resolver, port), timeout=5
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

# ---------------------------------------------------
# Safe async runner
# ---------------------------------------------------
def run_async(coro):
    """Run async code in Streamlit"""
    try:
        loop = asyncio.get_running_loop()
        with asyncio.Runner() as runner:
            return runner.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

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
            latency REAL,
            last_tested TIMESTAMP,
            country TEXT,
            country_code TEXT,
            city TEXT,
            lat REAL,
            lon REAL,
            PRIMARY KEY (host, port)
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
# Models
# ---------------------------------------------------
@dataclass
class ProxyInfo:
    host: str
    port: int
    protocol: str = "http"
    latency: Optional[float] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    is_valid: bool = False

    def __hash__(self):
        return hash((self.host, self.port))

    def as_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

# ---------------------------------------------------
# Validation
# ---------------------------------------------------
async def validate_proxy(proxy: ProxyInfo, timeout: int = 15) -> Tuple[bool, float]:
    """Validate proxy using normal websites"""
    proxy_url = proxy.as_url()
    
    for endpoint in VALIDATION_ENDPOINTS:
        try:
            start = time.perf_counter_ns()
            async with httpx.AsyncClient(
                proxies=proxy_url,
                timeout=timeout,
                verify=False
            ) as client:
                r = await client.get(endpoint)
                if r.status_code in [200, 301, 302, 403, 404]:
                    latency = (time.perf_counter_ns() - start) / 1_000_000
                    return True, latency
        except:
            continue
    
    return False, 0

async def validate_batch(proxies: List[ProxyInfo]) -> List[ProxyInfo]:
    """Validate and enrich proxies"""
    validated = []
    
    for proxy in proxies[:10]:  # Limit for demo
        ok, latency = await validate_proxy(proxy)
        if ok:
            proxy.is_valid = True
            proxy.latency = latency
            
            # Get geo data
            geo = geo_lookup(proxy.host, st.session_state.geo_db)
            proxy.country = geo.get('country')
            proxy.country_code = geo.get('country_code')
            proxy.city = geo.get('city')
            proxy.lat = geo.get('lat')
            proxy.lon = geo.get('lon')
            
            validated.append(proxy)
    
    # Save to database
    if validated:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for p in validated:
            cur.execute("""
                INSERT OR REPLACE INTO proxies
                (host, port, protocol, latency, last_tested, country, country_code, city, lat, lon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (p.host, p.port, p.protocol, p.latency, datetime.now(),
                  p.country, p.country_code, p.city, p.lat, p.lon))
        conn.commit()
        conn.close()
    
    return validated

def parse_proxy(line: str) -> Optional[ProxyInfo]:
    """Parse proxy from line"""
    line = line.strip()
    if not line or '#' in line:
        return None
    
    # Simple IP:PORT parser
    match = re.match(r'^(\d+\.\d+\.\d+\.\d+):(\d+)$', line)
    if match:
        host, port = match.groups()
        return ProxyInfo(host=host, port=int(port))
    return None

async def fetch_proxies(url: str) -> List[ProxyInfo]:
    """Fetch proxy list"""
    proxies = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url)
            if r.status_code == 200:
                for line in r.text.splitlines()[:100]:
                    p = parse_proxy(line)
                    if p:
                        proxies.append(p)
    except:
        pass
    return proxies

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🔒 ProxyStream with Auto-Setup")

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📍 Get My IP", use_container_width=True):
        with st.spinner("Getting IP via DNS..."):
            ip = run_async(get_exit_ip_dns())
            if ip:
                geo = geo_lookup(ip, st.session_state.geo_db)
                st.success(f"Your IP: {ip} ({geo.get('country_code', 'XX')})")

with col2:
    db_status = "✅ Ready" if st.session_state.geo_db else "⚠️ No DB"
    st.metric("Geo Database", db_status)

with col3:
    st.metric("Validated", len(st.session_state.proxies_validated))

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Database Setup
    st.subheader("📊 Database Setup")
    
    if st.session_state.geo_db:
        if GEOLITE2_DB_PATH.exists():
            st.success("✅ Using GeoLite2")
        elif IP2LOCATION_DB.exists():
            st.success("✅ Using IP2Location")
    else:
        st.warning("⚠️ No geo database")
        
        # Option 1: Auto-download IP2Location (no key needed)
        if st.button("📥 Auto-Setup (Free)", use_container_width=True):
            with st.spinner("Downloading IP2Location..."):
                if run_async(download_ip2location()):
                    st.success("✅ Database ready!")
                    st.session_state.geo_db = init_geo_database()
                    st.rerun()
        
        # Option 2: Use MaxMind with key
        with st.expander("Use MaxMind GeoLite2"):
            st.markdown("[Get free license key](https://www.maxmind.com/en/geolite2/signup)")
            key = st.text_input("License Key:", type="password")
            if key and st.button("Download GeoLite2"):
                with st.spinner("Downloading..."):
                    if run_async(download_geolite2(key)):
                        st.success("✅ GeoLite2 ready!")
                        st.session_state.geo_db = init_geo_database()
                        st.rerun()
    
    st.markdown("---")
    
    # Load proxies
    st.subheader("📥 Load Proxies")
    if st.button("🔄 Fetch Proxies", use_container_width=True):
        with st.spinner("Fetching..."):
            all_proxies = []
            for src in PROXY_SOURCES:
                proxies = run_async(fetch_proxies(src))
                all_proxies.extend(proxies)
            st.session_state.proxies_raw = all_proxies
            st.success(f"Loaded {len(all_proxies)} proxies")
    
    # Validate
    if st.session_state.proxies_raw:
        st.subheader("✅ Validate")
        if st.button("🧪 Validate", use_container_width=True):
            with st.spinner("Validating..."):
                validated = run_async(validate_batch(st.session_state.proxies_raw))
                st.session_state.proxies_validated = validated
                st.success(f"✅ {len(validated)} working")
    
    # Chain builder
    if st.session_state.proxies_validated:
        st.subheader("🔗 Chain Builder")
        for p in st.session_state.proxies_validated[:5]:
            if st.button(f"{p.host}:{p.port} ({p.country_code})", key=p.host):
                if len(st.session_state.proxy_chain) < 3:
                    st.session_state.proxy_chain.append(p)
                    st.success(f"Added hop {len(st.session_state.proxy_chain)}")
    
    # Current chain
    if st.session_state.proxy_chain:
        st.write("**Current Chain:**")
        for i, p in enumerate(st.session_state.proxy_chain):
            st.write(f"{i+1}. {p.host}:{p.port}")
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.proxy_chain = []
            st.rerun()

# Main area
st.subheader("📊 Validated Proxies")

if st.session_state.proxies_validated:
    data = [{
        'Host': p.host,
        'Port': p.port,
        'Country': p.country or 'Unknown',
        'City': p.city or 'Unknown',
        'Latency': f"{p.latency:.0f}ms" if p.latency else 'N/A'
    } for p in st.session_state.proxies_validated]
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No validated proxies yet")
    
    if not st.session_state.geo_db:
        st.warning("⚠️ Please setup geo database first (click Auto-Setup in sidebar)")

st.markdown("---")
st.caption("ProxyStream - Auto-downloads geo databases, no API dependencies")
