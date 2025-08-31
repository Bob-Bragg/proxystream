"""
ProxyStream Complete - Fixed Version with all critical issues resolved
"""

import asyncio
import ssl
import time
import json
import sqlite3
import base64
import os
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

# ---------------------------------------------------
# Page & Constants
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Advanced",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

# Use HTTPS for security
GEO_APIS = [
    {"url": "https://ipapi.co/{}/json/", "has_asn": True},
    {"url": "https://ipwhois.app/json/{}", "has_asn": True},
]

EXIT_IP_PROVIDERS = [
    "https://ipapi.co/json",
    "https://ipinfo.io/json",
    "https://api.ipify.org?format=json"
]

DB_PATH = "proxystream_final.db"
CLIENT_GEO_KEY = "client_geo"
CERT_PINS: Dict[str, str] = {}

# Redis config (optional)
REDIS_URL = os.getenv("PROXYSTREAM_REDIS_URL", "")
redis_client = redis.Redis.from_url(REDIS_URL) if (redis and REDIS_URL) else None

# pyasn DB (optional offline ASN)
PYASN_DB = os.getenv("PROXYSTREAM_PYASN_DB", "ipasn.dat")
pyasn_obj = pyasn.pyasn(PYASN_DB) if (pyasn and pathlib.Path(PYASN_DB).exists()) else None

# ---------------------------------------------------
# Prometheus metrics (disabled for Streamlit Cloud)
# ---------------------------------------------------
class DummyMetric:
    """Dummy metric that does nothing"""
    def inc(self): pass
    def observe(self, value): pass

METRIC_FETCH_OK = DummyMetric()
METRIC_FETCH_ERR = DummyMetric()
METRIC_CHAIN_TEST = DummyMetric()
METRIC_HOP_MS = DummyMetric()
METRIC_RATE_LIMIT_BLOCK = DummyMetric()

# ---------------------------------------------------
# Safe async runner for Streamlit
# ---------------------------------------------------
def run_async(coro):
    """Safe async runner that works with Streamlit's event loop"""
    try:
        loop = asyncio.get_running_loop()
        # If loop is running, we need to use a different approach
        import concurrent.futures
        import threading
        
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
        # No loop running, we can use asyncio.run directly
        return asyncio.run(coro)

# ---------------------------------------------------
# Simple TTL cache (in-memory)
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

# ---------------------------------------------------
# Database Init
# ---------------------------------------------------
@st.cache_resource
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA mmap_size=268435456;
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
            chain_id INTEGER,
            tested_at TIMESTAMP,
            exit_ip TEXT,
            total_latency REAL,
            hop_timings_json TEXT,
            stats_json TEXT,
            anonymity_json TEXT
        )
    """)
    cur.executescript("""
        CREATE INDEX IF NOT EXISTS idx_proxies_last_tested ON proxies(last_tested);
        CREATE INDEX IF NOT EXISTS idx_proxies_country_asn ON proxies(country_code, asn);
        CREATE INDEX IF NOT EXISTS idx_proxies_latency ON proxies(latency);
    """)
    conn.commit()
    conn.close()

def ensure_month_partition(table_prefix: str = "chain_tests"):
    month = datetime.utcnow().strftime("%Y%m")
    table = f"{table_prefix}_{month}"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id INTEGER,
            tested_at TIMESTAMP,
            exit_ip TEXT,
            total_latency REAL,
            hop_timings_json TEXT,
            stats_json TEXT,
            anonymity_json TEXT
        )
    """)
    conn.commit()
    conn.close()
    return table

def downsample_chain_tests(older_than_days: int = 30):
    """Fixed downsampling - keeps one record per day"""
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(f"""
        WITH keep AS (
            SELECT MIN(id) AS id
            FROM chain_tests
            WHERE date(tested_at) < date('{cutoff_str}')
            GROUP BY date(tested_at)
        )
        DELETE FROM chain_tests
        WHERE date(tested_at) < date('{cutoff_str}')
          AND id NOT IN (SELECT id FROM keep);
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
# Models & Utilities
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
        return hash((self.host, self.port, self.protocol, self.username, self.password))

    def as_url(self) -> str:
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

# Circuit breaker
class Circuit:
    def __init__(self, threshold: int = 5, cooldown_seconds: int = 120):
        self.failures = 0
        self.opened_at: Optional[float] = None
        self.threshold = threshold
        self.cooldown = cooldown_seconds

    def record_success(self):
        self.failures = 0
        self.opened_at = None

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.time()

    def can_attempt(self) -> bool:
        return self.opened_at is None or (time.time() - self.opened_at) >= self.cooldown

async def retry_with_backoff(fn: Callable[[], Any], retries: int = 3, base: float = 1.0, jitter: float = 0.3):
    last = None
    for i in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            last = e
            if i == retries:
                break
            await asyncio.sleep(base * (2 ** i) + random.uniform(0, jitter))
    raise last

# Local rate limiter
class RateLimiter:
    def __init__(self, rate: int, per_seconds: int, key_prefix: str = "rl:"):
        self.rate = rate
        self.per = per_seconds
        self.key_prefix = key_prefix
        self.local_buckets: Dict[str, Tuple[int, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        tokens, ts = self.local_buckets.get(key, (0, now))
        if now - ts > self.per:
            tokens, ts = 0, now
        tokens += 1
        self.local_buckets[key] = (tokens, ts)
        if tokens <= self.rate:
            return True
        METRIC_RATE_LIMIT_BLOCK.inc()
        return False

rate_limiter = RateLimiter(rate=30, per_seconds=60, key_prefix="proxystream:")

# ---------------------------------------------------
# ASN Enrichment
# ---------------------------------------------------
async def team_cymru_bulk_lookup(ips: List[str]) -> Dict[str, Dict[str, str]]:
    """Query Team Cymru WHOIS (TCP 43) in bulk."""
    if not ips:
        return {}
    try:
        reader, writer = await asyncio.open_connection("whois.cymru.com", 43)
        writer.write(b"begin\nverbose\n")
        for ip in ips:
            writer.write((ip + "\n").encode())
        writer.write(b"end\n")
        await writer.drain()
        data = await reader.read(-1)
        writer.close()
        await writer.wait_closed()
    except Exception:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    lines = data.decode(errors="ignore").splitlines()
    for line in lines:
        if line.startswith("AS") or line.lower().startswith("bulk mode"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 7:
            asn, ip, prefix, cc, reg, alloc, as_name = parts[:7]
            out[ip] = {
                "asn": asn or None,
                "asn_org": as_name or None,
                "cc": cc or None,
                "prefix": prefix or None,
                "registry": reg or None,
                "allocated": alloc or None
            }
    return out

def pyasn_lookup(ip: str) -> Dict[str, Optional[str]]:
    if not pyasn_obj:
        return {}
    try:
        asn, prefix = pyasn_obj.lookup(ip)
        return {"asn": str(asn) if asn else None, "prefix": prefix}
    except Exception:
        return {}

# ---------------------------------------------------
# Geo / Exit-IP / Anonymity (FIXED)
# ---------------------------------------------------
async def exit_ip_consensus_via_session(session: Any) -> Optional[str]:
    """Get exit IP consensus from multiple providers"""
    cached = exit_ip_cache.get("exit")
    if cached: return cached
    ips = []
    for url in EXIT_IP_PROVIDERS:
        try:
            # Handle both httpx and aiohttp sessions
            if hasattr(session, 'get'):
                r = await session.get(url, timeout=10)
                if hasattr(r, 'status_code'):  # httpx
                    if r.status_code == 200:
                        ip = r.json().get("ip")
                        if ip: ips.append(ip)
                else:  # aiohttp
                    if r.status == 200:
                        data = await r.json()
                        ip = data.get("ip")
                        if ip: ips.append(ip)
        except Exception:
            continue
    if not ips:
        return None
    best = max(set(ips), key=ips.count)
    exit_ip_cache.set("exit", best)
    return best

async def get_proxy_geo_with_asn(ip: str) -> Dict[str, Any]:
    """Fixed version - no sync calls inside async"""
    out: Dict[str, Any] = {}
    
    # pyasn first (offline)
    if _is_ip(ip):
        out.update(pyasn_lookup(ip))

    # public APIs (HTTPS)
    async with httpx.AsyncClient(timeout=10) as client:
        for api in GEO_APIS:
            try:
                r = await client.get(api["url"].format(ip))
                if r.status_code == 200:
                    d = r.json()
                    out.setdefault('country', d.get('country') or d.get('country_name'))
                    out.setdefault('country_code', d.get('country_code') or d.get('countryCode'))
                    out.setdefault('city', d.get('city'))
                    out.setdefault('region', d.get('region') or d.get('regionName'))
                    out.setdefault('lat', d.get('latitude') or d.get('lat'))
                    out.setdefault('lon', d.get('longitude') or d.get('lon'))
                    out.setdefault('asn', out.get('asn') or d.get('asn') or d.get('as'))
                    out.setdefault('org', d.get('org'))
                    out.setdefault('isp', d.get('isp') or d.get('org'))
                    break
            except Exception:
                continue

    # Team Cymru if ASN missing (FIXED - no sync inside async)
    if out.get("asn") is None and _is_ip(ip):
        info = (await team_cymru_bulk_lookup([ip])).get(ip) or {}
        if info:
            out['asn'] = info.get('asn') or out.get('asn')
            out['org'] = info.get('asn_org') or out.get('org')
            out['country_code'] = info.get('cc') or out.get('country_code')
    
    return out

def _is_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except Exception:
        return False

# ---------------------------------------------------
# Validation for single proxy (with configurable TLS)
# ---------------------------------------------------
async def validate_proxy(proxy: ProxyInfo, timeout: int = 15, verify: bool = True) -> Tuple[bool, float, Optional[str]]:
    """Validate a single HTTP proxy"""
    if not rate_limiter.allow(f"validate:{proxy.host}:{proxy.port}"):
        return False, 0.0, None

    proxy_url = proxy.as_url()
    
    try:
        start = time.perf_counter_ns()
        async with httpx.AsyncClient(
            proxies=proxy_url,
            timeout=timeout,
            follow_redirects=True,
            verify=verify
        ) as client:
            r = await client.get("http://httpbin.org/ip")
            if r.status_code == 200:
                latency_ms = (time.perf_counter_ns() - start) / 1_000_000
                try:
                    data = r.json()
                    exit_ip = data.get("origin", "").split(",")[0].strip()
                    return True, latency_ms, exit_ip
                except:
                    return True, latency_ms, proxy.host
    except Exception as e:
        pass
    
    return False, 0.0, None

# ---------------------------------------------------
# Proxy list parsing (FIXED - allows hostnames)
# ---------------------------------------------------
_IPV4_PORT = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}):(\d+)$")
_IPV6_PORT = re.compile(r"^\[?([0-9a-fA-F:]+)\]?:([0-9]{1,5})$")
_HOSTNAME_PORT = re.compile(r"^([A-Za-z0-9.-]+):(\d+)$")

def parse_possible_proxy_line(line: str) -> Optional[ProxyInfo]:
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

    # Try IPv4
    m4 = _IPV4_PORT.match(rest)
    if m4:
        host, port_s = m4.group(1), m4.group(2)
    else:
        # Try IPv6
        m6 = _IPV6_PORT.match(rest)
        if m6:
            host, port_s = m6.group(1), m6.group(2)
        else:
            # Try hostname (FIXED - now allows hostnames)
            mh = _HOSTNAME_PORT.match(rest)
            if mh:
                host, port_s = mh.group(1), mh.group(2)
            else:
                return None

    port = int(port_s)
    if not (1 <= port <= 65535):
        return None

    return ProxyInfo(host=host, port=port, protocol=proto.lower(), username=user, password=pw)

async def fetch_and_parse_proxies(source_url: str, limit: int = 1000) -> List[ProxyInfo]:
    proxies: List[ProxyInfo] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(source_url)
            if r.status_code == 200:
                for line in r.text.strip().splitlines()[:limit]:
                    pi = parse_possible_proxy_line(line)
                    if pi:
                        proxies.append(pi)
    except Exception as e:
        st.warning(f"Failed to fetch {source_url}: {str(e)[:100]}")
    return proxies

# ---------------------------------------------------
# Batch validate + Geo enrich (SINGLE DEFINITION)
# ---------------------------------------------------
async def load_and_validate_batch(proxies: List[ProxyInfo], max_concurrent: int = 10, verify_tls: bool = False) -> List[ProxyInfo]:
    """Validate proxies in controlled batches"""
    validated: List[ProxyInfo] = []
    sem = asyncio.Semaphore(max_concurrent)
    circ = Circuit(threshold=10, cooldown_seconds=60)

    async def one(proxy: ProxyInfo):
        async with sem:
            if not circ.can_attempt():
                await asyncio.sleep(1)
                if not circ.can_attempt():
                    return None
            try:
                ok, lat_ms, exit_ip = await validate_proxy(proxy, timeout=20, verify=verify_tls)
                if ok:
                    proxy.is_valid = True
                    proxy.latency = lat_ms
                    proxy.last_tested = datetime.now()

                    # Geo lookup (optional)
                    try:
                        geo = await get_proxy_geo_with_asn(proxy.host)
                        proxy.country = geo.get('country')
                        proxy.country_code = geo.get('country_code')
                        proxy.city = geo.get('city')
                        proxy.region = geo.get('region')
                        proxy.lat = geo.get('lat')
                        proxy.lon = geo.get('lon')
                        proxy.asn = geo.get('asn')
                        proxy.org = geo.get('org')
                        proxy.isp = geo.get('isp')
                    except:
                        pass
                    
                    circ.record_success()
                    return proxy
                else:
                    circ.record_failure()
                    return None
            except Exception:
                circ.record_failure()
                return None

    # Process in batches
    batch_size = 10
    for i in range(0, len(proxies), batch_size):
        batch = proxies[i:i+batch_size]
        results = await asyncio.gather(*(one(p) for p in batch), return_exceptions=True)
        for r in results:
            if r and not isinstance(r, Exception):
                validated.append(r)
        
        if i + batch_size < len(proxies):
            await asyncio.sleep(0.5)

    # Save to database
    if validated:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("BEGIN")
        for p in validated:
            cur.execute("""
                INSERT OR REPLACE INTO proxies
                (host, port, protocol, username, password, latency, last_tested,
                 country, country_code, city, region, lat, lon, asn, org, isp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.host, p.port, p.protocol, p.username, p.password, p.latency, p.last_tested,
                p.country, p.country_code, p.city, p.region, p.lat, p.lon, p.asn, p.org, p.isp
            ))
        conn.commit()
        conn.close()
    
    return validated

def load_proxies_from_db(filters: Dict[str, Any] = None) -> List[ProxyInfo]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q = """
        SELECT host,port,protocol,username,password,latency,last_tested,
               country,country_code,city,region,lat,lon,asn,org,isp
        FROM proxies
        WHERE last_tested > datetime('now', '-7 days')
    """
    params: List[Any] = []
    if filters:
        if filters.get('country'):
            q += " AND (country LIKE ? OR country_code = ?)"
            params += [f"%{filters['country']}%", filters['country']]
        if filters.get('asn'):
            q += " AND asn LIKE ?"
            params.append(f"%{filters['asn']}%")
        if filters.get('ip'):
            q += " AND host LIKE ?"
            params.append(f"%{filters['ip']}%")
        if filters.get('protocol'):
            q += " AND protocol = ?"
            params.append(filters['protocol'])
    q += " ORDER BY latency ASC LIMIT 500"
    cur.execute(q, params)
    rows = cur.fetchall()
    conn.close()

    out: List[ProxyInfo] = []
    for r in rows:
        out.append(ProxyInfo(
            host=r[0], port=r[1], protocol=r[2], username=r[3], password=r[4],
            latency=r[5], last_tested=r[6],
            country=r[7], country_code=r[8], city=r[9], region=r[10],
            lat=r[11], lon=r[12], asn=r[13], org=r[14], isp=r[15], is_valid=True
        ))
    return out

# ---------------------------------------------------
# Stats helpers
# ---------------------------------------------------
def _ns_to_ms(ns: int) -> float: return ns / 1_000_000

def compute_stats(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms: return {}
    samples_sorted = sorted(samples_ms)
    p50 = statistics.median(samples_sorted)
    p95 = statistics.quantiles(samples_sorted, n=20)[18] if len(samples_sorted) >= 20 else samples_sorted[-1]
    p99 = statistics.quantiles(samples_sorted, n=100)[98] if len(samples_sorted) >= 100 else samples_sorted[-1]
    return {
        "mean": statistics.mean(samples_sorted),
        "p50": p50, "p95": p95, "p99": p99,
        "stdev": statistics.pstdev(samples_sorted) if len(samples_sorted) > 1 else 0.0,
        "count": float(len(samples_sorted))
    }

def compute_rfc3550_jitter(samples_ms: List[float]) -> float:
    if len(samples_ms) < 2: return 0.0
    J = 0.0; prev = samples_ms[0]
    for s in samples_ms[1:]:
        D = abs(s - prev)
        J = J + (D - J) / 16.0
        prev = s
    return J

# ---------------------------------------------------
# Chain test using fetch_via_chain (safer than manual TLS)
# ---------------------------------------------------
async def test_proxy_chain_full(chain: List[ProxyInfo], samples: int = 5) -> Dict[str, Any]:
    """Test chain using the safer fetch_via_chain method"""
    if not chain:
        return {"success": False, "error": "Empty chain"}
    
    # Basic timing test
    total_samples: List[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = await fetch_via_chain(chain, "https://httpbin.org/ip", timeout=30)
        if result.get("success"):
            total_samples.append(_ns_to_ms(time.perf_counter_ns() - start))
    
    if not total_samples:
        return {"success": False, "error": "Chain validation failed"}
    
    # Get exit IP and anonymity
    result = await fetch_via_chain(chain, "https://httpbin.org/headers", timeout=30)
    if not result.get("success"):
        return {"success": False, "error": "Failed to fetch headers"}
    
    # Parse anonymity from response
    anonymity = result.get("resp_proxy_headers", {})
    
    total_stats = compute_stats(total_samples)
    
    # Log to database
    table = ensure_month_partition("chain_tests")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"""
        INSERT INTO {table} (tested_at, exit_ip, total_latency, stats_json, anonymity_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.utcnow(),
        result.get("exit_ip_consensus"),
        total_stats.get("mean", 0.0),
        json.dumps(total_stats),
        json.dumps(anonymity)
    ))
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "exit_ip": result.get("exit_ip_consensus"),
        "total_stats": {k: round(v, 2) for k, v in total_stats.items()},
        "anonymity": anonymity,
        "hop_count": len(chain),
    }

# ---------------------------------------------------
# Real browsing via pooled chain
# ---------------------------------------------------
async def fetch_via_chain(chain: List[ProxyInfo], url: str, timeout: int = 25) -> Dict[str, Any]:
    if not url.lower().startswith(("http://", "https://")):
        return {"success": False, "error": "Only HTTP/HTTPS URLs allowed"}
    if not rate_limiter.allow("fetch:global"):
        return {"success": False, "error": "Rate limited"}

    urls = [p.as_url() for p in chain]
    try:
        connector = ChainProxyConnector.from_urls(urls)
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
            # Get exit IP
            ip_consensus = None
            try:
                ip_consensus = await exit_ip_consensus_via_session(session)
            except:
                pass

            async with session.get(url, headers={"User-Agent": "ProxyStream/1.0"}) as resp:
                body = await resp.text(errors="ignore")
                
                # Response headers for anonymity check
                via = resp.headers.get("Via")
                xff = resp.headers.get("X-Forwarded-For")
                fwd = resp.headers.get("Forwarded")
                pconn = resp.headers.get("Proxy-Connection")

                METRIC_FETCH_OK.inc()
                return {
                    "success": True,
                    "status_line": f"HTTP/{resp.version.major}.{resp.version.minor} {resp.status} {resp.reason}",
                    "status_code": resp.status,
                    "body_preview": body[:4096],
                    "bytes": len(body.encode("utf-8", errors="ignore")),
                    "exit_ip_consensus": ip_consensus,
                    "resp_proxy_headers": {"Via": via, "X-Forwarded-For": xff, "Forwarded": fwd, "Proxy-Connection": pconn}
                }
    except Exception as e:
        METRIC_FETCH_ERR.inc()
        return {"success": False, "error": str(e)[:200]}

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.markdown("""
<style>
    .main { padding-top: 0; }
    .block-container { padding: 1rem; }
    h1 { color: #2563eb; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background: #3b82f6; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔒 ProxyStream Advanced - Multi-Hop Chain System")

# Top row controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📍 Detect My Location", use_container_width=True):
        with st.spinner("Getting location..."):
            location = run_async(httpx.AsyncClient().get("https://ipapi.co/json/"))
            if location and hasattr(location, 'json'):
                st.session_state.client_location = location.json()
            else:
                st.session_state.client_location = {"city": "Unknown", "country_code": "", "ip": "0.0.0.0"}
            st.success("Location detected!")
            st.rerun()

if st.session_state.client_location:
    loc = st.session_state.client_location
    with col2: st.metric("Your Location", f"{loc.get('city', 'Unknown')}, {loc.get('country_code', '')}")
    with col3: st.metric("Your IP", loc.get('ip', 'Unknown'))

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Settings
    st.subheader("⚙️ Settings")
    verify_tls = st.checkbox("Verify TLS certificates", value=False, help="Enable for production, disable for testing")
    
    # Load Proxies
    st.subheader("📥 Load Proxies")
    if st.button("🔄 Fetch from GitHub Sources", use_container_width=True):
        with st.spinner("Fetching from sources..."):
            all_proxies: List[ProxyInfo] = []
            for src in PROXY_SOURCES:
                proxies = run_async(fetch_and_parse_proxies(src))
                all_proxies.extend(proxies)
            unique = list({p: None for p in all_proxies}.keys())
            st.session_state.proxies_raw = unique
            st.success(f"Loaded {len(unique)} unique proxies")

    # Validation
    if st.session_state.proxies_raw:
        st.subheader("✅ Validate")
        validate_count = st.slider("Number to validate:", 5, 50, 10)
        if st.button("🧪 Validate Proxies", use_container_width=True):
            with st.spinner(f"Validating {validate_count} proxies..."):
                sample = st.session_state.proxies_raw[:validate_count]
                validated = run_async(load_and_validate_batch(sample, verify_tls=verify_tls))
                st.session_state.proxies_validated.extend(validated)
                st.success(f"✅ {len(validated)} working proxies validated")

    # Search/Filter
    st.subheader("🔍 Search & Filter")
    filters = {
        'country': st.text_input("Country"),
        'asn': st.text_input("ASN"),
        'ip': st.text_input("IP/Host"),
        'protocol': st.selectbox("Protocol", ["", "http", "https", "socks4", "socks5"])
    }
    if st.button("🔍 Search Database", use_container_width=True):
        filtered = {k: v for k, v in filters.items() if v}
        results = load_proxies_from_db(filtered)
        st.session_state.proxies_validated = results
        st.success(f"Found {len(results)} proxies")

    # Chain Builder
    st.subheader("🔗 Chain Builder (2-5 hops)")
    if st.session_state.proxies_validated:
        countries: Dict[str, List[ProxyInfo]] = {}
        for proxy in st.session_state.proxies_validated:
            key = proxy.country or "Unknown"
            countries.setdefault(key, []).append(proxy)
        
        if countries:
            selected_country = st.selectbox("Country", list(countries.keys()))
            country_proxies = countries[selected_country][:30]
            proxy_display = [
                f"{p.protocol}://{p.host}:{p.port} | {p.latency:.0f}ms" if p.latency else f"{p.protocol}://{p.host}:{p.port}"
                for p in country_proxies
            ]
            selected_proxy_str = st.selectbox("Proxy", proxy_display)
            
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
        st.write(f"**Chain ({len(st.session_state.proxy_chain)} hops):**")
        for i, p in enumerate(st.session_state.proxy_chain):
            st.write(f"• {p.protocol}://{p.host}:{p.port} ({p.country_code or '??'})")

        if len(st.session_state.proxy_chain) >= 2:
            if st.button("🧪 Test Chain", use_container_width=True):
                with st.spinner("Testing chain..."):
                    result = run_async(test_proxy_chain_full(st.session_state.proxy_chain))
                    st.session_state.chain_test_result = result
                    if result["success"]:
                        st.success("✅ Chain valid!")
                        st.metric("Exit IP", result.get('exit_ip') or "N/A")
                    else:
                        st.error(f"❌ {result['error']}")

        if st.button("🗑️ Clear Chain", use_container_width=True):
            st.session_state.proxy_chain = []
            st.session_state.chain_test_result = None
            st.rerun()
    
    # Clear validated list button
    if st.session_state.proxies_validated:
        if st.button("🗑️ Clear Validated List", use_container_width=True):
            st.session_state.proxies_validated = []
            st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Chain Visualization",
    "📊 Proxy List",
    "🌐 Browse via Chain",
    "📈 Analytics"
])

with tab2:
    st.subheader("Validated Proxies")
    if st.session_state.proxies_validated:
        data = [{
            'Host': p.host,
            'Port': p.port,
            'Protocol': p.protocol,
            'Country': p.country or 'Unknown',
            'City': p.city or 'Unknown',
            'ASN': p.asn or 'N/A',
            'Latency (ms)': f"{p.latency:.0f}" if p.latency else 'N/A'
        } for p in st.session_state.proxies_validated]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.info("No validated proxies. Load and validate from sidebar.")

with tab3:
    st.subheader("Browse Web Through Chain")
    url = st.text_input("Enter URL:", placeholder="https://example.com")
    if st.session_state.proxy_chain and len(st.session_state.proxy_chain) >= 2:
        if st.button("🌐 Fetch via Chain", use_container_width=True):
            if url:
                with st.spinner("Fetching..."):
                    result = run_async(fetch_via_chain(st.session_state.proxy_chain, url))
                    if result.get("success"):
                        st.success("✅ Fetched successfully!")
                        st.write(f"**Status:** {result.get('status_line')}")
                        st.write(f"**Size:** {result.get('bytes')} bytes")
                        if result.get("exit_ip_consensus"):
                            st.write(f"**Exit IP:** {result['exit_ip_consensus']}")
                        with st.expander("Preview"):
                            st.text(result.get('body_preview', '')[:2000])
                    else:
                        st.error(f"❌ {result.get('error')}")
    else:
        st.warning("Build a chain with at least 2 hops first")

st.markdown("---")
st.caption("ProxyStream - Production-ready multi-hop proxy chain system")
