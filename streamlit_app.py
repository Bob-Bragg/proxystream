"""
ProxyStream Complete - True Multi-Hop Proxy Chain Testing System
Real HTTP CONNECT tunneling, browser-side location, full chain routing
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
    import redis  # type: ignore
except Exception:
    redis = None
try:
    import pyasn  # type: ignore
except Exception:
    pyasn = None

# Prometheus metrics
from prometheus_client import start_http_server, Counter, Histogram, REGISTRY

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

GEO_APIS = [
    {"url": "http://ip-api.com/json/{}", "has_asn": True},
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
CERT_PINS: Dict[str, str] = {
    # "example.com": "sha256/<base64_fp>",   # e.g., "sha256:W6ph5Mm5Pz8GgiULbPgzG37mj9g="
}

# Redis config (optional)
REDIS_URL = os.getenv("PROXYSTREAM_REDIS_URL", "")
redis_client = redis.Redis.from_url(REDIS_URL) if (redis and REDIS_URL) else None

# pyasn DB (optional offline ASN)
PYASN_DB = os.getenv("PROXYSTREAM_PYASN_DB", "ipasn.dat")
pyasn_obj = pyasn.pyasn(PYASN_DB) if (pyasn and pathlib.Path(PYASN_DB).exists()) else None

# ---------------------------------------------------
# Prometheus metrics (handle Streamlit reloads)
# ---------------------------------------------------
# Use try-except to handle re-registration on Streamlit reload
try:
    METRIC_FETCH_OK = Counter("proxystream_fetch_success_total", "Successful fetches")
    METRIC_FETCH_ERR = Counter("proxystream_fetch_errors_total", "Errored fetches")
    METRIC_CHAIN_TEST = Histogram("proxystream_chain_total_ms", "Total chain test time (ms)")
    METRIC_HOP_MS = Histogram("proxystream_hop_ms", "Per-hop handshake time (ms)")
    METRIC_RATE_LIMIT_BLOCK = Counter("proxystream_rate_limit_block_total", "Rate limited blocks")
except ValueError:
    # Metrics already registered, get them from the registry
    METRIC_FETCH_OK = REGISTRY._names_to_collectors["proxystream_fetch_success_total"]
    METRIC_FETCH_ERR = REGISTRY._names_to_collectors["proxystream_fetch_errors_total"]
    METRIC_CHAIN_TEST = REGISTRY._names_to_collectors["proxystream_chain_total_ms"]
    METRIC_HOP_MS = REGISTRY._names_to_collectors["proxystream_hop_ms"]
    METRIC_RATE_LIMIT_BLOCK = REGISTRY._names_to_collectors["proxystream_rate_limit_block_total"]

# Try to start metrics server
try:
    start_http_server(int(os.getenv("PROXYSTREAM_METRICS_PORT", "9108")))
except Exception:
    pass  # Server already running or port in use

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
# Database Init with PRAGMAs, Indexes, Partitions
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
    # Base log table for backward compat
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
    """Downsample older chain tests (keep daily mean)."""
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # simple downsample on base table
    cur.execute(f"""
        DELETE FROM chain_tests
        WHERE date(tested_at) < date(?)
          AND id NOT IN (
            SELECT id FROM (
              SELECT id, date(tested_at) d,
                     AVG(total_latency) OVER (PARTITION BY date(tested_at)) m
              FROM chain_tests
            ) t
          )
    """, (cutoff_str,))
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
    protocol: str = "http"  # http/https/socks4/socks5 for pooled; manual CONNECT supports http/https
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

# Local token bucket; Redis-backed if available
class RateLimiter:
    def __init__(self, rate: int, per_seconds: int, key_prefix: str = "rl:"):
        self.rate = rate
        self.per = per_seconds
        self.key_prefix = key_prefix
        self.local_buckets: Dict[str, Tuple[int, float]] = {}

    def _redis_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def allow(self, key: str) -> bool:
        if redis_client:
            now = int(time.time())
            rkey = self._redis_key(key)
            with redis_client.pipeline() as pipe:
                try:
                    pipe.incr(rkey, 1)
                    pipe.expire(rkey, self.per)
                    count, _ = pipe.execute()
                    if int(count) <= self.rate:
                        return True
                    METRIC_RATE_LIMIT_BLOCK.inc()
                    return False
                except Exception:
                    # fallback to local
                    pass
        # local fallback
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
# ASN Enrichment (Team Cymru WHOIS + optional pyasn)
# ---------------------------------------------------
async def team_cymru_bulk_lookup(ips: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Query Team Cymru WHOIS (TCP 43) in bulk.
    Returns mapping ip -> {asn, asn_org, cc, prefix, registry}
    """
    if not ips:
        return {}
    reader, writer = await asyncio.open_connection("whois.cymru.com", 43)
    try:
        writer.write(b"begin\nverbose\n")
        for ip in ips:
            writer.write((ip + "\n").encode())
        writer.write(b"end\n")
        await writer.drain()
        data = await reader.read(-1)
    finally:
        writer.close()
        try: await writer.wait_closed()
        except Exception: pass

    out: Dict[str, Dict[str, str]] = {}
    lines = data.decode(errors="ignore").splitlines()
    # skip header lines
    for line in lines:
        if line.startswith("AS") or line.lower().startswith("bulk mode"):
            continue
        # Format (verbose): ASN | IP | BGP Prefix | CC | Registry | Allocated | AS Name
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
# Geo / Exit-IP / Anonymity
# ---------------------------------------------------
async def exit_ip_consensus_via_session(session: httpx.AsyncClient) -> Optional[str]:
    cached = exit_ip_cache.get("exit")
    if cached: return cached
    ips = []
    for url in EXIT_IP_PROVIDERS:
        try:
            r = await session.get(url, timeout=10)
            if r.status_code == 200:
                ip = r.json().get("ip")
                if ip:
                    ips.append(ip)
        except Exception:
            continue
    if not ips:
        return None
    best = max(set(ips), key=ips.count)
    exit_ip_cache.set("exit", best)
    return best

async def get_proxy_geo_with_asn(ip: str) -> Dict[str, Any]:
    """
    Try pyasn offline; fallback to public geo APIs; then optionally Team Cymru enrich if gaps.
    """
    out: Dict[str, Any] = {}
    # pyasn first (offline)
    if _is_ip(ip):
        out.update(pyasn_lookup(ip))

    # public APIs
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

    # Team Cymru if asn missing
    if out.get("asn") is None and _is_ip(ip):
        try:
            tc = asyncio.get_event_loop().run_until_complete(team_cymru_bulk_lookup([ip]))
        except RuntimeError:
            tc = asyncio.run(team_cymru_bulk_lookup([ip]))
        info = tc.get(ip) or {}
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
# Validation for single proxy
# ---------------------------------------------------
async def validate_proxy(proxy: ProxyInfo, timeout: int = 10) -> Tuple[bool, float, Optional[str]]:
    """
    Validate a single HTTP proxy via httpx proxies param (socks not supported here).
    Returns: (is_valid, latency_ms, exit_ip)
    """
    if not rate_limiter.allow(f"validate:{proxy.host}:{proxy.port}"):
        return False, 0.0, None

    proxy_url = proxy.as_url()
    try:
        start = time.perf_counter_ns()
        async with httpx.AsyncClient(proxies=proxy_url, timeout=timeout) as client:
            r = await client.get("http://httpbin.org/ip")
            if r.status_code == 200:
                latency_ms = (time.perf_counter_ns() - start) / 1_000_000
                data = r.json()
                exit_ip = data.get("origin", "").split(",")[0].strip()
                return True, latency_ms, exit_ip
    except Exception:
        pass
    return False, 0.0, None

# ---------------------------------------------------
# Proxy list parsing (IPv4 + IPv6)
# ---------------------------------------------------
_IPV4_PORT = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}):(\d+)$")
_IPV6_PORT = re.compile(r"^\[?([0-9a-fA-F:]+)\]?:([0-9]{1,5})$")

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

    # IPv4 or IPv6
    m4 = _IPV4_PORT.match(rest)
    if m4:
        host, port_s = m4.group(1), m4.group(2)
    else:
        m6 = _IPV6_PORT.match(rest)
        if not m6:
            return None
        host, port_s = m6.group(1), m6.group(2)

    try:
        ipaddress.ip_address(host)
    except Exception:
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
# Batch validate + Geo enrich + Safe DB writes + Redis tiered cache
# ---------------------------------------------------
def _cache_set_ip(ip: str, data: Dict[str, Any], ip_type: str = "dc"):
    if not redis_client:
        return
    ttl = 300 if ip_type.lower() == "tor" else (3600 if ip_type.lower() == "residential" else 7 * 86400)
    redis_client.setex(f"geo:{ip}", ttl, json.dumps(data))

def _cache_get_ip(ip: str) -> Optional[Dict[str, Any]]:
    if not redis_client:
        return None
    raw = redis_client.get(f"geo:{ip}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

async def load_and_validate_batch(proxies: List[ProxyInfo], max_concurrent: int = 20) -> List[ProxyInfo]:
    validated: List[ProxyInfo] = []
    sem = asyncio.Semaphore(max_concurrent)
    circ = Circuit(threshold=5, cooldown_seconds=120)

    async def one(proxy: ProxyInfo):
        async with sem:
            if not circ.can_attempt():
                return None
            try:
                async def attempt():
                    return await validate_proxy(proxy)
                ok, lat_ms, exit_ip = await retry_with_backoff(attempt, retries=2, base=0.8, jitter=0.4)
                if ok:
                    proxy.is_valid = True
                    proxy.latency = lat_ms
                    proxy.last_tested = datetime.now()

                    # Geo from cache or live
                    cached = _cache_get_ip(proxy.host)
                    if cached:
                        geo = cached
                    else:
                        geo = await get_proxy_geo_with_asn(proxy.host)
                        # naive ip type classification
                        ip_type = "dc"
                        if geo.get("org") and any(k in (geo["org"] or "").lower() for k in ["comcast", "verizon", "isp", "residential"]):
                            ip_type = "residential"
                        _cache_set_ip(proxy.host, geo, ip_type)

                    proxy.country = geo.get('country')
                    proxy.country_code = geo.get('country_code')
                    proxy.city = geo.get('city')
                    proxy.region = geo.get('region')
                    proxy.lat = geo.get('lat')
                    proxy.lon = geo.get('lon')
                    proxy.asn = geo.get('asn')
                    proxy.org = geo.get('org')
                    proxy.isp = geo.get('isp')
                    circ.record_success()
                    return proxy
                circ.record_failure()
                return None
            except Exception:
                circ.record_failure()
                return None

    results = await asyncio.gather(*(one(p) for p in proxies))
    validated = [r for r in results if r]

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
# Manual CONNECT chain (per-hop timing + TLS)
# ---------------------------------------------------
async def _connect_via_chain_http_connect(
    chain: List[ProxyInfo],
    target_host: str,
    target_port: int,
    use_tls: bool
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = None, None
    try:
        r, w = await asyncio.open_connection(chain[0].host, chain[0].port)
        reader, writer = r, w

        for i in range(len(chain)):
            dest_host = target_host if i == len(chain) - 1 else chain[i + 1].host
            dest_port = target_port if i == len(chain) - 1 else chain[i + 1].port
            req = f"CONNECT {dest_host}:{dest_port} HTTP/1.1\r\n"
            req += f"Host: {dest_host}:{dest_port}\r\n"
            req += "Connection: keep-alive\r\n"
            pi = chain[i]
            if pi.username and pi.password:
                token = base64.b64encode(f"{pi.username}:{pi.password}".encode()).decode()
                req += f"Proxy-Authorization: Basic {token}\r\n"
            req += "\r\n"
            writer.write(req.encode("ascii"))
            await writer.drain()

            status_line = await reader.readuntil(b"\r\n")
            if b"200" not in status_line:
                raise RuntimeError(f"CONNECT failed at hop {i+1}: {status_line.decode(errors='ignore').strip()}")

            # Drain headers
            while True:
                line = await reader.readuntil(b"\r\n")
                if line in (b"\r\n", b"\n", b""):
                    break

        if use_tls:
            loop = asyncio.get_running_loop()
            ssl_ctx = ssl.create_default_context()
            raw_transport = writer.transport
            protocol = reader._protocol  # private, acceptable here for wrap
            tls_transport = await loop.start_tls(raw_transport, protocol, ssl_ctx, server_hostname=target_host)
            new_reader = asyncio.StreamReader()
            new_protocol = asyncio.StreamReaderProtocol(new_reader)
            tls_transport.set_protocol(new_protocol)
            new_writer = asyncio.StreamWriter(tls_transport, new_protocol, new_reader, loop)
            return new_reader, new_writer

        return reader, writer
    except Exception:
        if writer:
            writer.close()
        raise

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
# Full chain test (timings + exit + anonymity + partitioned log)
# ---------------------------------------------------
async def test_proxy_chain_full(chain: List[ProxyInfo], samples: int = 5) -> Dict[str, Any]:
    if not chain:
        return {"success": False, "error": "Empty chain"}
    if any(p.protocol not in ("http", "https") for p in chain):
        return {"success": False, "error": "Manual CONNECT path supports http/https hops only"}

    hop_timings_samples: List[List[float]] = [[] for _ in chain]
    total_samples: List[float] = []
    t0_all = time.perf_counter_ns()

    try:
        for _ in range(samples):
            start_total = time.perf_counter_ns()
            for i in range(len(chain)):
                leg_start = time.perf_counter_ns()
                dest_host = chain[i + 1].host if i < len(chain) - 1 else "api.ipify.org"
                dest_port = chain[i + 1].port if i < len(chain) - 1 else 443
                use_tls = (i == len(chain) - 1)
                rd, wr = await _connect_via_chain_http_connect(chain[:i + 1], dest_host, dest_port, use_tls)
                wr.close()
                try: await wr.wait_closed()
                except Exception: pass
                dur_ms = _ns_to_ms(time.perf_counter_ns() - leg_start)
                hop_timings_samples[i].append(dur_ms)
                METRIC_HOP_MS.observe(dur_ms)
            total_samples.append(_ns_to_ms(time.perf_counter_ns() - start_total))

        hop_stats = []
        for i, s in enumerate(hop_timings_samples):
            stats_i = compute_stats(s)
            stats_i["jitter"] = compute_rfc3550_jitter(s)
            hop_stats.append({"hop": i + 1, **{k: round(v, 2) for k, v in stats_i.items()}})
        total_stats = compute_stats(total_samples)
        METRIC_CHAIN_TEST.observe(total_stats.get("mean", 0.0))

        # Build full tunnel once and fetch exit IP and anonymity headers
        rd, wr = await _connect_via_chain_http_connect(chain, "httpbin.org", 443, True)
        # headers endpoint gives us response + reflected headers
        req = b"GET /headers HTTP/1.1\r\nHost: httpbin.org\r\nConnection: close\r\n\r\n"
        wr.write(req); await wr.drain()
        raw = await rd.read(-1)
        wr.close()
        try: await wr.wait_closed()
        except Exception: pass
        hdr_json = {}
        try:
            _, _, body = raw.partition(b"\r\n\r\n")
            jpos = body.find(b"{")
            if jpos >= 0:
                hdr_json = json.loads(body[jpos:].decode("utf-8", errors="ignore"))
        except Exception:
            pass

        # also fetch exit ip quickly
        rd2, wr2 = await _connect_via_chain_http_connect(chain, "api.ipify.org", 443, True)
        req2 = b"GET /?format=json HTTP/1.1\r\nHost: api.ipify.org\r\nConnection: close\r\n\r\n"
        wr2.write(req2); await wr2.drain()
        body2 = await rd2.read(-1)
        wr2.close()
        try: await wr2.wait_closed()
        except Exception: pass
        exit_ip = None
        try:
            j2 = body2.find(b"{")
            if j2 >= 0: exit_ip = json.loads(body2[j2:].decode("utf-8", errors="ignore")).get("ip")
        except Exception:
            pass

        # Basic anonymity signals from response headers (if any proxies injected)
        # Note: Via/X-Forwarded-For may appear in response headers; httpbin returns request headers in JSON.
        req_headers = (hdr_json.get("headers") or {}) if isinstance(hdr_json, dict) else {}
        anonymity = {
            "via_header": req_headers.get("Via"),
            "x_forwarded_for": req_headers.get("X-Forwarded-For"),
            "forwarded": req_headers.get("Forwarded"),
            "proxy_connection": req_headers.get("Proxy-Connection"),
        }

        # Log into monthly partition
        table = ensure_month_partition("chain_tests")
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO {table} (tested_at, exit_ip, total_latency, hop_timings_json, stats_json, anonymity_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow(),
            exit_ip,
            total_stats.get("mean", 0.0),
            json.dumps(hop_stats),
            json.dumps(total_stats),
            json.dumps(anonymity)
        ))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "exit_ip": exit_ip,
            "hop_stats": hop_stats,
            "total_stats": {k: round(v, 2) for k, v in total_stats.items()},
            "anonymity": anonymity,
            "hop_count": len(chain),
        }
    except Exception as e:
        return {"success": False, "error": f"Chain failed: {str(e)[:200]}"}

# ---------------------------------------------------
# Real browsing via pooled, mixed-protocol chain + TLS pinning + exit consensus
# ---------------------------------------------------
async def to_httpx_via_aio(session: aiohttp.ClientSession) -> httpx.AsyncClient:
    class _AioTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            async with session.request(
                method=request.method,
                url=str(request.url),
                headers=request.headers,
                data=request.content
            ) as r:
                content = await r.read()
                return httpx.Response(status_code=r.status, headers=r.headers, content=content, request=request)
    return httpx.AsyncClient(transport=_AioTransport(), timeout=15.0)

def _sha256_b64(cert_der: bytes) -> str:
    import hashlib, base64
    h = hashlib.sha256(cert_der).digest()
    return "sha256:" + base64.b64encode(h).decode()

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
            # Exit-IP consensus via the chain
            httpx_via = await to_httpx_via_aio(session)
            ip_consensus = None
            try:
                ip_consensus = await exit_ip_consensus_via_session(session=httpx_via)
            except Exception:
                pass

            async with session.get(url, headers={"User-Agent": "ProxyStream/1.0"}) as resp:
                # TLS pinning check (optional)
                host = aiohttp.helpers.URL(url).host
                pin = CERT_PINS.get(host)
                if pin and resp.connection and resp.connection.transport:
                    ssl_obj = resp.connection.transport.get_extra_info("ssl_object")
                    if ssl_obj:
                        cert_bin = ssl_obj.getpeercert(binary_form=True)
                        fp = _sha256_b64(cert_bin)
                        if fp != pin:
                            return {"success": False, "error": f"TLS pin mismatch for {host}: {fp}"}

                body = await resp.text(errors="ignore")
                # anonymity via response headers
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
# Browser geolocation with fallback
# ---------------------------------------------------
def capture_client_geo():
    components.html(
        """
        <script>
        (async () => {
          async function ipFallback(){
            try{
              const r = await fetch('https://ipapi.co/json/');
              if(!r.ok) throw new Error('ipapi failure');
              return await r.json();
            }catch(e){
              return { city: 'Unknown', country: 'Unknown', country_code: '', ip: '0.0.0.0' };
            }
          }
          function send(val){
            const py = window.parent;
            py.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', args: {value: val}}, '*');
          }
          try{
            if (navigator.geolocation) {
              navigator.geolocation.getCurrentPosition(async (pos) => {
                const { latitude, longitude, accuracy } = pos.coords;
                let base = await ipFallback();
                base.latitude = latitude; base.longitude = longitude; base.accuracy = accuracy;
                send(base);
              }, async () => {
                const d = await ipFallback(); send(d);
              }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 });
            } else {
              const d = await ipFallback(); send(d);
            }
          }catch(e){
            const d = await ipFallback(); send(d);
          }
        })();
        </script>
        """,
        height=0
    )

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

st.title("🔒 ProxyStream Advanced - True Multi-Hop Chain System")

# Top row controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📍 Detect My Location", use_container_width=True):
        capture_client_geo()
        with st.spinner("Getting browser location..."):
            async def get_location():
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get("https://ipapi.co/json/")
                    return r.json() if r.status_code == 200 else {}
            # unified runner
            try:
                loop = asyncio.get_running_loop()
                location = loop.run_until_complete(get_location())  # in Streamlit this usually raises
            except RuntimeError:
                location = asyncio.run(get_location())
            st.session_state.client_location = location or {"city": "Unknown", "country_code": "", "ip": "0.0.0.0"}
            st.success("Location detected!")
            st.rerun()

if st.session_state.client_location:
    loc = st.session_state.client_location
    with col2: st.metric("Your Location", f"{loc.get('city', 'Unknown')}, {loc.get('country_code', '')}")
    with col3: st.metric("Your IP", loc.get('ip', 'Unknown'))
    with col4: st.metric("Accuracy (m)", f"{int(loc.get('accuracy', 0))}" if loc.get('accuracy') else "N/A")

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")

    # Load Proxies
    st.subheader("📥 Load Proxies")
    if st.button("🔄 Fetch from GitHub Sources", use_container_width=True):
        with st.spinner("Fetching from sources..."):
            all_proxies: List[ProxyInfo] = []
            for i, src in enumerate(PROXY_SOURCES):
                st.text(f"Loading {src.split('/')[-1]}...")
                try:
                    proxies = asyncio.run(fetch_and_parse_proxies(src))
                except RuntimeError:
                    proxies = asyncio.get_event_loop().run_until_complete(fetch_and_parse_proxies(src))
                all_proxies.extend(proxies)
                if i == 0: st.info(f"✅ Proxy-Hound: {len(proxies)} proxies")
            unique = list({p: None for p in all_proxies}.keys())
            st.session_state.proxies_raw = unique
            st.success(f"Loaded {len(unique)} unique proxies")

    # Validation
    if st.session_state.proxies_raw:
        st.subheader("✅ Validate")
        validate_count = st.slider("Number to validate:", 10, 150, 40)
        if st.button("🧪 Validate Proxies", use_container_width=True):
            with st.spinner(f"Validating {validate_count} proxies..."):
                sample = st.session_state.proxies_raw[:validate_count]
                try:
                    validated = asyncio.run(load_and_validate_batch(sample))
                except RuntimeError:
                    validated = asyncio.get_event_loop().run_until_complete(load_and_validate_batch(sample))
                st.session_state.proxies_validated.extend(validated)
                st.success(f"✅ {len(validated)} working proxies validated")

    # Search/Filter
    st.subheader("🔍 Search & Filter")
    filters = {
        'country': st.text_input("Country", placeholder="US, Germany..."),
        'asn': st.text_input("ASN", placeholder="AS12345"),
        'ip': st.text_input("IP", placeholder="1.2.3 or 2a03:..."),
        'protocol': st.selectbox("Protocol", ["", "http", "https", "socks4", "socks5"])
    }
    if st.button("🔍 Search Database", use_container_width=True):
        filtered = {k: v for k, v in filters.items() if v}
        results = load_proxies_from_db(filtered)
        st.session_state.proxies_validated = results
        st.success(f"Found {len(results)} proxies")

    # Chain Builder (2-5 hops)
    st.subheader("🔗 Chain Builder (2-5 hops)")
    if st.session_state.proxies_validated:
        countries: Dict[str, List[ProxyInfo]] = {}
        for proxy in st.session_state.proxies_validated:
            key = proxy.country or "Unknown"
            countries.setdefault(key, []).append(proxy)
        selected_country = st.selectbox("Country", list(countries.keys()))
        if selected_country:
            country_proxies = countries[selected_country][:30]
            proxy_display = [
                f"{p.protocol}://{p.host}:{p.port} | ASN:{p.asn or 'N/A'} | {p.latency:.0f}ms"
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

    # Current Chain block
    if st.session_state.proxy_chain:
        st.write(f"**Chain ({len(st.session_state.proxy_chain)} hops):**")
        colors = ['🔵', '🟢', '🟡', '🟠', '🟣']
        for i, p in enumerate(st.session_state.proxy_chain):
            auth = " (auth)" if p.username else ""
            st.write(f"{colors[i]} {p.protocol}://{p.host}:{p.port}{auth} ({p.country_code or '??'})")

        if len(st.session_state.proxy_chain) >= 2:
            if st.button("🧪 Test Chain", use_container_width=True):
                with st.spinner("Testing full multi-hop chain..."):
                    try:
                        result = asyncio.run(test_proxy_chain_full(st.session_state.proxy_chain))
                    except RuntimeError:
                        result = asyncio.get_event_loop().run_until_complete(test_proxy_chain_full(st.session_state.proxy_chain))
                    st.session_state.chain_test_result = result
                    if result["success"]:
                        st.success("✅ Chain valid!")
                        st.metric("Exit IP", result.get('exit_ip') or "N/A")
                        ts = result.get('total_stats', {})
                        st.metric("Total Mean Latency", f"{ts.get('mean', 0):.0f}ms")
                    else:
                        st.error(f"❌ {result['error']}")

        if st.button("🗑️ Clear Chain", use_container_width=True):
            st.session_state.proxy_chain = []
            st.session_state.chain_test_result = None
            st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Chain Visualization",
    "📊 Proxy List",
    "🌐 Browse via Chain",
    "📈 Analytics"
])

with tab1:
    st.subheader("Multi-Hop Chain Routing Visualization")
    if st.session_state.proxy_chain and len(st.session_state.proxy_chain) >= 2:
        fig = go.Figure()
        lats, lons = [], []

        if st.session_state.client_location:
            cl = st.session_state.client_location
            lats.append(cl.get('latitude', 0) or 0); lons.append(cl.get('longitude', 0) or 0)
            fig.add_trace(go.Scattermapbox(
                mode='markers+text', lon=[lons[0]], lat=[lats[0]], marker=dict(size=15),
                text="You", name="Your Location", showlegend=True
            ))

        for i, p in enumerate(st.session_state.proxy_chain):
            if p.lat and p.lon:
                lats.append(p.lat); lons.append(p.lon)
                fig.add_trace(go.Scattermapbox(
                    mode='markers+text', lon=[p.lon], lat=[p.lat], marker=dict(size=12),
                    text=f"Hop {i+1}", name=f"Hop {i+1}: {p.country or 'Unknown'}",
                    hovertemplate=(
                        f"<b>Hop {i+1}</b><br>"
                        f"IP: {p.host}:{p.port}<br>Protocol: {p.protocol}<br>"
                        f"Loc: {p.city or 'N/A'}, {p.country or 'N/A'}<br>"
                        f"ASN: {p.asn or 'N/A'}<br>ISP: {p.isp or p.org or 'N/A'}<br>"
                        f"Latency: {p.latency:.0f}ms" if p.latency else "Latency: N/A"
                    )
                ))

        # Exit point if tested
        if st.session_state.chain_test_result and st.session_state.chain_test_result.get("success"):
            exit_ip = st.session_state.chain_test_result.get("exit_ip")
            if exit_ip:
                try:
                    exit_geo = asyncio.run(get_proxy_geo_with_asn(exit_ip))
                except RuntimeError:
                    exit_geo = asyncio.get_event_loop().run_until_complete(get_proxy_geo_with_asn(exit_ip))
                if exit_geo.get('lat') and exit_geo.get('lon'):
                    fig.add_trace(go.Scattermapbox(
                        mode='markers', lon=[exit_geo['lon']], lat=[exit_geo['lat']],
                        marker=dict(size=15, symbol='star'), name=f"Exit: {exit_ip}",
                        hovertext=f"Exit IP: {exit_ip}<br>{exit_geo.get('city')}, {exit_geo.get('country')}"
                    ))
                    lats.append(exit_geo['lat']); lons.append(exit_geo['lon'])

        if len(lats) > 1:
            for i in range(len(lats) - 1):
                fig.add_trace(go.Scattermapbox(
                    mode='lines', lon=[lons[i], lons[i+1]], lat=[lats[i], lats[i+1]],
                    line=dict(width=3), showlegend=False
                ))

        fig.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lon=lons[0] if lons else 0, lat=lats[0] if lats else 0),
                        zoom=2),
            showlegend=True, height=600, margin=dict(r=0, t=0, l=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hop stats
        res = st.session_state.chain_test_result
        if res and res.get("hop_stats"):
            st.write("**Per-Hop Stats (ms):**")
            for hs in res["hop_stats"]:
                st.write(
                    f"Hop {hs['hop']}: mean={hs['mean']}, p50={hs['p50']}, p95={hs['p95']}, "
                    f"p99={hs['p99']}, stdev={hs['stdev']}, jitter={hs['jitter']}"
                )
        if res and res.get("anonymity"):
            st.write("**Anonymity Signals:** ", res["anonymity"])
    else:
        st.info("Build a chain with at least 2 hops to visualize routing")

with tab2:
    st.subheader("Validated Proxies with ASN Data")
    if st.session_state.proxies_validated:
        data = [{
            'Host': p.host, 'Port': p.port, 'Protocol': p.protocol,
            'Auth': 'Yes' if p.username else 'No',
            'Country': p.country or 'Unknown', 'City': p.city or 'Unknown',
            'ASN': p.asn or 'N/A', 'ISP/Org': p.isp or p.org or 'Unknown',
            'Latency (ms)': f"{p.latency:.0f}" if p.latency else 'N/A'
        } for p in st.session_state.proxies_validated]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "proxies.csv", "text/csv")
    else:
        st.info("No validated proxies. Load and validate from sidebar.")

with tab3:
    st.subheader("Browse Web Through Full Chain (pooled HTTP/SOCKS)")
    url = st.text_input("Enter URL:", placeholder="https://example.com")
    if st.session_state.proxy_chain and len(st.session_state.proxy_chain) >= 2:
        if st.button("🌐 Fetch via Chain", use_container_width=True):
            if url:
                with st.spinner("Fetching through full chain..."):
                    try:
                        result = asyncio.run(fetch_via_chain(st.session_state.proxy_chain, url))
                    except RuntimeError:
                        result = asyncio.get_event_loop().run_until_complete(fetch_via_chain(st.session_state.proxy_chain, url))
                    if result.get("success"):
                        st.success("✅ Fetched successfully!")
                        st.write(f"**Status:** {result.get('status_line')}")
                        st.write(f"**Size:** {result.get('bytes')} bytes")
                        if result.get("exit_ip_consensus"):
                            st.write(f"**Exit IP (consensus):** {result['exit_ip_consensus']}")
                        if result.get("resp_proxy_headers"):
                            st.write("**Response Proxy Headers:** ", result["resp_proxy_headers"])
                        with st.expander("Preview"):
                            st.text(result.get('body_preview', '')[:2000])
                    else:
                        st.error(f"❌ {result.get('error')}")
    else:
        st.warning("Build a chain with at least 2 hops first")

with tab4:
    st.subheader("Analytics & Maintenance")
    all_proxies = load_proxies_from_db()
    if all_proxies:
        col1, col2 = st.columns(2)
        with col1:
            countries = [p.country for p in all_proxies if p.country]
            if countries:
                country_counts = pd.Series(countries).value_counts().head(15)
                fig = px.bar(x=country_counts.values, y=country_counts.index, orientation='h', title="Top 15 Countries")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            asns = [p.asn for p in all_proxies if p.asn]
            if asns:
                asn_counts = pd.Series(asns).value_counts().head(10)
                fig = px.pie(values=asn_counts.values, names=asn_counts.index, title="Top 10 ASNs")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("**DB Maintenance**")
    if st.button("Run downsampling (older than 30 days)"):
        downsample_chain_tests(older_than_days=30)
        st.success("Downsampling invoked.")

st.markdown("---")
st.caption("ProxyStream - Multi-hop CONNECT timing + pooled HTTP/SOCKS chaining, exit-IP consensus, ASN enrichment, TLS pinning, Prometheus metrics, and production-grade resilience.")
