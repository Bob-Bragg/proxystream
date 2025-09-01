"""
ProxyStream Cloud - Optimized Edition
Enhanced with better validation, performance optimizations, and data persistence
"""

import asyncio
import time
import json
import base64
import os
import re
import random
import socket
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import streamlit as st
import pandas as pd
import plotly.express as px
import httpx
import aiohttp
from aiohttp_socks import ProxyConnector

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="ProxyStream Cloud Optimized",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Proxy Sources
PROXY_SOURCES = {
    "Primary": [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
        "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    ],
    "SOCKS": [
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
        "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
    ],
    "GitHub Lists": [
        "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
        "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
    ]
}

# Test endpoints for validation
TEST_ENDPOINTS = [
    "http://httpbin.org/ip",
    "http://icanhazip.com",
    "https://api.ipify.org",
    "http://checkip.amazonaws.com",
    "http://ipecho.net/plain",
]

# ---------------------------------------------------
# Session State Management
# ---------------------------------------------------
def init_session_state():
    """Initialize session state with optimized defaults"""
    defaults = {
        "validated_proxies": deque(maxlen=1000),  # Limit to prevent memory issues
        "proxy_buffer": [],  # Temporary buffer for fetched proxies
        "validation_queue": deque(maxlen=5000),  # Queue for validation
        "known_good_proxies": set(),  # Set of known good proxies
        "stats": {
            "total_fetched": 0,
            "total_validated": 0,
            "total_working": 0,
            "last_validation": None
        },
        "settings": {
            "max_concurrent": 50,
            "timeout": 10,
            "auto_validate": True,
            "persist_to_gist": False,
            "gist_token": "",
            "gist_id": ""
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------------------------------------
# Data Models
# ---------------------------------------------------
@dataclass
class Proxy:
    host: str
    port: int
    protocol: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    latency: Optional[float] = None
    last_tested: Optional[datetime] = None
    is_valid: bool = False
    test_count: int = 0
    
    def __hash__(self):
        return hash((self.host, self.port, self.protocol))
    
    def __eq__(self, other):
        if isinstance(other, Proxy):
            return self.host == other.host and self.port == other.port
        return False
    
    def as_url(self) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"
    
    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "latency": self.latency,
            "url": self.as_url()
        }

# ---------------------------------------------------
# Optimized Validation System
# ---------------------------------------------------
class OptimizedValidator:
    """Optimized proxy validator with better concurrency handling"""
    
    def __init__(self, max_concurrent: int = 50, timeout: int = 10):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout_config = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout_config)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_proxy(self, proxy: Proxy) -> bool:
        """Test a single proxy"""
        if proxy.protocol in ["socks4", "socks5"]:
            # Use ProxyConnector for SOCKS
            connector = ProxyConnector.from_url(proxy.as_url())
            session = aiohttp.ClientSession(connector=connector)
        else:
            session = self.session
        
        for endpoint in TEST_ENDPOINTS[:2]:  # Test with first 2 endpoints
            try:
                start = time.perf_counter()
                
                if proxy.protocol in ["http", "https"]:
                    async with session.get(
                        endpoint,
                        proxy=proxy.as_url(),
                        ssl=False
                    ) as response:
                        if response.status in [200, 301, 302]:
                            proxy.latency = (time.perf_counter() - start) * 1000
                            proxy.is_valid = True
                            proxy.last_tested = datetime.now()
                            if proxy.protocol in ["socks4", "socks5"]:
                                await session.close()
                            return True
                else:
                    # SOCKS proxy
                    async with session.get(endpoint, ssl=False) as response:
                        if response.status in [200, 301, 302]:
                            proxy.latency = (time.perf_counter() - start) * 1000
                            proxy.is_valid = True
                            proxy.last_tested = datetime.now()
                            await session.close()
                            return True
                            
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
        
        if proxy.protocol in ["socks4", "socks5"] and session != self.session:
            await session.close()
        return False
    
    async def validate_batch(self, proxies: List[Proxy], progress_callback=None) -> List[Proxy]:
        """Validate a batch of proxies concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        valid_proxies = []
        
        async def validate_with_semaphore(proxy):
            async with semaphore:
                if await self.test_proxy(proxy):
                    valid_proxies.append(proxy)
                    if progress_callback:
                        progress_callback(1)
                return proxy
        
        tasks = [validate_with_semaphore(proxy) for proxy in proxies]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return valid_proxies

def run_async_validation(proxies: List[Proxy], max_concurrent: int = 50, progress_callback=None) -> List[Proxy]:
    """Run async validation in a thread-safe manner"""
    async def validate():
        async with OptimizedValidator(max_concurrent=max_concurrent) as validator:
            return await validator.validate_batch(proxies, progress_callback)
    
    # Create new event loop for thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(validate())
    finally:
        loop.close()

# ---------------------------------------------------
# Synchronous Fallback Validator
# ---------------------------------------------------
def validate_proxy_sync(proxy: Proxy, timeout: int = 10) -> bool:
    """Synchronous proxy validation as fallback"""
    proxy_dict = {
        'http': proxy.as_url(),
        'https': proxy.as_url()
    }
    
    for endpoint in TEST_ENDPOINTS[:2]:
        try:
            start = time.perf_counter()
            response = httpx.get(
                endpoint,
                proxies=proxy_dict,
                timeout=timeout,
                verify=False,
                follow_redirects=True
            )
            if response.status_code in [200, 301, 302]:
                proxy.latency = (time.perf_counter() - start) * 1000
                proxy.is_valid = True
                proxy.last_tested = datetime.now()
                return True
        except:
            continue
    return False

def validate_batch_threaded(proxies: List[Proxy], max_workers: int = 20, progress_callback=None) -> List[Proxy]:
    """Validate proxies using thread pool"""
    valid_proxies = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_proxy_sync, proxy): proxy for proxy in proxies}
        
        for future in as_completed(futures):
            proxy = futures[future]
            try:
                if future.result():
                    valid_proxies.append(proxy)
                    if progress_callback:
                        progress_callback(1)
            except:
                pass
    
    return valid_proxies

# ---------------------------------------------------
# Proxy Parsing and Fetching
# ---------------------------------------------------
def parse_proxy_line(line: str) -> Optional[Proxy]:
    """Parse a proxy line in various formats"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
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
        if not parts.hostname or not parts.port:
            # Try simple format host:port
            if ":" in line and "/" not in line:
                host, port = line.rsplit(":", 1)
                return Proxy(host=host, port=int(port), protocol="http")
            return None
        
        return Proxy(
            host=parts.hostname,
            port=parts.port,
            protocol=parts.scheme or "http",
            username=parts.username,
            password=parts.password
        )
    except:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_proxies_from_url(url: str, limit: int = 1000) -> List[Proxy]:
    """Fetch and parse proxies from URL"""
    proxies = []
    try:
        response = httpx.get(url, timeout=20, follow_redirects=True)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')[:limit]
            for line in lines:
                proxy = parse_proxy_line(line)
                if proxy:
                    proxies.append(proxy)
    except Exception as e:
        st.warning(f"Failed to fetch from {url}: {str(e)[:50]}")
    return proxies

# ---------------------------------------------------
# Data Persistence (GitHub Gist)
# ---------------------------------------------------
def save_to_gist(proxies: List[Proxy], token: str, gist_id: str = None) -> Optional[str]:
    """Save proxies to GitHub Gist"""
    if not token:
        return None
    
    content = "\n".join([p.as_url() for p in proxies if p.is_valid])
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "description": f"Validated Proxies - {datetime.now().isoformat()}",
        "public": False,
        "files": {
            "proxies.txt": {"content": content}
        }
    }
    
    try:
        if gist_id:
            # Update existing gist
            response = httpx.patch(
                f"https://api.github.com/gists/{gist_id}",
                headers=headers,
                json=data,
                timeout=10
            )
        else:
            # Create new gist
            response = httpx.post(
                "https://api.github.com/gists",
                headers=headers,
                json=data,
                timeout=10
            )
        
        if response.status_code in [200, 201]:
            return response.json().get("id")
    except Exception as e:
        st.error(f"Failed to save to Gist: {str(e)[:100]}")
    
    return None

def load_from_gist(token: str, gist_id: str) -> List[Proxy]:
    """Load proxies from GitHub Gist"""
    if not token or not gist_id:
        return []
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = httpx.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["files"]["proxies.txt"]["content"]
            
            proxies = []
            for line in content.split('\n'):
                proxy = parse_proxy_line(line)
                if proxy:
                    proxy.is_valid = True  # These are pre-validated
                    proxies.append(proxy)
            return proxies
    except Exception as e:
        st.error(f"Failed to load from Gist: {str(e)[:100]}")
    
    return []

# ---------------------------------------------------
# UI Components
# ---------------------------------------------------
def render_stats():
    """Render statistics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fetched", st.session_state.stats["total_fetched"])
    
    with col2:
        st.metric("Total Validated", st.session_state.stats["total_validated"])
    
    with col3:
        working = len([p for p in st.session_state.validated_proxies if p.is_valid])
        st.metric("Working Proxies", working)
    
    with col4:
        if st.session_state.stats["total_validated"] > 0:
            success_rate = (st.session_state.stats["total_working"] / st.session_state.stats["total_validated"]) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")

def render_proxy_input():
    """Render manual proxy input section"""
    st.subheader("📝 Add Known Good Proxies")
    
    proxy_text = st.text_area(
        "Enter proxies (one per line):",
        height=100,
        placeholder="http://1.2.3.4:8080\nsocks5://user:pass@5.6.7.8:1080",
        key="proxy_input"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Add to Known Good", type="primary", use_container_width=True):
            if proxy_text:
                lines = proxy_text.strip().split('\n')
                added = 0
                for line in lines:
                    proxy = parse_proxy_line(line)
                    if proxy:
                        proxy.is_valid = True
                        st.session_state.known_good_proxies.add(proxy)
                        st.session_state.validated_proxies.append(proxy)
                        added += 1
                
                if added:
                    st.success(f"✅ Added {added} proxies to known good list")
                    st.session_state.stats["total_working"] += added
                else:
                    st.error("No valid proxies found in input")
    
    with col2:
        if st.button("🧪 Validate & Add", use_container_width=True):
            if proxy_text:
                lines = proxy_text.strip().split('\n')
                proxies_to_test = []
                for line in lines:
                    proxy = parse_proxy_line(line)
                    if proxy:
                        proxies_to_test.append(proxy)
                
                if proxies_to_test:
                    with st.spinner(f"Validating {len(proxies_to_test)} proxies..."):
                        valid = run_async_validation(proxies_to_test, max_concurrent=10)
                        
                        for proxy in valid:
                            st.session_state.known_good_proxies.add(proxy)
                            st.session_state.validated_proxies.append(proxy)
                        
                        st.success(f"✅ {len(valid)}/{len(proxies_to_test)} proxies are working")
                        st.session_state.stats["total_validated"] += len(proxies_to_test)
                        st.session_state.stats["total_working"] += len(valid)

def render_proxy_table():
    """Render the proxy list table"""
    if not st.session_state.validated_proxies:
        st.info("No validated proxies yet. Fetch and validate some proxies to see them here.")
        return
    
    # Convert to DataFrame
    data = []
    for proxy in list(st.session_state.validated_proxies):
        if proxy.is_valid:
            data.append({
                'Protocol': proxy.protocol.upper(),
                'Host': proxy.host,
                'Port': proxy.port,
                'Latency': f"{proxy.latency:.0f}ms" if proxy.latency else "N/A",
                'Status': "✅ Working" if proxy.is_valid else "❌ Failed",
                'URL': proxy.as_url()
            })
    
    if data:
        df = pd.DataFrame(data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            protocols = ["All"] + sorted(df['Protocol'].unique().tolist())
            filter_protocol = st.selectbox("Filter by Protocol", protocols)
        
        with col2:
            max_latency = st.slider("Max Latency (ms)", 0, 10000, 10000)
        
        with col3:
            limit = st.number_input("Show top N", min_value=10, max_value=1000, value=100)
        
        # Apply filters
        filtered_df = df
        if filter_protocol != "All":
            filtered_df = filtered_df[filtered_df['Protocol'] == filter_protocol]
        
        # Sort by latency and limit
        filtered_df = filtered_df.head(limit)
        
        st.dataframe(filtered_df, use_container_width=True, height=400, hide_index=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "proxies.csv", "text/csv", use_container_width=True)
        
        with col2:
            urls = "\n".join(filtered_df['URL'].tolist())
            st.download_button("📥 Download URLs", urls, "proxy_urls.txt", "text/plain", use_container_width=True)
        
        with col3:
            json_data = json.dumps([p.to_dict() for p in st.session_state.validated_proxies if p.is_valid], indent=2)
            st.download_button("📥 Download JSON", json_data, "proxies.json", "application/json", use_container_width=True)
    else:
        st.warning("No working proxies found yet")

# ---------------------------------------------------
# Main Application
# ---------------------------------------------------
st.title("🚀 ProxyStream Cloud - Optimized Edition")
st.caption("High-performance proxy harvesting and validation with smart caching")

# Stats Dashboard
render_stats()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Performance Settings
    with st.expander("🎯 Performance", expanded=True):
        st.session_state.settings["max_concurrent"] = st.slider(
            "Max Concurrent Tests",
            min_value=10,
            max_value=100,
            value=st.session_state.settings["max_concurrent"],
            help="Higher = Faster but more resource intensive"
        )
        
        st.session_state.settings["timeout"] = st.slider(
            "Timeout (seconds)",
            min_value=5,
            max_value=30,
            value=st.session_state.settings["timeout"]
        )
        
        validation_mode = st.radio(
            "Validation Mode",
            ["Async (Fast)", "Threaded (Stable)"],
            help="Async is faster but may have issues on some systems"
        )
    
    # Data Persistence
    with st.expander("💾 Data Persistence"):
        use_gist = st.checkbox("Use GitHub Gist Storage")
        
        if use_gist:
            token = st.text_input("GitHub Token", type="password", value=st.session_state.settings.get("gist_token", ""))
            gist_id = st.text_input("Gist ID (optional)", value=st.session_state.settings.get("gist_id", ""))
            
            if token:
                st.session_state.settings["gist_token"] = token
                st.session_state.settings["gist_id"] = gist_id
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save to Gist", use_container_width=True):
                        working_proxies = [p for p in st.session_state.validated_proxies if p.is_valid]
                        if working_proxies:
                            gist_id = save_to_gist(working_proxies, token, gist_id)
                            if gist_id:
                                st.session_state.settings["gist_id"] = gist_id
                                st.success(f"Saved {len(working_proxies)} proxies to Gist")
                                st.code(f"Gist ID: {gist_id}")
                
                with col2:
                    if st.button("📥 Load from Gist", use_container_width=True):
                        if gist_id:
                            loaded = load_from_gist(token, gist_id)
                            if loaded:
                                for proxy in loaded:
                                    st.session_state.validated_proxies.append(proxy)
                                st.success(f"Loaded {len(loaded)} proxies from Gist")
    
    # Quick Actions
    st.markdown("---")
    st.subheader("🎬 Quick Actions")
    
    if st.button("🚀 Harvest & Validate All", type="primary", use_container_width=True):
        with st.spinner("Fetching proxies..."):
            all_proxies = []
            progress = st.progress(0)
            
            sources = []
            for category, urls in PROXY_SOURCES.items():
                sources.extend(urls)
            
            for i, url in enumerate(sources):
                proxies = fetch_proxies_from_url(url, limit=500)
                all_proxies.extend(proxies)
                progress.progress((i + 1) / len(sources))
            
            # Remove duplicates
            unique_proxies = list({p: None for p in all_proxies}.keys())
            st.session_state.stats["total_fetched"] += len(unique_proxies)
            
            st.success(f"Fetched {len(unique_proxies)} unique proxies")
            
            # Validate all without limits
            progress_bar = st.progress(0)
            validated_count = 0
            
            def update_progress(count):
                nonlocal validated_count
                validated_count += count
                progress_bar.progress(min(validated_count / len(unique_proxies), 1.0))
            
            with st.spinner(f"Validating {len(unique_proxies)} proxies..."):
                if validation_mode == "Async (Fast)":
                    valid_proxies = run_async_validation(
                        unique_proxies,
                        max_concurrent=st.session_state.settings["max_concurrent"],
                        progress_callback=update_progress
                    )
                else:
                    valid_proxies = validate_batch_threaded(
                        unique_proxies,
                        max_workers=st.session_state.settings["max_concurrent"],
                        progress_callback=update_progress
                    )
                
                # Update stats
                st.session_state.stats["total_validated"] += len(unique_proxies)
                st.session_state.stats["total_working"] += len(valid_proxies)
                st.session_state.stats["last_validation"] = datetime.now()
                
                # Add to validated list
                for proxy in valid_proxies:
                    st.session_state.validated_proxies.append(proxy)
                
                st.success(f"✅ Found {len(valid_proxies)} working proxies!")
                
                # Auto-save if configured
                if use_gist and token:
                    gist_id = save_to_gist(valid_proxies, token, st.session_state.settings.get("gist_id"))
                    if gist_id:
                        st.session_state.settings["gist_id"] = gist_id
                        st.info(f"Auto-saved to Gist: {gist_id}")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.validated_proxies.clear()
        st.session_state.proxy_buffer.clear()
        st.session_state.validation_queue.clear()
        st.session_state.known_good_proxies.clear()
        st.session_state.stats = {
            "total_fetched": 0,
            "total_validated": 0,
            "total_working": 0,
            "last_validation": None
        }
        st.rerun()

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Proxy List", "➕ Add Proxies", "📊 Analytics", "🔧 Advanced"])

with tab1:
    render_proxy_table()

with tab2:
    render_proxy_input()
    
    st.markdown("---")
    
    st.subheader("🌐 Fetch from Sources")
    
    selected_sources = []
    for category, urls in PROXY_SOURCES.items():
        if st.checkbox(f"{category} ({len(urls)} sources)", value=True, key=f"src_{category}"):
            selected_sources.extend(urls)
    
    if st.button("🔄 Fetch Selected Sources", type="primary"):
        if selected_sources:
            with st.spinner(f"Fetching from {len(selected_sources)} sources..."):
                all_proxies = []
                for url in selected_sources:
                    proxies = fetch_proxies_from_url(url)
                    all_proxies.extend(proxies)
                
                unique = list({p: None for p in all_proxies}.keys())
                st.session_state.proxy_buffer = unique
                st.session_state.stats["total_fetched"] += len(unique)
                st.success(f"Fetched {len(unique)} unique proxies")

with tab3:
    st.subheader("📊 Proxy Analytics")
    
    if st.session_state.validated_proxies:
        working_proxies = [p for p in st.session_state.validated_proxies if p.is_valid]
        
        if working_proxies:
            # Protocol distribution
            col1, col2 = st.columns(2)
            
            with col1:
                protocol_counts = defaultdict(int)
                for p in working_proxies:
                    protocol_counts[p.protocol.upper()] += 1
                
                df_proto = pd.DataFrame(
                    list(protocol_counts.items()),
                    columns=["Protocol", "Count"]
                )
                fig = px.pie(df_proto, values="Count", names="Protocol", title="Protocol Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Latency distribution
                latencies = [p.latency for p in working_proxies if p.latency]
                if latencies:
                    df_latency = pd.DataFrame({"Latency (ms)": latencies})
                    fig = px.histogram(df_latency, x="Latency (ms)", nbins=30, title="Latency Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Success rate over time
            if st.session_state.stats["last_validation"]:
                st.metric("Last Validation", st.session_state.stats["last_validation"].strftime("%Y-%m-%d %H:%M:%S"))
    else:
        st.info("No data available. Start fetching and validating proxies to see analytics.")

with tab4:
    st.subheader("🔧 Advanced Tools")
    
    # Memory optimization
    st.write("**Memory Optimization**")
    col1, col2 = st.columns(2)
    
    with col1:
        current_size = len(st.session_state.validated_proxies)
        st.metric("Proxies in Memory", current_size)
    
    with col2:
        if st.button("🧹 Optimize Memory", use_container_width=True):
            # Keep only the best proxies
            working = [p for p in st.session_state.validated_proxies if p.is_valid]
            working.sort(key=lambda p: p.latency or float('inf'))
            st.session_state.validated_proxies = deque(working[:500], maxlen=1000)
            st.success(f"Optimized! Kept best {len(st.session_state.validated_proxies)} proxies")
    
    st.markdown("---")
    
    # Bulk operations
    st.write("**Bulk Operations**")
    
    url_input = st.text_input("Custom proxy list URL:")
    
    if url_input and st.button("📥 Fetch & Validate Custom URL"):
        with st.spinner("Fetching..."):
            proxies = fetch_proxies_from_url(url_input, limit=1000)
            if proxies:
                st.info(f"Found {len(proxies)} proxies, validating...")
                
                if validation_mode == "Async (Fast)":
                    valid = run_async_validation(proxies, max_concurrent=st.session_state.settings["max_concurrent"])
                else:
                    valid = validate_batch_threaded(proxies, max_workers=st.session_state.settings["max_concurrent"])
                
                for proxy in valid:
                    st.session_state.validated_proxies.append(proxy)
                
                st.success(f"Added {len(valid)} working proxies!")

# Footer
st.markdown("---")
st.caption(f"ProxyStream Cloud Optimized | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if st.session_state.stats["last_validation"]:
    st.caption(f"Last validation: {st.session_state.stats['last_validation'].strftime('%Y-%m-%d %H:%M:%S')}")
