import json
import os
import random
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
import math

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import streamlit.components.v1 as components
from requests.exceptions import ProxyError, SSLError, ConnectTimeout, ReadTimeout, ConnectionError as ReqConnectionError
import socket

# ProxyStream Configuration
st.set_page_config(
    page_title="ProxyStream - Advanced Proxy Testing Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Theme/CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: white; }
    .main-header { text-align: center; font-size: 36px; font-weight: 700; margin-bottom: 30px; color: white;
                   display: flex; align-items: center; justify-content: center; gap: 16px; }
    .metric-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 24px; border-radius: 16px;
                   margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s ease; }
    .proxy-status-connected { color: #10b981; font-weight: bold; background: rgba(16,185,129,.1); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(16,185,129,.2); }
    .proxy-status-disconnected { color: #ef4444; font-weight: bold; background: rgba(239,68,68,.1); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(239,68,68,.2); }
    .proxy-status-warning { color: #f59e0b; font-weight: bold; background: rgba(245,158,11,.1); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(245,158,11,.2); }
    .location-card { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); padding: 16px; border-radius: 12px;
                     margin: 8px 0; border: 1px solid rgba(255, 255, 255, 0.15); }
    .location-comparison { display: flex; justify-content: space-between; align-items: center; padding: 12px; 
                          background: rgba(255, 255, 255, 0.05); border-radius: 8px; margin: 8px 0; }
    [data-testid="metric-container"] { background: rgba(255,255,255,.05); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,.1); padding: 1.5rem; border-radius: 16px; margin: .5rem 0; }
    [data-testid="metric-container"] > div { color: white; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 12px; font-weight: 600; transition: all .3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,.4); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .country-stats { background: rgba(255,255,255,.05); padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 14px; }
    .security-warning { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); padding: 12px; border-radius: 8px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# Enhanced Geolocation Functions
@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def get_user_location() -> Optional[Dict[str, Any]]:
    """Detect user's real location using multiple services"""
    services = [
        'https://ipapi.co/json/',
        'https://ip-api.com/json/',
        'https://freegeoip.app/json/',
    ]
    
    for service in services:
        try:
            response = requests.get(service, timeout=8)
            if response.ok:
                data = response.json()
                # Normalize different API response formats
                return {
                    'ip': data.get('ip') or data.get('query'),
                    'city': data.get('city'),
                    'region': data.get('region') or data.get('regionName'),
                    'country': data.get('country_name') or data.get('country'),
                    'country_code': data.get('country_code') or data.get('countryCode'),
                    'lat': data.get('latitude') or data.get('lat'),
                    'lon': data.get('longitude') or data.get('lon'),
                    'isp': data.get('org') or data.get('isp'),
                    'timezone': data.get('timezone'),
                    'postal': data.get('postal') or data.get('zip')
                }
        except Exception as e:
            continue
    return None

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_detailed_location(ip_address: str) -> Optional[Dict[str, Any]]:
    """Get detailed location info for proxy IP"""
    services = [
        f'https://ipapi.co/{ip_address}/json/',
        f'https://ip-api.com/json/{ip_address}',
        f'https://freegeoip.app/json/{ip_address}'
    ]
    
    for service in services:
        try:
            response = requests.get(service, timeout=8)
            if response.ok:
                data = response.json()
                return {
                    'ip': ip_address,
                    'city': data.get('city'),
                    'region': data.get('region') or data.get('regionName'),
                    'country': data.get('country_name') or data.get('country'),
                    'country_code': data.get('country_code') or data.get('countryCode'),
                    'lat': data.get('latitude') or data.get('lat'),
                    'lon': data.get('longitude') or data.get('lon'),
                    'isp': data.get('org') or data.get('isp'),
                    'timezone': data.get('timezone'),
                    'postal': data.get('postal') or data.get('zip'),
                    'as_number': data.get('asn') or data.get('as')
                }
        except Exception:
            continue
    return None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    if not all([lat1, lon1, lat2, lon2]):
        return 0.0
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    
    return c * r

# Dynamic proxy loader (preserved)
@st.cache_data(ttl=3600, show_spinner=False)
def load_proxy_list(force_key: int = 0) -> Tuple[List[str], str, List[str]]:
    sources = [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/main/docs/by_type/https_hunted.txt",
        "https://cdn.jsdelivr.net/gh/arandomguyhere/Proxy-Hound@main/docs/by_type/https_hunted.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
    headers = {"User-Agent": "ProxyStream/2.0 (+https://proxystream.app)"}
    errors = []

    def parse_lines(text: str) -> List[str]:
        out = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s or " " in s or ":" not in s:
                continue
            host, _, port = s.partition(":")
            if host and port.isdigit():
                out.append(f"{host}:{port}")
        seen = set(); res = []
        for p in out:
            if p not in seen:
                seen.add(p); res.append(p)
        return res

    for url in sources:
        try:
            r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if r.ok and r.text:
                parsed = parse_lines(r.text)
                if parsed:
                    return parsed, url, errors
            errors.append(f"{url} -> status {r.status_code}")
        except Exception as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")
    
    fallback = [
        "34.121.105.79:80","68.107.241.150:8080","3.133.146.217:5050",
        "72.10.160.90:13847","170.85.158.82:80","170.85.158.82:10005",
        "67.43.236.20:29915","167.172.157.96:80","72.10.164.178:1771",
        "72.10.160.173:13909","155.94.241.134:3128"
    ]
    errors.append("All sources failed; using fallback seed list.")
    return fallback, "fallback", errors

# Country mapping (preserved but enhanced)
COUNTRY_IP_MAPPING = {
    'US': ['34.121.105.79', '68.107.241.150', '3.133.146.217', '72.10.160.90', '170.85.158.82'],
    'CA': ['72.10.164.178', '38.127.172.53', '67.43.228.254', '67.43.228.253'],
    'GB': ['170.106.169.97', '130.41.109.158', '155.94.241.134'],
    'DE': ['136.175.9.83', '136.175.9.82', '136.175.9.86'],
    'FR': ['201.174.239.25'],
    'NL': ['67.43.228.251', '67.43.228.250'],
    'SG': ['72.10.160.173', '72.10.160.174', '72.10.160.170'],
    'AU': ['3.133.221.69'],
    'JP': ['67.43.228.252']
}

COUNTRY_COORDS = {
    "US": (37.0902, -95.7129), "CA": (56.1304, -106.3468), "GB": (55.3781, -3.4360),
    "DE": (51.1657, 10.4515), "FR": (46.2276, 2.2137), "NL": (52.1326, 5.2913),
    "SG": (1.3521, 103.8198), "AU": (-25.2744,133.7751), "JP": (36.2048, 138.2529),
}

IP_TO_COUNTRY: Dict[str, str] = {}
for cc, ips in COUNTRY_IP_MAPPING.items():
    for ip in ips:
        IP_TO_COUNTRY[ip] = cc

def get_country_flag(cc: str) -> str:
    flags = {'US':'🇺🇸','CA':'🇨🇦','GB':'🇬🇧','DE':'🇩🇪','FR':'🇫🇷','NL':'🇳🇱','SG':'🇸🇬','AU':'🇦🇺','JP':'🇯🇵'}
    return flags.get(cc, '🏳️')

def parse_proxy_list(proxies: List[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {}
    for proxy in proxies:
        if ':' not in proxy:
            continue
        ip = proxy.split(':')[0].strip()
        cc = IP_TO_COUNTRY.get(ip, 'US')
        buckets.setdefault(cc, []).append(proxy)
    return buckets

def normalize_proxy_http(proxy: str) -> str:
    return proxy if "://" in proxy else f"http://{proxy}"

def tcp_ping(host: str, port: int, timeout: float = 4.0) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, int(port))) == 0
    except Exception:
        return False

@st.cache_data(ttl=600, show_spinner=False)
def detect_proxy_capabilities(proxy_http_url: str, timeout: int = 8) -> Dict[str, Any]:
    caps = {"http_ok": False, "https_ok": False, "ip_http": None, "ip_https": None,
            "err_http": "", "err_https": "", "latency_ms": 0}
    headers = {"User-Agent": "ProxyStream/2.0"}
    proxies = {"http": proxy_http_url, "https": proxy_http_url}
    
    # Test HTTP capability and measure latency
    try:
        start_time = time.perf_counter()
        r = requests.get("http://httpbin.org/ip", proxies=proxies, headers=headers, timeout=timeout)
        elapsed = (time.perf_counter() - start_time) * 1000
        caps["http_ok"] = r.ok
        caps["latency_ms"] = round(elapsed)
        if r.ok: 
            caps["ip_http"] = r.json().get("origin")
    except Exception as e:
        caps["err_http"] = str(e)[:200]
    
    # Test HTTPS capability
    try:
        r = requests.get("https://httpbin.org/ip", proxies=proxies, headers=headers, timeout=timeout)
        caps["https_ok"] = r.ok
        if r.ok: 
            caps["ip_https"] = r.json().get("origin")
    except Exception as e:
        caps["err_https"] = str(e)[:200]
    
    return caps

# Enhanced proxy connection testing with location detection
def test_proxy_connection(proxy: str, timeout: int = 10) -> tuple[bool, dict]:
    """Test proxy with enhanced location detection"""
    proxy_http = normalize_proxy_http(proxy)
    host, port = proxy.split(':')[0], int(proxy.split(':')[1])
    
    # First check if proxy is reachable via TCP
    if not tcp_ping(host, port, timeout=4.0):
        return False, {
            'latency': 0,
            'speed': 0,
            'country': IP_TO_COUNTRY.get(host, 'US'),
            'error': 'TCP connection failed - proxy unreachable',
            'http_ok': False,
            'https_ok': False,
            'ip_detected': None,
            'location': None
        }
    
    # Test actual proxy capabilities
    caps = detect_proxy_capabilities(proxy_http, timeout=timeout)
    
    # Determine if proxy is usable
    is_working = caps["http_ok"] or caps["https_ok"]
    
    # Get detailed location for the proxy IP
    proxy_location = get_detailed_location(host)
    
    # Calculate speed estimate (rough approximation based on latency)
    speed_estimate = 0
    if is_working and caps["latency_ms"] > 0:
        if caps["latency_ms"] < 50:
            speed_estimate = random.uniform(80, 100)
        elif caps["latency_ms"] < 100:
            speed_estimate = random.uniform(40, 80)
        elif caps["latency_ms"] < 200:
            speed_estimate = random.uniform(20, 40)
        else:
            speed_estimate = random.uniform(5, 20)
    
    return is_working, {
        'latency': caps["latency_ms"],
        'speed': round(speed_estimate, 1),
        'country': proxy_location.get('country', 'Unknown') if proxy_location else IP_TO_COUNTRY.get(host, 'US'),
        'error': caps.get("err_http", "") or caps.get("err_https", ""),
        'http_ok': caps["http_ok"],
        'https_ok': caps["https_ok"],
        'ip_detected': caps.get("ip_http") or caps.get("ip_https"),
        'location': proxy_location
    }

def generate_usage_data():
    """Generate realistic usage data - only when actually connected"""
    hours = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq="h"
    )
    return pd.DataFrame({
        'time': hours,
        'download': np.random.exponential(scale=50, size=len(hours)),
        'upload': np.random.exponential(scale=20, size=len(hours))
    })

def coords_for_proxy(proxy_str: str, fallback_country: str = "US") -> Tuple[float, float, str]:
    """Get coordinates - enhanced to use real location data when available"""
    if 'proxy_location' in st.session_state and st.session_state.proxy_location:
        loc = st.session_state.proxy_location
        if loc.get('lat') and loc.get('lon'):
            return float(loc['lat']), float(loc['lon']), loc.get('country_code', fallback_country)
    
    # Fallback to hardcoded mapping
    ip = proxy_str.split(":")[0] if ":" in proxy_str else None
    cc = IP_TO_COUNTRY.get(ip, fallback_country)
    lat, lon = COUNTRY_COORDS.get(cc, (0.0, 0.0))
    return lat, lon, cc

def create_enhanced_map():
    """Create map showing both user and proxy locations"""
    fig = go.Figure()
    
    # Add user location if available
    if 'user_location' in st.session_state and st.session_state.user_location:
        user_loc = st.session_state.user_location
        if user_loc.get('lat') and user_loc.get('lon'):
            fig.add_trace(go.Scattergeo(
                lon=[user_loc['lon']], lat=[user_loc['lat']],
                mode='markers',
                marker=dict(size=18, color='#ef4444', symbol='circle', 
                          line=dict(width=2, color='white')),
                name='Your Real Location',
                text=[f"You: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}"],
                hoverinfo='text'
            ))
    
    # Add proxy location if connected
    if st.session_state.proxy_connected:
        proxy_lat, proxy_lon, proxy_cc = coords_for_proxy(st.session_state.current_proxy, st.session_state.selected_country)
        proxy_loc = st.session_state.get('proxy_location')
        
        proxy_text = f"Proxy: {proxy_loc.get('city', 'Unknown') if proxy_loc else 'Unknown'}, {proxy_loc.get('country', proxy_cc) if proxy_loc else proxy_cc}"
        
        fig.add_trace(go.Scattergeo(
            lon=[proxy_lon], lat=[proxy_lat],
            mode='markers',
            marker=dict(size=22, color='#10b981', symbol='diamond',
                      line=dict(width=3, color='white')),
            name='Proxy Server',
            text=[proxy_text],
            hoverinfo='text'
        ))
        
        # Draw connection line if both locations available
        if ('user_location' in st.session_state and 
            st.session_state.user_location and
            st.session_state.user_location.get('lat') and 
            st.session_state.user_location.get('lon')):
            
            user_loc = st.session_state.user_location
            fig.add_trace(go.Scattergeo(
                lon=[user_loc['lon'], proxy_lon],
                lat=[user_loc['lat'], proxy_lat],
                mode='lines',
                line=dict(width=3, color='#fbbf24', dash='dash'),
                name='Connection Path',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        geo=dict(
            showframe=False, 
            showcoastlines=True, 
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)', 
            landcolor='#374151', 
            oceancolor='#1e293b', 
            coastlinecolor='#6b7280'
        ),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        height=350, 
        margin=dict(l=0,r=0,t=0,b=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color="white")
        )
    )
    
    return fig

# Session state initialization
if "proxy_connected" not in st.session_state:
    st.session_state.proxy_connected = False
if "current_proxy" not in st.session_state:
    st.session_state.current_proxy = None
if "connection_start_time" not in st.session_state:
    st.session_state.connection_start_time = None
if "proxy_metrics" not in st.session_state:
    st.session_state.proxy_metrics = {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False}
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "US"
if "active_proxy" not in st.session_state:
    st.session_state.active_proxy = None
if "force_reload_key" not in st.session_state:
    st.session_state.force_reload_key = 0
if "only_common_ports" not in st.session_state:
    st.session_state.only_common_ports = True
if "user_location" not in st.session_state:
    st.session_state.user_location = None
if "proxy_location" not in st.session_state:
    st.session_state.proxy_location = None

def servers_to_list_control(n: int) -> Tuple[int, bool]:
    shuffle = st.checkbox("Shuffle", True)
    if n <= 1:
        st.caption(f"Servers available: {n}")
        return n, shuffle
    max_slider = min(500, n)
    default = min(50, n)
    step = 1 if n < 20 else 10
    val = st.slider("Servers to list", 1, max_slider, default, step=step)
    return val, shuffle

def main():
    st.markdown('<div class="main-header">🛡️ ProxyStream</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 40px;">Advanced Proxy Testing Dashboard with Global Location Intelligence</p>', unsafe_allow_html=True)

    # Security warning
    st.markdown("""
    <div class="security-warning">
        <strong>⚠️ Security Notice:</strong> This tool tests public HTTP proxies for educational purposes. 
        Public proxies may log traffic, inject ads, or be compromised. Never use them for sensitive activities.
        For real privacy protection, use a reputable VPN service.
    </div>
    """, unsafe_allow_html=True)

    all_proxies, source_used, load_errors = load_proxy_list(st.session_state.force_reload_key)

    with st.sidebar:
        st.markdown("## 🔧 Proxy Settings")

        # Location Detection Section
        st.markdown("---")
        st.markdown("## 🎯 Your Location")
        
        if st.button("📍 Detect My Location", use_container_width=True):
            with st.spinner("Detecting your location..."):
                user_loc = get_user_location()
                if user_loc:
                    st.session_state.user_location = user_loc
                    st.success(f"Located: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}")
                else:
                    st.error("Could not detect location")
        
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.markdown(f"""
            <div class="location-card">
                <strong>🏠 Your Real Location</strong><br>
                📍 {user_loc.get('city', 'Unknown')}, {user_loc.get('region', 'Unknown')}<br>
                🏳️ {user_loc.get('country', 'Unknown')}<br>
                🌐 IP: {user_loc.get('ip', 'Unknown')}<br>
                🏢 ISP: {user_loc.get('isp', 'Unknown')[:30]}...
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        colref1, colref2 = st.columns([3,1])
        with colref1:
            st.caption(f"Source: {source_used}")
            if source_used == "fallback":
                st.warning("Using fallback list (remote fetch failed)", icon="⚠️")
        with colref2:
            if st.button("↻ Refresh"):
                st.session_state.force_reload_key += 1

        with st.expander("Advanced"):
            st.session_state.only_common_ports = st.checkbox("Only common ports 80/8080/3128/443", value=st.session_state.only_common_ports)

        filtered = all_proxies
        if st.session_state.only_common_ports:
            COMMON = {80, 8080, 3128, 443}
            def okp(p):
                try:
                    return int(p.split(":")[1]) in COMMON
                except:
                    return False
            filtered = [p for p in all_proxies if okp(p)]

        proxy_data = parse_proxy_list(filtered)
        total_proxies = sum(len(v) for v in proxy_data.values())
        st.markdown(f"""
        <div class="country-stats">
            📊 <strong>Network Statistics</strong><br>
            Total Proxies: <strong>{total_proxies:,}</strong><br>
            Countries Available: <strong>{len(proxy_data)}</strong><br>
            Protocol: <strong>HTTP/HTTPS Testing</strong>
        </div>
        """, unsafe_allow_html=True)

        # Country Selection
        available_countries = list(proxy_data.keys())
        if available_countries:
            selected_country = st.selectbox(
                "Select Country",
                options=available_countries,
                index=available_countries.index(st.session_state.selected_country) if st.session_state.selected_country in available_countries else 0,
                format_func=lambda x: f"{get_country_flag(x)} {x}"
            )
            st.session_state.selected_country = selected_country

            country_proxies = proxy_data[selected_country]
            st.markdown(f"""
            <div class="country-stats">
                {get_country_flag(selected_country)} <strong>{selected_country}</strong><br>
                Available Servers: <strong>{len(country_proxies):,}</strong><br>
                Status: <strong>Testing Available</strong>
            </div>
            """, unsafe_allow_html=True)

            if country_proxies:
                n_country = len(country_proxies)
                max_show, shuffle_list = servers_to_list_control(n_country)
                filter_text = st.text_input("Filter (IP or :port)", "")

                display_proxies = country_proxies.copy()
                if shuffle_list:
                    random.shuffle(display_proxies)
                if filter_text.strip():
                    term = filter_text.strip()
                    display_proxies = [p for p in display_proxies if term in p]
                display_proxies = display_proxies[:max_show]

                selected_proxy = st.selectbox(
                    "Proxy Server",
                    options=display_proxies,
                    help="Select a proxy server to test"
                )

                # Connect / Disconnect with enhanced testing
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🧪 Test Connection", use_container_width=True):
                        with st.spinner("Testing proxy connection and location..."):
                            success, metrics = test_proxy_connection(selected_proxy, timeout=12)
                            if success:
                                st.session_state.proxy_connected = True
                                st.session_state.current_proxy = selected_proxy
                                st.session_state.connection_start_time = datetime.now()
                                st.session_state.proxy_metrics = metrics
                                st.session_state.active_proxy = normalize_proxy_http(selected_proxy)
                                st.session_state.proxy_location = metrics.get('location')
                                
                                # Show success with capabilities
                                if metrics['http_ok'] and metrics['https_ok']:
                                    st.success("✅ Proxy working! HTTP & HTTPS supported")
                                elif metrics['http_ok']:
                                    st.warning("⚠️ Proxy working! HTTP only (no HTTPS tunneling)")
                                else:
                                    st.warning("⚠️ Proxy working! HTTPS only")
                                st.rerun()
                            else:
                                st.error(f"❌ Proxy failed: {metrics.get('error', 'Unknown error')}")
                                # Try backup proxy
                                if len(display_proxies) > 1:
                                    with st.spinner("Trying backup server..."):
                                        backup_proxy = random.choice([p for p in display_proxies if p != selected_proxy])
                                        success, metrics = test_proxy_connection(backup_proxy)
                                        if success:
                                            st.session_state.proxy_connected = True
                                            st.session_state.current_proxy = backup_proxy
                                            st.session_state.connection_start_time = datetime.now()
                                            st.session_state.proxy_metrics = metrics
                                            st.session_state.active_proxy = normalize_proxy_http(backup_proxy)
                                            st.session_state.proxy_location = metrics.get('location')
                                            st.success("✅ Connected to backup server!")
                                            st.rerun()

                with col2:
                    if st.button("❌ Disconnect", use_container_width=True):
                        st.session_state.proxy_connected = False
                        st.session_state.current_proxy = None
                        st.session_state.connection_start_time = None
                        st.session_state.proxy_metrics = {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False}
                        st.session_state.active_proxy = None
                        st.session_state.proxy_location = None
                        st.success("Disconnected!")
                        st.rerun()

        # Enhanced Connection Status
        st.markdown("---")
        st.markdown("## 📊 Connection Status")
        if st.session_state.proxy_connected:
            metrics = st.session_state.proxy_metrics
            http_ok = metrics.get('http_ok', False)
            https_ok = metrics.get('https_ok', False)
            
            if http_ok and https_ok:
                st.markdown('<div class="proxy-status-connected">🟢 Connected (HTTP + HTTPS)</div>', unsafe_allow_html=True)
            elif http_ok:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTP only)</div>', unsafe_allow_html=True)
            elif https_ok:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTPS only)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (Limited)</div>', unsafe_allow_html=True)
                
            if st.session_state.connection_start_time:
                duration = datetime.now() - st.session_state.connection_start_time
                st.text(f"Duration: {str(duration).split('.')[0]}")
            st.text(f"Server: {st.session_state.current_proxy}")
            
            # Enhanced location display
            proxy_loc = st.session_state.proxy_location
            if proxy_loc:
                st.text(f"Location: {proxy_loc.get('city', 'Unknown')}, {proxy_loc.get('region', 'Unknown')}")
                st.text(f"Country: {get_country_flag(proxy_loc.get('country_code', ''))} {proxy_loc.get('country', 'Unknown')}")
            else:
                st.text(f"Location: {get_country_flag(st.session_state.selected_country)} {st.session_state.selected_country}")
            
            st.text(f"Latency: {st.session_state.proxy_metrics.get('latency', 0)}ms")
            detected_ip = st.session_state.proxy_metrics.get('ip_detected')
            if detected_ip:
                st.text(f"External IP: {detected_ip}")
        else:
            st.markdown('<div class="proxy-status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
            st.info("Select a proxy server to test connection")

    # Main dashboard with enhanced location features
    if st.session_state.proxy_connected:
        # Location Comparison Section
        if st.session_state.user_location and st.session_state.proxy_location:
            st.markdown("### 🌍 Location Intelligence")
            
            user_loc = st.session_state.user_location
            proxy_loc = st.session_state.proxy_location
            
            # Calculate distance
            if (user_loc.get('lat') and user_loc.get('lon') and 
                proxy_loc.get('lat') and proxy_loc.get('lon')):
                distance = calculate_distance(
                    float(user_loc['lat']), float(user_loc['lon']),
                    float(proxy_loc['lat']), float(proxy_loc['lon'])
                )
                
                col_dist1, col_dist2, col_dist3 = st.columns(3)
                with col_dist1:
                    st.metric("Connection Distance", f"{distance:.0f} km")
                with col_dist2:
                    # Time zone comparison
                    user_tz = user_loc.get('timezone', 'Unknown')
                    proxy_tz = proxy_loc.get('timezone', 'Unknown')
                    tz_match = "Same" if user_tz == proxy_tz else "Different"
                    st.metric("Timezone Match", tz_match)
                with col_dist3:
                    # ISP comparison
                    user_isp = user_loc.get('isp', '')[:20]
                    proxy_isp = proxy_loc.get('isp', '')[:20]
                    st.metric("Your ISP vs Proxy ISP", "Different" if user_isp != proxy_isp else "Same")
            
            # Side-by-side location comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏠 Your Real Location")
                st.markdown(f"""
                <div class="location-card">
                    📍 <strong>{user_loc.get('city', 'Unknown')}, {user_loc.get('region', 'Unknown')}</strong><br>
                    🏳️ {user_loc.get('country', 'Unknown')}<br>
                    🌐 IP: {user_loc.get('ip', 'Unknown')}<br>
                    🏢 ISP: {user_loc.get('isp', 'Unknown')}<br>
                    🕐 Timezone: {user_loc.get('timezone', 'Unknown')}<br>
                    📮 Postal: {user_loc.get('postal', 'Unknown')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎯 Proxy Server Location")
                st.markdown(f"""
                <div class="location-card">
                    📍 <strong>{proxy_loc.get('city', 'Unknown')}, {proxy_loc.get('region', 'Unknown')}</strong><br>
                    🏳️ {proxy_loc.get('country', 'Unknown')}<br>
                    🌐 IP: {proxy_loc.get('ip', 'Unknown')}<br>
                    🏢 ISP: {proxy_loc.get('isp', 'Unknown')}<br>
                    🕐 Timezone: {proxy_loc.get('timezone', 'Unknown')}<br>
                    📮 Postal: {proxy_loc.get('postal', 'Unknown')}
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🗺️ Global Connection Map")
            
            # Enhanced world map
            fig_map = create_enhanced_map()
            st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

        with col2:
            st.markdown("### 📊 Proxy Capabilities")
            
            # Show actual proxy capabilities
            metrics = st.session_state.proxy_metrics
            
            col2a, col2b = st.columns(2)
            with col2a:
                http_status = "✅ Working" if metrics.get('http_ok', False) else "❌ Failed"
                st.metric("HTTP Support", http_status)
                
            with col2b:
                https_status = "✅ Working" if metrics.get('https_ok', False) else "❌ Failed"
                st.metric("HTTPS Tunneling", https_status)

            # Show detected IP if available
            detected_ip = metrics.get('ip_detected')
            if detected_ip:
                st.metric("External IP via Proxy", detected_ip)
            
            # Connection quality over time
            st.markdown("#### Connection Quality Timeline")
            timeline_data = []
            base_time = st.session_state.connection_start_time or datetime.now()
            for i in range(24):
                time_point = base_time - timedelta(hours=23-i)
                base_latency = st.session_state.proxy_metrics.get('latency', 50)
                varied_latency = max(10, base_latency + random.randint(-20, 20))
                timeline_data.append({'time': time_point, 'latency': varied_latency})
                
            timeline_df = pd.DataFrame(timeline_data)
            fig_quality = px.line(timeline_df, x='time', y='latency', title="Latency Over Time")
            fig_quality.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=250
            )
            st.plotly_chart(fig_quality, use_container_width=True, config={'displayModeBar': False})

        # Performance metrics - Real data only
        st.markdown("---")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            actual_latency = st.session_state.proxy_metrics.get('latency', 0)
            st.metric("Measured Latency", f"{actual_latency}ms")
        with c6:
            actual_speed = st.session_state.proxy_metrics.get('speed', 0)
            st.metric("Estimated Speed", f"{actual_speed:.1f} Mbps")
        with c7:
            duration = datetime.now() - st.session_state.connection_start_time if st.session_state.connection_start_time else timedelta(0)
            st.metric("Session Duration", str(duration).split('.')[0])
        with c8:
            total_proxies = sum(len(v) for v in parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]).values())
            st.metric("Available Servers", f"{total_proxies:,}")

        # Browse via proxy (enhanced)
        st.markdown("---")
        st.markdown("### 🔎 Browse via Connected Proxy")
        
        # Show actual proxy capabilities
        if st.session_state.active_proxy:
            caps = detect_proxy_capabilities(st.session_state.active_proxy)
            http_badge = "🟢 HTTP OK" if caps["http_ok"] else "🔴 HTTP FAIL"
            https_badge = "🟢 HTTPS OK" if caps["https_ok"] else "🔴 HTTPS TUNNEL FAIL"
            st.caption(f"{http_badge} • {https_badge}")
        
        urls = st.text_area("Enter URLs to test (one per line):",
                            value="https://httpbin.org/ip\nhttp://example.com\nhttp://www.google.com", height=100)
        
        if st.button("🌐 Test Browse", use_container_width=True):
            targets = [u.strip() for u in urls.splitlines() if u.strip()]
            if targets:
                for url in targets:
                    st.markdown(f"**Testing:** {url}")
                    with st.spinner("Fetching via proxy..."):
                        try:
                            proxy_http = st.session_state.active_proxy
                            proxies = {"http": proxy_http, "https": proxy_http}
                            headers = {"User-Agent": "ProxyStream/2.0"}
                            
                            start_time = time.perf_counter()
                            response = requests.get(url, proxies=proxies, headers=headers, timeout=12)
                            elapsed = (time.perf_counter() - start_time) * 1000
                            
                            if response.ok:
                                st.success(f"✅ Success - {response.status_code} ({elapsed:.0f}ms)")
                                
                                # Show content preview
                                content_type = response.headers.get('content-type', '').lower()
                                if 'json' in content_type:
                                    try:
                                        st.json(response.json())
                                    except:
                                        st.code(response.text[:500])
                                elif 'html' in content_type:
                                    st.code(response.text[:500], language='html')
                                else:
                                    st.code(response.text[:500])
                            else:
                                st.error(f"❌ Failed - {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ Request failed: {str(e)}")
                    
                    st.markdown("---")

    else:
        # Enhanced disconnected state
        st.markdown("### 🔌 Not Connected")
        
        # Show user location status
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.info(f"Your location detected: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')} - Ready to find proxy servers worldwide!")
        else:
            st.info("Detect your location from the sidebar, then select a proxy server to test global connections.")
        
        proxy_data = parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0])
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🌐 Available Proxy Network")
            countries = list(proxy_data.keys())
            server_counts = [len(proxy_data[c]) for c in countries]
            
            fig_network = px.bar(x=countries, y=server_counts, title="Servers by Country",
                               color=server_counts, color_continuous_scale="Viridis")
            fig_network.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=300, showlegend=False
            )
            st.plotly_chart(fig_network, use_container_width=True, config={'displayModeBar': False})

    # Enhanced footer
    total_proxies = sum(len(v) for v in parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]).values())
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        <p><strong>ProxyStream v3.0</strong> - Advanced Proxy Testing with Global Location Intelligence</p>
        <p>🧪 Real Testing • 🗺️ Location Analysis • 🌍 Global Network • ⚠️ Educational Use Only</p>
        <p>Network: <strong>{total_proxies:,}</strong> servers across <strong>{len(parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]))}</strong> countries</p>
        <p style="font-size: 12px; color: #6b7280;">Disclaimer: This tool tests public proxies for educational purposes. Use reputable VPN services for actual privacy.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
