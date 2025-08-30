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

# Professional ProxyStream Configuration
st.set_page_config(
    page_title="ProxyStream - Professional Proxy Testing",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Theme - Clean Enterprise Style
st.markdown("""
<style>
    /* Base Layout */
    .stApp { 
        background: #f8fafc; 
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }
    
    /* Headers */
    .main-header {
        color: #0f172a;
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 8px;
        text-align: left;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 32px;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        margin: 4px 0;
    }
    
    .status-connected {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-disconnected {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .status-warning {
        background: #fefce8;
        color: #a16207;
        border: 1px solid #fde68a;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .dot-green { background: #10b981; }
    .dot-red { background: #ef4444; }
    .dot-yellow { background: #f59e0b; }
    .dot-gray { background: #6b7280; }
    
    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #0f172a;
    }
    
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 500;
    }
    
    /* Tables */
    .dataframe {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Alert Boxes */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #1e40af;
    }
    
    .warning-box {
        background: #fefce8;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #a16207;
    }
    
    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #991b1b;
    }
    
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #166534;
    }
    
    /* Location Cards */
    .location-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .location-header {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 12px;
        font-size: 16px;
    }
    
    .location-detail {
        color: #64748b;
        margin: 6px 0;
        font-size: 14px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 24px;
            text-align: center;
        }
        .metric-card {
            padding: 16px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Geolocation Functions (preserved from previous version)
@st.cache_data(ttl=1800, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Earth's radius in kilometers
    
    return c * r

# Proxy loading and testing functions (preserved)
@st.cache_data(ttl=3600, show_spinner=False)
def load_proxy_list(force_key: int = 0) -> Tuple[List[str], str, List[str]]:
    sources = [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/main/docs/by_type/https_hunted.txt",
        "https://cdn.jsdelivr.net/gh/arandomguyhere/Proxy-Hound@main/docs/by_type/https_hunted.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
    headers = {"User-Agent": "ProxyStream/3.0 Professional"}
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

# Country mapping
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
    headers = {"User-Agent": "ProxyStream/3.0 Professional"}
    proxies = {"http": proxy_http_url, "https": proxy_http_url}
    
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
    
    try:
        r = requests.get("https://httpbin.org/ip", proxies=proxies, headers=headers, timeout=timeout)
        caps["https_ok"] = r.ok
        if r.ok: 
            caps["ip_https"] = r.json().get("origin")
    except Exception as e:
        caps["err_https"] = str(e)[:200]
    
    return caps

def test_proxy_connection(proxy: str, timeout: int = 10) -> tuple[bool, dict]:
    """Test proxy with enhanced location detection"""
    proxy_http = normalize_proxy_http(proxy)
    host, port = proxy.split(':')[0], int(proxy.split(':')[1])
    
    if not tcp_ping(host, port, timeout=4.0):
        return False, {
            'latency': 0, 'speed': 0, 'country': IP_TO_COUNTRY.get(host, 'US'),
            'error': 'TCP connection failed - proxy unreachable',
            'http_ok': False, 'https_ok': False, 'ip_detected': None, 'location': None
        }
    
    caps = detect_proxy_capabilities(proxy_http, timeout=timeout)
    is_working = caps["http_ok"] or caps["https_ok"]
    proxy_location = get_detailed_location(host)
    
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
        'latency': caps["latency_ms"], 'speed': round(speed_estimate, 1),
        'country': proxy_location.get('country', 'Unknown') if proxy_location else IP_TO_COUNTRY.get(host, 'US'),
        'error': caps.get("err_http", "") or caps.get("err_https", ""),
        'http_ok': caps["http_ok"], 'https_ok': caps["https_ok"],
        'ip_detected': caps.get("ip_http") or caps.get("ip_https"), 'location': proxy_location
    }

def create_professional_map():
    """Create clean, professional map visualization"""
    fig = go.Figure()
    
    # Add user location
    if 'user_location' in st.session_state and st.session_state.user_location:
        user_loc = st.session_state.user_location
        if user_loc.get('lat') and user_loc.get('lon'):
            fig.add_trace(go.Scattergeo(
                lon=[user_loc['lon']], lat=[user_loc['lat']],
                mode='markers',
                marker=dict(size=12, color='#ef4444', symbol='circle'),
                name='Your Location',
                text=[f"You: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}"],
                hoverinfo='text'
            ))
    
    # Add proxy location
    if st.session_state.proxy_connected:
        proxy_loc = st.session_state.get('proxy_location')
        if proxy_loc and proxy_loc.get('lat') and proxy_loc.get('lon'):
            fig.add_trace(go.Scattergeo(
                lon=[proxy_loc['lon']], lat=[proxy_loc['lat']],
                mode='markers',
                marker=dict(size=14, color='#2563eb', symbol='square'),
                name='Proxy Server',
                text=[f"Proxy: {proxy_loc.get('city', 'Unknown')}, {proxy_loc.get('country', 'Unknown')}"],
                hoverinfo='text'
            ))
            
            # Connection line
            if ('user_location' in st.session_state and 
                st.session_state.user_location and
                st.session_state.user_location.get('lat') and 
                st.session_state.user_location.get('lon')):
                
                user_loc = st.session_state.user_location
                fig.add_trace(go.Scattergeo(
                    lon=[user_loc['lon'], proxy_loc['lon']],
                    lat=[user_loc['lat'], proxy_loc['lat']],
                    mode='lines',
                    line=dict(width=2, color='#64748b', dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        geo=dict(
            showframe=False, showcoastlines=True, projection_type='natural earth',
            bgcolor='#f8fafc', landcolor='#e2e8f0', oceancolor='#cbd5e1', coastlinecolor='#94a3b8'
        ),
        plot_bgcolor='#f8fafc', paper_bgcolor='#f8fafc', height=300, margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False
    )
    
    return fig

# Session state initialization
session_defaults = {
    "proxy_connected": False, "current_proxy": None, "connection_start_time": None,
    "proxy_metrics": {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False},
    "selected_country": "US", "active_proxy": None, "force_reload_key": 0,
    "only_common_ports": True, "user_location": None, "proxy_location": None
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def main():
    # Professional Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown('<div class="main-header">ProxyStream Professional</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Enterprise-grade proxy testing and network analysis</div>', unsafe_allow_html=True)
    with col_header2:
        if st.button("Refresh Network", type="secondary"):
            st.session_state.force_reload_key += 1
            st.rerun()

    # Load proxy data
    all_proxies, source_used, load_errors = load_proxy_list(st.session_state.force_reload_key)

    # Professional Alert Box
    if source_used == "fallback":
        st.markdown("""
        <div class="warning-box">
            <strong>Network Notice:</strong> Using fallback proxy list. External proxy sources unavailable.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>Security Notice:</strong> This tool tests public HTTP proxies for network analysis and educational purposes. 
            Public proxies should not be used for sensitive data transmission.
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Network Configuration")

        # Location Detection
        st.markdown("#### Your Location")
        
        if st.button("Detect Location", use_container_width=True, type="primary"):
            with st.spinner("Detecting location..."):
                user_loc = get_user_location()
                if user_loc:
                    st.session_state.user_location = user_loc
                    st.success("Location detected successfully")
                    st.rerun()
                else:
                    st.error("Location detection failed")
        
        # Display user location professionally
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.markdown(f"""
            <div class="location-card">
                <div class="location-header">Your Network Location</div>
                <div class="location-detail"><strong>City:</strong> {user_loc.get('city', 'Unknown')}</div>
                <div class="location-detail"><strong>Region:</strong> {user_loc.get('region', 'Unknown')}</div>
                <div class="location-detail"><strong>Country:</strong> {user_loc.get('country', 'Unknown')}</div>
                <div class="location-detail"><strong>ISP:</strong> {user_loc.get('isp', 'Unknown')[:30]}...</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Proxy Configuration")

        # Filter options
        with st.expander("Filter Options"):
            st.session_state.only_common_ports = st.checkbox(
                "Standard ports only (80, 8080, 3128, 443)", 
                value=st.session_state.only_common_ports
            )

        # Process proxy data
        filtered = all_proxies
        if st.session_state.only_common_ports:
            COMMON = {80, 8080, 3128, 443}
            filtered = [p for p in all_proxies if int(p.split(":")[1]) in COMMON]

        proxy_data = parse_proxy_list(filtered)
        total_proxies = sum(len(v) for v in proxy_data.values())

        # Network statistics
        st.markdown(f"""
        <div class="status-card">
            <strong>Network Statistics</strong><br>
            Total Servers: <strong>{total_proxies:,}</strong><br>
            Countries: <strong>{len(proxy_data)}</strong><br>
            Source: <strong>{source_used.split('/')[-1] if source_used != 'fallback' else 'Fallback'}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Country selection
        available_countries = list(proxy_data.keys())
        if available_countries:
            selected_country = st.selectbox(
                "Target Country",
                options=available_countries,
                index=available_countries.index(st.session_state.selected_country) if st.session_state.selected_country in available_countries else 0,
                format_func=lambda x: f"{get_country_flag(x)} {x}"
            )
            st.session_state.selected_country = selected_country

            country_proxies = proxy_data[selected_country]
            
            # Proxy server selection
            if country_proxies:
                # Limit display for performance
                display_proxies = country_proxies[:50]
                if len(country_proxies) > 50:
                    st.caption(f"Showing first 50 of {len(country_proxies)} servers")
                
                selected_proxy = st.selectbox(
                    "Proxy Server",
                    options=display_proxies,
                    format_func=lambda x: f"{x} ({get_country_flag(selected_country)})"
                )

                # Connection controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Test Connection", use_container_width=True, type="primary"):
                        with st.spinner("Testing proxy..."):
                            success, metrics = test_proxy_connection(selected_proxy, timeout=12)
                            if success:
                                st.session_state.proxy_connected = True
                                st.session_state.current_proxy = selected_proxy
                                st.session_state.connection_start_time = datetime.now()
                                st.session_state.proxy_metrics = metrics
                                st.session_state.active_proxy = normalize_proxy_http(selected_proxy)
                                st.session_state.proxy_location = metrics.get('location')
                                st.success("Connection successful")
                                st.rerun()
                            else:
                                st.error(f"Connection failed: {metrics.get('error', 'Unknown error')}")

                with col2:
                    if st.button("Disconnect", use_container_width=True, type="secondary"):
                        for key in ["proxy_connected", "current_proxy", "connection_start_time", "active_proxy", "proxy_location"]:
                            st.session_state[key] = None if key != "proxy_connected" else False
                        st.session_state.proxy_metrics = {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False}
                        st.success("Disconnected")
                        st.rerun()

        # Connection Status
        st.markdown("---")
        st.markdown("#### Connection Status")
        
        if st.session_state.proxy_connected:
            metrics = st.session_state.proxy_metrics
            http_ok = metrics.get('http_ok', False)
            https_ok = metrics.get('https_ok', False)
            
            if http_ok and https_ok:
                status_html = '<div class="status-indicator status-connected"><span class="status-dot dot-green"></span>Connected (Full)</div>'
            elif http_ok:
                status_html = '<div class="status-indicator status-warning"><span class="status-dot dot-yellow"></span>Connected (HTTP)</div>'
            elif https_ok:
                status_html = '<div class="status-indicator status-warning"><span class="status-dot dot-yellow"></span>Connected (HTTPS)</div>'
            else:
                status_html = '<div class="status-indicator status-warning"><span class="status-dot dot-yellow"></span>Connected (Limited)</div>'
            
            st.markdown(status_html, unsafe_allow_html=True)
            
            # Connection details
            if st.session_state.connection_start_time:
                duration = datetime.now() - st.session_state.connection_start_time
                st.text(f"Duration: {str(duration).split('.')[0]}")
            st.text(f"Server: {st.session_state.current_proxy}")
            st.text(f"Latency: {st.session_state.proxy_metrics.get('latency', 0)}ms")
            
        else:
            st.markdown('<div class="status-indicator status-disconnected"><span class="status-dot dot-red"></span>Disconnected</div>', unsafe_allow_html=True)

    # Main Dashboard
    if st.session_state.proxy_connected:
        # Connection Overview
        st.markdown("### Connection Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Connected", delta="Active")
        with col2:
            st.metric("Latency", f"{st.session_state.proxy_metrics.get('latency', 0)}ms")
        with col3:
            st.metric("Speed Est.", f"{st.session_state.proxy_metrics.get('speed', 0):.1f} Mbps")
        with col4:
            duration = datetime.now() - st.session_state.connection_start_time if st.session_state.connection_start_time else timedelta(0)
            st.metric("Session", str(duration).split('.')[0])

        # Location Analysis
        if st.session_state.user_location and st.session_state.proxy_location:
            st.markdown("---")
            st.markdown("### Location Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_loc = st.session_state.user_location
                st.markdown(f"""
                <div class="location-card">
                    <div class="location-header">Your Real Location</div>
                    <div class="location-detail"><strong>City:</strong> {user_loc.get('city', 'Unknown')}</div>
                    <div class="location-detail"><strong>Region:</strong> {user_loc.get('region', 'Unknown')}</div>
                    <div class="location-detail"><strong>Country:</strong> {user_loc.get('country', 'Unknown')}</div>
                    <div class="location-detail"><strong>ISP:</strong> {user_loc.get('isp', 'Unknown')}</div>
                    <div class="location-detail"><strong>Timezone:</strong> {user_loc.get('timezone', 'Unknown')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                proxy_loc = st.session_state.proxy_location
                st.markdown(f"""
                <div class="location-card">
                    <div class="location-header">Proxy Server Location</div>
                    <div class="location-detail"><strong>City:</strong> {proxy_loc.get('city', 'Unknown')}</div>
                    <div class="location-detail"><strong>Region:</strong> {proxy_loc.get('region', 'Unknown')}</div>
                    <div class="location-detail"><strong>Country:</strong> {proxy_loc.get('country', 'Unknown')}</div>
                    <div class="location-detail"><strong>ISP:</strong> {proxy_loc.get('isp', 'Unknown')}</div>
                    <div class="location-detail"><strong>Timezone:</strong> {proxy_loc.get('timezone', 'Unknown')}</div>
                </div>
                """, unsafe_allow_html=True)

            # Distance calculation
            if (user_loc.get('lat') and user_loc.get('lon') and 
                proxy_loc.get('lat') and proxy_loc.get('lon')):
                distance = calculate_distance(
                    float(user_loc['lat']), float(user_loc['lon']),
                    float(proxy_loc['lat']), float(proxy_loc['lon'])
                )
                
                col_dist1, col_dist2, col_dist3 = st.columns(3)
                with col_dist1:
                    st.metric("Distance", f"{distance:.0f} km")
                with col_dist2:
                    user_tz = user_loc.get('timezone', 'Unknown')
                    proxy_tz = proxy_loc.get('timezone', 'Unknown')
                    tz_status = "Same" if user_tz == proxy_tz else "Different"
                    st.metric("Timezone", tz_status)
                with col_dist3:
                    st.metric("Connection Type", "Proxy Tunnel")

        # Network Map and Capabilities
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Network Topology")
            fig_map = create_professional_map()
            st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
            
        with col2:
            st.markdown("### Protocol Support")
            metrics = st.session_state.proxy_metrics
            
            http_status = "✅ Supported" if metrics.get('http_ok', False) else "❌ Failed"
            https_status = "✅ Supported" if metrics.get('https_ok', False) else "❌ Failed"
            
            st.markdown(f"""
            <div class="status-card">
                <div style="margin-bottom: 12px;"><strong>HTTP:</strong> {http_status}</div>
                <div style="margin-bottom: 12px;"><strong>HTTPS Tunneling:</strong> {https_status}</div>
                <div><strong>External IP:</strong> {metrics.get('ip_detected', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Testing Interface
        st.markdown("---")
        st.markdown("### Connection Testing")
        
        urls = st.text_area(
            "Test URLs (one per line):",
            value="https://httpbin.org/ip\nhttp://example.com\nhttps://www.google.com",
            height=100
        )
        
        if st.button("Run Tests", type="primary"):
            targets = [u.strip() for u in urls.splitlines() if u.strip()]
            if targets:
                for url in targets:
                    with st.expander(f"Testing: {url}", expanded=True):
                        with st.spinner("Connecting..."):
                            try:
                                proxy_http = st.session_state.active_proxy
                                proxies = {"http": proxy_http, "https": proxy_http}
                                headers = {"User-Agent": "ProxyStream/3.0 Professional"}
                                
                                start_time = time.perf_counter()
                                response = requests.get(url, proxies=proxies, headers=headers, timeout=12)
                                elapsed = (time.perf_counter() - start_time) * 1000
                                
                                if response.ok:
                                    st.markdown(f"""
                                    <div class="success-box">
                                        <strong>Success:</strong> HTTP {response.status_code} ({elapsed:.0f}ms)
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    content_type = response.headers.get('content-type', '').lower()
                                    if 'json' in content_type:
                                        try:
                                            st.json(response.json())
                                        except:
                                            st.code(response.text[:500])
                                    else:
                                        st.code(response.text[:500])
                                else:
                                    st.markdown(f"""
                                    <div class="error-box">
                                        <strong>Failed:</strong> HTTP {response.status_code}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>Request Failed:</strong> {str(e)}
                                </div>
                                """, unsafe_allow_html=True)

    else:
        # Disconnected State
        st.markdown("### Network Dashboard")
        
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.markdown(f"""
            <div class="info-box">
                Your location: <strong>{user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}</strong> 
                — Select a proxy server from the sidebar to begin network analysis.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                Click "Detect Location" in the sidebar to identify your network location, then select a proxy server for testing.
            </div>
            """, unsafe_allow_html=True)
        
        # Network overview
        proxy_data = parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0])
        
        if proxy_data:
            st.markdown("### Available Network Infrastructure")
            countries = list(proxy_data.keys())
            server_counts = [len(proxy_data[c]) for c in countries]
            
            # Create professional bar chart
            fig = px.bar(
                x=countries, y=server_counts, 
                title="Proxy Servers by Country",
                color=server_counts,
                color_continuous_scale=["#e2e8f0", "#2563eb"]
            )
            fig.update_layout(
                plot_bgcolor='#f8fafc',
                paper_bgcolor='#f8fafc',
                font_color='#1e293b',
                title_font_size=16,
                height=300,
                showlegend=False,
                xaxis_title="Country",
                yaxis_title="Available Servers"
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Professional Footer
    total_proxies = sum(len(v) for v in parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]).values())
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 14px; padding: 24px 0;">
        <p><strong>ProxyStream Professional v3.0</strong></p>
        <p>Network Testing • Location Analysis • Security Assessment</p>
        <p>Active Network: <strong>{total_proxies:,}</strong> servers across <strong>{len(parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]))}</strong> countries</p>
        <p style="font-size: 12px; margin-top: 12px;">For educational and network analysis purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
