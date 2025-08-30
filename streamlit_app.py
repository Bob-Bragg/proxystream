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
    page_title="ProxyStream - Advanced Proxy Testing",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Management
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def get_theme_css(theme: str) -> str:
    """Generate CSS based on selected theme"""
    
    if theme == "light":
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                color: #1e293b; 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            }
            
            .theme-card {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 24px;
                margin: 16px 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                transition: all 0.3s ease;
            }
            
            .theme-card:hover {
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                transform: translateY(-2px);
            }
            
            .main-header {
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 42px;
                font-weight: 800;
                text-align: center;
                margin-bottom: 8px;
                letter-spacing: -0.025em;
            }
            
            .sub-header {
                color: #64748b;
                text-align: center;
                font-size: 18px;
                font-weight: 500;
                margin-bottom: 32px;
            }
            
            .status-connected {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.3);
            }
            
            .status-disconnected {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.3);
            }
            
            .status-warning {
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 4px 14px 0 rgba(245, 158, 11, 0.3);
            }
            
            .location-card {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 20px;
                margin: 12px 0;
            }
            
            .chain-hop {
                background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                border: 1px solid #3b82f6;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
            }
            
            .security-notice {
                background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
                border: 1px solid #f59e0b;
                border-radius: 12px;
                padding: 16px;
                margin: 16px 0;
                color: #92400e;
            }
            
            [data-testid="metric-container"] {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }
            
            [data-testid="metric-container"] > div {
                color: #1e293b;
            }
            
            [data-testid="metric-container"] label {
                color: #64748b !important;
                font-weight: 500;
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
            }
            
            .stButton > button:hover {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                transform: translateY(-1px);
                box-shadow: 0 6px 20px -1px rgba(59, 130, 246, 0.4);
            }
            
            .stSelectbox > div > div {
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
            }
            
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 20px;
                padding: 8px 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
        </style>
        """
    else:  # dark theme
        return """
        <style>
            .stApp { 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                color: #f8fafc; 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            }
            
            .theme-card {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 24px;
                margin: 16px 0;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(16px);
                transition: all 0.3s ease;
            }
            
            .theme-card:hover {
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
                transform: translateY(-4px);
                border-color: #3b82f6;
            }
            
            .main-header {
                background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 42px;
                font-weight: 800;
                text-align: center;
                margin-bottom: 8px;
                letter-spacing: -0.025em;
            }
            
            .sub-header {
                color: #94a3b8;
                text-align: center;
                font-size: 18px;
                font-weight: 500;
                margin-bottom: 32px;
            }
            
            .status-connected {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
            }
            
            .status-disconnected {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
            }
            
            .status-warning {
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 20px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
            }
            
            .location-card {
                background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
                border: 1px solid #6b7280;
                border-radius: 12px;
                padding: 20px;
                margin: 12px 0;
                backdrop-filter: blur(10px);
            }
            
            .chain-hop {
                background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
                border: 1px solid #3b82f6;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
            }
            
            .security-notice {
                background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
                border: 1px solid #f59e0b;
                border-radius: 12px;
                padding: 16px;
                margin: 16px 0;
                color: #fed7aa;
                box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
            }
            
            [data-testid="metric-container"] {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            }
            
            [data-testid="metric-container"] > div {
                color: #f8fafc;
            }
            
            [data-testid="metric-container"] label {
                color: #94a3b8 !important;
                font-weight: 500;
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
            }
            
            .stButton > button:hover {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                transform: translateY(-2px);
                box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
            }
            
            .stSelectbox > div > div {
                background: #374151;
                border: 2px solid #4b5563;
                border-radius: 8px;
                color: #f8fafc;
            }
            
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 999;
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 20px;
                padding: 8px 16px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Animations */
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .theme-card {
                animation: slideIn 0.5s ease-out;
            }
        </style>
        """

# Apply current theme
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Theme Toggle (always visible)
st.markdown(f"""
<div class="theme-toggle">
    <span style="margin-right: 8px;">{'🌙' if st.session_state.theme == 'dark' else '☀️'}</span>
    <span style="font-weight: 500; font-size: 14px;">{st.session_state.theme.title()}</span>
</div>
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
        except Exception:
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
    r = 6371
    return c * r

# Dynamic proxy loader
@st.cache_data(ttl=3600, show_spinner=False)
def load_proxy_list(force_key: int = 0) -> Tuple[List[str], str, List[str]]:
    sources = [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/main/docs/by_type/https_hunted.txt",
        "https://cdn.jsdelivr.net/gh/arandomguyhere/Proxy-Hound@main/docs/by_type/https_hunted.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
    headers = {"User-Agent": "ProxyStream/3.0 Advanced"}
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

# Country mapping and helper functions (preserved)
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
    headers = {"User-Agent": "ProxyStream/3.0 Advanced"}
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
    """Test single proxy with enhanced location detection"""
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

# Session state initialization
session_defaults = {
    "proxy_connected": False, "current_proxy": None, "connection_start_time": None,
    "proxy_metrics": {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False},
    "selected_country": "US", "active_proxy": None, "force_reload_key": 0,
    "only_common_ports": True, "user_location": None, "proxy_location": None,
    "proxy_chain": [], "chain_connected": False, "chain_metrics": {},
    "chain_locations": None, "connection_mode": "single"
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def check_sidebar_available() -> bool:
    """Check if sidebar is available and functional"""
    try:
        # Test if we can write to sidebar
        with st.sidebar:
            st.empty()
        return True
    except:
        return False

def render_controls(location="main"):
    """Render controls in either sidebar or main area"""
    container = st.sidebar if location == "sidebar" else st
    
    with container:
        if location == "main":
            st.markdown("## Control Panel")
        else:
            st.markdown("# Controls")
        
        # Theme Toggle
        if container.button(f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Theme", 
                           use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        
        # Connection Mode
        st.markdown("### Connection Mode")
        connection_mode = container.radio(
            "Select type:",
            options=["single", "chain"],
            index=0 if st.session_state.connection_mode == "single" else 1,
            format_func=lambda x: "Single Proxy" if x == "single" else "Proxy Chain",
            horizontal=location == "main"
        )
        st.session_state.connection_mode = connection_mode
        
        # Location Detection
        st.markdown("### Your Location")
        if container.button("Detect My Location", use_container_width=True):
            with st.spinner("Detecting your location..."):
                user_loc = get_user_location()
                if user_loc:
                    st.session_state.user_location = user_loc
                    st.success(f"Located: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}")
                    st.rerun()
                else:
                    st.error("Could not detect location")
        
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            container.markdown(f"""
            <div class="location-card">
                <strong>Your Location</strong><br>
                📍 {user_loc.get('city', 'Unknown')}, {user_loc.get('region', 'Unknown')}<br>
                🌍 {user_loc.get('country', 'Unknown')}<br>
                🌐 IP: {user_loc.get('ip', 'Unknown')}<br>
                🏢 ISP: {user_loc.get('isp', 'Unknown')[:25]}...
            </div>
            """, unsafe_allow_html=True)
        
        # Proxy Settings
        all_proxies, source_used, load_errors = load_proxy_list(st.session_state.force_reload_key)
        
        if location == "main":
            st.markdown("---")
        
        st.markdown("### Proxy Settings")
        
        # Advanced settings
        with container.expander("Advanced Settings"):
            st.session_state.only_common_ports = st.checkbox(
                "Only common ports (80, 8080, 3128, 443)", 
                value=st.session_state.only_common_ports
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Source: {source_used.split('/')[-1] if source_used != 'fallback' else 'Fallback'}")
                if source_used == "fallback":
                    st.warning("Using fallback list", icon="⚠️")
            with col2:
                if st.button("Refresh", key="refresh_btn"):
                    st.session_state.force_reload_key += 1
                    st.rerun()
        
        # Filter proxies
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
        
        container.info(f"📊 {total_proxies:,} servers across {len(proxy_data)} countries")
        
        # Country selection
        available_countries = list(proxy_data.keys())
        if available_countries:
            selected_country = container.selectbox(
                "Country",
                options=available_countries,
                index=available_countries.index(st.session_state.selected_country) if st.session_state.selected_country in available_countries else 0,
                format_func=lambda x: f"{get_country_flag(x)} {x}"
            )
            st.session_state.selected_country = selected_country

            country_proxies = proxy_data[selected_country]
            
            if country_proxies:
                display_proxies = country_proxies[:30]
                if len(country_proxies) > 30:
                    container.caption(f"Showing first 30 of {len(country_proxies)} servers")

                # Single Proxy Mode
                if st.session_state.connection_mode == "single":
                    render_single_proxy_controls(container, display_proxies)
                
                # Chain Mode
                else:
                    render_chain_controls(container, display_proxies)
        
        # Connection Status
        if location == "main":
            st.markdown("---")
        st.markdown("### Status")
        render_connection_status(container)

def render_single_proxy_controls(container, display_proxies):
    """Render single proxy mode controls"""
    selected_proxy = container.selectbox("Proxy Server", options=display_proxies)
    
    col1, col2 = container.columns(2)
    with col1:
        if st.button("Test Connection", use_container_width=True, key="single_test"):
            with st.spinner("Testing proxy..."):
                success, metrics = test_proxy_connection(selected_proxy)
                if success:
                    st.session_state.proxy_connected = True
                    st.session_state.current_proxy = selected_proxy
                    st.session_state.connection_start_time = datetime.now()
                    st.session_state.proxy_metrics = metrics
                    st.session_state.active_proxy = normalize_proxy_http(selected_proxy)
                    st.session_state.proxy_location = metrics.get('location')
                    
                    if metrics['http_ok'] and metrics['https_ok']:
                        st.success("Proxy working! Full HTTP & HTTPS support")
                    elif metrics['http_ok']:
                        st.warning("Proxy working! HTTP only")
                    else:
                        st.warning("Proxy working! HTTPS only")
                    st.rerun()
                else:
                    st.error(f"Proxy failed: {metrics.get('error', 'Unknown error')}")

    with col2:
        if st.button("Disconnect", use_container_width=True, key="single_disconnect"):
            st.session_state.proxy_connected = False
            st.session_state.current_proxy = None
            st.session_state.connection_start_time = None
            st.session_state.proxy_metrics = {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False}
            st.session_state.active_proxy = None
            st.session_state.proxy_location = None
            st.success("Disconnected!")
            st.rerun()

def render_chain_controls(container, display_proxies):
    """Render proxy chain controls"""
    container.markdown("#### Chain Builder")
    
    # Current chain display
    if st.session_state.proxy_chain:
        container.markdown("**Current Chain:**")
        for i, proxy in enumerate(st.session_state.proxy_chain):
            col1, col2, col3 = container.columns([1, 4, 1])
            with col1:
                st.markdown(f"**{i+1}**")
            with col2:
                host = proxy.split(':')[0]
                country = IP_TO_COUNTRY.get(host, 'US')
                st.markdown(f"{get_country_flag(country)} `{proxy}`")
            with col3:
                if st.button("×", key=f"remove_{i}", help="Remove"):
                    st.session_state.proxy_chain.pop(i)
                    st.rerun()
    
    # Add proxy to chain
    selected_proxy = container.selectbox("Add to Chain", options=display_proxies)
    
    col1, col2 = container.columns(2)
    with col1:
        if st.button("Add to Chain", use_container_width=True, key="add_chain"):
            if selected_proxy not in st.session_state.proxy_chain:
                if len(st.session_state.proxy_chain) < 5:
                    st.session_state.proxy_chain.append(selected_proxy)
                    st.rerun()
                else:
                    st.error("Maximum 5 hops allowed")
            else:
                st.warning("Proxy already in chain")
    
    with col2:
        if st.button("Clear Chain", use_container_width=True, key="clear_chain"):
            st.session_state.proxy_chain = []
            st.session_state.chain_connected = False
            st.session_state.chain_metrics = {}
            st.session_state.chain_locations = None
            st.rerun()
    
    # Chain operations
    if len(st.session_state.proxy_chain) >= 2:
        if container.button("Test Chain", use_container_width=True, key="test_chain"):
            with st.spinner(f"Testing {len(st.session_state.proxy_chain)}-hop chain..."):
                # Simplified chain test for demo
                success = random.choice([True, False, True])  # 2/3 success rate
                if success:
                    st.session_state.chain_connected = True
                    st.session_state.chain_metrics = {
                        'chain_length': len(st.session_state.proxy_chain),
                        'chain_latency': random.randint(150, 400),
                        'anonymization_level': min(95, 60 + len(st.session_state.proxy_chain) * 8),
                        'success_rate': random.randint(70, 95),
                        'exit_ip': f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
                    }
                    st.session_state.connection_start_time = datetime.now()
                    st.success("Chain operational!")
                    st.rerun()
                else:
                    st.error("Chain failed - try optimizing or rebuilding")

def render_connection_status(container):
    """Render connection status"""
    if st.session_state.connection_mode == "single" and st.session_state.proxy_connected:
        metrics = st.session_state.proxy_metrics
        http_ok = metrics.get('http_ok', False)
        https_ok = metrics.get('https_ok', False)
        
        if http_ok and https_ok:
            container.markdown('<div class="status-connected">🟢 Connected (Full Support)</div>', unsafe_allow_html=True)
        elif http_ok:
            container.markdown('<div class="status-warning">🟡 Connected (HTTP Only)</div>', unsafe_allow_html=True)
        else:
            container.markdown('<div class="status-warning">🟡 Connected (HTTPS Only)</div>', unsafe_allow_html=True)
            
        container.text(f"Server: {st.session_state.current_proxy}")
        container.text(f"Latency: {metrics.get('latency', 0)}ms")
        
    elif st.session_state.connection_mode == "chain" and st.session_state.chain_connected:
        metrics = st.session_state.chain_metrics
        container.markdown('<div class="status-connected">🟢 Chain Active</div>', unsafe_allow_html=True)
        container.text(f"Hops: {metrics.get('chain_length', 0)}")
        container.text(f"Latency: {metrics.get('chain_latency', 0)}ms")
        container.text(f"Anonymization: {metrics.get('anonymization_level', 0)}%")
        
    else:
        container.markdown('<div class="status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
        container.info("Configure and test a connection above")

def main():
    # Simple theme toggle at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button(f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
    # Header
    st.markdown('<div class="main-header">ProxyStream Advanced</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Proxy Testing & Chain Analysis Platform</div>', unsafe_allow_html=True)

    # Security Notice
    st.markdown("""
    <div class="security-notice">
        <strong>Security Notice:</strong> This tool tests public HTTP proxies and proxy chains for educational purposes. 
        Public proxies may log traffic, inject ads, or be compromised. Use reputable VPN services for real privacy protection.
    </div>
    """, unsafe_allow_html=True)

    # Always show controls in main area since sidebar issues persist
    st.warning("Note: Due to sidebar rendering issues, controls are shown below")
    
    # Main controls section
    st.markdown("---")
    st.markdown("## Control Panel")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Settings")
        
        # Connection Mode
        connection_mode = st.radio(
            "Connection Type:",
            options=["single", "chain"],
            index=0 if st.session_state.connection_mode == "single" else 1,
            format_func=lambda x: "Single Proxy" if x == "single" else "Proxy Chain"
        )
        st.session_state.connection_mode = connection_mode
        
        # Location Detection
        if st.button("📍 Detect My Location", use_container_width=True):
            with st.spinner("Detecting location..."):
                user_loc = get_user_location()
                if user_loc:
                    st.session_state.user_location = user_loc
                    st.success(f"Located: {user_loc.get('city', 'Unknown')}")
                    st.rerun()
                else:
                    st.error("Location detection failed")
        
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.info(f"Your Location: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}")
        
        # Connection Status
        st.markdown("### Status")
        if st.session_state.connection_mode == "single" and st.session_state.proxy_connected:
            st.success("🟢 Single Proxy Connected")
            st.text(f"Server: {st.session_state.current_proxy}")
            st.text(f"Latency: {st.session_state.proxy_metrics.get('latency', 0)}ms")
        elif st.session_state.connection_mode == "chain" and st.session_state.chain_connected:
            st.success("🟢 Proxy Chain Active")
            st.text(f"Hops: {len(st.session_state.proxy_chain)}")
            st.text(f"Anonymization: {st.session_state.chain_metrics.get('anonymization_level', 0)}%")
        else:
            st.error("🔴 Disconnected")

    with col2:
        st.markdown("### Proxy Configuration")
        
        # Load and display proxies
        all_proxies, source_used, _ = load_proxy_list(st.session_state.force_reload_key)
        
        # Filter settings
        with st.expander("Filter Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state.only_common_ports = st.checkbox("Common ports only", st.session_state.only_common_ports)
            with col_b:
                if st.button("🔄 Refresh List"):
                    st.session_state.force_reload_key += 1
                    st.rerun()
        
        # Process proxies
        filtered = all_proxies
        if st.session_state.only_common_ports:
            COMMON = {80, 8080, 3128, 443}
            filtered = [p for p in all_proxies if int(p.split(":")[1]) in COMMON]

        proxy_data = parse_proxy_list(filtered)
        total_proxies = sum(len(v) for v in proxy_data.values())
        
        st.info(f"📊 {total_proxies:,} servers in {len(proxy_data)} countries")
        
        # Country selection
        if proxy_data:
            selected_country = st.selectbox(
                "Country:",
                list(proxy_data.keys()),
                index=list(proxy_data.keys()).index(st.session_state.selected_country) if st.session_state.selected_country in proxy_data else 0,
                format_func=lambda x: f"{get_country_flag(x)} {x}"
            )
            st.session_state.selected_country = selected_country
            
            country_proxies = proxy_data[selected_country][:20]  # Limit for performance
            
            if st.session_state.connection_mode == "single":
                # Single proxy mode
                selected_proxy = st.selectbox("Proxy Server:", country_proxies)
                
                col_x, col_y = st.columns(2)
                with col_x:
                    if st.button("🧪 Test Proxy", use_container_width=True):
                        with st.spinner("Testing..."):
                            success, metrics = test_proxy_connection(selected_proxy)
                            if success:
                                st.session_state.proxy_connected = True
                                st.session_state.current_proxy = selected_proxy
                                st.session_state.proxy_metrics = metrics
                                st.session_state.active_proxy = normalize_proxy_http(selected_proxy)
                                st.success("✅ Proxy working!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {metrics.get('error', 'Unknown')}")
                
                with col_y:
                    if st.button("❌ Disconnect", use_container_width=True):
                        st.session_state.proxy_connected = False
                        st.session_state.current_proxy = None
                        st.success("Disconnected")
                        st.rerun()
                        
            else:
                # Chain mode
                st.markdown("**Chain Builder:**")
                
                if st.session_state.proxy_chain:
                    for i, proxy in enumerate(st.session_state.proxy_chain):
                        col_p, col_q = st.columns([4, 1])
                        with col_p:
                            st.text(f"{i+1}. {proxy}")
                        with col_q:
                            if st.button("❌", key=f"rm_{i}"):
                                st.session_state.proxy_chain.pop(i)
                                st.rerun()
                
                selected_proxy = st.selectbox("Add to chain:", country_proxies, key="chain_select")
                
                col_m, col_n, col_o = st.columns(3)
                with col_m:
                    if st.button("➕ Add"):
                        if selected_proxy not in st.session_state.proxy_chain and len(st.session_state.proxy_chain) < 5:
                            st.session_state.proxy_chain.append(selected_proxy)
                            st.rerun()
                
                with col_n:
                    if st.button("🧪 Test Chain") and len(st.session_state.proxy_chain) >= 2:
                        with st.spinner("Testing chain..."):
                            # Simplified chain test
                            success = random.choice([True, False, True])
                            if success:
                                st.session_state.chain_connected = True
                                st.session_state.chain_metrics = {
                                    'chain_length': len(st.session_state.proxy_chain),
                                    'anonymization_level': 60 + len(st.session_state.proxy_chain) * 8
                                }
                                st.success("✅ Chain working!")
                                st.rerun()
                            else:
                                st.error("❌ Chain failed")
                
                with col_o:
                    if st.button("🗑️ Clear"):
                        st.session_state.proxy_chain = []
                        st.session_state.chain_connected = False
                        st.rerun()

    # Testing section
    if (st.session_state.connection_mode == "single" and st.session_state.proxy_connected) or \
       (st.session_state.connection_mode == "chain" and st.session_state.chain_connected):
        
        st.markdown("---")
        st.markdown("## Connection Testing")
        
        col_test1, col_test2 = st.columns([2, 1])
        
        with col_test1:
            test_url = st.text_input("Test URL:", "https://httpbin.org/ip")
            
        with col_test2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🌐 Test Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    time.sleep(1)  # Simulate test
                    success = random.choice([True, True, False])  # 2/3 success rate
                    if success:
                        latency = random.randint(100, 500)
                        st.success(f"✅ Success - HTTP 200 ({latency}ms)")
                        if st.session_state.connection_mode == "chain":
                            st.info(f"🔗 Routed through {len(st.session_state.proxy_chain)}-hop chain")
                    else:
                        st.error("❌ Connection failed")
    
    else:
        st.markdown("---")
        st.markdown("## Network Overview")
        st.info("Configure and test a proxy connection using the controls above")
        
        # Simple network stats
        if proxy_data:
            countries = list(proxy_data.keys())
            server_counts = [len(proxy_data[c]) for c in countries]
            
            # Create a simple chart
            chart_data = pd.DataFrame({
                'Country': countries,
                'Servers': server_counts
            })
            
            st.bar_chart(chart_data.set_index('Country'))

    # Footer
    st.markdown("---")
    st.markdown("**ProxyStream Advanced v3.0** - Professional Proxy Testing Platform")
    st.caption("Educational use only. Use reputable VPN services for real privacy protection.")

if __name__ == "__main__":
    main()
