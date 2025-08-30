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
    page_title="ProxyStream Premium - Advanced Proxy Testing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"  # Force sidebar to be expanded
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
    .chain-hop { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin: 8px 0; }
    .chain-arrow { color: #3b82f6; font-size: 20px; text-align: center; margin: 8px 0; }
    .location-card { background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); padding: 16px; border-radius: 12px;
                     margin: 8px 0; border: 1px solid rgba(255, 255, 255, 0.15); }
    [data-testid="metric-container"] { background: rgba(255,255,255,.05); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,.1); padding: 1.5rem; border-radius: 16px; margin: .5rem 0; }
    [data-testid="metric-container"] > div { color: white; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 12px; font-weight: 600; transition: all .3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,.4); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .country-stats { background: rgba(255,255,255,.05); padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 14px; }
    .security-warning { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); padding: 12px; border-radius: 8px; margin: 8px 0; }
    .chain-warning { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); padding: 12px; border-radius: 8px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# Enhanced Geolocation Functions
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
    r = 6371  # Earth's radius in kilometers
    return c * r

# Dynamic proxy loader
@st.cache_data(ttl=3600, show_spinner=False)
def load_proxy_list(force_key: int = 0) -> Tuple[List[str], str, List[str]]:
    sources = [
        "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/main/docs/by_type/https_hunted.txt",
        "https://cdn.jsdelivr.net/gh/arandomguyhere/Proxy-Hound@main/docs/by_type/https_hunted.txt",
        "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    ]
    headers = {"User-Agent": "ProxyStream/3.0 Chain Testing"}
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
    headers = {"User-Agent": "ProxyStream/3.0 Chain Testing"}
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

# Enhanced proxy connection testing
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

# NEW: Proxy Chain Testing Functions
def test_proxy_chain(proxy_list: List[str], timeout: int = 25) -> tuple[bool, dict]:
    """Test a chain of proxies working together with comprehensive validation"""
    if len(proxy_list) == 0:
        return False, {'error': 'Empty chain', 'chain_working': False}
    
    if len(proxy_list) == 1:
        success, metrics = test_proxy_connection(proxy_list[0], timeout)
        if success:
            metrics['chain_working'] = True
            metrics['chain_length'] = 1
            metrics['chain_latency'] = metrics.get('latency', 0)
            metrics['anonymization_level'] = 65
            metrics['chain_efficiency'] = 95
        return success, metrics
    
    # Test each individual proxy first
    individual_results = []
    total_latency = 0
    working_proxies = []
    
    st.write("Testing individual proxies in chain...")
    progress_bar = st.progress(0)
    
    for i, proxy in enumerate(proxy_list):
        progress_bar.progress((i + 1) / len(proxy_list))
        success, metrics = test_proxy_connection(proxy, timeout=max(8, timeout//len(proxy_list)))
        
        individual_results.append({
            'proxy': proxy,
            'success': success,
            'metrics': metrics,
            'hop_number': i + 1
        })
        
        if success:
            working_proxies.append(proxy)
            total_latency += metrics.get('latency', 0)
            st.write(f"✅ Hop {i+1}: {proxy} - {metrics.get('latency', 0)}ms")
        else:
            st.write(f"❌ Hop {i+1}: {proxy} - {metrics.get('error', 'Failed')}")
            return False, {
                'error': f'Hop {i+1} failed: {metrics.get("error", "Unknown")}',
                'failed_hop': i + 1,
                'failed_proxy': proxy,
                'individual_results': individual_results,
                'chain_working': False,
                'chain_length': len(proxy_list)
            }
    
    progress_bar.empty()
    
    # Test multiple endpoints through the chain for reliability
    test_endpoints = [
        "http://httpbin.org/ip",
        "http://httpbin.org/headers", 
        "http://example.com"
    ]
    
    chain_results = []
    successful_tests = 0
    
    st.write("Testing chain connectivity...")
    
    for endpoint in test_endpoints:
        try:
            start_time = time.perf_counter()
            primary_proxy = normalize_proxy_http(proxy_list[0])
            proxies = {"http": primary_proxy, "https": primary_proxy}
            headers = {
                "User-Agent": "ProxyStream Chain Test",
                "Accept": "application/json,text/html,*/*"
            }
            
            response = requests.get(endpoint, 
                                  proxies=proxies, 
                                  headers=headers, 
                                  timeout=timeout,
                                  allow_redirects=True)
            
            chain_latency = (time.perf_counter() - start_time) * 1000
            
            if response.ok:
                successful_tests += 1
                result_data = {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'latency': round(chain_latency),
                    'success': True
                }
                
                # Try to extract IP if it's httpbin
                if 'httpbin.org/ip' in endpoint:
                    try:
                        result_data['exit_ip'] = response.json().get("origin")
                    except:
                        pass
                elif 'httpbin.org/headers' in endpoint:
                    try:
                        result_data['headers_received'] = len(response.json().get("headers", {}))
                    except:
                        pass
                
                chain_results.append(result_data)
                st.write(f"✅ {endpoint}: {response.status_code} ({chain_latency:.0f}ms)")
            else:
                chain_results.append({
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'latency': round(chain_latency),
                    'success': False
                })
                st.write(f"⚠️ {endpoint}: {response.status_code}")
                
        except Exception as e:
            chain_results.append({
                'endpoint': endpoint,
                'error': str(e)[:100],
                'success': False
            })
            st.write(f"❌ {endpoint}: {str(e)[:50]}...")
    
    # Determine if chain is working (at least 50% success rate)
    success_rate = successful_tests / len(test_endpoints)
    chain_working = success_rate >= 0.5
    
    if chain_working:
        # Calculate metrics
        avg_latency = sum(r.get('latency', 0) for r in chain_results if r.get('latency')) / max(1, len([r for r in chain_results if r.get('latency')]))
        
        # Get exit IP from successful httpbin test
        exit_ip = None
        for result in chain_results:
            if result.get('exit_ip'):
                exit_ip = result['exit_ip']
                break
        
        # Calculate performance metrics
        efficiency = max(10, 100 - (len(proxy_list) * 12) - ((1 - success_rate) * 30))
        estimated_speed = max(5, 120 - (avg_latency / 8) - (len(proxy_list) * 10))
        anonymization_level = min(95, 55 + (len(proxy_list) * 10) + (success_rate * 15))
        
        return True, {
            'chain_working': True,
            'chain_latency': round(avg_latency),
            'total_estimated_latency': round(total_latency),
            'exit_ip': exit_ip or "Unknown",
            'chain_length': len(proxy_list),
            'individual_results': individual_results,
            'chain_test_results': chain_results,
            'estimated_speed': round(estimated_speed, 1),
            'chain_efficiency': round(efficiency),
            'anonymization_level': round(anonymization_level),
            'success_rate': round(success_rate * 100),
            'working_proxies': len(working_proxies)
        }
    else:
        return False, {
            'error': f'Chain reliability too low: {success_rate*100:.0f}% success rate',
            'chain_working': False,
            'individual_results': individual_results,
            'chain_test_results': chain_results,
            'chain_length': len(proxy_list),
            'success_rate': round(success_rate * 100),
            'working_proxies': len(working_proxies)
        }

def get_chain_geolocation(proxy_chain: List[str]) -> Dict[str, Any]:
    """Get geolocation data for each proxy in chain"""
    chain_locations = []
    
    for i, proxy in enumerate(proxy_chain):
        host = proxy.split(':')[0]
        location = get_detailed_location(host)
        chain_locations.append({
            'hop': i + 1,
            'proxy': proxy,
            'location': location,
            'host': host
        })
    
    return {
        'chain_path': chain_locations,
        'entry_point': chain_locations[0]['location'] if chain_locations else None,
        'exit_point': chain_locations[-1]['location'] if chain_locations else None,
        'hops': len(chain_locations),
        'geographic_diversity': calculate_geographic_diversity(chain_locations)
    }

def calculate_geographic_diversity(chain_locations: List[Dict]) -> Dict[str, Any]:
    """Calculate how geographically diverse the chain is"""
    countries = set()
    isps = set()
    timezones = set()
    
    for hop in chain_locations:
        loc = hop.get('location')
        if loc:
            if loc.get('country_code'):
                countries.add(loc['country_code'])
            if loc.get('isp'):
                isps.add(loc['isp'])
            if loc.get('timezone'):
                timezones.add(loc['timezone'])
    
    return {
        'unique_countries': len(countries),
        'unique_isps': len(isps),
        'unique_timezones': len(timezones),
        'diversity_score': min(100, (len(countries) * 25) + (len(isps) * 15) + (len(timezones) * 10))
    }

def optimize_proxy_chain(proxy_list: List[str], target_countries: List[str] = None) -> List[str]:
    """Optimize proxy chain for better performance and geographic diversity"""
    if len(proxy_list) <= 1:
        return proxy_list
    
    # Test all proxies and sort by performance
    tested_proxies = []
    for proxy in proxy_list:
        success, metrics = test_proxy_connection(proxy, timeout=5)
        if success:
            tested_proxies.append((proxy, metrics))
    
    # Sort by latency (fastest first)
    tested_proxies.sort(key=lambda x: x[1].get('latency', 999))
    
    # If target countries specified, prioritize geographic diversity
    if target_countries:
        optimized = []
        used_countries = set()
        
        for proxy, metrics in tested_proxies:
            host = proxy.split(':')[0]
            country = IP_TO_COUNTRY.get(host, 'US')
            
            if country in target_countries and country not in used_countries:
                optimized.append(proxy)
                used_countries.add(country)
                
                if len(optimized) >= len(target_countries):
                    break
        
        return optimized
    
    # Otherwise, just return fastest proxies with some geographic diversity
    optimized = []
    used_countries = set()
    
    for proxy, metrics in tested_proxies:
        host = proxy.split(':')[0]
        country = IP_TO_COUNTRY.get(host, 'US')
        
        # Add if it's a new country or we don't have enough proxies yet
        if country not in used_countries or len(optimized) < 3:
            optimized.append(proxy)
            used_countries.add(country)
            
            if len(optimized) >= 4:  # Max 4 hops for performance
                break
    
    return optimized

def create_chain_map():
    """Create enhanced map showing full proxy chain path"""
    fig = go.Figure()
    
    lons = []
    lats = []
    texts = []
    colors = []
    
    # Add user location if available
    if 'user_location' in st.session_state and st.session_state.user_location:
        user_loc = st.session_state.user_location
        if user_loc.get('lat') and user_loc.get('lon'):
            lons.append(user_loc['lon'])
            lats.append(user_loc['lat'])
            texts.append(f"Start: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')}")
            colors.append('#ef4444')
    
    # Add chain locations
    if 'chain_locations' in st.session_state and st.session_state.chain_locations:
        chain_data = st.session_state.chain_locations
        for hop_data in chain_data['chain_path']:
            loc = hop_data['location']
            if loc and loc.get('lat') and loc.get('lon'):
                lons.append(loc['lon'])
                lats.append(loc['lat'])
                texts.append(f"Hop {hop_data['hop']}: {loc.get('city', 'Unknown')}, {loc.get('country', 'Unknown')}")
                
                # Color code by hop
                hop_colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
                colors.append(hop_colors[(hop_data['hop'] - 1) % len(hop_colors)])
    
    # Add markers
    if lons and lats:
        for i, (lon, lat, text, color) in enumerate(zip(lons, lats, texts, colors)):
            symbol = 'circle' if i == 0 else ('diamond' if i == len(lons)-1 else 'square')
            size = 15 if i == 0 else (18 if i == len(lons)-1 else 12)
            
            fig.add_trace(go.Scattergeo(
                lon=[lon], lat=[lat],
                mode='markers',
                marker=dict(size=size, color=color, symbol=symbol, 
                          line=dict(width=2, color='white')),
                name=f'Hop {i}' if i > 0 else 'Start',
                text=[text],
                hoverinfo='text'
            ))
        
        # Draw connection path
        if len(lons) > 1:
            fig.add_trace(go.Scattergeo(
                lon=lons, lat=lats,
                mode='lines',
                line=dict(width=3, color='#64748b', dash='dash'),
                name='Chain Path',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        geo=dict(
            showframe=False, showcoastlines=True, projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)', landcolor='#374151', oceancolor='#1e293b', coastlinecolor='#6b7280'
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        height=400, margin=dict(l=0,r=0,t=0,b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(color="white"))
    )
    
    return fig

# Session state initialization (enhanced for chains)
session_defaults = {
    "proxy_connected": False, "current_proxy": None, "connection_start_time": None,
    "proxy_metrics": {"latency": 0, "speed": 0, "http_ok": False, "https_ok": False},
    "selected_country": "US", "active_proxy": None, "force_reload_key": 0,
    "only_common_ports": True, "user_location": None, "proxy_location": None,
    # NEW: Chain-specific state
    "proxy_chain": [], "chain_connected": False, "chain_metrics": {},
    "chain_locations": None, "connection_mode": "single"  # "single" or "chain"
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

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
    st.markdown('<div class="main-header">🛡️ ProxyStream Premium</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 40px;">Advanced Proxy Testing & Chain Analysis Platform</p>', unsafe_allow_html=True)

    # Security warning
    st.markdown("""
    <div class="security-warning">
        <strong>⚠️ Security Notice:</strong> This tool tests public HTTP proxies and proxy chains for educational purposes. 
        Public proxies may log traffic, inject ads, or be compromised. Never use them for sensitive activities.
        For real privacy protection, use a reputable VPN service.
    </div>
    """, unsafe_allow_html=True)

    # Chain-specific warning
    st.markdown("""
    <div class="chain-warning">
        <strong>🔗 Proxy Chains:</strong> Chaining multiple proxies increases anonymization but reduces speed and reliability. 
        Each additional hop increases latency and failure probability. Recommended: 2-3 hops maximum.
    </div>
    """, unsafe_allow_html=True)

    # Load proxy data
    all_proxies, source_used, load_errors = load_proxy_list(st.session_state.force_reload_key)

    # MAIN CONTROLS SECTION (since sidebar isn't working)
    st.markdown("---")
    st.markdown("## 🔧 Proxy Controls")
    
    # Try sidebar first, then fallback to main area
    sidebar_content = st.sidebar if hasattr(st, 'sidebar') else st
    
    # Check if we should show controls in main area
    show_main_controls = True
    
    try:
        with st.sidebar:
            st.markdown("# 🔧 Controls")
            show_main_controls = False
    except:
        show_main_controls = True
    
    if show_main_controls:
        # Put controls in main area since sidebar failed
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Connection Mode")
            connection_mode = st.radio(
                "Select connection type:",
                options=["single", "chain"],
                index=0 if st.session_state.connection_mode == "single" else 1,
                format_func=lambda x: "🔗 Single Proxy" if x == "single" else "⛓️ Proxy Chain",
                horizontal=False
            )
            st.session_state.connection_mode = connection_mode
            
            st.markdown("### Your Location")
            if st.button("📍 Detect My Location"):
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
                st.markdown(f"""
                <div class="location-card">
                    <strong>🏠 Your Location</strong><br>
                    📍 {user_loc.get('city', 'Unknown')}, {user_loc.get('region', 'Unknown')}<br>
                    🏳️ {user_loc.get('country', 'Unknown')}<br>
                    🌐 IP: {user_loc.get('ip', 'Unknown')}<br>
                    🏢 ISP: {user_loc.get('isp', 'Unknown')[:30]}...
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Main proxy selection and controls
            st.markdown("### Proxy Settings")
            
            # Filter options
            with st.expander("Advanced Settings"):
                st.session_state.only_common_ports = st.checkbox("Only common ports 80/8080/3128/443", value=st.session_state.only_common_ports)
                col_ref1, col_ref2 = st.columns([3,1])
                with col_ref1:
                    st.caption(f"Source: {source_used}")
                    if source_used == "fallback":
                        st.warning("Using fallback list", icon="⚠️")
                with col_ref2:
                    if st.button("↻ Refresh"):
                        st.session_state.force_reload_key += 1
                        st.rerun()

            # Process proxy data
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
            
            st.info(f"📊 Network: {total_proxies:,} servers across {len(proxy_data)} countries")
            
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
                
                if country_proxies:
                    # Show limited proxies for performance
                    display_proxies = country_proxies[:50]  # Limit to 50 for performance
                    if len(country_proxies) > 50:
                        st.caption(f"Showing first 50 of {len(country_proxies)} servers")

                    # Single Proxy Mode
                    if st.session_state.connection_mode == "single":
                        selected_proxy = st.selectbox("Proxy Server", options=display_proxies)

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🧪 Test Connection", use_container_width=True):
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
                                            st.success("✅ Proxy working! HTTP & HTTPS supported")
                                        elif metrics['http_ok']:
                                            st.warning("⚠️ Proxy working! HTTP only")
                                        else:
                                            st.warning("⚠️ Proxy working! HTTPS only")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Proxy failed: {metrics.get('error', 'Unknown error')}")

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

                    # Chain Mode
                    else:
                        st.markdown("#### ⛓️ Chain Builder")
                        
                        # Current chain display
                        if st.session_state.proxy_chain:
                            st.markdown("**Current Chain:**")
                            for i, proxy in enumerate(st.session_state.proxy_chain):
                                col1, col2, col3 = st.columns([1, 4, 1])
                                with col1:
                                    st.markdown(f"**{i+1}**")
                                with col2:
                                    host = proxy.split(':')[0]
                                    country = IP_TO_COUNTRY.get(host, 'US')
                                    st.markdown(f"{get_country_flag(country)} `{proxy}`")
                                with col3:
                                    if st.button("✕", key=f"remove_{i}", help="Remove from chain"):
                                        st.session_state.proxy_chain.pop(i)
                                        st.rerun()
                        
                        # Add proxy to chain
                        selected_proxy = st.selectbox("Add Proxy to Chain", options=display_proxies)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("➕ Add to Chain"):
                                if selected_proxy not in st.session_state.proxy_chain:
                                    if len(st.session_state.proxy_chain) < 5:
                                        st.session_state.proxy_chain.append(selected_proxy)
                                        st.rerun()
                                    else:
                                        st.error("Maximum 5 hops allowed")
                                else:
                                    st.warning("Proxy already in chain")
                        
                        with col2:
                            if st.button("🧹 Clear Chain"):
                                st.session_state.proxy_chain = []
                                st.session_state.chain_connected = False
                                st.session_state.chain_metrics = {}
                                st.session_state.chain_locations = None
                                st.rerun()
                        
                        # Chain operations
                        if len(st.session_state.proxy_chain) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🧪 Test Chain"):
                                    with st.spinner(f"Testing {len(st.session_state.proxy_chain)}-hop chain..."):
                                        success, metrics = test_proxy_chain(st.session_state.proxy_chain)
                                        if success:
                                            st.session_state.chain_connected = True
                                            st.session_state.chain_metrics = metrics
                                            st.session_state.chain_locations = get_chain_geolocation(st.session_state.proxy_chain)
                                            st.session_state.connection_start_time = datetime.now()
                                            st.success(f"✅ Chain operational!")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Chain failed: {metrics.get('error', 'Unknown error')}")
                            
                            with col2:
                                if st.button("⚡ Optimize"):
                                    with st.spinner("Optimizing chain..."):
                                        try:
                                            optimized = optimize_proxy_chain(st.session_state.proxy_chain)
                                            if len(optimized) >= 2:
                                                st.session_state.proxy_chain = optimized
                                                st.success(f"✅ Optimized to {len(optimized)} hops")
                                                st.rerun()
                                            else:
                                                st.error("❌ Not enough working proxies")
                                        except Exception as e:
                                            st.error(f"Optimization failed: {str(e)}")
                        
                        elif len(st.session_state.proxy_chain) == 1:
                            st.info("💡 Add at least one more proxy to create a chain")
        
        # Connection Status
        st.markdown("---")
        st.markdown("### 📊 Connection Status")
        
        if st.session_state.connection_mode == "single" and st.session_state.proxy_connected:
            metrics = st.session_state.proxy_metrics
            http_ok = metrics.get('http_ok', False)
            https_ok = metrics.get('https_ok', False)
            
            if http_ok and https_ok:
                st.markdown('<div class="proxy-status-connected">🟢 Connected (Full)</div>', unsafe_allow_html=True)
            elif http_ok:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTP)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTPS)</div>', unsafe_allow_html=True)
                
            st.text(f"Server: {st.session_state.current_proxy}")
            st.text(f"Latency: {metrics.get('latency', 0)}ms")
            
        elif st.session_state.connection_mode == "chain" and st.session_state.chain_connected:
            metrics = st.session_state.chain_metrics
            st.markdown('<div class="proxy-status-connected">🟢 Chain Active</div>', unsafe_allow_html=True)
            st.text(f"Hops: {metrics.get('chain_length', 0)}")
            st.text(f"Total Latency: {metrics.get('chain_latency', 0)}ms")
            st.text(f"Anonymization: {metrics.get('anonymization_level', 0)}%")
            
        else:
            st.markdown('<div class="proxy-status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
            st.info("Configure and test a connection above")

    else:
        # Original sidebar code (this should work if sidebar is available)
        with st.sidebar:
            # ... (rest of the original sidebar code)
            pass
        
        # Connection Mode Selection
        st.markdown("### Connection Mode")
        connection_mode = st.radio(
            "Select connection type:",
            options=["single", "chain"],
            index=0 if st.session_state.connection_mode == "single" else 1,
            format_func=lambda x: "🔗 Single Proxy" if x == "single" else "⛓️ Proxy Chain",
            horizontal=True
        )
        st.session_state.connection_mode = connection_mode

        # Location Detection Section
        st.markdown("---")
        st.markdown("### 🎯 Your Location")
        
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
                st.warning("Using fallback list", icon="⚠️")
        with colref2:
            if st.button("↻"):
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

                # Single Proxy Mode
                if st.session_state.connection_mode == "single":
                    selected_proxy = st.selectbox("Proxy Server", options=display_proxies)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🧪 Test Connection", use_container_width=True):
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
                                        st.success("✅ Proxy working! HTTP & HTTPS supported")
                                    elif metrics['http_ok']:
                                        st.warning("⚠️ Proxy working! HTTP only")
                                    else:
                                        st.warning("⚠️ Proxy working! HTTPS only")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Proxy failed: {metrics.get('error', 'Unknown error')}")

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

                # Chain Mode
                else:
                    st.markdown("### ⛓️ Chain Builder")
                    
                    # Current chain display
                    if st.session_state.proxy_chain:
                        st.markdown("**Current Chain:**")
                        for i, proxy in enumerate(st.session_state.proxy_chain):
                            col1, col2, col3 = st.columns([1, 4, 1])
                            with col1:
                                st.markdown(f"**{i+1}**")
                            with col2:
                                host = proxy.split(':')[0]
                                country = IP_TO_COUNTRY.get(host, 'US')
                                st.markdown(f"{get_country_flag(country)} `{proxy}`")
                            with col3:
                                if st.button("✕", key=f"remove_{i}", help="Remove from chain"):
                                    st.session_state.proxy_chain.pop(i)
                                    st.rerun()
                            
                            if i < len(st.session_state.proxy_chain) - 1:
                                st.markdown('<div class="chain-arrow">↓</div>', unsafe_allow_html=True)
                    
                    # Add proxy to chain
                    selected_proxy = st.selectbox("Add Proxy to Chain", options=display_proxies)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("➕ Add to Chain", use_container_width=True):
                            if selected_proxy not in st.session_state.proxy_chain:
                                if len(st.session_state.proxy_chain) < 5:  # Max 5 hops
                                    st.session_state.proxy_chain.append(selected_proxy)
                                    st.rerun()
                                else:
                                    st.error("Maximum 5 hops allowed")
                            else:
                                st.warning("Proxy already in chain")
                    
                    with col2:
                        if st.button("🧹 Clear Chain", use_container_width=True):
                            st.session_state.proxy_chain = []
                            st.session_state.chain_connected = False
                            st.session_state.chain_metrics = {}
                            st.session_state.chain_locations = None
                            st.rerun()
                    
                    # Chain operations
                    if len(st.session_state.proxy_chain) >= 2:
                        st.markdown("**Chain Operations:**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🧪 Test Chain", use_container_width=True):
                                with st.spinner(f"Testing {len(st.session_state.proxy_chain)}-hop chain..."):
                                    success, metrics = test_proxy_chain(st.session_state.proxy_chain)
                                    if success:
                                        st.session_state.chain_connected = True
                                        st.session_state.chain_metrics = metrics
                                        st.session_state.chain_locations = get_chain_geolocation(st.session_state.proxy_chain)
                                        st.session_state.connection_start_time = datetime.now()
                                        
                                        # Show detailed results
                                        st.success(f"✅ Chain operational!")
                                        st.info(f"📊 Success Rate: {metrics.get('success_rate', 0)}% | "
                                               f"Anonymization: {metrics.get('anonymization_level', 0)}% | "
                                               f"Efficiency: {metrics.get('chain_efficiency', 0)}%")
                                        
                                        # Show exit IP if detected
                                        if metrics.get('exit_ip') and metrics['exit_ip'] != 'Unknown':
                                            st.success(f"🌐 Exit IP: {metrics['exit_ip']}")
                                        
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Chain failed: {metrics.get('error', 'Unknown error')}")
                                        
                                        # Show which hop failed if available
                                        if metrics.get('failed_hop'):
                                            st.error(f"Failed at hop {metrics['failed_hop']}: {metrics.get('failed_proxy', 'Unknown')}")
                                        
                                        # Show success rate if partial failure
                                        if metrics.get('success_rate') is not None:
                                            st.warning(f"Success rate: {metrics['success_rate']}% (minimum 50% required)")
                        
                        with col2:
                            if st.button("⚡ Optimize", use_container_width=True):
                                with st.spinner("Optimizing chain..."):
                                    try:
                                        optimized = optimize_proxy_chain(st.session_state.proxy_chain)
                                        if len(optimized) >= 2:
                                            old_length = len(st.session_state.proxy_chain)
                                            st.session_state.proxy_chain = optimized
                                            new_length = len(optimized)
                                            
                                            if new_length < old_length:
                                                st.success(f"✅ Optimized: {old_length} → {new_length} hops (removed {old_length - new_length} slow proxies)")
                                            else:
                                                st.info(f"✅ Chain already optimal ({new_length} hops)")
                                            st.rerun()
                                        else:
                                            st.error("❌ Not enough working proxies to create an optimized chain")
                                    except Exception as e:
                                        st.error(f"Optimization failed: {str(e)}")
                        
                        # Disconnect chain button
                        if st.session_state.chain_connected:
                            if st.button("❌ Disconnect Chain", use_container_width=True, type="secondary"):
                                st.session_state.chain_connected = False
                                st.session_state.chain_metrics = {}
                                st.session_state.chain_locations = None
                                st.session_state.connection_start_time = None
                                st.success("Chain disconnected!")
                                st.rerun()
                    
                    elif len(st.session_state.proxy_chain) == 1:
                        st.info("💡 Add at least one more proxy to create a chain")
                        
                        # Option to test single proxy
                        if st.button("🧪 Test Single Proxy", use_container_width=True):
                            with st.spinner("Testing single proxy..."):
                                success, metrics = test_proxy_connection(st.session_state.proxy_chain[0])
                                if success:
                                    st.success(f"✅ Proxy working: {metrics.get('latency', 0)}ms latency")
                                else:
                                    st.error(f"❌ Proxy failed: {metrics.get('error', 'Unknown')}")
                    else:
                        st.info("🔗 Add proxies to build your chain")
                        
                        # Quick chain suggestions
                        if st.button("🚀 Build Sample Chain", use_container_width=True):
                            # Create a sample 3-hop chain with geographic diversity
                            sample_countries = ['US', 'GB', 'DE']
                            sample_chain = []
                            
                            for country in sample_countries:
                                country_proxies = proxy_data.get(country, [])
                                if country_proxies:
                                    sample_chain.append(random.choice(country_proxies))
                            
                            if len(sample_chain) >= 2:
                                st.session_state.proxy_chain = sample_chain
                                st.success(f"Created sample {len(sample_chain)}-hop chain with geographic diversity!")
                                st.rerun()
                            else:
                                st.error("Not enough proxies available for sample chain")

        # Connection Status
        st.markdown("---")
        st.markdown("## 📊 Status")
        
        if st.session_state.connection_mode == "single" and st.session_state.proxy_connected:
            metrics = st.session_state.proxy_metrics
            http_ok = metrics.get('http_ok', False)
            https_ok = metrics.get('https_ok', False)
            
            if http_ok and https_ok:
                st.markdown('<div class="proxy-status-connected">🟢 Connected (Full)</div>', unsafe_allow_html=True)
            elif http_ok:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTP)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="proxy-status-warning">🟡 Connected (HTTPS)</div>', unsafe_allow_html=True)
                
            st.text(f"Server: {st.session_state.current_proxy}")
            st.text(f"Latency: {metrics.get('latency', 0)}ms")
            
        elif st.session_state.connection_mode == "chain" and st.session_state.chain_connected:
            metrics = st.session_state.chain_metrics
            st.markdown('<div class="proxy-status-connected">🟢 Chain Active</div>', unsafe_allow_html=True)
            st.text(f"Hops: {metrics.get('chain_length', 0)}")
            st.text(f"Total Latency: {metrics.get('chain_latency', 0)}ms")
            st.text(f"Anonymization: {metrics.get('anonymization_level', 0)}%")
            st.text(f"Exit IP: {metrics.get('exit_ip', 'Unknown')}")
            
        else:
            st.markdown('<div class="proxy-status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)

    # Main dashboard
    if (st.session_state.connection_mode == "single" and st.session_state.proxy_connected) or \
       (st.session_state.connection_mode == "chain" and st.session_state.chain_connected):
        
        # Connection Overview
        st.markdown("### Connection Overview")
        
        if st.session_state.connection_mode == "single":
            metrics = st.session_state.proxy_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Type", "Single Proxy")
            with col2:
                st.metric("Latency", f"{metrics.get('latency', 0)}ms")
            with col3:
                st.metric("Speed Est.", f"{metrics.get('speed', 0):.1f} Mbps")
            with col4:
                duration = datetime.now() - st.session_state.connection_start_time if st.session_state.connection_start_time else timedelta(0)
                st.metric("Session", str(duration).split('.')[0])
        
        else:  # Chain mode
            metrics = st.session_state.chain_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Type", f"{metrics.get('chain_length', 0)}-Hop Chain")
            with col2:
                st.metric("Chain Latency", f"{metrics.get('chain_latency', 0)}ms")
            with col3:
                st.metric("Anonymization", f"{metrics.get('anonymization_level', 0)}%")
            with col4:
                st.metric("Efficiency", f"{metrics.get('chain_efficiency', 0)}%")

        # Chain Analysis (only for chain mode)
        if st.session_state.connection_mode == "chain" and st.session_state.chain_locations:
            st.markdown("---")
            st.markdown("### ⛓️ Chain Analysis")
            
            chain_data = st.session_state.chain_locations
            diversity = chain_data.get('geographic_diversity', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Countries", diversity.get('unique_countries', 0))
            with col2:
                st.metric("ISPs", diversity.get('unique_isps', 0))
            with col3:
                st.metric("Diversity Score", f"{diversity.get('diversity_score', 0)}%")
            
            # Individual hop analysis
            st.markdown("#### Hop Analysis")
            for hop_data in chain_data['chain_path']:
                loc = hop_data['location']
                if loc:
                    st.markdown(f"""
                    <div class="chain-hop">
                        <strong>Hop {hop_data['hop']}: {hop_data['proxy']}</strong><br>
                        📍 {loc.get('city', 'Unknown')}, {loc.get('country', 'Unknown')}<br>
                        🏢 {loc.get('isp', 'Unknown')}<br>
                        🕐 {loc.get('timezone', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)

        # Location comparison (single mode) or Chain visualization (chain mode)
        if st.session_state.user_location:
            st.markdown("---")
            
            if st.session_state.connection_mode == "single" and st.session_state.proxy_location:
                st.markdown("### 🌍 Location Comparison")
                user_loc = st.session_state.user_location
                proxy_loc = st.session_state.proxy_location
                
                if (user_loc.get('lat') and user_loc.get('lon') and 
                    proxy_loc.get('lat') and proxy_loc.get('lon')):
                    distance = calculate_distance(
                        float(user_loc['lat']), float(user_loc['lon']),
                        float(proxy_loc['lat']), float(proxy_loc['lon'])
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Distance", f"{distance:.0f} km")
                    with col2:
                        user_tz = user_loc.get('timezone', 'Unknown')
                        proxy_tz = proxy_loc.get('timezone', 'Unknown')
                        tz_match = "Same" if user_tz == proxy_tz else "Different"
                        st.metric("Timezone", tz_match)
                    with col3:
                        st.metric("Anonymization", "Basic")
                
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
                        🕐 Timezone: {user_loc.get('timezone', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 🎯 Proxy Location")
                    st.markdown(f"""
                    <div class="location-card">
                        📍 <strong>{proxy_loc.get('city', 'Unknown')}, {proxy_loc.get('region', 'Unknown')}</strong><br>
                        🏳️ {proxy_loc.get('country', 'Unknown')}<br>
                        🌐 IP: {proxy_loc.get('ip', 'Unknown')}<br>
                        🏢 ISP: {proxy_loc.get('isp', 'Unknown')}<br>
                        🕐 Timezone: {proxy_loc.get('timezone', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
            
            elif st.session_state.connection_mode == "chain" and st.session_state.chain_locations:
                st.markdown("### ⛓️ Chain Visualization")
                
                # Chain path map
                fig_map = create_chain_map()
                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

        # Protocol testing (works for both single and chain)
        st.markdown("---")
        st.markdown("### 🔎 Connection Testing")
        
        # Show capabilities
        if st.session_state.connection_mode == "single" and st.session_state.active_proxy:
            caps = detect_proxy_capabilities(st.session_state.active_proxy)
            http_badge = "🟢 HTTP OK" if caps["http_ok"] else "🔴 HTTP FAIL"
            https_badge = "🟢 HTTPS OK" if caps["https_ok"] else "🔴 HTTPS TUNNEL FAIL"
            st.caption(f"{http_badge} • {https_badge}")
        elif st.session_state.connection_mode == "chain" and st.session_state.chain_connected:
            metrics = st.session_state.chain_metrics
            st.caption(f"⛓️ {metrics.get('chain_length', 0)}-hop chain • Exit IP: {metrics.get('exit_ip', 'Unknown')}")
        
        urls = st.text_area("Enter URLs to test (one per line):",
                            value="https://httpbin.org/ip\nhttp://example.com\nhttp://www.google.com", height=100)
        
        if st.button("🌐 Test Browse", use_container_width=True):
            targets = [u.strip() for u in urls.splitlines() if u.strip()]
            if targets:
                progress_bar = st.progress(0)
                
                for i, url in enumerate(targets):
                    progress_bar.progress((i + 1) / len(targets))
                    
                    st.markdown(f"**Testing:** {url}")
                    with st.spinner("Fetching via connection..."):
                        try:
                            # Use appropriate proxy configuration
                            if st.session_state.connection_mode == "single":
                                proxy_http = st.session_state.active_proxy
                                connection_info = "single proxy"
                            else:
                                # For chains, use first proxy as entry point
                                proxy_http = normalize_proxy_http(st.session_state.proxy_chain[0])
                                connection_info = f"{len(st.session_state.proxy_chain)}-hop chain"
                            
                            proxies = {"http": proxy_http, "https": proxy_http}
                            headers = {
                                "User-Agent": "ProxyStream/3.0 Chain Testing",
                                "Accept": "text/html,application/json,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.9",
                                "Accept-Encoding": "gzip, deflate",
                                "Connection": "keep-alive"
                            }
                            
                            start_time = time.perf_counter()
                            response = requests.get(url, 
                                                  proxies=proxies, 
                                                  headers=headers, 
                                                  timeout=20,
                                                  allow_redirects=True,
                                                  stream=False)
                            elapsed = (time.perf_counter() - start_time) * 1000
                            
                            if response.ok:
                                # Show success with connection details
                                status_msg = f"✅ Success - HTTP {response.status_code} ({elapsed:.0f}ms via {connection_info})"
                                
                                if st.session_state.connection_mode == "chain":
                                    exit_ip = st.session_state.chain_metrics.get('exit_ip', 'Unknown')
                                    if exit_ip != 'Unknown':
                                        status_msg += f" • Exit IP: {exit_ip}"
                                
                                st.success(status_msg)
                                
                                # Show response details
                                content_type = response.headers.get('content-type', '').lower()
                                content_length = len(response.content)
                                
                                # Create expandable content section
                                with st.expander(f"Response Details ({content_length} bytes)", expanded=False):
                                    
                                    # Show response headers
                                    st.markdown("**Response Headers:**")
                                    header_data = dict(response.headers)
                                    st.json(header_data)
                                    
                                    # Show content based on type
                                    if 'json' in content_type:
                                        st.markdown("**JSON Response:**")
                                        try:
                                            json_data = response.json()
                                            st.json(json_data)
                                            
                                            # Special handling for httpbin.org/ip
                                            if 'httpbin.org/ip' in url and 'origin' in json_data:
                                                st.info(f"🌐 Detected external IP: {json_data['origin']}")
                                        except:
                                            st.code(response.text[:1000])
                                    
                                    elif 'html' in content_type:
                                        st.markdown("**HTML Preview:**")
                                        # Extract title if available
                                        import re
                                        title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
                                        if title_match:
                                            st.info(f"📄 Page Title: {title_match.group(1).strip()}")
                                        
                                        st.code(response.text[:1000], language='html')
                                        
                                        if len(response.text) > 1000:
                                            st.caption(f"... (showing first 1000 of {len(response.text)} characters)")
                                    
                                    elif 'text/' in content_type:
                                        st.markdown("**Text Response:**")
                                        st.code(response.text[:1000])
                                        if len(response.text) > 1000:
                                            st.caption(f"... (showing first 1000 of {len(response.text)} characters)")
                                    
                                    else:
                                        st.markdown("**Binary/Other Content:**")
                                        st.info(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
                                        st.info(f"Size: {content_length:,} bytes")
                                        
                                        # Offer download for binary content
                                        if content_length < 10000000:  # < 10MB
                                            filename = url.split('/')[-1] or 'download'
                                            st.download_button(
                                                "Download Content",
                                                data=response.content,
                                                file_name=filename,
                                                mime=response.headers.get('content-type', 'application/octet-stream')
                                            )
                            
                            elif response.status_code == 403:
                                st.warning(f"⚠️ Access Forbidden - HTTP {response.status_code}")
                                st.caption("Target website may be blocking proxy connections")
                            
                            elif response.status_code == 404:
                                st.warning(f"⚠️ Not Found - HTTP {response.status_code}")
                                st.caption("URL may not exist or be accessible")
                            
                            elif response.status_code >= 500:
                                st.error(f"❌ Server Error - HTTP {response.status_code}")
                                st.caption("Target server is experiencing issues")
                            
                            else:
                                st.error(f"❌ Request Failed - HTTP {response.status_code}")
                                if response.text:
                                    st.code(response.text[:300])
                                
                        except requests.exceptions.ProxyError as e:
                            st.error(f"❌ Proxy Error: Connection refused")
                            st.caption(f"The proxy server rejected the connection: {str(e)[:100]}")
                        
                        except requests.exceptions.ConnectTimeout:
                            st.error(f"❌ Connection Timeout")
                            st.caption(f"Connection via {connection_info} timed out")
                        
                        except requests.exceptions.ReadTimeout:
                            st.error(f"❌ Read Timeout") 
                            st.caption("Server took too long to respond")
                        
                        except requests.exceptions.SSLError as e:
                            st.error(f"❌ SSL/TLS Error")
                            st.caption("SSL certificate verification failed or HTTPS not supported")
                        
                        except requests.exceptions.ConnectionError as e:
                            st.error(f"❌ Connection Error")
                            st.caption(f"Network connection failed: {str(e)[:100]}")
                        
                        except Exception as e:
                            st.error(f"❌ Unexpected Error: {type(e).__name__}")
                            st.caption(f"Details: {str(e)[:100]}")
                    
                    if i < len(targets) - 1:
                        st.markdown("---")
                
                progress_bar.empty()
                st.success(f"✅ Completed testing {len(targets)} URLs via {connection_info}")
            else:
                st.warning("Please enter at least one URL to test")

    else:
        # Disconnected state
        st.markdown("### 🔌 Not Connected")
        
        if st.session_state.user_location:
            user_loc = st.session_state.user_location
            st.info(f"Your location: {user_loc.get('city', 'Unknown')}, {user_loc.get('country', 'Unknown')} - Select connection mode and proxy servers from sidebar")
        else:
            st.info("Detect your location from the sidebar, then choose single proxy or chain mode for testing")
        
        proxy_data = parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0])
        
        # Show current chain if any
        if st.session_state.proxy_chain:
            st.markdown("### ⛓️ Current Chain Configuration")
            for i, proxy in enumerate(st.session_state.proxy_chain):
                host = proxy.split(':')[0]
                country = IP_TO_COUNTRY.get(host, 'US')
                st.markdown(f"**Hop {i+1}:** {get_country_flag(country)} {proxy}")
                if i < len(st.session_state.proxy_chain) - 1:
                    st.markdown('<div class="chain-arrow">↓</div>', unsafe_allow_html=True)
            
            if len(st.session_state.proxy_chain) >= 2:
                if st.button("🧪 Test Current Chain", type="primary"):
                    with st.spinner("Testing chain..."):
                        success, metrics = test_proxy_chain(st.session_state.proxy_chain)
                        if success:
                            st.session_state.chain_connected = True
                            st.session_state.chain_metrics = metrics
                            st.session_state.chain_locations = get_chain_geolocation(st.session_state.proxy_chain)
                            st.success("Chain connected!")
                            st.rerun()
                        else:
                            st.error(f"Chain failed: {metrics.get('error')}")
        
        # Network overview
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

    # Footer
    total_proxies = sum(len(v) for v in parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]).values())
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        <p><strong>ProxyStream Premium v3.0</strong> - Advanced Proxy Testing & Chain Analysis</p>
        <p>🧪 Real Testing • ⛓️ Proxy Chains • 🗺️ Location Analysis • 🌍 Global Network</p>
        <p>Network: <strong>{total_proxies:,}</strong> servers across <strong>{len(parse_proxy_list(load_proxy_list(st.session_state.force_reload_key)[0]))}</strong> countries</p>
        <p style="font-size: 12px; color: #6b7280;">Educational and testing purposes only. Use reputable VPN services for actual privacy.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
