"""
ProxyStream Advanced - Professional Proxy Testing & Chain Analysis Platform
With persistence, geolocation, and validation features
"""

import asyncio
import time
import random
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import ipaddress
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="ProxyStream Advanced",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

GEO_APIS = [
    "https://ipapi.co/{}/json/",
    "http://ip-api.com/json/{}",
    "https://ipwhois.app/json/{}"
]

COMMON_PORTS = {80, 8080, 3128, 443, 1080, 8888}
DB_PATH = "proxystream.db"

# Database Setup
def init_database():
    """Initialize SQLite database for persistence."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Good proxies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS good_proxies (
            host TEXT,
            port INTEGER,
            latency REAL,
            last_tested TIMESTAMP,
            success_rate REAL,
            country TEXT,
            city TEXT,
            lat REAL,
            lon REAL,
            PRIMARY KEY (host, port)
        )
    """)
    
    # Proxy chains table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proxy_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_json TEXT,
            total_latency REAL,
            created_at TIMESTAMP,
            success_count INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()

def save_good_proxy(proxy_data):
    """Save working proxy to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO good_proxies 
        (host, port, latency, last_tested, success_rate, country, city, lat, lon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        proxy_data['host'], proxy_data['port'], proxy_data['latency'],
        datetime.now(), proxy_data.get('success_rate', 100),
        proxy_data.get('country'), proxy_data.get('city'),
        proxy_data.get('lat'), proxy_data.get('lon')
    ))
    conn.commit()
    conn.close()

def load_good_proxies(limit=100):
    """Load known good proxies from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM good_proxies 
        WHERE last_tested > datetime('now', '-7 days')
        ORDER BY latency ASC
        LIMIT ?
    """, (limit,))
    results = cursor.fetchall()
    conn.close()
    return results

def save_proxy_chain(chain_data):
    """Save successful proxy chain."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO proxy_chains (chain_json, total_latency, created_at)
        VALUES (?, ?, ?)
    """, (json.dumps(chain_data), chain_data.get('latency', 0), datetime.now()))
    conn.commit()
    conn.close()

# Initialize database
init_database()

# Session State
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "proxies" not in st.session_state:
    st.session_state.proxies = []
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "good_proxies" not in st.session_state:
    st.session_state.good_proxies = []
if "proxy_chain" not in st.session_state:
    st.session_state.proxy_chain = []
if "geo_cache" not in st.session_state:
    st.session_state.geo_cache = {}

# Theme CSS
def get_theme_css():
    if st.session_state.theme == "light":
        return """
        <style>
        .main-header { color: #3b82f6; font-size: 42px; font-weight: 800; text-align: center; }
        .sub-header { color: #64748b; text-align: center; font-size: 18px; margin-bottom: 32px; }
        .status-badge { padding: 8px 16px; border-radius: 20px; display: inline-block; }
        .status-working { background: #10b981; color: white; }
        .status-failed { background: #ef4444; color: white; }
        </style>
        """
    else:
        return """
        <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .main-header { color: #60a5fa; font-size: 42px; font-weight: 800; text-align: center; }
        .sub-header { color: #94a3b8; text-align: center; font-size: 18px; margin-bottom: 32px; }
        .status-badge { padding: 8px 16px; border-radius: 20px; display: inline-block; }
        .status-working { background: #10b981; color: white; }
        .status-failed { background: #ef4444; color: white; }
        </style>
        """

st.markdown(get_theme_css(), unsafe_allow_html=True)

# Data Models
@dataclass
class ProxyInfo:
    host: str
    port: int
    protocol: str = "http"
    latency: Optional[float] = None
    status: str = "unknown"
    country: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    last_tested: Optional[datetime] = None

# Async Functions
async def get_geolocation(ip: str) -> Dict[str, Any]:
    """Fetch geolocation data for IP."""
    if ip in st.session_state.geo_cache:
        return st.session_state.geo_cache[ip]
    
    for api_url in GEO_APIS:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(api_url.format(ip))
                if response.status_code == 200:
                    data = response.json()
                    # Normalize response
                    geo_data = {
                        'country': data.get('country') or data.get('country_name'),
                        'city': data.get('city'),
                        'lat': data.get('latitude') or data.get('lat'),
                        'lon': data.get('longitude') or data.get('lon'),
                        'isp': data.get('isp') or data.get('org')
                    }
                    st.session_state.geo_cache[ip] = geo_data
                    return geo_data
        except:
            continue
    return {}

async def validate_proxy(proxy: ProxyInfo, timeout: int = 5) -> Tuple[bool, float, Optional[str]]:
    """Validate proxy and get response time."""
    proxy_url = f"http://{proxy.host}:{proxy.port}"
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
            response = await client.get("http://httpbin.org/ip")
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return True, elapsed, data.get("origin")
    except:
        pass
    
    return False, 0, None

async def fetch_and_validate_proxies(source_url: str) -> List[ProxyInfo]:
    """Fetch proxies and validate them immediately."""
    valid_proxies = []
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(source_url)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                
                # Parse proxies
                proxies_to_test = []
                for line in lines[:100]:  # Limit initial validation
                    line = line.strip()
                    if ':' in line and not line.startswith('#'):
                        try:
                            host, port = line.rsplit(':', 1)
                            port = int(port)
                            if 1 <= port <= 65535:
                                ipaddress.ip_address(host)
                                proxies_to_test.append(ProxyInfo(host=host, port=port))
                        except:
                            continue
                
                # Validate in parallel
                if proxies_to_test:
                    tasks = [validate_proxy(p, timeout=3) for p in proxies_to_test[:20]]
                    results = await asyncio.gather(*tasks)
                    
                    for proxy, (is_valid, latency, detected_ip) in zip(proxies_to_test[:20], results):
                        if is_valid:
                            proxy.status = "working"
                            proxy.latency = latency
                            
                            # Get geo data
                            geo = await get_geolocation(proxy.host)
                            proxy.country = geo.get('country')
                            proxy.city = geo.get('city')
                            proxy.lat = geo.get('lat')
                            proxy.lon = geo.get('lon')
                            
                            valid_proxies.append(proxy)
                            
                            # Save to database
                            save_good_proxy({
                                'host': proxy.host,
                                'port': proxy.port,
                                'latency': latency,
                                'country': proxy.country,
                                'city': proxy.city,
                                'lat': proxy.lat,
                                'lon': proxy.lon
                            })
    except:
        pass
    
    return valid_proxies

async def load_proxies_with_validation():
    """Load and validate proxies from all sources."""
    all_valid_proxies = []
    
    # Load from database first
    db_proxies = load_good_proxies()
    for row in db_proxies:
        proxy = ProxyInfo(
            host=row[0], port=row[1], latency=row[2],
            country=row[5], city=row[6], lat=row[7], lon=row[8],
            status="working"
        )
        all_valid_proxies.append(proxy)
    
    # Fetch and validate new proxies
    tasks = [fetch_and_validate_proxies(url) for url in PROXY_SOURCES]
    results = await asyncio.gather(*tasks)
    
    for proxy_list in results:
        all_valid_proxies.extend(proxy_list)
    
    return all_valid_proxies

# UI Components
def render_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button(f"Switch to {'Dark' if st.session_state.theme == 'light' else 'Light'} Theme",
                    type="primary"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    st.markdown('<div class="main-header">ProxyStream Advanced</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Proxy Testing & Chain Analysis Platform</div>', unsafe_allow_html=True)

def render_control_panel():
    st.markdown("## Control Panel")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Quick Actions")
        
        # Load and validate proxies
        if st.button("🔄 Load & Validate Proxies", use_container_width=True):
            with st.spinner("Loading and validating proxies..."):
                proxies = asyncio.run(load_proxies_with_validation())
                st.session_state.proxies = proxies
                st.session_state.good_proxies = [p for p in proxies if p.status == "working"]
                st.success(f"Loaded {len(proxies)} proxies ({len(st.session_state.good_proxies)} validated)")
        
        # Quick stats from database
        db_count = len(load_good_proxies(1000))
        st.info(f"📊 {db_count} good proxies in database")
        
        # Load saved chains
        if st.button("📥 Load Saved Chains", use_container_width=True):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM proxy_chains ORDER BY created_at DESC LIMIT 5")
            chains = cursor.fetchall()
            conn.close()
            if chains:
                st.success(f"Loaded {len(chains)} saved chains")
    
    with col2:
        if st.session_state.good_proxies:
            st.markdown("### Working Proxies")
            
            # Display working proxies with geo data
            for proxy in st.session_state.good_proxies[:5]:
                location = f"{proxy.city}, {proxy.country}" if proxy.city else proxy.country or "Unknown"
                st.markdown(f"""
                <div class="status-badge status-working">
                    {proxy.host}:{proxy.port} | {location} | {proxy.latency:.0f}ms
                </div>
                """, unsafe_allow_html=True)
                
            # Quick connect to best
            if st.button("⚡ Connect to Fastest", use_container_width=True):
                fastest = min(st.session_state.good_proxies, key=lambda p: p.latency or 9999)
                st.success(f"Connected to {fastest.host}:{fastest.port} ({fastest.latency:.0f}ms)")

def render_map():
    """Render actual geographic map with real proxy locations."""
    if st.session_state.good_proxies:
        # Prepare map data
        map_data = []
        for proxy in st.session_state.good_proxies:
            if proxy.lat and proxy.lon:
                map_data.append({
                    "lat": proxy.lat,
                    "lon": proxy.lon,
                    "proxy": f"{proxy.host}:{proxy.port}",
                    "city": proxy.city or "Unknown",
                    "country": proxy.country or "Unknown",
                    "latency": proxy.latency or 0
                })
        
        if map_data:
            df_map = pd.DataFrame(map_data)
            
            fig = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                hover_name="proxy",
                hover_data=["city", "country", "latency"],
                color="latency",
                size="latency",
                color_continuous_scale="Viridis",
                size_max=20,
                zoom=2,
                title="Working Proxy Locations"
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                height=500,
                margin={"r": 0, "t": 30, "l": 0, "b": 0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geographic data available for proxies")
    else:
        st.info("Load proxies to see geographic distribution")

def render_analytics():
    """Show analytics from database."""
    st.markdown("### Historical Analytics")
    
    # Load all good proxies from DB
    all_good = load_good_proxies(1000)
    
    if all_good:
        df = pd.DataFrame(all_good, columns=['Host', 'Port', 'Latency', 'Last Tested', 'Success Rate', 
                                             'Country', 'City', 'Lat', 'Lon'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Country distribution
            country_counts = df['Country'].value_counts().head(10)
            fig = px.bar(x=country_counts.values, y=country_counts.index, 
                        orientation='h', title="Top 10 Countries")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Latency distribution
            fig = px.histogram(df['Latency'].dropna(), nbins=30, 
                             title="Latency Distribution (ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Best proxies table
        st.markdown("### Top Performing Proxies")
        best_df = df.nsmallest(10, 'Latency')[['Host', 'Port', 'Latency', 'Country', 'City']]
        st.dataframe(best_df, use_container_width=True, hide_index=True)

# Main App
def main():
    render_header()
    
    st.warning("""
    **Security Notice:** This tool tests public HTTP proxies for educational purposes. 
    Public proxies may log traffic or be compromised. Use reputable VPN services for real privacy.
    """)
    
    render_control_panel()
    
    st.markdown("---")
    
    tabs = st.tabs(["🗺️ Map", "📊 Analytics", "⛓️ Chains", "💾 Database"])
    
    with tabs[0]:
        render_map()
    
    with tabs[1]:
        render_analytics()
    
    with tabs[2]:
        st.info("Chain management - configure proxy chains from validated proxies")
        if st.session_state.good_proxies:
            chain = st.multiselect("Build chain:", 
                                  [f"{p.host}:{p.port}" for p in st.session_state.good_proxies])
            if len(chain) >= 2 and st.button("Save Chain"):
                save_proxy_chain({"chain": chain, "latency": 0})
                st.success("Chain saved to database")
    
    with tabs[3]:
        st.markdown("### Database Management")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Good Proxies", len(load_good_proxies(10000)))
        with col2:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM proxy_chains")
            chain_count = cursor.fetchone()[0]
            conn.close()
            st.metric("Saved Chains", chain_count)
        with col3:
            if st.button("🗑️ Clear Old Data"):
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM good_proxies WHERE last_tested < datetime('now', '-30 days')")
                conn.commit()
                conn.close()
                st.success("Cleared old data")

if __name__ == "__main__":
    main()
