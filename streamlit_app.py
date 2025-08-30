"""
ProxyStream Advanced - Professional Proxy Testing Platform
Clean UX version with improved layout and styling
"""

import asyncio
import time
import random
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import ipaddress
import re

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

# Initialize database
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS good_proxies (
            host TEXT,
            port INTEGER,
            latency REAL,
            last_tested TIMESTAMP,
            country TEXT,
            city TEXT,
            lat REAL,
            lon REAL,
            PRIMARY KEY (host, port)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proxy_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_json TEXT,
            total_latency REAL,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_database()

# Clean CSS
st.markdown("""
<style>
    /* Clean, modern styling */
    .main { padding: 0; }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 100%;
    }
    
    h1 {
        color: #2563eb;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e40af;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #334155;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #2563eb;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Remove extra padding */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "proxies" not in st.session_state:
    st.session_state.proxies = []
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "good_proxies" not in st.session_state:
    st.session_state.good_proxies = []

# Data Models
@dataclass
class ProxyInfo:
    host: str
    port: int
    latency: Optional[float] = None
    country: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

# Core Functions
async def get_geolocation(ip: str) -> Dict[str, Any]:
    for api_url in GEO_APIS:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(api_url.format(ip))
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'country': data.get('country') or data.get('country_name'),
                        'city': data.get('city'),
                        'lat': data.get('latitude') or data.get('lat'),
                        'lon': data.get('longitude') or data.get('lon')
                    }
        except:
            continue
    return {}

async def validate_proxy(proxy: ProxyInfo) -> Tuple[bool, float]:
    proxy_url = f"http://{proxy.host}:{proxy.port}"
    try:
        start = time.time()
        async with httpx.AsyncClient(proxy=proxy_url, timeout=5) as client:
            response = await client.get("http://httpbin.org/ip")
            if response.status_code == 200:
                return True, (time.time() - start) * 1000
    except:
        pass
    return False, 0

async def load_and_validate_proxies():
    all_proxies = []
    
    # Fetch from sources
    async with httpx.AsyncClient() as client:
        for source in PROXY_SOURCES:
            try:
                response = await client.get(source, timeout=10)
                if response.status_code == 200:
                    lines = response.text.strip().split('\n')[:100]
                    for line in lines:
                        if ':' in line and not line.startswith('#'):
                            try:
                                host, port = line.rsplit(':', 1)
                                port = int(port)
                                if 1 <= port <= 65535:
                                    ipaddress.ip_address(host)
                                    all_proxies.append(ProxyInfo(host=host, port=port))
                            except:
                                continue
            except:
                continue
    
    # Validate sample
    validated = []
    if all_proxies:
        sample = random.sample(all_proxies, min(20, len(all_proxies)))
        for proxy in sample:
            is_valid, latency = await validate_proxy(proxy)
            if is_valid:
                proxy.latency = latency
                geo = await get_geolocation(proxy.host)
                proxy.country = geo.get('country')
                proxy.city = geo.get('city')
                proxy.lat = geo.get('lat')
                proxy.lon = geo.get('lon')
                validated.append(proxy)
                
                # Save to DB
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO good_proxies 
                    (host, port, latency, last_tested, country, city, lat, lon)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (proxy.host, proxy.port, latency, datetime.now(),
                     proxy.country, proxy.city, proxy.lat, proxy.lon))
                conn.commit()
                conn.close()
    
    return all_proxies, validated

def load_from_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM good_proxies 
        WHERE last_tested > datetime('now', '-7 days')
        ORDER BY latency ASC LIMIT 100
    """)
    results = cursor.fetchall()
    conn.close()
    
    proxies = []
    for row in results:
        proxies.append(ProxyInfo(
            host=row[0], port=row[1], latency=row[2],
            country=row[4], city=row[5], lat=row[6], lon=row[7]
        ))
    return proxies

# UI Components
st.title("🔒 ProxyStream Advanced")
st.caption("Professional Proxy Testing & Chain Analysis Platform")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
db_proxies = load_from_database()

with col1:
    st.metric("Database", f"{len(db_proxies)} proxies")
with col2:
    st.metric("Loaded", f"{len(st.session_state.proxies)} proxies")
with col3:
    if st.session_state.good_proxies:
        st.metric("Validated", f"{len(st.session_state.good_proxies)} working")
    else:
        st.metric("Validated", "0 working")
with col4:
    if st.session_state.good_proxies:
        avg_latency = np.mean([p.latency for p in st.session_state.good_proxies if p.latency])
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    else:
        st.metric("Avg Latency", "—")

# Main Controls
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    
    if st.button("🔄 Load & Validate Proxies", use_container_width=True):
        with st.spinner("Loading and validating..."):
            all_proxies, validated = asyncio.run(load_and_validate_proxies())
            st.session_state.proxies = all_proxies
            st.session_state.good_proxies = validated
            st.success(f"Loaded {len(all_proxies)} proxies, {len(validated)} validated")
    
    if st.button("📥 Load from Database", use_container_width=True):
        st.session_state.good_proxies = load_from_database()
        st.success(f"Loaded {len(st.session_state.good_proxies)} proxies from database")
    
    if st.session_state.good_proxies and st.button("⚡ Connect to Fastest", use_container_width=True):
        fastest = min(st.session_state.good_proxies, key=lambda p: p.latency or 9999)
        st.success(f"Connected: {fastest.host}:{fastest.port}")

with col2:
    st.subheader("Working Proxies")
    
    if st.session_state.good_proxies:
        # Create clean table
        data = []
        for p in st.session_state.good_proxies[:10]:
            data.append({
                "Proxy": f"{p.host}:{p.port}",
                "Location": f"{p.city or 'Unknown'}, {p.country or 'Unknown'}",
                "Latency": f"{p.latency:.0f}ms" if p.latency else "—"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No validated proxies yet. Click 'Load & Validate Proxies' to start.")

# Tabs
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🗺️ Map", "📊 Analytics", "💾 Database"])

with tab1:
    if st.session_state.good_proxies:
        map_data = []
        for p in st.session_state.good_proxies:
            if p.lat and p.lon:
                map_data.append({
                    "lat": p.lat, "lon": p.lon,
                    "proxy": f"{p.host}:{p.port}",
                    "location": f"{p.city or 'Unknown'}, {p.country or 'Unknown'}",
                    "latency": p.latency or 0
                })
        
        if map_data:
            df_map = pd.DataFrame(map_data)
            fig = px.scatter_mapbox(
                df_map, lat="lat", lon="lon",
                hover_name="proxy", hover_data=["location", "latency"],
                color="latency", size="latency",
                color_continuous_scale="Viridis",
                zoom=2, height=500
            )
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load proxies to see geographic distribution")

with tab2:
    if db_proxies:
        col1, col2 = st.columns(2)
        
        with col1:
            # Country distribution
            countries = pd.Series([p.country for p in db_proxies if p.country]).value_counts().head(10)
            fig = px.bar(x=countries.values, y=countries.index, orientation='h',
                        labels={'x': 'Count', 'y': 'Country'})
            fig.update_layout(title="Top Countries", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Latency histogram
            latencies = [p.latency for p in db_proxies if p.latency]
            fig = px.histogram(latencies, nbins=20,
                             labels={'value': 'Latency (ms)', 'count': 'Count'})
            fig.update_layout(title="Latency Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2, col3 = st.columns(3)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with col1:
        cursor.execute("SELECT COUNT(*) FROM good_proxies")
        count = cursor.fetchone()[0]
        st.metric("Total Proxies", count)
    
    with col2:
        cursor.execute("SELECT COUNT(*) FROM proxy_chains")
        chains = cursor.fetchone()[0]
        st.metric("Saved Chains", chains)
    
    with col3:
        cursor.execute("SELECT MIN(last_tested) FROM good_proxies")
        oldest = cursor.fetchone()[0]
        if oldest:
            st.metric("Oldest Entry", oldest[:10])
    
    conn.close()
    
    if st.button("🗑️ Clear Old Entries (30+ days)"):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM good_proxies WHERE last_tested < datetime('now', '-30 days')")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        st.success(f"Deleted {deleted} old entries")
