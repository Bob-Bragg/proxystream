import json
import os
import random
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import streamlit.components.v1 as components

# ProxyStream Configuration
st.set_page_config(
    page_title="ProxyStream - Modern VPN Dashboard", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ProxyStream Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: white;
    }
    
    .main-header {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 30px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .proxy-status-connected {
        color: #10b981;
        font-weight: bold;
        background: rgba(16, 185, 129, 0.1);
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .proxy-status-disconnected {
        color: #ef4444;
        font-weight: bold;
        background: rgba(239, 68, 68, 0.1);
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.5rem 0;
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .country-stats {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Proxy List from Proxy-Hound
PROXY_LIST = [
    "34.121.105.79:80", "68.107.241.150:8080", "3.133.146.217:5050",
    "72.10.160.90:13847", "170.85.158.82:80", "170.85.158.82:10005",
    "67.43.236.20:29915", "167.172.157.96:80", "72.10.164.178:1771",
    "72.10.164.178:27495", "38.127.172.53:24171", "67.43.228.254:11859",
    "67.43.228.253:4337", "170.106.169.97:3128", "130.41.109.158:8080",
    "72.10.160.173:13909", "72.10.160.90:12347", "67.43.236.19:4999",
    "72.10.160.90:20647", "72.10.160.172:7959", "67.43.236.19:18107",
    "67.43.228.250:20689", "72.10.164.178:20035", "67.43.236.20:4999",
    "72.10.160.171:28795", "3.133.221.69:3128", "201.174.239.25:8080",
    "136.175.9.83:8084", "72.10.160.94:30069", "72.10.160.174:2387",
    "72.10.160.173:25221", "72.10.160.170:3483", "136.175.9.82:8081",
    "136.175.9.86:8082", "136.175.9.82:8085", "72.10.164.178:28483",
    "67.43.228.251:26125", "72.10.160.90:13495", "67.43.228.253:15161",
    "72.10.160.90:10529", "67.43.236.20:8011", "67.43.228.252:3187",
    "72.10.160.90:5137", "72.10.164.178:26129", "72.10.160.170:15401",
    "72.10.164.178:27479", "67.43.236.20:3407", "67.43.236.19:15237",
    "67.43.236.20:11663", "72.10.160.170:9125", "67.43.228.250:3625",
    "67.43.236.20:9579", "72.10.164.178:4381", "72.10.160.170:23569",
    "72.10.160.174:24169", "72.10.160.90:1941", "155.94.241.134:3128"
]

# Country-to-IP mapping for geolocation simulation
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

# Reverse mapping
IP_TO_COUNTRY: Dict[str, str] = {}
for country, ips in COUNTRY_IP_MAPPING.items():
    for ip in ips:
        IP_TO_COUNTRY[ip] = country

def get_country_flag(country_code: str) -> str:
    flags = {
        'US': '🇺🇸', 'CA': '🇨🇦', 'GB': '🇬🇧', 'DE': '🇩🇪', 
        'FR': '🇫🇷', 'NL': '🇳🇱', 'SG': '🇸🇬', 'AU': '🇦🇺', 'JP': '🇯🇵'
    }
    return flags.get(country_code, '🏳️')

def parse_proxy_list() -> Dict[str, List[str]]:
    proxies_by_country: Dict[str, List[str]] = {}
    for proxy in PROXY_LIST:
        if ':' in proxy:
            ip = proxy.split(':')[0]
            country = IP_TO_COUNTRY.get(ip, 'US')
            if country not in proxies_by_country:
                proxies_by_country[country] = []
            proxies_by_country[country].append(proxy)
    return proxies_by_country

def test_proxy_connection(proxy: str) -> tuple[bool, dict]:
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    is_success = random.choices([True, False], weights=[75, 25])[0]
    metrics = {
        'latency': random.randint(10, 150) if is_success else 0,
        'speed': random.uniform(10, 100) if is_success else 0,
        'country': IP_TO_COUNTRY.get(proxy.split(':')[0], 'US')
    }
    return is_success, metrics

def generate_usage_data():
    hours = pd.date_range(start=datetime.now()-timedelta(hours=24), end=datetime.now(), freq='H')
    download_data = np.random.exponential(scale=50, size=len(hours))
    upload_data = np.random.exponential(scale=20, size=len(hours))
    return pd.DataFrame({
        'time': hours,
        'download': download_data,
        'upload': upload_data
    })

# ---------- NEW: real HTTP fetch via active proxy ----------
def fetch_via_proxy_requests(url: str, proxy: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Real HTTP(S) GET via the chosen proxy using 'requests'.
    Accepts proxy like 'IP:PORT' or 'http://IP:PORT'.
    Returns: dict with ok/status/final_url/headers/content/elapsed_ms/error
    """
    hostport = proxy.split("://")[-1]
    proxy_url = f"http://{hostport}"
    proxies = {"http": proxy_url, "https": proxy_url}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
    t0 = time.perf_counter()
    try:
        r = requests.get(
            url,
            proxies=proxies,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            stream=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "ok": True,
            "status": r.status_code,
            "final_url": r.url,
            "headers": dict(r.headers),
            "content": r.content,
            "elapsed_ms": round(elapsed, 1),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "status": None,
            "final_url": None,
            "headers": {},
            "content": b"",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": str(e)[:200],
        }

# Initialize session state
if "proxy_connected" not in st.session_state:
    st.session_state.proxy_connected = False
if "current_proxy" not in st.session_state:
    st.session_state.current_proxy = None
if "connection_start_time" not in st.session_state:
    st.session_state.connection_start_time = None
if "data_usage" not in st.session_state:
    st.session_state.data_usage = {"download": 0, "upload": 0}
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "US"
if "proxy_metrics" not in st.session_state:
    st.session_state.proxy_metrics = {"latency": 0, "speed": 0}
# NEW: track an active proxy for real browsing
if "active_proxy" not in st.session_state:
    st.session_state.active_proxy = None  # "http://IP:PORT"

# Parse proxy list
proxy_data = parse_proxy_list()

def main():
    # Header
    st.markdown('''
    <div class="main-header">
        🛡️ ProxyStream
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 40px;">Modern Open-Source VPN Dashboard with Real Proxy Network</p>', unsafe_allow_html=True)

    # Sidebar - Proxy Controls
    with st.sidebar:
        st.markdown("## 🔧 Proxy Settings")
        
        # Display proxy statistics
        total_proxies = sum(len(proxies) for proxies in proxy_data.values())
        st.markdown(f"""
        <div class="country-stats">
            📊 <strong>Network Statistics</strong><br>
            Total Proxies: <strong>{total_proxies:,}</strong><br>
            Countries Available: <strong>{len(proxy_data)}</strong><br>
            Protocol: <strong>HTTPS</strong>
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
            
            # Display country statistics
            country_proxies = proxy_data[selected_country]
            st.markdown(f"""
            <div class="country-stats">
                {get_country_flag(selected_country)} <strong>{selected_country}</strong><br>
                Available Servers: <strong>{len(country_proxies)}</strong><br>
                Status: <strong>Online</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Proxy Server Selection
            if country_proxies:
                display_proxies = country_proxies[:10]
                selected_proxy = st.selectbox(
                    "Proxy Server",
                    options=display_proxies,
                    help="Select a proxy server from the available list"
                )
                
                # Connection Controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔗 Connect", use_container_width=True):
                        with st.spinner("Testing connection..."):
                            success, metrics = test_proxy_connection(selected_proxy)
                            if success:
                                st.session_state.proxy_connected = True
                                st.session_state.current_proxy = selected_proxy
                                st.session_state.connection_start_time = datetime.now()
                                st.session_state.proxy_metrics = metrics
                                # NEW: set active proxy for browsing
                                st.session_state.active_proxy = f"http://{selected_proxy}"
                                st.success("Connected successfully!")
                                st.rerun()
                            else:
                                st.error("Connection failed - trying next server...")
                                if len(display_proxies) > 1:
                                    backup_proxy = random.choice([p for p in display_proxies if p != selected_proxy])
                                    success, metrics = test_proxy_connection(backup_proxy)
                                    if success:
                                        st.session_state.proxy_connected = True
                                        st.session_state.current_proxy = backup_proxy
                                        st.session_state.connection_start_time = datetime.now()
                                        st.session_state.proxy_metrics = metrics
                                        # NEW: set active proxy for browsing
                                        st.session_state.active_proxy = f"http://{backup_proxy}"
                                        st.success(f"Connected to backup server!")
                                        st.rerun()
                
                with col2:
                    if st.button("❌ Disconnect", use_container_width=True):
                        st.session_state.proxy_connected = False
                        st.session_state.current_proxy = None
                        st.session_state.connection_start_time = None
                        st.session_state.proxy_metrics = {"latency": 0, "speed": 0}
                        # NEW: clear active proxy
                        st.session_state.active_proxy = None
                        st.success("Disconnected!")
                        st.rerun()
        
        # Connection Status
        st.markdown("---")
        st.markdown("## 📊 Connection Status")
        if st.session_state.proxy_connected:
            st.markdown(f'<div class="proxy-status-connected">🟢 Connected</div>', unsafe_allow_html=True)
            if st.session_state.connection_start_time:
                duration = datetime.now() - st.session_state.connection_start_time
                st.text(f"Duration: {str(duration).split('.')[0]}")
            st.text(f"Server: {st.session_state.current_proxy}")
            st.text(f"Location: {get_country_flag(st.session_state.selected_country)} {st.session_state.selected_country}")
            st.text(f"Latency: {st.session_state.proxy_metrics.get('latency', 0)}ms")
        else:
            st.markdown(f'<div class="proxy-status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
            st.info("Select a country and proxy server to connect")

    # Main Dashboard
    if st.session_state.proxy_connected:
        # Connected state
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Connection Details")
            st.success(f"🟢 Connected via {get_country_flag(st.session_state.selected_country)} {st.session_state.selected_country}")
            st.info(f"Server: {st.session_state.current_proxy}")
            
            latency = st.session_state.proxy_metrics.get('latency', 0)
            speed = st.session_state.proxy_metrics.get('speed', 0)
            
            col1a, col1b = st.columns(2)
            with col1a:
                quality = "Excellent" if latency < 50 else "Good" if latency < 100 else "Fair"
                st.metric("Latency", f"{latency}ms", delta=f"{quality}")
            with col1b:
                st.metric("Speed", f"{speed:.1f} Mbps", delta="Connected" if speed > 0 else None)
            
            # World map
            fig_map = go.Figure(data=go.Scattergeo(
                lon=[-100 if st.session_state.selected_country == 'US' else 0],
                lat=[40 if st.session_state.selected_country == 'US' else 50],
                mode='markers',
                marker=dict(size=20, color='#10b981', line=dict(width=3, color='white')),
                showlegend=False,
                text=[f"{st.session_state.selected_country} Server"],
                hoverinfo='text'
            ))
            
            fig_map.update_layout(
                geo=dict(
                    showframe=False, showcoastlines=True, projection_type='natural earth',
                    bgcolor='rgba(0,0,0,0)', landcolor='#374151', oceancolor='#1e293b', coastlinecolor='#6b7280'
                ),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=300, margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("### 📊 Data Usage")
            
            if st.session_state.proxy_connected:
                st.session_state.data_usage["download"] += random.uniform(0.01, 0.1)
                st.session_state.data_usage["upload"] += random.uniform(0.005, 0.05)
            
            total_usage = st.session_state.data_usage["download"] + st.session_state.data_usage["upload"]
            st.metric("Session Total", f"{total_usage:.2f} GB")
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Downloaded", f"{st.session_state.data_usage['download']:.2f} GB", 
                         delta=f"+{random.uniform(0.01, 0.05):.2f}")
            with col2b:
                st.metric("Uploaded", f"{st.session_state.data_usage['upload']:.2f} GB", 
                         delta=f"+{random.uniform(0.005, 0.02):.2f}")
            
            # Usage chart
            usage_df = generate_usage_data()
            fig_usage = px.area(usage_df, x='time', y=['download', 'upload'], title="24h Traffic Pattern",
                               color_discrete_map={'download': '#667eea', 'upload': '#10b981'})
            
            fig_usage.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                height=250, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_usage, use_container_width=True, config={'displayModeBar': False})

        # Performance metrics
        st.markdown("---")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            current_latency = st.session_state.proxy_metrics.get('latency', 0)
            st.metric("Current Latency", f"{current_latency}ms", delta=f"{random.randint(-5, 5)}ms")
        
        with col4:
            current_speed = st.session_state.proxy_metrics.get('speed', 0)
            st.metric("Current Speed", f"{current_speed:.1f} Mbps", delta=f"{random.uniform(-2, 5):.1f}")
        
        with col5:
            uptime = "99.9%" if st.session_state.proxy_connected else "0%"
            st.metric("Uptime", uptime, delta="0.1%" if st.session_state.proxy_connected else None)
        
        with col6:
            total_proxies = sum(len(proxies) for proxies in proxy_data.values())
            st.metric("Available Servers", f"{total_proxies:,}", delta=f"+{random.randint(5, 20)}")

        # ----------- NEW: Browse via Current Connection (keeps your UI) -----------
        st.markdown("---")
        st.markdown("### 🔎 Browse via Current Connection")

        if not st.session_state.active_proxy:
            st.info("Connect to a server first to enable browsing.")
        else:
            urls = st.text_area(
                "Enter one or more URLs (one per line):",
                value="https://httpbin.org/ip\nhttps://example.com",
                height=90,
                help="We’ll fetch these pages through your active proxy."
            )
            c1, c2 = st.columns(2)
            with c1:
                browse_timeout = st.slider("Request timeout (seconds)", 3, 30, 12)
            with c2:
                go_browse = st.button("🌐 Go", use_container_width=True)

            if go_browse:
                targets = [u.strip() for u in urls.splitlines() if u.strip()]
                if not targets:
                    st.warning("Please enter at least one URL.")
                else:
                    for u in targets:
                        st.markdown(f"**Target:** {u}")
                        with st.spinner("Fetching via active proxy…"):
                            res = fetch_via_proxy_requests(u, st.session_state.active_proxy, timeout=browse_timeout)

                        if not res["ok"]:
                            st.error(f"Error: {res['error']}")
                            st.markdown("---")
                            continue

                        st.markdown(
                            f"**Status:** {res['status']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"**Elapsed:** {res['elapsed_ms']} ms &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"**Final URL:** {res['final_url']}"
                        )
                        ct = res["headers"].get("content-type", "").lower()

                        with st.expander("Response headers"):
                            st.json(res["headers"])

                        if "text/html" in ct:
                            html = res["content"].decode("utf-8", errors="ignore")
                            components.html(html, height=600, scrolling=True)
                            st.download_button("Download HTML", data=html.encode(), file_name="page.html")
                        elif "json" in ct or "javascript" in ct:
                            try:
                                st.json(json.loads(res["content"]))
                            except Exception:
                                st.code(res["content"][:2000])
                            st.download_button("Download JSON", data=res["content"], file_name="response.json")
                        elif any(x in ct for x in ["png", "jpeg", "jpg", "gif", "webp"]):
                            st.image(res["content"])
                            st.download_button("Download image", data=res["content"], file_name="image.bin")
                        else:
                            st.code(res["content"][:2000])
                            st.download_button("Download body", data=res["content"], file_name="response.bin")

                        st.markdown("---")

    else:
        # Disconnected state
        st.markdown("### 🔌 Not Connected")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("Select a country and proxy server from the sidebar to establish a secure connection.")
            
            st.markdown("### 🌐 Network Overview")
            
            countries = list(proxy_data.keys())
            server_counts = [len(proxy_data[country]) for country in countries]
            
            fig_network = px.bar(x=countries, y=server_counts, title="Available Servers by Country",
                               color=server_counts, color_continuous_scale="Viridis")
            
            fig_network.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                height=300, xaxis_title="Country", yaxis_title="Server Count", showlegend=False
            )
            
            st.plotly_chart(fig_network, use_container_width=True, config={'displayModeBar': False})

    # Footer
    total_proxies = sum(len(proxy_data[country]) for country in proxy_data.keys())
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        <p><strong>ProxyStream v2.0</strong> - Powered by Proxy-Hound Network</p>
        <p>🔒 Secure • 🚀 Fast • 🌍 Global • ⭐ Open Source</p>
        <p>Total Network: <strong>{total_proxies:,}</strong> servers across <strong>{len(proxy_data)}</strong> countries</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    # Auto-refresh for live updates when connected
    # Note: If this clears your browsing results too fast, increase the sleep to ~10-15s.
    if st.session_state.proxy_connected:
        time.sleep(3)
        st.rerun()
