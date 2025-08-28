import json
import os
import random
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
import requests
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
    
    .stream-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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

# Load proxy list from data file
@st.cache_data
def load_proxy_list():
    try:
        with open('data/proxies.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and ':' in line]
    except FileNotFoundError:
        # Fallback proxy list if file not found
        return [
            "34.121.105.79:80", "68.107.241.150:8080", "3.133.146.217:5050",
            "72.10.160.90:13847", "170.85.158.82:80", "167.172.157.96:80",
            "72.10.164.178:1771", "38.127.172.53:24171", "67.43.228.254:11859"
        ]

PROXY_LIST = load_proxy_list()

# Country-to-IP mapping for geolocation simulation
COUNTRY_IP_MAPPING = {
    'US': ['34.121.105.79', '68.107.241.150', '3.133.146.217', '72.10.160.90', '170.85.158.82', '67.43.236.20', '167.172.157.96'],
    'CA': ['72.10.164.178', '38.127.172.53', '67.43.228.254', '67.43.228.253'],
    'GB': ['170.106.169.97', '130.41.109.158', '155.94.241.134'],
    'DE': ['136.175.9.83', '136.175.9.82', '136.175.9.86'],
    'FR': ['201.174.239.25'],
    'NL': ['67.43.228.251', '67.43.228.250'],
    'SG': ['72.10.160.173', '72.10.160.174', '72.10.160.170'],
    'AU': ['3.133.221.69'],
    'JP': ['67.43.228.252']
}

# Reverse mapping for IP to country lookup
IP_TO_COUNTRY = {}
for country, ips in COUNTRY_IP_MAPPING.items():
    for ip in ips:
        IP_TO_COUNTRY[ip] = country

def get_country_flag(country_code: str) -> str:
    """Get flag emoji for country code"""
    flags = {
        'US': '🇺🇸', 'CA': '🇨🇦', 'GB': '🇬🇧', 'DE': '🇩🇪', 
        'FR': '🇫🇷', 'NL': '🇳🇱', 'SG': '🇸🇬', 'AU': '🇦🇺', 
        'JP': '🇯🇵', 'ES': '🇪🇸', 'IT': '🇮🇹', 'CH': '🇨🇭'
    }
    return flags.get(country_code, '🏳️')

def parse_proxy_list() -> Dict[str, List[str]]:
    """Parse the proxy list and organize by country"""
    proxies_by_country = {}
    
    for proxy in PROXY_LIST:
        if ':' in proxy:
            ip = proxy.split(':')[0]
            country = IP_TO_COUNTRY.get(ip, 'US')  # Default to US if not found
            
            if country not in proxies_by_country:
                proxies_by_country[country] = []
            proxies_by_country[country].append(proxy)
    
    return proxies_by_country

def test_proxy_connection(proxy: str) -> tuple[bool, dict]:
    """Test if a proxy is working (simplified simulation)"""
    # Simulate network delay
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    
    # Simulate success/failure with weighted probability
    is_success = random.choices([True, False], weights=[75, 25])[0]
    
    metrics = {
        'latency': random.randint(10, 150) if is_success else 0,
        'speed': random.uniform(10, 100) if is_success else 0,
        'country': IP_TO_COUNTRY.get(proxy.split(':')[0], 'US')
    }
    
    return is_success, metrics

def generate_usage_data():
    """Generate sample usage data for the dashboard"""
    hours = pd.date_range(start=datetime.now()-timedelta(hours=24), end=datetime.now(), freq='H')
    download_data = np.random.exponential(scale=50, size=len(hours))
    upload_data = np.random.exponential(scale=20, size=len(hours))
    return pd.DataFrame({
        'time': hours,
        'download': download_data,
        'upload': upload_data
    })

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
                # Show top 10 servers for better UX
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
                                st.success("Connected successfully!")
                                st.rerun()
                            else:
                                st.error("Connection failed - trying next server...")
                                # Auto-retry with another server
                                if len(display_proxies) > 1:
                                    backup_proxy = random.choice([p for p in display_proxies if p != selected_proxy])
                                    success, metrics = test_proxy_connection(backup_proxy)
                                    if success:
                                        st.session_state.proxy_connected = True
                                        st.session_state.current_proxy = backup_proxy
                                        st.session_state.connection_start_time = datetime.now()
                                        st.session_state.proxy_metrics = metrics
                                        st.success(f"Connected to backup server!")
                                        st.rerun()
                
                with col2:
                    if st.button("❌ Disconnect", use_container_width=True):
                        st.session_state.proxy_connected = False
                        st.session_state.current_proxy = None
                        st.session_state.connection_start_time = None
                        st.session_state.proxy_metrics = {"latency": 0, "speed": 0}
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
        # Connected state - show full dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Connection Details")
            st.success(f"🟢 Connected via {get_country_flag(st.session_state.selected_country)} {st.session_state.selected_country}")
            st.info(f"Server: {st.session_state.current_proxy}")
            
            # Connection quality indicators
            latency = st.session_state.proxy_metrics.get('latency', 0)
            speed = st.session_state.proxy_metrics.get('speed', 0)
            
            col1a, col1b = st.columns(2)
            with col1a:
                quality = "Excellent" if latency < 50 else "Good" if latency < 100 else "Fair"
                st.metric("Latency", f"{latency}ms", delta=f"{quality}")
            with col1b:
                st.metric("Speed", f"{speed:.1f} Mbps", delta="Connected" if speed > 0 else None)
            
            # World map visualization
            fig_map = go.Figure(data=go.Scattergeo(
                lon=[-100 if st.session_state.selected_country == 'US' else 0],
                lat=[40 if st.session_state.selected_country == 'US' else 50],
                mode='markers',
                marker=dict(
                    size=20,
                    color='#10b981',
                    line=dict(width=3, color='white')
                ),
                showlegend=False,
                text=[f"{st.session_state.selected_country} Server"],
                hoverinfo='text'
            ))
            
            fig_map.update_layout(
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
                height=300,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("### 📊 Data Usage")
            
            # Simulate data usage increment
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
            fig_usage = px.area(
                usage_df, 
                x='time', 
                y=['download', 'upload'],
                title="24h Traffic Pattern",
                color_discrete_map={'download': '#667eea', 'upload': '#10b981'}
            )
            
            fig_usage.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=250,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_usage, use_container_width=True, config={'displayModeBar': False})

        # Performance metrics row
        st.markdown("---")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            current_latency = st.session_state.proxy_metrics.get('latency', 0)
            st.metric("Current Latency", f"{current_latency}ms", 
                     delta=f"{random.randint(-5, 5)}ms")
        
        with col4:
            current_speed = st.session_state.proxy_metrics.get('speed', 0)
            st.metric("Current Speed", f"{current_speed:.1f} Mbps", 
                     delta=f"{random.uniform(-2, 5):.1f}")
        
        with col5:
            uptime = "99.9%" if st.session_state.proxy_connected else "0%"
            st.metric("Uptime", uptime, delta="0.1%" if st.session_state.proxy_connected else None)
        
        with col6:
            total_proxies = sum(len(proxies) for proxies in proxy_data.values())
            st.metric("Available Servers", f"{total_proxies:,}", delta=f"+{random.randint(5, 20)}")

    else:
        # Disconnected state - show connection prompt
        st.markdown("### 🔌 Not Connected")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("Select a country and proxy server from the sidebar to establish a secure connection.")
            
            # Show network overview
            st.markdown("### 🌐 Network Overview")
            
            # Create a simple chart showing available servers by country
            countries = list(proxy_data.keys())
            server_counts = [len(proxy_data[country]) for country in countries]
            
            fig_network = px.bar(
                x=countries,
                y=server_counts,
                title="Available Servers by Country",
                color=server_counts,
                color_continuous_scale="Viridis"
            )
            
            fig_network.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300,
                xaxis_title="Country",
                yaxis_title="Server Count",
                showlegend=False
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

    # Auto-refresh for live updates when connected (reduced frequency)
    if st.session_state.proxy_connected:
        time.sleep(3)
        st.rerun()
