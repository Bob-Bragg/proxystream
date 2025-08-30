"""
ProxyStream Advanced - Professional Proxy Testing & Chain Analysis Platform
Educational tool for testing public HTTP proxies and proxy chains
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
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
    "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/proxy_hound_results.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

COMMON_PORTS = {80, 8080, 3128, 443, 1080, 8888}

# Session State
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "proxies" not in st.session_state:
    st.session_state.proxies = []
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "proxy_chain" not in st.session_state:
    st.session_state.proxy_chain = []
if "connection_mode" not in st.session_state:
    st.session_state.connection_mode = "single"
if "connected" not in st.session_state:
    st.session_state.connected = False
if "selected_proxy" not in st.session_state:
    st.session_state.selected_proxy = None
if "only_common_ports" not in st.session_state:
    st.session_state.only_common_ports = True

# Theme CSS
def get_theme_css():
    if st.session_state.theme == "light":
        return """
        <style>
        .stApp { 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
            color: #1e293b;
        }
        .main-header {
            color: #3b82f6;
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 8px;
        }
        .sub-header {
            color: #64748b;
            text-align: center;
            font-size: 18px;
            margin-bottom: 32px;
        }
        [data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
        }
        .status-connected {
            background: #10b981;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        .status-disconnected {
            background: #ef4444;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        </style>
        """
    else:
        return """
        <style>
        .stApp { 
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
            color: #f8fafc;
        }
        .main-header {
            color: #60a5fa;
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 8px;
        }
        .sub-header {
            color: #94a3b8;
            text-align: center;
            font-size: 18px;
            margin-bottom: 32px;
        }
        [data-testid="metric-container"] {
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 20px;
        }
        .status-connected {
            background: #10b981;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
        .status-disconnected {
            background: #ef4444;
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            font-weight: 600;
            text-align: center;
        }
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

@dataclass
class TestResult:
    proxy: ProxyInfo
    success: bool
    response_time: float
    error: Optional[str] = None
    timestamp: datetime = None
    detected_ip: Optional[str] = None

# Async Functions
async def fetch_proxies_from_source(session: httpx.AsyncClient, url: str) -> List[str]:
    try:
        response = await session.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip().split('\n')
    except:
        pass
    return []

async def load_all_proxies() -> List[ProxyInfo]:
    all_proxies = set()
    
    async with httpx.AsyncClient() as session:
        tasks = [fetch_proxies_from_source(session, url) for url in PROXY_SOURCES]
        results = await asyncio.gather(*tasks)
        
        for proxy_list in results:
            for line in proxy_list:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    try:
                        host, port = line.rsplit(':', 1)
                        port = int(port)
                        if 1 <= port <= 65535:
                            try:
                                ipaddress.ip_address(host)
                                all_proxies.add((host, port))
                            except:
                                pass
                    except:
                        continue
    
    return [ProxyInfo(host=host, port=port) for host, port in all_proxies]

async def test_proxy(proxy: ProxyInfo, timeout: int = 5) -> TestResult:
    proxy_url = f"http://{proxy.host}:{proxy.port}"
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
            response = await client.get("http://httpbin.org/ip")
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return TestResult(
                    proxy=proxy,
                    success=True,
                    response_time=elapsed,
                    detected_ip=data.get("origin"),
                    timestamp=datetime.now()
                )
    except Exception as e:
        return TestResult(
            proxy=proxy,
            success=False,
            response_time=0,
            error=str(e)[:100],
            timestamp=datetime.now()
        )
    
    return TestResult(proxy=proxy, success=False, response_time=0, timestamp=datetime.now())

async def batch_test_proxies(proxies: List[ProxyInfo], max_concurrent: int = 50) -> List[TestResult]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def test_with_limit(proxy):
        async with semaphore:
            return await test_proxy(proxy)
    
    tasks = [test_with_limit(proxy) for proxy in proxies]
    return await asyncio.gather(*tasks)

# UI Components
def render_header():
    # Theme toggle button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button(f"Switch to {'Dark' if st.session_state.theme == 'light' else 'Light'} Theme",
                    key="theme_toggle",
                    type="primary"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    st.markdown('<div class="main-header">ProxyStream Advanced</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Proxy Testing & Chain Analysis Platform</div>', unsafe_allow_html=True)

def render_security_notice():
    st.warning("""
    **Security Notice:** This tool tests public HTTP proxies and proxy chains for educational purposes. 
    Public proxies may log traffic, inject ads, or be compromised. Use reputable VPN services for real privacy protection.
    """)
    
    if st.sidebar.available:
        st.info("Sidebar is available - controls shown in sidebar")
    else:
        st.info("Note: Due to sidebar rendering issues, controls are shown below")

def render_control_panel():
    st.markdown("## Control Panel")
    
    # Settings Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Settings")
        
        # Connection Mode
        st.session_state.connection_mode = st.radio(
            "Connection Type:",
            ["Single Proxy", "Proxy Chain"],
            index=0 if st.session_state.connection_mode == "Single Proxy" else 1
        )
        
        # Filters
        st.session_state.only_common_ports = st.checkbox(
            "Only common ports",
            value=st.session_state.only_common_ports
        )
        
        # Load Proxies
        if st.button("🔄 Load Proxies", use_container_width=True):
            with st.spinner("Loading proxy lists..."):
                proxies = asyncio.run(load_all_proxies())
                st.session_state.proxies = proxies
                st.success(f"Loaded {len(proxies):,} proxies!")
        
        # Status
        st.markdown("### Status")
        if st.session_state.connected and st.session_state.selected_proxy:
            st.markdown('<div class="status-connected">🟢 Connected</div>', unsafe_allow_html=True)
            st.text(f"Server: {st.session_state.selected_proxy.host}:{st.session_state.selected_proxy.port}")
        else:
            st.markdown('<div class="status-disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Proxy Configuration")
        
        # Apply filters
        filtered_proxies = st.session_state.proxies
        if st.session_state.only_common_ports:
            filtered_proxies = [p for p in filtered_proxies if p.port in COMMON_PORTS]
        
        if filtered_proxies:
            st.info(f"📊 {len(filtered_proxies):,} proxies available")
            
            if st.session_state.connection_mode == "Single Proxy":
                # Single proxy mode
                sample_proxies = filtered_proxies[:100]
                proxy_options = [f"{p.host}:{p.port}" for p in sample_proxies]
                selected = st.selectbox("Select Proxy:", proxy_options)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🧪 Test Proxy", use_container_width=True):
                        for p in sample_proxies:
                            if f"{p.host}:{p.port}" == selected:
                                with st.spinner("Testing..."):
                                    result = asyncio.run(test_proxy(p))
                                    if result.success:
                                        st.session_state.selected_proxy = p
                                        st.session_state.connected = True
                                        st.success(f"✅ Working! Latency: {result.response_time:.0f}ms")
                                    else:
                                        st.error(f"❌ Failed: {result.error}")
                                break
                
                with col_b:
                    if st.button("❌ Disconnect", use_container_width=True):
                        st.session_state.connected = False
                        st.session_state.selected_proxy = None
                        st.info("Disconnected")
                
                # Batch test
                test_count = st.slider("Batch test count:", 10, min(500, len(filtered_proxies)), 50)
                if st.button("🚀 Test Batch", use_container_width=True):
                    with st.spinner(f"Testing {test_count} proxies..."):
                        sample = random.sample(filtered_proxies, test_count)
                        results = asyncio.run(batch_test_proxies(sample))
                        st.session_state.test_results = results
                        success_count = sum(1 for r in results if r.success)
                        st.success(f"Tested {len(results)} proxies: {success_count} working")
            
            else:
                # Chain mode
                st.markdown("**Current Chain:**")
                if st.session_state.proxy_chain:
                    for i, p in enumerate(st.session_state.proxy_chain):
                        col_p, col_q = st.columns([4, 1])
                        with col_p:
                            st.text(f"{i+1}. {p.host}:{p.port}")
                        with col_q:
                            if st.button("❌", key=f"rm_{i}"):
                                st.session_state.proxy_chain.pop(i)
                                st.rerun()
                else:
                    st.info("No proxies in chain")
                
                sample_proxies = filtered_proxies[:50]
                proxy_options = [f"{p.host}:{p.port}" for p in sample_proxies]
                selected = st.selectbox("Add to chain:", proxy_options)
                
                if st.button("➕ Add to Chain", use_container_width=True):
                    for p in sample_proxies:
                        if f"{p.host}:{p.port}" == selected:
                            if len(st.session_state.proxy_chain) < 5:
                                st.session_state.proxy_chain.append(p)
                                st.rerun()
                            else:
                                st.error("Maximum 5 proxies in chain")
                            break
                
                if len(st.session_state.proxy_chain) >= 2:
                    if st.button("🧪 Test Chain", use_container_width=True):
                        st.success(f"Chain configured with {len(st.session_state.proxy_chain)} hops")
        else:
            st.info("Load proxies to start testing")

def render_results_section():
    st.markdown("---")
    
    tabs = st.tabs(["📊 Results", "📈 Analytics", "🗺️ Map"])
    
    with tabs[0]:
        if st.session_state.test_results:
            results = st.session_state.test_results
            working = [r for r in results if r.success]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tested", len(results))
            with col2:
                st.metric("Working", len(working))
            with col3:
                st.metric("Success Rate", f"{(len(working)/len(results)*100):.1f}%")
            with col4:
                if working:
                    avg_latency = np.mean([r.response_time for r in working])
                    st.metric("Avg Latency", f"{avg_latency:.0f}ms")
            
            # Results table
            df = pd.DataFrame([
                {
                    "Host": r.proxy.host,
                    "Port": r.proxy.port,
                    "Status": "✅" if r.success else "❌",
                    "Latency (ms)": round(r.response_time) if r.success else None,
                    "Detected IP": r.detected_ip if r.success else None,
                    "Error": r.error if not r.success else None
                }
                for r in results
            ])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "proxy_results.csv",
                "text/csv"
            )
        else:
            st.info("No test results yet. Run tests from the Control Panel above.")
    
    with tabs[1]:
        if st.session_state.test_results:
            working = [r for r in st.session_state.test_results if r.success]
            if working:
                # Latency distribution
                latencies = [r.response_time for r in working]
                fig = px.histogram(latencies, nbins=30, title="Latency Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Port analysis
                port_data = {}
                for r in st.session_state.test_results:
                    port = r.proxy.port
                    if port not in port_data:
                        port_data[port] = {"total": 0, "success": 0}
                    port_data[port]["total"] += 1
                    if r.success:
                        port_data[port]["success"] += 1
                
                if port_data:
                    df_ports = pd.DataFrame([
                        {"Port": p, "Success Rate": d["success"]/d["total"]*100, "Total": d["total"]}
                        for p, d in port_data.items()
                    ])
                    fig = px.bar(df_ports, x="Port", y="Success Rate", title="Success Rate by Port")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to analyze yet.")
    
    with tabs[2]:
        st.info("Geographic mapping coming soon!")

# Main App
def main():
    render_header()
    render_security_notice()
    render_control_panel()
    render_results_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
        ProxyStream Advanced v2.0 | Educational Tool for Proxy Testing
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
