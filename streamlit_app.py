# streamlit_app.py
"""
ProxyStream - Modern VPN Dashboard Interface
A comprehensive proxy management and testing platform
"""

import asyncio
import time
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math
import ipaddress
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# Page Configuration
st.set_page_config(
    page_title="ProxyStream - VPN Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Bob-Bragg/proxystream',
        'Report a bug': "https://github.com/Bob-Bragg/proxystream/issues",
        'About': "ProxyStream v2.0 - Modern VPN Dashboard"
    }
)

# Constants
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

GEO_API_ENDPOINTS = [
    "https://ipapi.co/{}/json/",
    "http://ip-api.com/json/{}",
    "https://ipwhois.app/json/{}"
]

COMMON_PORTS = {80, 8080, 3128, 443, 1080, 8888, 8000, 3000}

# Data Models
@dataclass
class ProxyInfo:
    host: str
    port: int
    protocol: str = "http"
    country: Optional[str] = None
    city: Optional[str] = None
    isp: Optional[str] = None
    latency: Optional[float] = None
    last_checked: Optional[datetime] = None
    status: str = "unknown"
    anonymity: str = "unknown"
    speed_score: float = 0.0

@dataclass
class TestResult:
    proxy: ProxyInfo
    success: bool
    response_time: float
    http_status: Optional[int] = None
    detected_ip: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

# Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    /* Dark theme optimizations */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    
    /* Success/Error badges */
    .status-success {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Improve button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Utility Functions
async def fetch_proxies_from_source(session: httpx.AsyncClient, url: str) -> List[str]:
    """Fetch proxy list from a single source."""
    try:
        response = await session.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip().split('\n')
    except Exception as e:
        st.warning(f"Failed to fetch from {url}: {str(e)}")
    return []

async def load_all_proxies() -> List[ProxyInfo]:
    """Load proxies from all configured sources."""
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
                            # Validate IP
                            try:
                                ipaddress.ip_address(host)
                                all_proxies.add((host, port))
                            except ValueError:
                                pass
                    except (ValueError, IndexError):
                        continue
    
    return [ProxyInfo(host=host, port=port) for host, port in all_proxies]

async def test_proxy(proxy: ProxyInfo, timeout: int = 5) -> TestResult:
    """Test a single proxy for connectivity."""
    proxy_url = f"http://{proxy.host}:{proxy.port}"
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(proxy=proxy_url, timeout=timeout) as client:
            response = await client.get("http://httpbin.org/ip")
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return TestResult(
                    proxy=proxy,
                    success=True,
                    response_time=elapsed * 1000,
                    http_status=response.status_code,
                    detected_ip=data.get("origin", "")
                )
    except Exception as e:
        return TestResult(
            proxy=proxy,
            success=False,
            response_time=0,
            error=str(e)
        )
    
    return TestResult(proxy=proxy, success=False, response_time=0)

async def get_geolocation(ip: str) -> Dict[str, Any]:
    """Get geolocation data for an IP address."""
    for endpoint in GEO_API_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(endpoint.format(ip))
                if response.status_code == 200:
                    return response.json()
        except:
            continue
    return {}

async def batch_test_proxies(proxies: List[ProxyInfo], max_concurrent: int = 50) -> List[TestResult]:
    """Test multiple proxies concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def test_with_limit(proxy):
        async with semaphore:
            return await test_proxy(proxy)
    
    tasks = [test_with_limit(proxy) for proxy in proxies]
    return await asyncio.gather(*tasks)

# Streamlit Session State
def init_session_state():
    if 'proxies' not in st.session_state:
        st.session_state.proxies = []
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    if 'selected_proxy' not in st.session_state:
        st.session_state.selected_proxy = None
    if 'proxy_chain' not in st.session_state:
        st.session_state.proxy_chain = []
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'common_ports_only': True,
            'min_speed': 0,
            'country_filter': 'All'
        }

# UI Components
def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">ProxyStream Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888; margin-bottom: 30px;">Modern VPN & Proxy Management Interface</p>', unsafe_allow_html=True)

def render_metrics_row():
    """Render key metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_proxies = len(st.session_state.proxies)
        st.metric("Total Proxies", f"{total_proxies:,}")
    
    with col2:
        if st.session_state.test_results:
            success_rate = sum(1 for r in st.session_state.test_results if r.success) / len(st.session_state.test_results) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "N/A")
    
    with col3:
        if st.session_state.test_results:
            avg_latency = np.mean([r.response_time for r in st.session_state.test_results if r.success])
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        else:
            st.metric("Avg Latency", "N/A")
    
    with col4:
        status = "🟢 Connected" if st.session_state.connected else "🔴 Disconnected"
        st.metric("Status", status)

def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Connection Mode
        mode = st.radio("Connection Mode", ["Single Proxy", "Proxy Chain", "Auto-Rotate"])
        
        # Load Proxies
        if st.button("🔄 Load Proxies", use_container_width=True):
            with st.spinner("Loading proxy lists..."):
                proxies = asyncio.run(load_all_proxies())
                st.session_state.proxies = proxies
                st.success(f"Loaded {len(proxies):,} proxies!")
        
        # Filters
        st.markdown("### 🔍 Filters")
        st.session_state.filter_settings['common_ports_only'] = st.checkbox(
            "Common Ports Only", 
            value=st.session_state.filter_settings['common_ports_only']
        )
        
        # Apply filters
        filtered_proxies = st.session_state.proxies
        if st.session_state.filter_settings['common_ports_only']:
            filtered_proxies = [p for p in filtered_proxies if p.port in COMMON_PORTS]
        
        # Testing
        if filtered_proxies:
            st.markdown("### 🧪 Testing")
            
            test_count = st.slider("Proxies to Test", 10, min(500, len(filtered_proxies)), 50)
            
            if st.button("🚀 Test Proxies", use_container_width=True):
                with st.spinner(f"Testing {test_count} proxies..."):
                    sample = random.sample(filtered_proxies, min(test_count, len(filtered_proxies)))
                    results = asyncio.run(batch_test_proxies(sample))
                    st.session_state.test_results = results
                    
                    success_count = sum(1 for r in results if r.success)
                    st.success(f"Tested {len(results)} proxies: {success_count} working")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("📊 Export Results", use_container_width=True):
            if st.session_state.test_results:
                df = pd.DataFrame([
                    {
                        "Host": r.proxy.host,
                        "Port": r.proxy.port,
                        "Status": "✅ Working" if r.success else "❌ Failed",
                        "Latency (ms)": r.response_time if r.success else None,
                        "Timestamp": r.timestamp
                    }
                    for r in st.session_state.test_results
                ])
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "proxy_results.csv",
                    "text/csv"
                )

def render_main_dashboard():
    """Render main dashboard content."""
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 Test Results", "📈 Analytics", "🗺️ Geo Map"])
    
    with tab1:
        render_dashboard_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_map_tab()

def render_dashboard_tab():
    """Render main dashboard tab."""
    # Quick Stats
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Proxy Distribution")
        if st.session_state.proxies:
            # Port distribution
            port_counts = {}
            for proxy in st.session_state.proxies[:1000]:  # Limit for performance
                port_counts[proxy.port] = port_counts.get(proxy.port, 0) + 1
            
            df_ports = pd.DataFrame(
                list(port_counts.items()), 
                columns=["Port", "Count"]
            ).sort_values("Count", ascending=False).head(10)
            
            fig = px.bar(df_ports, x="Port", y="Count", 
                        title="Top 10 Ports",
                        color="Count",
                        color_continuous_scale="Viridis")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔄 Recent Activity")
        if st.session_state.test_results:
            recent = st.session_state.test_results[-5:]
            for result in reversed(recent):
                if result.success:
                    st.success(f"✅ {result.proxy.host}:{result.proxy.port} - {result.response_time:.0f}ms")
                else:
                    st.error(f"❌ {result.proxy.host}:{result.proxy.port} - Failed")
        else:
            st.info("No recent tests")
    
    # Connection Panel
    st.markdown("### 🔌 Quick Connect")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌍 Connect to Fastest", use_container_width=True):
            if st.session_state.test_results:
                working = [r for r in st.session_state.test_results if r.success]
                if working:
                    fastest = min(working, key=lambda x: x.response_time)
                    st.session_state.selected_proxy = fastest.proxy
                    st.session_state.connected = True
                    st.success(f"Connected to {fastest.proxy.host}:{fastest.proxy.port}")
    
    with col2:
        if st.button("🎲 Random Proxy", use_container_width=True):
            if st.session_state.proxies:
                selected = random.choice(st.session_state.proxies)
                st.session_state.selected_proxy = selected
                st.info(f"Selected {selected.host}:{selected.port}")
    
    with col3:
        if st.button("🔴 Disconnect", use_container_width=True):
            st.session_state.connected = False
            st.session_state.selected_proxy = None
            st.info("Disconnected")

def render_results_tab():
    """Render test results tab."""
    if not st.session_state.test_results:
        st.info("No test results yet. Run a test from the sidebar.")
        return
    
    # Summary stats
    results = st.session_state.test_results
    working = [r for r in results if r.success]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Working Proxies", len(working))
    with col2:
        st.metric("Failed Proxies", len(results) - len(working))
    with col3:
        if working:
            fastest = min(working, key=lambda x: x.response_time)
            st.metric("Fastest", f"{fastest.response_time:.0f}ms")
    
    # Results table
    st.markdown("### 📋 Test Results")
    
    df = pd.DataFrame([
        {
            "Host": r.proxy.host,
            "Port": r.proxy.port,
            "Status": "✅" if r.success else "❌",
            "Latency (ms)": round(r.response_time) if r.success else None,
            "HTTP Status": r.http_status if r.success else None,
            "Error": r.error if not r.success else None,
            "Tested": r.timestamp.strftime("%H:%M:%S")
        }
        for r in results
    ])
    
    # Filter options
    show_only_working = st.checkbox("Show only working proxies")
    if show_only_working:
        df = df[df["Status"] == "✅"]
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Latency (ms)": st.column_config.ProgressColumn(
                "Latency (ms)",
                min_value=0,
                max_value=5000,
                format="%d ms"
            )
        }
    )

def render_analytics_tab():
    """Render analytics tab."""
    if not st.session_state.test_results:
        st.info("No data to analyze. Run tests first.")
        return
    
    results = st.session_state.test_results
    working = [r for r in results if r.success]
    
    if not working:
        st.warning("No working proxies found in test results.")
        return
    
    # Latency distribution
    col1, col2 = st.columns(2)
    
    with col1:
        latencies = [r.response_time for r in working]
        fig = px.histogram(latencies, nbins=30, 
                          title="Latency Distribution",
                          labels={"value": "Latency (ms)", "count": "Count"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Success rate over time
        df_time = pd.DataFrame([
            {"Time": r.timestamp, "Success": 1 if r.success else 0}
            for r in results
        ])
        df_time = df_time.groupby(pd.Grouper(key="Time", freq="1Min")).mean()
        
        fig = px.line(df_time, y="Success", 
                     title="Success Rate Over Time",
                     labels={"Success": "Success Rate", "Time": "Time"})
        fig.update_yaxis(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    # Port analysis
    st.markdown("### 🔌 Port Analysis")
    port_stats = {}
    for r in results:
        port = r.proxy.port
        if port not in port_stats:
            port_stats[port] = {"total": 0, "success": 0}
        port_stats[port]["total"] += 1
        if r.success:
            port_stats[port]["success"] += 1
    
    df_ports = pd.DataFrame([
        {
            "Port": port,
            "Total": stats["total"],
            "Working": stats["success"],
            "Success Rate": stats["success"] / stats["total"] * 100
        }
        for port, stats in port_stats.items()
    ]).sort_values("Success Rate", ascending=False)
    
    fig = px.bar(df_ports, x="Port", y="Success Rate",
                 title="Success Rate by Port",
                 color="Success Rate",
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)

def render_map_tab():
    """Render geographic map tab."""
    st.markdown("### 🗺️ Proxy Geographic Distribution")
    
    # Simulate some geographic data for demonstration
    if st.session_state.test_results:
        # Create sample geographic data
        sample_locations = [
            {"lat": 40.7128, "lon": -74.0060, "city": "New York", "count": random.randint(10, 50)},
            {"lat": 51.5074, "lon": -0.1278, "city": "London", "count": random.randint(10, 50)},
            {"lat": 48.8566, "lon": 2.3522, "city": "Paris", "count": random.randint(10, 50)},
            {"lat": 35.6762, "lon": 139.6503, "city": "Tokyo", "count": random.randint(10, 50)},
            {"lat": -33.8688, "lon": 151.2093, "city": "Sydney", "count": random.randint(10, 50)},
        ]
        
        df_map = pd.DataFrame(sample_locations)
        
        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            size="count",
            color="count",
            hover_name="city",
            hover_data=["count"],
            color_continuous_scale="Viridis",
            size_max=20,
            zoom=1,
            title="Proxy Server Locations"
        )
        
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            height=500,
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run proxy tests to see geographic distribution")

# Main Application
def main():
    load_custom_css()
    init_session_state()
    
    render_header()
    render_metrics_row()
    
    # Sidebar
    render_sidebar()
    
    # Main content
    render_main_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            ProxyStream v2.0 | 
            <a href='https://github.com/Bob-Bragg/proxystream' style='color: #667eea;'>GitHub</a> | 
            <a href='https://proxystream.streamlit.app' style='color: #667eea;'>Live Demo</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# requirements.txt
streamlit>=1.29.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
httpx>=0.25.0
tenacity>=8.2.0

# README.md
# 🔒 ProxyStream

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proxystream.streamlit.app)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)

Modern open-source VPN dashboard interface with integrated proxy management.

## ✨ Features

- **🚀 Real-time Proxy Testing** - Test hundreds of proxies simultaneously with async operations
- **📊 Advanced Analytics** - Visualize proxy performance, latency distribution, and success rates
- **🗺️ Geographic Mapping** - See proxy locations worldwide on an interactive map
- **🔄 Auto-Rotation** - Automatically switch between working proxies
- **⚡ Batch Operations** - Test up to 500 proxies concurrently
- **📈 Performance Metrics** - Track latency, success rates, and response times
- **🎨 Modern UI** - Beautiful dark theme with animated gradients
- **📱 Responsive Design** - Works on desktop and mobile devices

## 🚀 Live Demo

Try the live demo: [https://proxystream.streamlit.app](https://proxystream.streamlit.app)

## 🛠️ Installation

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Bob-Bragg/proxystream.git
cd proxystream
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

### Docker

```bash
# Build the image
docker build -t proxystream .

# Run the container
docker run -p 8501:8501 proxystream
```

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 📖 Usage

1. **Load Proxies**: Click "Load Proxies" in the sidebar to fetch proxy lists from multiple sources
2. **Test Proxies**: Select the number of proxies to test and click "Test Proxies"
3. **View Results**: Check the results in the Dashboard, Test Results, and Analytics tabs
4. **Connect**: Use "Connect to Fastest" to automatically connect to the best performing proxy
5. **Export**: Download test results as CSV for further analysis

## 🏗️ Architecture

```
proxystream/
├── streamlit_app.py      # Main application
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
├── LICENSE              # Apache 2.0 license
└── .github/
    └── workflows/       # CI/CD pipelines
```

## 🔧 Configuration

The application can be configured through environment variables:

```bash
# Streamlit configuration
STREAMLIT_THEME_BASE="dark"
STREAMLIT_THEME_PRIMARY_COLOR="#667eea"

# Proxy settings
MAX_CONCURRENT_TESTS=100
DEFAULT_TIMEOUT=5
```

## 📊 Features in Detail

### Proxy Testing
- Concurrent testing of multiple proxies
- HTTP/HTTPS protocol support
- Latency measurement
- Success rate calculation
- Error tracking and reporting

### Analytics Dashboard
- Real-time metrics display
- Latency distribution charts
- Success rate over time
- Port analysis
- Geographic distribution map

### Data Management
- Export results to CSV
- Filter working proxies
- Sort by performance metrics
- Search functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Charts powered by [Plotly](https://plotly.com)
- Async operations with [httpx](https://www.python-httpx.org)

## 📧 Contact

- GitHub: [@Bob-Bragg](https://github.com/Bob-Bragg)
- Project Link: [https://github.com/Bob-Bragg/proxystream](https://github.com/Bob-Bragg/proxystream)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Bob-Bragg/proxystream&type=Date)](https://star-history.com/#Bob-Bragg/proxystream&Date)

---

Made with ❤️ by the ProxyStream Team

# LICENSE
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction,
and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by
the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all
other entities that control, are controlled by, or are under common
control with that entity.

"You" (or "Your") shall mean an individual or Legal Entity
exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation
source, and configuration files.

"Object" form shall mean any form resulting from mechanical
transformation or translation of a Source form, including but
not limited to compiled object code, generated documentation,
and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or
Object form, made available under the License.

"Derivative Works" shall mean any work, whether in Source or Object
form, that is based on (or derived from) the Work.

"Contribution" shall mean any work of authorship, including
the original version of the Work and any modifications or additions
to that Work or Derivative Works thereof.

"Contributor" shall mean Licensor and any individual or Legal Entity
on behalf of whom a Contribution has been received by Licensor.

2. Grant of Copyright License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
copyright license to reproduce, prepare Derivative Works of,
publicly display, publicly perform, sublicense, and distribute the
Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
(except as stated in this section) patent license to make, have made,
use, offer to sell, sell, import, and otherwise transfer the Work.

4. Redistribution. You may reproduce and distribute copies of the
Work or Derivative Works thereof in any medium, with or without
modifications, and in Source or Object form, provided that You
meet the following conditions:

(a) You must give any other recipients of the Work or
    Derivative Works a copy of this License; and

(b) You must cause any modified files to carry prominent notices
    stating that You changed the files; and

(c) You must retain, in the Source form of any Derivative Works
    that You distribute, all copyright, patent, trademark, and
    attribution notices from the Source form of the Work; and

(d) If the Work includes a "NOTICE" text file as part of its
    distribution, then any Derivative Works that You distribute must
    include a readable copy of the attribution notices.

5. Submission of Contributions. Unless You explicitly state otherwise,
any Contribution intentionally submitted for inclusion in the Work
by You to the Licensor shall be under the terms and conditions of
this License, without any additional terms or conditions.

6. Trademarks. This License does not grant permission to use the trade
names, trademarks, service marks, or product names of the Licensor,
except as required for reasonable and customary use in describing the
origin of the Work.

7. Disclaimer of Warranty. Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE.

8. Limitation of Liability. In no event and under no legal theory,
whether in tort (including negligence), contract, or otherwise,
unless required by applicable law (such as deliberate and grossly
negligent acts) or agreed to in writing, shall any Contributor be
liable to You for damages.

9. Accepting Warranty or Additional Liability. While redistributing
the Work or Derivative Works thereof, You may choose to offer,
and charge a fee for, acceptance of support, warranty, indemnity,
or other liability obligations and/or rights consistent with this
License.

END OF TERMS AND CONDITIONS

Copyright 2024 ProxyStream Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# macOS
.DS_Store

# Windows
Thumbs.db
ehthumbs.db

# Streamlit
.streamlit/secrets.toml

# Project specific
*.csv
*.json
!package.json
!package-lock.json
logs/
temp/
cache/
