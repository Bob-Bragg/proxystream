import re, os, json, time, random, socket, asyncio, hashlib
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import requests
import aiohttp
from aiohttp import ClientTimeout
try:
    from aiohttp_socks import ProxyConnector  # for socks4/5
except Exception:
    ProxyConnector = None

# ------------------ Page config & theme ------------------
st.set_page_config(
    page_title="ProxyStream",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background:#1e1e1e; color:#f3f3f3; }
  .main-header { text-align:center; font-size:28px; font-weight:800; color:#4CAF50; margin:-8px 0 8px; }
  [data-testid="metric-container"] { background:#2b2b2b; border:1px solid #3b3b3b; border-radius:12px; padding:.85rem; }
  [data-testid="metric-container"] label { color:#d8d8d8 !important; }
  #MainMenu, header, footer { visibility:hidden; }
  .chip { display:inline-flex; gap:.5rem; align-items:center; padding:.25rem .6rem; border-radius:999px; border:1px solid #3a3a3a; background:#2a2a2a; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌐 Proxy<span style="color:#2196F3;">Stream</span></div>', unsafe_allow_html=True)
st.caption("Full-list ingestion • async validation • geolocation • latency analytics • exports")

# ------------------ Utils ------------------
PROXY_RE = re.compile(
    r"""^(?:(?P<scheme>https?|socks4|socks5)://)?(?:(?P<user>[^:@\s]+):(?P<pw>[^@\s]+)@)?(?P<host>\[[0-9a-fA-F:]+\]|[^:\s]+):(?P<port>\d{2,5})$""",
    re.X,
)

def parse_proxy(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith("#"): return None
    m = PROXY_RE.match(line)
    if m:
        d = m.groupdict()
        return {
            "scheme": (d["scheme"] or "http").lower(),
            "user": d["user"], "pw": d["pw"],
            "host": d["host"].strip("[]"),
            "port": int(d["port"]),
            "raw": line
        }
    # user:pass@ip:port without scheme
    if "@" in line and line.count(":") >= 2:
        try:
            creds, hp = line.split("@", 1)
            u, p = creds.split(":", 1)
            h, pr = hp.rsplit(":", 1)
            return {"scheme":"http","user":u,"pw":p,"host":h,"port":int(pr),"raw":line}
        except: return None
    # ip:port only
    if line.count(":") == 1:
        h, pr = line.split(":")
        return {"scheme":"http","user":None,"pw":None,"host":h,"port":int(pr),"raw":line}
    return None

def norm_proxy(p: Dict) -> str:
    auth = f"{p['user']}:{p['pw']}@" if p.get("user") and p.get("pw") else ""
    return f"{p['scheme']}://{auth}{p['host']}:{p['port']}"

def tcp_ping(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, int(port))) == 0
    except: return False

# ------------------ Data sources ------------------
RAW_URL = "https://raw.githubusercontent.com/arandomguyhere/Proxy-Hound/refs/heads/main/docs/by_type/https_hunted.txt"

@st.cache_data(ttl=3600, show_spinner=False)
def load_proxy_list() -> List[str]:
    try:
        r = requests.get(RAW_URL, timeout=12)
        if r.ok:
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip() and ":" in ln]
            return lines
    except: pass
    # tiny fallback
    return ["34.121.105.79:80","68.107.241.150:8080","3.133.146.217:5050"]

# ------------------ Geolocation (ip-api batch) ------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geolocate_ips(ips: List[str]) -> Dict[str, Dict]:
    out = {}
    if not ips: return out
    url = "http://ip-api.com/batch?fields=status,country,countryCode,lat,lon,query"
    uniq = list(dict.fromkeys(ips))
    for i in range(0, len(uniq), 100):
        chunk = uniq[i:i+100]
        try:
            resp = requests.post(url, json=[{"query": ip} for ip in chunk], timeout=12)
            if resp.ok:
                for row in resp.json():
                    if row.get("status") == "success":
                        out[row["query"]] = {
                            "country": row.get("country"),
                            "countryCode": row.get("countryCode"),
                            "lat": row.get("lat"),
                            "lon": row.get("lon"),
                        }
        except: pass
    return out

# ------------------ Async validation ------------------
async def _probe_http_https(proxy: Dict, url: str, timeout_s: float) -> Dict:
    t0 = time.perf_counter()
    to = ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=to) as sess:
        async with sess.get(url, proxy=norm_proxy(proxy)) as r:
            text = await r.text()
    dt = (time.perf_counter() - t0) * 1000.0
    ip = None
    try:
        j = json.loads(text)
        ip = j.get("origin") or j.get("ip")
    except: pass
    return {"alive": True, "latency_ms": round(dt,1), "outward_ip": ip, "error": ""}

async def _probe_socks(proxy: Dict, url: str, timeout_s: float) -> Dict:
    if not ProxyConnector:
        raise RuntimeError("Install aiohttp-socks for SOCKS support")
    t0 = time.perf_counter()
    to = ClientTimeout(total=timeout_s)
    conn = ProxyConnector.from_url(norm_proxy(proxy))
    async with aiohttp.ClientSession(timeout=to, connector=conn) as sess:
        async with sess.get(url) as r:
            text = await r.text()
    dt = (time.perf_counter() - t0) * 1000.0
    ip = None
    try:
        j = json.loads(text)
        ip = j.get("origin") or j.get("ip")
    except: pass
    return {"alive": True, "latency_ms": round(dt,1), "outward_ip": ip, "error": ""}

async def _validate_one(proxy: Dict, url: str, timeout_s: float) -> Dict:
    try:
        if proxy["scheme"].startswith("socks"):
            res = await _probe_socks(proxy, url, timeout_s)
        else:
            res = await _probe_http_https(proxy, url, timeout_s)
        return {**proxy, **res}
    except Exception as e:
        # TCP fallback: if port open, mark “alive” but without confirmed egress
        alive = tcp_ping(proxy["host"], proxy["port"], timeout=min(5, timeout_s))
        return {**proxy, "alive": bool(alive), "latency_ms": np.nan, "outward_ip": None, "error": str(e)[:140]}

async def validate_batch(proxies: List[Dict], url: str, timeout_s: float, max_conc: int) -> List[Dict]:
    sem = asyncio.Semaphore(max_conc)
    async def wrap(p): 
        async with sem: 
            return await _validate_one(p, url, timeout_s)
    return await asyncio.gather(*[wrap(p) for p in proxies])

def validate_proxies(proxies: List[Dict], url: str, timeout_s: float, max_conc: int) -> pd.DataFrame:
    try:
        rows = asyncio.run(validate_batch(proxies, url, timeout_s, max_conc))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        rows = loop.run_until_complete(validate_batch(proxies, url, timeout_s, max_conc))
    return pd.DataFrame(rows)

# ------------------ Quality badge & metrics ------------------
def connection_quality(latency: float, speed_guess: float) -> Dict:
    if (latency or 999) < 50 and speed_guess > 50:
        return {"quality": "Excellent", "color": "#10b981", "icon": "🟢"}
    if (latency or 999) < 100 and speed_guess > 25:
        return {"quality": "Good", "color": "#f59e0b", "icon": "🟡"}
    return {"quality": "Poor", "color": "#ef4444", "icon": "🔴"}

def enhanced_vpn_metrics() -> Dict:
    return {
        "encryption": "AES-256-GCM",
        "protocol": "OpenVPN",
        "dns_leak_protection": True,
        "kill_switch": True,
        "bandwidth_unlimited": True,
        "simultaneous_connections": 5,
        "logs_policy": "No logs",
        "server_load": random.randint(5, 85),
    }

# ------------------ Sidebar settings ------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    TEST_URL = st.text_input("Test URL", "https://httpbin.org/ip", help="Keep tiny & fast.")
    TIMEOUT = st.slider("Timeout (sec)", 2, 30, 8)
    MAX_CONC = st.slider("Max concurrency", 1, 200, 32)
    GEOLOCATE = st.toggle("Geolocate outgoing IPs", True)
    st.markdown("—")
    st.markdown("### 🧪 Connection prefs")
    auto_reconnect = st.checkbox("Auto-reconnect on failure", True)
    preferred_protocol = st.selectbox("Protocol", ["OpenVPN","IKEv2","WireGuard"], index=0)
    kill_switch = st.checkbox("Kill switch", True)
    dns_choice = st.selectbox("DNS", ["Auto","Cloudflare","Google","OpenDNS"], index=0)

# ------------------ Input & merge sources ------------------
st.markdown("### 📥 Proxies")
colA, colB = st.columns([2,1])
with colA:
    user_txt = st.text_area(
        "Paste proxies (one per line). Supports http(s), socks4/5, with or without auth.",
        height=140,
        placeholder="http://user:pass@1.2.3.4:8080\nsocks5://5.6.7.8:1080\n9.9.9.9:3128",
    )
with colB:
    uploaded = st.file_uploader("Or upload a .txt list", type=["txt"])
    if uploaded:
        user_txt = (user_txt + "\n" + uploaded.read().decode("utf-8")).strip()

# Load full list (cached) and merge with user input
repo_list = load_proxy_list()
raw_lines = (user_txt or "").splitlines() + repo_list
parsed = [p for p in (parse_proxy(ln) for ln in raw_lines) if p]
dedup = {norm_proxy(p): p for p in parsed}
proxies = list(dedup.values())

st.write(f"Loaded **{len(repo_list)}** from repo, merged to **{len(proxies)}** total after parsing/dedup.")

# ------------------ Connection logs ------------------
if "connection_log" not in st.session_state:
    st.session_state.connection_log = []

def log_event(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.connection_log.append(f"[{ts}] {msg}")
    if len(st.session_state.connection_log) > 200:
        st.session_state.connection_log = st.session_state.connection_log[-200:]

# ------------------ Validate button & progress ------------------
go_validate = st.button("✅ Validate Proxies", type="primary")
results_df = None

if go_validate:
    if not proxies:
        st.warning("No valid proxies found. Paste or upload some.")
    else:
        with st.spinner("Checking proxies…"):
            progress = st.progress(0)
            # small warm-up to show motion
            for i in range(10): time.sleep(0.01); progress.progress(i+1)
            results_df = validate_proxies(proxies, TEST_URL, TIMEOUT, MAX_CONC)
            progress.progress(100)
        log_event(f"Validated {len(proxies)} proxies via {TEST_URL}")

# ------------------ Results & analytics ------------------
if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    # Choose an estimated speed from latency (purely cosmetic; replace with real speed test if desired)
    results_df["speed_guess_mbps"] = results_df["latency_ms"].apply(
        lambda x: 80.0 if pd.notna(x) and x < 50 else (40.0 if pd.notna(x) and x < 100 else 8.0)
    )
    # Geolocate outward IPs (prefer outward_ip; fallback to host if IPv4)
    if GEOLOCATE:
        out_ips = []
        for _, r in results_df.iterrows():
            ip = (r.get("outward_ip") or r.get("host") or "")
            ip = ip.split(",")[0].strip()
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
                out_ips.append(ip)
        geo_map = geolocate_ips(out_ips)
        results_df["country"] = results_df.apply(
            lambda r: geo_map.get((r.get("outward_ip") or r.get("host") or "").split(",")[0].strip(), {}).get("country"), axis=1
        )
        results_df["countryCode"] = results_df.apply(
            lambda r: geo_map.get((r.get("outward_ip") or r.get("host") or "").split(",")[0].strip(), {}).get("countryCode"), axis=1
        )
        results_df["lat"] = results_df.apply(
            lambda r: geo_map.get((r.get("outward_ip") or r.get("host") or "").split(",")[0].strip(), {}).get("lat"), axis=1
        )
        results_df["lon"] = results_df.apply(
            lambda r: geo_map.get((r.get("outward_ip") or r.get("host") or "").split(",")[0].strip(), {}).get("lon"), axis=1
        )

    alive = results_df["alive"].mean() * 100.0
    median_lat = results_df.loc[results_df["alive"], "latency_ms"].median() if results_df["alive"].any() else np.nan
    best = results_df.loc[results_df["alive"]].sort_values("latency_ms").head(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌐 Total", f"{len(results_df)}")
    c2.metric("✅ Alive", f"{alive:.1f}%")
    c3.metric("⚡ Median Latency", f"{median_lat:.0f} ms" if not np.isnan(median_lat) else "—")
    if not best.empty:
        b = best.iloc[0]
        c4.metric("🥇 Fastest", f"{b['latency_ms']:.0f} ms", help=norm_proxy(b.to_dict())[:60] + "…")
    else:
        c4.metric("🥇 Fastest", "—")

    # connection “quality” chip (based on median)
    q = connection_quality(median_lat if not np.isnan(median_lat) else 999, 40.0)
    st.markdown(f'<div class="chip">{q["icon"]} <b>{q["quality"]}</b> quality</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Latency (alive only)")
        alive_df = results_df[results_df["alive"] & results_df["latency_ms"].notna()]
        if not alive_df.empty:
            fig = px.histogram(alive_df, x="latency_ms", nbins=30)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10,r=10,t=10,b=10), xaxis_title="ms", yaxis_title="count")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No alive proxies to plot.")

    with col2:
        st.markdown("#### Status breakdown")
        vc = results_df["alive"].map({True:"alive", False:"dead"}).value_counts()
        fig2 = px.pie(pd.DataFrame({"status": vc.index, "count": vc.values}), names="status", values="count", hole=0.55)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    if GEOLOCATE and results_df["lat"].notna().any():
        st.markdown("#### 🌍 Map of outward IPs")
        mdf = results_df[results_df["lat"].notna()]
        figm = go.Figure(go.Scattergeo(
            lon=mdf["lon"], lat=mdf["lat"],
            text=[f"{norm_proxy(r)}<br>{r['country'] or ''} · {r['latency_ms'] if pd.notna(r['latency_ms']) else '—'} ms" for _, r in mdf.iterrows()],
            mode='markers',
            marker=dict(
                size=np.clip(1200.0 / (mdf["latency_ms"].fillna(999) + 50), 4, 16),
                line=dict(width=.8, color='white'),
                color=np.where(mdf["alive"], "#4CAF50", "#F44336"),
            )
        ))
        figm.update_layout(
            geo=dict(projection_type='natural earth', showland=True, landcolor='#3d3d3d', showocean=True, oceancolor='#1b1b1b', coastlinecolor='#808080'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0), height=360
        )
        st.plotly_chart(figm, use_container_width=True, config={"displayModeBar": False})

    # Results table & export
    show_cols = ["alive","latency_ms","scheme","host","port","user","outward_ip","country","countryCode","error","speed_guess_mbps"]
    for c in show_cols:
        if c not in results_df.columns: results_df[c] = None
    st.markdown("#### Results")
    st.dataframe(results_df[show_cols], use_container_width=True, hide_index=True)

    colx, coly = st.columns(2)
    colx.download_button("⬇️ CSV", data=results_df.to_csv(index=False).encode(), file_name=f"proxystream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    coly.download_button("⬇️ JSON", data=results_df.to_json(orient="records").encode(), file_name=f"proxystream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# ------------------ Advanced settings / VPN metrics ------------------
with st.expander("🔧 Advanced / VPN-style metrics"):
    m = enhanced_vpn_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Protocol", m["protocol"])
    c2.metric("Encryption", m["encryption"])
    c3.metric("Kill switch", "On" if m["kill_switch"] else "Off")
    c4.metric("Server load", f"{m['server_load']}%")
    st.caption("These are illustrative; wire to your real backend as it matures.")

# ------------------ Connection logs ------------------
with st.expander("📜 Connection log"):
    if st.button("Clear log"): st.session_state.connection_log = []
    for line in st.session_state.connection_log[-40:]:
        st.text(line)
