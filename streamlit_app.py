<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ProxyStream - Live Demo</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
        }

        .sidebar {
            width: 350px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            overflow-y: auto;
            position: fixed;
            height: 100vh;
        }

        .main-content {
            margin-left: 350px;
            padding: 24px;
            width: calc(100% - 350px);
            min-height: 100vh;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 36px;
            font-weight: 700;
            color: white;
            margin-bottom: 8px;
        }

        .header .subtitle {
            color: #94a3b8;
            font-size: 16px;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }

        .stats-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 16px;
            border-radius: 12px;
            margin: 12px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 14px;
        }

        select {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: white;
            font-size: 14px;
            margin: 8px 0;
        }

        select option {
            background: #1a1f2e;
            color: white;
        }

        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 16px 0;
        }

        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-connect {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-connect:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-disconnect {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }

        .status-connected {
            color: #10b981;
            font-weight: bold;
            background: rgba(16, 185, 129, 0.1);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            margin: 16px 0;
        }

        .status-disconnected {
            color: #ef4444;
            font-weight: bold;
            background: rgba(239, 68, 68, 0.1);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            margin: 16px 0;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 4px;
        }

        .metric-label {
            font-size: 14px;
            color: #94a3b8;
        }

        .world-map {
            height: 250px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }

        .map-dot {
            position: absolute;
            width: 16px;
            height: 16px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 50%;
            border: 3px solid white;
            animation: pulse 2s infinite;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.6);
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            padding: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }

        .disconnected-prompt {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            border: 2px dashed rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Proxy Settings</h2>
        
        <div class="stats-card">
            <strong>Network Statistics</strong><br>
            Total Proxies: <strong>50</strong><br>
            Countries Available: <strong>9</strong><br>
            Protocol: <strong>HTTPS</strong>
        </div>

        <select id="countrySelect">
            <option value="US">🇺🇸 United States</option>
            <option value="CA">🇨🇦 Canada</option>
            <option value="GB">🇬🇧 United Kingdom</option>
            <option value="DE">🇩🇪 Germany</option>
            <option value="FR">🇫🇷 France</option>
            <option value="NL">🇳🇱 Netherlands</option>
            <option value="SG">🇸🇬 Singapore</option>
            <option value="AU">🇦🇺 Australia</option>
            <option value="JP">🇯🇵 Japan</option>
        </select>

        <div class="stats-card" id="countryInfo">
            🇺🇸 <strong>United States</strong><br>
            Available Servers: <strong>5</strong><br>
            Status: <strong>Online</strong>
        </div>

        <select id="proxySelect">
            <option>34.121.105.79:80</option>
            <option>68.107.241.150:8080</option>
            <option>3.133.146.217:5050</option>
            <option>72.10.160.90:13847</option>
            <option>170.85.158.82:80</option>
        </select>

        <div class="button-group">
            <button class="btn btn-connect" id="connectBtn">🔗 Connect</button>
            <button class="btn btn-disconnect" id="disconnectBtn" disabled>❌ Disconnect</button>
        </div>

        <div id="connectionStatus" class="status-disconnected">
            🔴 Disconnected
        </div>

        <div class="card">
            <h3>Connection Info</h3>
            <div id="connectionInfo">
                Select a country and proxy server to connect
            </div>
        </div>
    </div>

    <div class="main-content">
        <div class="header">
            <h1>🛡️ ProxyStream</h1>
            <p class="subtitle">Modern Open-Source VPN Dashboard Demo</p>
        </div>

        <div id="connectedDashboard" style="display: none;">
            <div class="dashboard-grid">
                <div class="card">
                    <h3>🌍 Connection Details</h3>
                    <div style="margin-bottom: 16px; padding: 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; color: #10b981; font-weight: 600;">
                        🟢 Connected via <span id="connectedCountry">🇺🇸 United States</span>
                    </div>
                    <div style="margin-bottom: 20px; color: #94a3b8;">
                        Server: <span id="connectedServer">34.121.105.79:80</span>
                    </div>
                    
                    <div class="world-map">
                        <div class="map-dot" style="top: 45%; left: 30%;"></div>
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Data Usage</h3>
                    <div class="metric-card" style="margin-bottom: 20px;">
                        <div class="metric-value">1.2 GB</div>
                        <div class="metric-label">Session Total</div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div class="metric-card">
                            <div class="metric-value" style="color: #667eea;">0.8</div>
                            <div class="metric-label">Downloaded (GB)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" style="color: #10b981;">0.4</div>
                            <div class="metric-label">Uploaded (GB)</div>
                        </div>
                    </div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div class="metric-card">
                    <div class="metric-value">42ms</div>
                    <div class="metric-label">Latency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">87.3</div>
                    <div class="metric-label">Speed (Mbps)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">99.9%</div>
                    <div class="metric-label">Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">50</div>
                    <div class="metric-label">Total Servers</div>
                </div>
            </div>
        </div>

        <div id="disconnectedDashboard">
            <div class="disconnected-prompt">
                <h2>🔌 Not Connected</h2>
                <p style="margin: 16px 0; color: #94a3b8;">
                    Select a country and proxy server from the sidebar to connect.
                </p>
                
                <h3 style="margin: 24px 0;">🌐 Network Overview</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                    <div class="metric-card">
                        <div class="metric-value">5</div>
                        <div class="metric-label">🇺🇸 US Servers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">4</div>
                        <div class="metric-label">🇨🇦 CA Servers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">3</div>
                        <div class="metric-label">🇬🇧 UK Servers</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p><strong>ProxyStream v2.0</strong> - Modern Open-Source VPN Dashboard</p>
            <p>🔒 Secure • 🚀 Fast • 🌍 Global • ⭐ Open Source</p>
        </div>
    </div>

    <script>
        let isConnected = false;

        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const connectionStatus = document.getElementById('connectionStatus');
        const connectedDashboard = document.getElementById('connectedDashboard');
        const disconnectedDashboard = document.getElementById('disconnectedDashboard');

        function connect() {
            connectBtn.disabled = true;
            connectBtn.textContent = '⏳ Connecting...';
            
            setTimeout(() => {
                isConnected = true;
                
                connectionStatus.className = 'status-connected';
                connectionStatus.textContent = '🟢 Connected';
                
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
                
                connectedDashboard.style.display = 'block';
                disconnectedDashboard.style.display = 'none';
                
                connectBtn.textContent = '🔗 Connect';
            }, 2000);
        }

        function disconnect() {
            isConnected = false;
            
            connectionStatus.className = 'status-disconnected';
            connectionStatus.textContent = '🔴 Disconnected';
            
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            
            connectedDashboard.style.display = 'none';
            disconnectedDashboard.style.display = 'block';
        }

        connectBtn.addEventListener('click', connect);
        disconnectBtn.addEventListener('click', disconnect);
    </script>
</body>
</html>
