#!/usr/bin/env python3
"""Enhanced Flask server for comprehensive sensor telemetry visualization.

Features:
- Temperature & Humidity (DHT11)
- Heart Rate & SpO2 (MAX30100)
- GPS Location with interactive map
- Ultrasonic Distance Sensors (Front, Left, Right)
- Real-time charts and visual indicators
"""
import atexit
import json
import signal
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify

from sensor_monitor_enhanced import (
    get_latest_dht,
    get_latest_gps,
    get_latest_health,
    get_latest_ultrasonic,
    start_monitoring,
    stop_monitoring,
)

CONFIG_PATH = Path(__file__).parent / "config.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8877

app = Flask(__name__)

try:
    CONFIG = json.loads(CONFIG_PATH.read_text())
except FileNotFoundError:
    CONFIG = {}


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Smart Wheelchair Sensor Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1629 100%);
      color: #e8eef7;
      min-height: 100vh;
      padding: 20px;
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    
    header {
      text-align: center;
      margin-bottom: 30px;
      padding: 20px;
      background: rgba(15, 25, 50, 0.6);
      border-radius: 20px;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(100, 140, 255, 0.2);
    }
    
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, #4da6ff, #9575ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    
    .timestamp {
      color: #8fa8d4;
      font-size: 0.95rem;
    }
    
    .grid {
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    
    .card {
      background: rgba(20, 30, 60, 0.7);
      border-radius: 20px;
      padding: 24px;
      backdrop-filter: blur(15px);
      border: 1px solid rgba(100, 140, 255, 0.15);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 25px 70px rgba(0, 0, 0, 0.5);
    }
    
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: #b8d0ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .icon {
      font-size: 1.4rem;
    }
    
    .metric-value {
      font-size: 2.8rem;
      font-weight: 700;
      margin: 12px 0;
    }
    
    .metric-label {
      font-size: 0.85rem;
      color: #7a95c4;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    
    .temp-gradient {
      background: linear-gradient(135deg, #ff6b6b, #ff9f43);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    .humidity-gradient {
      background: linear-gradient(135deg, #4facfe, #00f2fe);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    .heart-gradient {
      background: linear-gradient(135deg, #ff6b9d, #c44569);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.7; transform: scale(1.05); }
    }
    
    .spo2-gradient {
      background: linear-gradient(135deg, #5ed3a3, #2ecc71);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    .distance-bar {
      width: 100%;
      height: 30px;
      background: rgba(40, 50, 80, 0.5);
      border-radius: 15px;
      overflow: hidden;
      margin: 10px 0;
      position: relative;
    }
    
    .distance-fill {
      height: 100%;
      background: linear-gradient(90deg, #4facfe, #00f2fe);
      transition: width 0.3s ease;
      border-radius: 15px;
    }
    
    .distance-fill.warning { background: linear-gradient(90deg, #ffcc00, #ff9f43); }
    .distance-fill.danger { background: linear-gradient(90deg, #ff6b6b, #c44569); }
    
    .distance-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-weight: 600;
      font-size: 0.9rem;
      text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    #map {
      width: 100%;
      height: 400px;
      border-radius: 16px;
      margin-top: 16px;
    }
    
    .map-card {
      grid-column: 1 / -1;
    }
    
    canvas {
      margin-top: 16px;
    }
    
    .sensor-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }
    
    .mini-card {
      background: rgba(30, 40, 70, 0.4);
      padding: 12px;
      border-radius: 12px;
      border: 1px solid rgba(100, 140, 255, 0.1);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🦽 Smart Wheelchair Telemetry</h1>
      <div class="timestamp" id="timestamp">Connecting...</div>
    </header>
    
    <div class="grid">
      <!-- Temperature & Humidity -->
      <div class="card">
        <div class="card-title"><span class="icon">🌡️</span>Environment</div>
        <div class="mini-card">
          <div class="metric-label">Temperature</div>
          <div class="metric-value temp-gradient" id="temperature">--</div>
        </div>
        <div class="mini-card" style="margin-top: 12px;">
          <div class="metric-label">Humidity</div>
          <div class="metric-value humidity-gradient" id="humidity">--</div>
        </div>
      </div>
      
      <!-- Health Monitoring -->
      <div class="card">
        <div class="card-title"><span class="icon">💓</span>Health Monitor</div>
        <div class="mini-card">
          <div class="metric-label">Heart Rate</div>
          <div class="metric-value heart-gradient" id="heartRate">--</div>
        </div>
        <div class="mini-card" style="margin-top: 12px;">
          <div class="metric-label">Blood Oxygen</div>
          <div class="metric-value spo2-gradient" id="spo2">--</div>
        </div>
      </div>
      
      <!-- Ultrasonic Distances -->
      <div class="card">
        <div class="card-title"><span class="icon">📡</span>Distance Sensors</div>
        <div class="sensor-grid">
          <div class="mini-card">
            <div class="metric-label">Front</div>
            <div id="frontDist" style="font-size: 1.5rem; font-weight: 600; margin: 8px 0;">-- cm</div>
            <div class="distance-bar">
              <div class="distance-fill" id="frontBar" style="width: 0%;"></div>
            </div>
          </div>
          <div class="mini-card">
            <div class="metric-label">Left</div>
            <div id="leftDist" style="font-size: 1.5rem; font-weight: 600; margin: 8px 0;">-- cm</div>
            <div class="distance-bar">
              <div class="distance-fill" id="leftBar" style="width: 0%;"></div>
            </div>
          </div>
          <div class="mini-card">
            <div class="metric-label">Right</div>
            <div id="rightDist" style="font-size: 1.5rem; font-weight: 600; margin: 8px 0;">-- cm</div>
            <div class="distance-bar">
              <div class="distance-fill" id="rightBar" style="width: 0%;"></div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- GPS Map -->
      <div class="card map-card">
        <div class="card-title"><span class="icon">🗺️</span>GPS Location</div>
        <div class="sensor-grid">
          <div class="mini-card">
            <div class="metric-label">Latitude</div>
            <div id="latitude" style="font-size: 1.2rem; font-weight: 600; margin-top: 4px;">--</div>
          </div>
          <div class="mini-card">
            <div class="metric-label">Longitude</div>
            <div id="longitude" style="font-size: 1.2rem; font-weight: 600; margin-top: 4px;">--</div>
          </div>
          <div class="mini-card">
            <div class="metric-label">Satellites</div>
            <div id="numSats" style="font-size: 1.2rem; font-weight: 600; margin-top: 4px;">--</div>
          </div>
        </div>
        <div id="map"></div>
      </div>
      
      <!-- Charts -->
      <div class="card" style="grid-column: 1 / -1;">
        <div class="card-title"><span class="icon">📊</span>Sensor Trends</div>
        <canvas id="envChart"></canvas>
        <canvas id="healthChart" style="margin-top: 30px;"></canvas>
      </div>
    </div>
  </div>
  
  <script>
    let envChart, healthChart, map, gpsMarker;
    const MAX_POINTS = 60;
    
    // Initialize map
    map = L.map('map').setView([20.5937, 78.9629], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Initialize charts
    function initCharts() {
      const chartOptions = {
        animation: false,
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { labels: { color: '#c8d6f0' } } },
        scales: {
          y: { ticks: { color: '#9fb3e0' }, grid: { color: 'rgba(100, 130, 200, 0.15)' } },
          x: { ticks: { color: '#9fb3e0' }, grid: { display: false } }
        }
      };
      
      envChart = new Chart(document.getElementById('envChart'), {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            { label: 'Temp (°C)', data: [], borderColor: '#ff9f43', backgroundColor: 'rgba(255, 159, 67, 0.1)', tension: 0.4 },
            { label: 'Humidity (%)', data: [], borderColor: '#4facfe', backgroundColor: 'rgba(79, 172, 254, 0.1)', tension: 0.4 }
          ]
        },
        options: chartOptions
      });
      
      healthChart = new Chart(document.getElementById('healthChart'), {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            { label: 'Heart Rate (bpm)', data: [], borderColor: '#ff6b9d', backgroundColor: 'rgba(255, 107, 157, 0.1)', tension: 0.4 },
            { label: 'Raw IR Signal', data: [], borderColor: '#5ed3a3', backgroundColor: 'rgba(94, 211, 163, 0.1)', tension: 0.4, yAxisID: 'y1' }
          ]
        },
        options: {
          ...chartOptions,
          scales: {
            ...chartOptions.scales,
            y1: { position: 'right', grid: { display: false }, ticks: { color: '#5ed3a3' } }
          }
        }
      });
    }
    
    function updateDistance(sensor, distance) {
      const maxDist = 200;
      const percent = distance ? Math.min((distance / maxDist) * 100, 100) : 0;
      const bar = document.getElementById(`${sensor}Bar`);
      const text = document.getElementById(`${sensor}Dist`);
      
      text.textContent = distance ? `${distance} cm` : '-- cm';
      bar.style.width = `${percent}%`;
      
      bar.className = 'distance-fill';
      if (distance && distance < 30) bar.classList.add('danger');
      else if (distance && distance < 60) bar.classList.add('warning');
    }
    
    function updateCharts(label, data) {
      [envChart, healthChart].forEach(chart => {
        chart.data.labels.push(label);
        if (chart.data.labels.length > MAX_POINTS) {
          chart.data.labels.shift();
          chart.data.datasets.forEach(d => d.data.shift());
        }
      });
      
      envChart.data.datasets[0].data.push(data.temperature);
      envChart.data.datasets[1].data.push(data.humidity);
      healthChart.data.datasets[0].data.push(data.heartRate);
      healthChart.data.datasets[1].data.push(data.spo2);
      
      envChart.update('none');
      healthChart.update('none');
    }
    
    async function fetchData() {
      try {
        const res = await fetch('/api/data');
        if (!res.ok) {
          console.error('API request failed:', res.status);
          return;
        }
        const payload = await res.json();
        console.log('Received data:', payload);
        const ts = new Date();
        
        document.getElementById('timestamp').textContent = `Last update: ${ts.toLocaleString()}`;
        
        const dht = payload.dht || {};
        const health = payload.health || {};
        const gps = payload.gps || {};
        const ultrasonic = payload.ultrasonic || {};
        
        // Temperature & Humidity
        document.getElementById('temperature').textContent = (dht.temperature_c != null) ? `${dht.temperature_c.toFixed(1)}°C` : '--';
        document.getElementById('humidity').textContent = (dht.humidity_percent != null) ? `${dht.humidity_percent.toFixed(1)}%` : '--';
        
        // Health
        const ir = health.ir_value || 0;
        const red = health.red_value || 0;
        const bpm = health.heart_rate_bpm;
        
        document.getElementById('heartRate').textContent = (bpm != null) ? `${Math.round(bpm)} bpm` : '--';
        document.getElementById('spo2').innerHTML = `IR: ${ir}<br><span style="font-size:0.6em">RED: ${red}</span>`;
        
        // Ultrasonic
        updateDistance('front', ultrasonic.front_cm);
        updateDistance('left', ultrasonic.left_cm);
        updateDistance('right', ultrasonic.right_cm);
        
        // GPS
        document.getElementById('latitude').textContent = (gps.latitude != null) ? gps.latitude.toFixed(6) : '--';
        document.getElementById('longitude').textContent = (gps.longitude != null) ? gps.longitude.toFixed(6) : '--';
        document.getElementById('numSats').textContent = (gps.num_sats != null) ? gps.num_sats : '--';
        
        if (gps.latitude != null && gps.longitude != null) {
          const pos = [gps.latitude, gps.longitude];
          map.setView(pos, Math.max(map.getZoom(), 15));
          if (gpsMarker) gpsMarker.setLatLng(pos);
          else gpsMarker = L.marker(pos).addTo(map).bindPopup('Wheelchair Location');
        }
        
        updateCharts(ts.toLocaleTimeString(), {
          temperature: dht.temperature_c,
          humidity: dht.humidity_percent,
          heartRate: bpm,
          rawIR: ir
        });
      } catch (err) {
        console.error('Failed to fetch data:', err);
        document.getElementById('timestamp').textContent = 'Error: ' + err.message;
      }
    }
    
    initCharts();
    fetchData();
    setInterval(fetchData, 1000);
  </script>
</body>
</html>
"""


@app.route("/")
def index() -> Response:
    """Serve the dashboard page."""
    return Response(HTML_TEMPLATE, mimetype="text/html")


@app.route("/api/data")
def api_data() -> Response:
    """Return the latest sensor readings as JSON."""
    payload = {
        "timestamp": time.time(),
        "dht": get_latest_dht(),
        "health": get_latest_health(),
        "gps": get_latest_gps(),
        "ultrasonic": get_latest_ultrasonic(),
    }
    return jsonify(payload)


def _graceful_shutdown(*_args) -> None:
    stop_monitoring()
    sys.exit(0)


def _register_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _graceful_shutdown)


def _init_monitoring() -> None:
    start_monitoring()
    atexit.register(stop_monitoring)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start sensor monitoring and run the Flask development server."""
    _init_monitoring()
    _register_signal_handlers()
    print(f"🚀 Sensor Dashboard running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    run_server()
