#!/usr/bin/env python3
"""Minimal Flask server for live sensor telemetry visualization.

This web server exposes live Raspberry Pi sensor data collected by
`sensor_monitor` and renders it in a browser with charts and a dynamic map.

Features
========
* Numeric readouts for temperature, humidity, heart rate, and SpO2
* Chart.js line charts that update every second
* Leaflet map showing the latest GPS fix from the Neo-6M receiver

Inputs
------
* Sensor data produced by `sensor_monitor` threads

Outputs
-------
* HTML dashboard at `/`
* JSON payload with the latest readings at `/api/data`
"""
from __future__ import annotations

import atexit
import json
import signal
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify

from sensor_monitor import (
    get_latest_dht,
    get_latest_gps,
    get_latest_health,
    start_monitoring,
    stop_monitoring,
)

CONFIG_PATH = Path(__file__).parent / "config.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8877

app = Flask(__name__)

# Preload configuration so the dashboard can display pin mappings if desired.
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
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-u0U0D7V0GX6zV7bWxX/HvYkUUxw0FKxCNiXZp0G0X6A="
    crossorigin=""
  />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-SmSRN6vV+3nHn9oIo9g9Nk30DJ6LmW5GtQD7C0hRaY8="
    crossorigin=""
  ></script>
  <style>
    :root {
      color-scheme: dark;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: #050b17;
      color: #f2f5fa;
    }

    body {
      margin: 0;
      padding: 24px;
      background: linear-gradient(135deg, #050b17 0%, #0f1a36 55%, #091027 100%);
      min-height: 100vh;
    }

    h1 {
      margin-top: 0;
      font-size: 2rem;
      font-weight: 600;
    }

    .grid {
      display: grid;
      gap: 20px;
    }

    @media (min-width: 1024px) {
      .grid {
        grid-template-columns: 360px 1fr;
        align-items: start;
      }
    }

    .card {
      background: rgba(7, 17, 38, 0.85);
      border-radius: 18px;
      box-shadow: 0 25px 60px rgba(0, 0, 0, 0.28);
      padding: 20px;
      backdrop-filter: blur(18px);
      border: 1px solid rgba(61, 108, 255, 0.12);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 16px;
    }

    .metric {
      background: rgba(15, 27, 54, 0.75);
      border-radius: 14px;
      padding: 14px 16px;
      border: 1px solid rgba(113, 140, 255, 0.1);
    }

    .metric .label {
      font-size: 0.8rem;
      color: #9fb4ff;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric .value {
      font-size: 1.5rem;
      font-weight: 600;
    }

    #map {
      width: 100%;
      height: 320px;
      border-radius: 16px;
    }

    .timestamp {
      font-size: 0.85rem;
      color: #8aa0d6;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="grid">
    <div class="card">
      <h1>Smart Wheelchair Telemetry</h1>
      <div class="timestamp" id="timestamp">Connecting...</div>

      <div class="metrics">
        <div class="metric">
          <div class="label">Temperature</div>
          <div class="value" id="temperature">–</div>
          <div class="label">Humidity</div>
          <div class="value" id="humidity">–</div>
        </div>

        <div class="metric">
          <div class="label">Heart Rate</div>
          <div class="value" id="heartRate">–</div>
          <div class="label">SpO₂</div>
          <div class="value" id="spo2">–</div>
        </div>

        <div class="metric">
          <div class="label">Latitude</div>
          <div class="value" id="latitude">–</div>
          <div class="label">Longitude</div>
          <div class="value" id="longitude">–</div>
        </div>
      </div>

      <div class="metric" style="margin-top: 16px;">
        <div class="label">Sensor Pins</div>
        <pre style="margin:0; font-size:0.78rem; color:#d9e3ff; white-space: pre-wrap;" id="pinConfig"></pre>
      </div>
    </div>

    <div class="card">
      <div id="map"></div>
      <canvas id="environmentChart" style="margin-top: 22px;"></canvas>
      <canvas id="healthChart" style="margin-top: 22px;"></canvas>
    </div>
  </div>

  <script>
    const pinConfig = document.getElementById('pinConfig');
    const configData = %CONFIG_JSON%;
    pinConfig.textContent = JSON.stringify(configData, null, 2);

    let envChart, healthChart;
    const MAX_POINTS = 120;

    function initCharts() {
      const envCtx = document.getElementById('environmentChart');
      envChart = new Chart(envCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Temperature (°C)',
              data: [],
              borderColor: '#4da8ff',
              backgroundColor: 'rgba(77, 168, 255, 0.1)',
              tension: 0.32,
              spanGaps: true,
            },
            {
              label: 'Humidity (%)',
              data: [],
              borderColor: '#9f7bff',
              backgroundColor: 'rgba(159, 123, 255, 0.1)',
              tension: 0.32,
              spanGaps: true,
            }
          ],
        },
        options: {
          animation: false,
          responsive: true,
          scales: {
            y: {
              beginAtZero: false,
              ticks: { color: '#9cb3ff' },
              grid: { color: 'rgba(100, 130, 200, 0.15)' },
            },
            x: {
              ticks: { color: '#9cb3ff' },
              grid: { display: false },
            },
          },
          plugins: {
            legend: { labels: { color: '#dce5ff' } },
          },
        },
      });

      const healthCtx = document.getElementById('healthChart');
      healthChart = new Chart(healthCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Heart Rate (bpm)',
              data: [],
              borderColor: '#ff6b81',
              backgroundColor: 'rgba(255, 107, 129, 0.1)',
              tension: 0.32,
              spanGaps: true,
            },
            {
              label: 'SpO₂ (%)',
              data: [],
              borderColor: '#5cd6b3',
              backgroundColor: 'rgba(92, 214, 179, 0.1)',
              tension: 0.32,
              spanGaps: true,
            }
          ],
        },
        options: {
          animation: false,
          responsive: true,
          scales: {
            y: {
              beginAtZero: false,
              ticks: { color: '#9cb3ff' },
              grid: { color: 'rgba(100, 130, 200, 0.15)' },
            },
            x: {
              ticks: { color: '#9cb3ff' },
              grid: { display: false },
            },
          },
          plugins: {
            legend: { labels: { color: '#dce5ff' } },
          },
        },
      });
    }

    const map = L.map('map').setView([20.5937, 78.9629], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);
    let gpsMarker = null;

    function updateCharts(timestampLabel, data) {
      if (!envChart || !healthChart) {
        return;
      }

      envChart.data.labels.push(timestampLabel);
      envChart.data.datasets[0].data.push(data.temperature);
      envChart.data.datasets[1].data.push(data.humidity);

      healthChart.data.labels.push(timestampLabel);
      healthChart.data.datasets[0].data.push(data.heartRate);
      healthChart.data.datasets[1].data.push(data.spo2);

      if (envChart.data.labels.length > MAX_POINTS) {
        envChart.data.labels.shift();
        envChart.data.datasets.forEach((dataset) => dataset.data.shift());
      }
      if (healthChart.data.labels.length > MAX_POINTS) {
        healthChart.data.labels.shift();
        healthChart.data.datasets.forEach((dataset) => dataset.data.shift());
      }

      envChart.update('none');
      healthChart.update('none');
    }

    function formatValue(value, fallback = '–', decimals = 1) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return fallback;
      }
      return Number.parseFloat(value).toFixed(decimals);
    }

    function updateMap(latitude, longitude) {
      if (latitude === null || longitude === null) {
        return;
      }
      const lat = Number(latitude);
      const lng = Number(longitude);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        return;
      }
      const position = [lat, lng];
      map.setView(position, Math.max(map.getZoom(), 15));
      if (gpsMarker) {
        gpsMarker.setLatLng(position);
      } else {
        gpsMarker = L.marker(position).addTo(map);
      }
    }

    async function fetchData() {
      try {
        const response = await fetch('/api/data');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        const ts = payload.timestamp ? new Date(payload.timestamp * 1000) : new Date();
        const label = ts.toLocaleTimeString();
        document.getElementById('timestamp').textContent = `Last update: ${ts.toLocaleString()}`;

        const dht = payload.dht || {};
        const health = payload.health || {};
        const gps = payload.gps || {};

        document.getElementById('temperature').textContent = `${formatValue(dht.temperature_c)} °C`;
        document.getElementById('humidity').textContent = `${formatValue(dht.humidity_percent)} %`;
        document.getElementById('heartRate').textContent = `${formatValue(health.heart_rate_bpm)} bpm`;
        document.getElementById('spo2').textContent = `${formatValue(health.spo2_percent)} %`;
        document.getElementById('latitude').textContent = formatValue(gps.latitude, '–', 6);
        document.getElementById('longitude').textContent = formatValue(gps.longitude, '–', 6);

        updateCharts(label, {
          temperature: dht.temperature_c ?? null,
          humidity: dht.humidity_percent ?? null,
          heartRate: health.heart_rate_bpm ?? null,
          spo2: health.spo2_percent ?? null,
        });

        updateMap(gps.latitude, gps.longitude);
      } catch (err) {
        console.error('Failed to fetch sensor data', err);
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
    sensor_config = CONFIG.get("raspberry_pi", {}).get("sensors", {})
    html = HTML_TEMPLATE.replace("%CONFIG_JSON%", json.dumps(sensor_config))
    return Response(html, mimetype="text/html")


@app.route("/api/data")
def api_data() -> Response:
    """Return the latest sensor readings as JSON."""
    payload = {
        "timestamp": time.time(),
        "dht": get_latest_dht(),
        "health": get_latest_health(),
        "gps": get_latest_gps(),
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
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    run_server()
