# ESP32 + GSM WebSocket Emergency Bridge

This folder contains firmware and notes for the ESP32-based bridge that
listens on the same WebSocket as the Raspberry Pi and sends an SMS via the
SIM800L module when an emergency alert arrives.

## Files

| File | Purpose |
|---|---|
| `esp32_ws_gsm_bridge.ino` | **New** – ESP32 firmware (WebSocket client → SIM800L) |
| `esp_http_gsm_bridge.ino` | Legacy – HTTP POST-based bridge (kept for reference) |
| `alert_service.py` | Python helper on RPi for building and dispatching alerts |
| `__init__.py` | Package init |

---

## Hardware Wiring

```
ESP32 DevKit               SIM800L
-----------               ---------
GPIO17 (TX)  ---------->  RX
GPIO16 (RX)  <----------  TX
GND          ---------->  GND
(external)   ---------->  VCC  3.7–4.2 V / 2 A  ← must NOT use ESP 3.3 V
```

> **Important:** GPIO16 and GPIO17 are right next to D15 on the ESP32 DevKit board.
> The SIM800L needs its own dedicated power supply — NEVER power it from the ESP32's 3.3 V rail.

---

## Libraries to Install (Arduino IDE)

1. **ArduinoJson** by Benoit Blanchon — `Tools → Manage Libraries → search "ArduinoJson"`
2. **WebSockets** by Markus Sattler — `Tools → Manage Libraries → search "WebSockets"` (the one by Markus Sattler)

---

## Flashing the ESP32 – Step by Step

### 1. Install Arduino IDE
Download from https://www.arduino.cc/en/software (v2.x recommended).

### 2. Add ESP32 board support
`File → Preferences → Additional Boards Manager URLs`:
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
Then `Tools → Board → Boards Manager` → search **esp32** → Install.

### 3. Install required libraries
`Tools → Manage Libraries`:
- Search **ArduinoJson** → Install (by Benoit Blanchon)
- Search **WebSockets** → Install (by Markus Sattler)

### 4. Open the sketch
`File → Open` → select `GSM/esp32_ws_gsm_bridge.ino`

### 5. Verify these constants at the top of the sketch

```cpp
static const char *WIFI_SSID     = "Pandey";       // your Wi-Fi
static const char *WIFI_PASSWORD = "RaMoSa171116*";

static const char *WS_HOST = "192.168.0.103";  // RPi IP (from config.json)
static const uint16_t WS_PORT = 8765;           // RPi WS port

static const char *GSM_RECEIVER = "+919326632001";  // emergency contact
static const char *GSM_SENDER   = "+910000000000";  // your SIM number
```

### 6. Select the board and port
- `Tools → Board → ESP32 Arduino → ESP32 Dev Module`
- `Tools → Port → (your ESP32 serial port, e.g. /dev/ttyUSB0)`

### 7. Upload
Click **Upload (→)**. Open Serial Monitor at **115200 baud** to confirm:
```
Wi-Fi OK  IP: 192.168.0.xxx
[GSM] Modem ready
[WS] Connected to ws://192.168.0.103:8765/
```

---

## How It Works

1. ESP32 boots and connects to Wi-Fi.
2. ESP32 connects to the RPi's WebSocket server as a client.
3. ESP32 sends a registration message: `{"type":"register","client":"esp32_gsm_bridge"}`.
4. The RPi WebSocket server broadcasts an `emergency_alert` JSON when triggered.
5. ESP32 receives the JSON, builds an SMS from the sensor data, and sends it to the emergency contact via SIM800L AT commands.
6. ESP32 sends an acknowledgement back: `{"type":"emergency_ack","status":"sms_sent"}`.

### Emergency Alert JSON Format (sent by RPi)
```json
{
  "type": "emergency_alert",
  "source": "voice",
  "reason": "obstacle",
  "receiver_number": "+919326632001",
  "temperature_c": 36.5,
  "humidity_pct": 55.0,
  "latitude": 19.07600,
  "longitude": 72.87770,
  "altitude_m": 14.0,
  "spo2_percent": 98.0,
  "heart_rate_bpm": 75.0
}
```

### SMS Sent to Emergency Contact
```
EMERGENCY! Wheelchair alert.
Reason: obstacle (voice)
Vitals: Temp 36.5C Hum 55% HR 75bpm SpO2 98%
GPS: 19.07600,72.87770 Alt 14m
maps.google.com/?q=19.07600,72.87770
```

---

## Configuration (config.json)

```json
"gsm": {
  "sender_number": "+910000000000",
  "receiver_number": "+919326632001",
  "message_template": "..."
},
"esp": {
  "host": "192.168.0.200",
  "websocket_port": 8765
}
```

After flashing, check the ESP32's Serial Monitor for its IP address, then update `config.json → esp → host` on the RPi if needed (only required for the legacy HTTP fallback path).

---

## Triggering an Emergency from the RPi

Send this JSON over WebSocket (the RPi server broadcasts it when emergency mode is triggered):
```json
{
  "type": "emergency_alert",
  "receiver_number": "+919326632001",
  "source": "manual",
  "reason": "test",
  "temperature_c": 37.0,
  "humidity_pct": 60.0,
  "spo2_percent": 97.0,
  "heart_rate_bpm": 80.0,
  "latitude": 19.0760,
  "longitude": 72.8777,
  "altitude_m": 12.0
}
```

Or test from your laptop (same Wi-Fi):
```bash
# Install wscat: npm install -g wscat
wscat -c ws://192.168.0.103:8765
# Then paste the JSON above and press Enter
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| ESP32 loops on "Connecting to Wi-Fi" | Check SSID/password; confirm 2.4 GHz (SIM800L won't work on 5 GHz) |
| `[WS] Disconnected – will retry...` | RPi server not running; check `rpi_websocket_server.py` is active |
| `[GSM] Modem not responding` | Check TX/RX wiring (they must be crossed), baud rate (9600), and power supply |
| SMS never arrives | Run `AT+CREG?` in Serial Monitor – response should be `+CREG: 0,1`; check SIM balance and antenna |
| SMS arrives but is garbled | Keep message under 160 chars; avoid special characters |
