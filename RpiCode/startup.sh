#!/bin/bash
# =============================================================================
# Smart Wheelchair - RPI Startup Script
# =============================================================================
# This script:
#   1. Starts a Bluetooth watchdog (keeps BT on, discoverable & pairable)
#   2. Runs the main WebSocket server
#   3. Optionally starts the camera stream
#
# Usage:
#   chmod +x ~/Code/startup.sh   (first time only)
#   ~/Code/startup.sh
#
# To run at boot, add this to /etc/rc.local (before "exit 0"):
#   sudo -u pi /home/pi/Code/startup.sh >> /home/pi/Code/logs/startup.log 2>&1 &
# =============================================================================

set -e

# ─────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────
DEVICE_ID="123"
API_URL="https://wheelchair-api.vercel.app"
API_PASSWORD="123"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --device-id) DEVICE_ID="$2"; shift ;;
        --url) API_URL="$2"; shift ;;
        --password) API_PASSWORD="$2"; shift ;;
        *) echo "Unknown parameter passed: $1" ;;
    esac
    shift
done

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CODE_DIR="$HOME/Code"
VENV_PYTHON="$CODE_DIR/venv/bin/python"
VENV2_ACTIVATE="$CODE_DIR/venv2/bin/activate"

WEBSOCKET_SCRIPT="$CODE_DIR/rpi_websocket_server.py"
CAMERA_SCRIPT="$CODE_DIR/rpi_stream_detect_ws.py"
SENSOR_SERVER_SCRIPT="$CODE_DIR/sensor_server_enhanced.py"

LOG_DIR="$CODE_DIR/logs"
WEBSOCKET_LOG="$LOG_DIR/websocket.log"
CAMERA_LOG="$LOG_DIR/camera.log"
BT_LOG="$LOG_DIR/bluetooth_watchdog.log"
SENSOR_SERVER_LOG="$LOG_DIR/sensor_server.log"

BT_REFRESH_INTERVAL=30   # seconds between bluetooth keep-alive checks
BT_DEVICE_NAME="SmartWheelchair"

# Camera stream settings (edit as needed)
CAMERA_HOST="127.0.0.1"
CAMERA_PORT="8765"
STREAM_NAME="front_camera"
STREAM_WIDTH="640"
STREAM_HEIGHT="480"
STREAM_FPS="10"

# Set to "true"/"false" to enable/disable camera stream at startup
ENABLE_CAMERA="true"

# ─────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✔  $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠  $*${NC}"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✖  $*${NC}" >&2; }
info() { echo -e "${BLUE}[$(date '+%H:%M:%S')] ℹ  $*${NC}"; }

# ─────────────────────────────────────────────
# Trap: clean everything up on exit / Ctrl+C
# ─────────────────────────────────────────────
# PIDs of background processes we spawn
BT_WATCHDOG_PID=""
WEBSOCKET_PID=""
CAMERA_PID=""
SENSOR_SERVER_PID=""

cleanup() {
    echo ""
    warn "Shutting down Smart Wheelchair services..."

    [ -n "$CAMERA_PID" ]     && kill "$CAMERA_PID"     2>/dev/null && info "Camera stream stopped."
    [ -n "$SENSOR_SERVER_PID" ] && kill "$SENSOR_SERVER_PID" 2>/dev/null && info "Sensor server stopped."
    [ -n "$BT_WATCHDOG_PID" ] && kill "$BT_WATCHDOG_PID" 2>/dev/null && info "Bluetooth watchdog stopped."
    [ -n "$WEBSOCKET_PID" ]  && kill "$WEBSOCKET_PID"  2>/dev/null && info "WebSocket server stopped."

    # Restore bluetooth to a sane non-discoverable state on exit
    bluetoothctl discoverable off 2>/dev/null || true
    bluetoothctl pairable off    2>/dev/null || true

    log "All services stopped. Goodbye."
    exit 0
}

trap cleanup INT TERM EXIT

# ─────────────────────────────────────────────
# 1. Sanity checks
# ─────────────────────────────────────────────
info "=== Smart Wheelchair Startup ==="
info "Code directory : $CODE_DIR"
info "Date / Time    : $(date)"

mkdir -p "$LOG_DIR"

if [ ! -f "$WEBSOCKET_SCRIPT" ]; then
    err "WebSocket server script not found: $WEBSOCKET_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    err "Python venv not found at: $VENV_PYTHON"
    exit 1
fi

# ─────────────────────────────────────────────
# 2. Bluetooth helpers
# ─────────────────────────────────────────────

# Ensure the bluetooth service is up
ensure_bt_service() {
    if ! systemctl is-active --quiet bluetooth; then
        warn "bluetooth service is not running – starting it..."
        sudo systemctl start bluetooth
        sleep 2
    fi
}

# Power on BT and make it discoverable + pairable
bt_enable() {
    bluetoothctl power on          2>/dev/null || true
    sleep 0.5
    bluetoothctl discoverable on   2>/dev/null || true
    bluetoothctl pairable on       2>/dev/null || true
    # Give the device a friendly name
    bluetoothctl system-alias "$BT_DEVICE_NAME" 2>/dev/null || true
}

# ─────────────────────────────────────────────
# 3. Bluetooth watchdog (background loop)
# ─────────────────────────────────────────────
# This function runs in a separate subshell and re-applies the BT settings
# every BT_REFRESH_INTERVAL seconds, so if the adapter resets or times out
# it automatically recovers without any manual ./blon.sh run.

bt_watchdog() {
    echo "[BT-watchdog] Starting – refresh every ${BT_REFRESH_INTERVAL}s" >> "$BT_LOG" 2>&1
    while true; do
        # Check if bluetooth adapter is present
        if hciconfig hci0 up 2>/dev/null; then
            bt_enable >> "$BT_LOG" 2>&1
            echo "[BT-watchdog] $(date '+%H:%M:%S') – BT refreshed (on/discoverable/pairable)" >> "$BT_LOG" 2>&1
        else
            echo "[BT-watchdog] $(date '+%H:%M:%S') – WARNING: hci0 not found, retrying..." >> "$BT_LOG" 2>&1
            # Try to bring it up
            sudo hciconfig hci0 up 2>/dev/null || true
            sleep 5
        fi
        sleep "$BT_REFRESH_INTERVAL"
    done
}

# ─────────────────────────────────────────────
# 4. Start Bluetooth
# ─────────────────────────────────────────────
info "Initialising Bluetooth..."
ensure_bt_service
bt_enable
log "Bluetooth is ON, discoverable & pairable (device: $BT_DEVICE_NAME)"

# Launch watchdog in background
bt_watchdog &
BT_WATCHDOG_PID=$!
log "Bluetooth watchdog started (PID $BT_WATCHDOG_PID)"

# ─────────────────────────────────────────────
# 5. Start WebSocket server
# ─────────────────────────────────────────────
info "Starting WebSocket server..."
info "  Script  : $WEBSOCKET_SCRIPT"
info "  Log     : $WEBSOCKET_LOG"

sudo "$VENV_PYTHON" "$WEBSOCKET_SCRIPT" >> "$WEBSOCKET_LOG" 2>&1 &
WEBSOCKET_PID=$!
log "WebSocket server started (PID $WEBSOCKET_PID)"

# Give the server a moment to initialise before starting the camera
sleep 3

# Verify it is still running
if ! kill -0 "$WEBSOCKET_PID" 2>/dev/null; then
    err "WebSocket server failed to start. Check $WEBSOCKET_LOG"
    exit 1
fi

# ─────────────────────────────────────────────
# 5.5 Start Sensor Server (Web Dashboard + Firebase Upload)
# ─────────────────────────────────────────────
info "Starting Sensor Server..."
info "  Script  : $SENSOR_SERVER_SCRIPT"
info "  Log     : $SENSOR_SERVER_LOG"
info "  Device  : $DEVICE_ID to $API_URL"

sudo "$VENV_PYTHON" "$SENSOR_SERVER_SCRIPT" \
    --device-id "$DEVICE_ID" \
    --url "$API_URL" \
    --password "$API_PASSWORD" \
    >> "$SENSOR_SERVER_LOG" 2>&1 &
SENSOR_SERVER_PID=$!
log "Sensor server started (PID $SENSOR_SERVER_PID)"
sleep 3


# ─────────────────────────────────────────────
# 6. (Optional) Camera stream
# ─────────────────────────────────────────────
if [ "$ENABLE_CAMERA" = "true" ]; then
    if [ ! -f "$CAMERA_SCRIPT" ]; then
        warn "Camera script not found ($CAMERA_SCRIPT) – skipping camera stream."
    elif [ ! -f "$VENV2_ACTIVATE" ]; then
        warn "venv2 not found ($VENV2_ACTIVATE) – skipping camera stream."
    else
        info "Starting camera stream..."
        info "  Script  : $CAMERA_SCRIPT"
        info "  Log     : $CAMERA_LOG"

        # Run inside venv2
        bash -c "
            source '$VENV2_ACTIVATE'
            python3 '$CAMERA_SCRIPT' \
                --host '$CAMERA_HOST' \
                --port '$CAMERA_PORT' \
                --stream-name '$STREAM_NAME' \
                --width '$STREAM_WIDTH' \
                --height '$STREAM_HEIGHT' \
                --stream-fps '$STREAM_FPS'
        " >> "$CAMERA_LOG" 2>&1 &
        CAMERA_PID=$!
        log "Camera stream started (PID $CAMERA_PID)"

        # Brief check
        sleep 2
        if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
            warn "Camera stream exited early. Check $CAMERA_LOG"
            warn "Common cause: camera not plugged in (Index out of bounds error)."
            CAMERA_PID=""
        fi
    fi
else
    info "Camera stream disabled (ENABLE_CAMERA=false). Edit startup.sh to enable."
fi

# ─────────────────────────────────────────────
# 7. Status summary
# ─────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All services are running:"
echo ""
info "  Bluetooth watchdog  PID: $BT_WATCHDOG_PID"
info "  WebSocket server    PID: $WEBSOCKET_PID"
info "  Sensor server       PID: $SENSOR_SERVER_PID"
[ -n "$CAMERA_PID" ] && info "  Camera stream       PID: $CAMERA_PID"
echo ""
info "Logs:"
info "  Bluetooth : $BT_LOG"
info "  WebSocket : $WEBSOCKET_LOG"
info "  Sensor    : $SENSOR_SERVER_LOG"
[ -n "$CAMERA_PID" ] && info "  Camera    : $CAMERA_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
warn "Press Ctrl+C to stop all services cleanly."

# ─────────────────────────────────────────────
# 8. Keep this script alive; monitor children
# ─────────────────────────────────────────────
while true; do
    # If WebSocket server dies, restart it automatically
    if ! kill -0 "$WEBSOCKET_PID" 2>/dev/null; then
        warn "WebSocket server crashed! Restarting..."
        sleep 2
        sudo "$VENV_PYTHON" "$WEBSOCKET_SCRIPT" >> "$WEBSOCKET_LOG" 2>&1 &
        WEBSOCKET_PID=$!
        log "WebSocket server restarted (PID $WEBSOCKET_PID)"
    fi

    # If Sensor server dies, restart it
    if ! kill -0 "$SENSOR_SERVER_PID" 2>/dev/null; then
        warn "Sensor server crashed! Restarting..."
        sleep 2
        sudo "$VENV_PYTHON" "$SENSOR_SERVER_SCRIPT" \
            --device-id "$DEVICE_ID" \
            --url "$API_URL" \
            --password "$API_PASSWORD" \
            >> "$SENSOR_SERVER_LOG" 2>&1 &
        SENSOR_SERVER_PID=$!
        log "Sensor server restarted (PID $SENSOR_SERVER_PID)"
    fi

    # If camera crashes, log it but don't restart automatically
    # (camera may simply not be connected)
    if [ -n "$CAMERA_PID" ] && ! kill -0 "$CAMERA_PID" 2>/dev/null; then
        warn "Camera stream stopped (camera may have been disconnected)."
        CAMERA_PID=""
    fi

    sleep 10
done
