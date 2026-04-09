#!/usr/bin/env python3
"""
Laptop Webcam Test GUI for Smart Wheelchair.

Uses the local webcam and YOLOv8 to detect close objects.
When an object crosses the danger threshold the app sends a real
STOP command to the Raspberry Pi over WebSocket.

Usage (offline – no RPi):
    python3 laptop_webcam_test_gui.py

Usage (connected to RPi):
    python3 laptop_webcam_test_gui.py --host 192.168.1.42 --port 8765
"""

import argparse
import asyncio
import json
import logging
import queue
import sys
import threading
import time

import cv2
import numpy as np
import websockets
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow,
    QTextEdit, QVBoxLayout, QWidget,
)
from ultralytics import YOLO

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("webcam_test")

# ─── Constants ───────────────────────────────────────────────────────────────
# Approximate real-world widths in cm (same as laptop_stream_processor.py)
REAL_WIDTHS_CM = {
    "person": 50, "bicycle": 40, "car": 180, "motorcycle": 70, "bus": 250, "truck": 250,
    "bench": 120, "cat": 15, "dog": 20, "horse": 60, "sheep": 30, "cow": 50,
    "chair": 50, "couch": 180, "potted plant": 30, "bed": 160, "dining table": 120,
    "toilet": 40, "tv": 80, "laptop": 35, "mouse": 6, "remote": 5, "keyboard": 45,
    "cell phone": 8, "microwave": 50, "oven": 60, "toaster": 30, "sink": 50,
    "refrigerator": 70, "book": 15, "clock": 30, "vase": 15, "scissors": 10,
    "teddy bear": 30, "hair drier": 20, "toothbrush": 2,
}
DEFAULT_WIDTH_CM = 50
FOCAL_LENGTH_FACTOR = 600          # calibrated for a typical laptop webcam
DANGER_DISTANCE_CM = 60            # threshold – anything closer triggers STOP
STOP_COOLDOWN_SECONDS = 5          # minimum gap between consecutive STOP commands
WS_RECONNECT_DELAY = 3.0           # seconds to wait before reconnecting


# =============================================================================
# WebSocket Sender (runs its own asyncio loop in a daemon thread)
# =============================================================================

class WebSocketSender(threading.Thread):
    """
    Maintains a persistent WebSocket connection to the RPi and drains a
    thread-safe queue of outgoing JSON payloads.

    If the connection drops it reconnects automatically.
    If `host` is None the sender thread exits immediately (offline mode).
    """

    def __init__(self, host: str | None, port: int):
        super().__init__(daemon=True, name="ws-sender")
        self.host = host
        self.port = port
        self._queue: queue.Queue[dict | None] = queue.Queue()
        self._connected = False
        self._stop_event = threading.Event()

    # Called from any thread ----------
    def enqueue(self, payload: dict) -> None:
        """Thread-safe: put a payload on the send queue."""
        if self.host is not None:
            self._queue.put(payload)

    def request_stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)  # unblock the get()

    @property
    def connected(self) -> bool:
        return self._connected

    # Internal asyncio helpers -----
    async def _send_loop(self, ws) -> None:
        """Drain the queue and send each payload; return on disconnect."""
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                # Poll the thread-safe queue without blocking the event loop
                payload = await loop.run_in_executor(None, self._queue.get, True, 0.1)
            except queue.Empty:
                continue
            if payload is None:
                return
            try:
                await ws.send(json.dumps(payload))
                await ws.drain()
            except Exception as exc:
                LOG.warning("WS send failed: %s", exc)
                self._queue.put(payload)   # re-queue for next connection
                return

    async def _run_async(self) -> None:
        uri = f"ws://{self.host}:{self.port}"
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    uri,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                    open_timeout=5,
                ) as ws:
                    self._connected = True
                    LOG.info("WebSocket sender connected to %s", uri)
                    # Register as a control client
                    await ws.send(json.dumps({
                        "type": "camera_register",
                        "stream": "webcam_laptop_gui",
                    }))
                    await self._send_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("WS sender error: %s – reconnecting in %.1fs", exc, WS_RECONNECT_DELAY)
            finally:
                self._connected = False

            if not self._stop_event.is_set():
                await asyncio.sleep(WS_RECONNECT_DELAY)

    def run(self) -> None:
        if self.host is None:
            LOG.info("No --host provided – running in offline (simulation) mode")
            return
        asyncio.run(self._run_async())


# =============================================================================
# Video capture + YOLO thread
# =============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_log_signal = pyqtSignal(str)
    stop_signal = pyqtSignal(str)           # reason string → UI
    ws_status_signal = pyqtSignal(str)      # "Connected" / "Offline" / "Disconnected"

    def __init__(self, ws_sender: WebSocketSender):
        super().__init__()
        self._run_flag = True
        self._ws_sender = ws_sender
        self.model = YOLO("yolov8n.pt")
        self.last_stop_time = 0.0
        self._last_ws_state: str | None = None

    def estimate_distance(self, label: str, box_width_px: int) -> float:
        real_width = REAL_WIDTHS_CM.get(label.lower(), DEFAULT_WIDTH_CM)
        if box_width_px == 0:
            return 999.0
        return (real_width * FOCAL_LENGTH_FACTOR) / box_width_px

    def _emit_ws_status(self) -> None:
        if self._ws_sender.host is None:
            state = "Offline (no --host)"
        elif self._ws_sender.connected:
            state = "Connected ✔"
        else:
            state = "Connecting…"
        if state != self._last_ws_state:
            self.ws_status_signal.emit(state)
            self._last_ws_state = state

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            # Emit WS connection status
            self._emit_ws_status()

            results = self.model.predict(frame, conf=0.5, verbose=False)
            annotated = frame.copy()
            stop_reason: str | None = None

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_w = x2 - x1
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    dist = self.estimate_distance(label, box_w)

                    # Colour coding
                    if dist < 20:
                        color = (0, 0, 255)       # Red – danger
                    elif dist < DANGER_DISTANCE_CM:
                        color = (0, 165, 255)     # Orange – warning
                    else:
                        color = (0, 255, 0)       # Green – safe

                    if dist < DANGER_DISTANCE_CM:
                        stop_reason = f"Object too close: {label} ({dist:.0f}cm)"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated, f"{label} {dist:.0f}cm",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )

            if stop_reason:
                now = time.time()
                if now - self.last_stop_time > STOP_COOLDOWN_SECONDS:
                    self.last_stop_time = now
                    self.stop_signal.emit(stop_reason)
                    self.update_log_signal.emit(f"STOP: {stop_reason}")
                    # ── Send real STOP command to RPi ──────────────────────
                    self._ws_sender.enqueue({
                        "type": "manual_control",
                        "mode": "manual",
                        "command": "stop",
                        "source": "webcam_laptop",
                        "triggered_by": "object_detection",
                        "reason": stop_reason,
                    })
                cv2.putText(
                    annotated, f"STOP: {stop_reason}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )

            self.change_pixmap_signal.emit(annotated)
            # ~30 FPS cap for the GUI; YOLO inference is the real bottleneck
            time.sleep(0.01)

        cap.release()

    def stop(self) -> None:
        self._run_flag = False
        self.wait()


# =============================================================================
# Main GUI window
# =============================================================================

class App(QMainWindow):
    def __init__(self, ws_sender: WebSocketSender):
        super().__init__()
        self._ws_sender = ws_sender
        self.setWindowTitle("Smart Wheelchair – Webcam YOLO Monitor")
        self.resize(1050, 750)
        self.display_width = 700
        self.display_height = 525

        # ── Layout ──────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Connection status bar
        self.ws_label = QLabel("WebSocket: —")
        self.ws_label.setStyleSheet(
            "font-size: 13px; padding: 4px 10px; background: #222; color: #aaa;"
        )
        root.addWidget(self.ws_label)

        # Video feed
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.display_width, self.display_height)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #444;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Status bar
        self.status_label = QLabel("Status: SAFE")
        self.status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #00cc44; padding: 8px;"
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        # Log
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        self.log_box.setStyleSheet("background: #111; color: #ccc; font-family: monospace;")
        root.addWidget(self.log_box)

        # ── Video thread ─────────────────────────────────────────────────────
        self.thread = VideoThread(ws_sender)
        self.thread.change_pixmap_signal.connect(self._update_image)
        self.thread.update_log_signal.connect(self._update_log)
        self.thread.stop_signal.connect(self._handle_stop)
        self.thread.ws_status_signal.connect(self._update_ws_status)
        self.thread.start()

    # ── Slots ────────────────────────────────────────────────────────────────
    def _update_ws_status(self, status: str) -> None:
        color = "#00cc44" if "✔" in status else ("#ff6622" if "Offline" in status else "#ffaa00")
        self.ws_label.setText(f"WebSocket: {status}")
        self.ws_label.setStyleSheet(
            f"font-size: 13px; padding: 4px 10px; background: #222; color: {color};"
        )

    def _handle_stop(self, reason: str) -> None:
        self.status_label.setText(f"⛔ STOP – {reason}")
        self.status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #ff2222; "
            "padding: 8px; background: #330000;"
        )
        QTimer.singleShot(STOP_COOLDOWN_SECONDS * 1000, self._reset_status)

    def _reset_status(self) -> None:
        self.status_label.setText("Status: SAFE")
        self.status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #00cc44; padding: 8px;"
        )

    def _update_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {text}")

    def _update_image(self, cv_img: np.ndarray) -> None:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = qt_img.scaled(
            self.display_width, self.display_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

    def closeEvent(self, event) -> None:
        self.thread.stop()
        self._ws_sender.request_stop()
        event.accept()


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam YOLO monitor with optional RPi STOP command")
    p.add_argument("--host", default=None,
                   help="IP address of the RPi WebSocket server (omit to run offline)")
    p.add_argument("--port", type=int, default=8765,
                   help="WebSocket port (default: 8765)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Start the WebSocket sender thread (no-op if host is None)
    ws_sender = WebSocketSender(host=args.host, port=args.port)
    ws_sender.start()

    app = QApplication(sys.argv)
    window = App(ws_sender)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
