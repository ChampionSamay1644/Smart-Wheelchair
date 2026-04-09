#!/usr/bin/env python3
"""
Laptop Stream Processor for Smart Wheelchair.

Connects to the Raspberry Pi WebSocket server, receives live camera frames,
runs YOLOv8 on the laptop (NOT on the RPi), and sends STOP commands when
a dangerous object is detected too close.

Key design:
  - YOLO inference runs in a ThreadPoolExecutor so the asyncio receive loop
    is never blocked.  New frames arriving while inference is running are
    DROPPED (skipped) rather than queued, preventing buffer build-up.
  - The RPi camera and stream are completely unaffected when no laptop client
    is connected, and stream at full speed when the laptop IS connected.

Usage:
    python3 laptop_stream_processor.py --host <RPI_IP> [--port 8765] [--model yolov8n.pt]
"""

import argparse
import asyncio
import base64
import concurrent.futures
import json
import logging
import time

import cv2
import numpy as np
import websockets
from ultralytics import YOLO

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("laptop_processor")

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE   = 0.5
DANGER_DISTANCE_CM   = 60   # send STOP when closer than this (cm)
STOP_COOLDOWN        = 10   # seconds between consecutive STOP commands
RECONNECT_DELAY      = 5    # seconds to wait before reconnecting
FOCAL_LENGTH_FACTOR  = 600  # calibrated for typical webcam/640px

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


# ─── CPU-bound inference (runs in thread pool) ────────────────────────────────

def decode_and_infer(model: YOLO, b64_data: str) -> tuple[np.ndarray | None, str | None]:
    """
    Decode a base64 JPEG and run YOLO inference.
    Returns (annotated_frame, stop_reason_or_None).
    This runs in a thread – never call asyncio APIs here.
    """
    try:
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        LOG.debug("Frame decode failed: %s", exc)
        return None, None

    if frame is None:
        return None, None

    results = model.predict(frame, conf=DEFAULT_CONFIDENCE, verbose=False)
    annotated = frame.copy()
    stop_reason: str | None = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w = x2 - x1
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            real_w = REAL_WIDTHS_CM.get(label.lower(), DEFAULT_WIDTH_CM)
            dist = (real_w * FOCAL_LENGTH_FACTOR / box_w) if box_w > 0 else 999.0

            if dist < 20:
                color = (0, 0, 255)       # red  – danger
            elif dist < DANGER_DISTANCE_CM:
                color = (0, 165, 255)     # orange – warning
            else:
                color = (0, 255, 0)       # green – safe

            if dist < DANGER_DISTANCE_CM and stop_reason is None:
                stop_reason = f"Object too close: {label} ({dist:.0f}cm)"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, f"{label} {dist:.0f}cm",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    return annotated, stop_reason


# ─── Main processor class ─────────────────────────────────────────────────────

class StreamProcessor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.uri = f"ws://{args.host}:{args.port}"
        self.model = YOLO(args.model)
        self.last_stop_time = 0.0
        self.running = True
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        LOG.info("Loaded YOLO model: %s", args.model)
        LOG.info("Target WebSocket URI: %s", self.uri)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send_stop(self, ws, reason: str) -> None:
        now = time.time()
        if now - self.last_stop_time < STOP_COOLDOWN:
            return
        LOG.warning("STOP command → RPi: %s", reason)
        await ws.send(json.dumps({
            "type": "manual_control",
            "mode": "manual",
            "command": "stop",
            "source": "webcam_laptop",
            "triggered_by": "object_detection",
            "reason": reason,
        }))
        self.last_stop_time = now

    # ── Connection logic ──────────────────────────────────────────────────────

    async def run(self) -> None:
        while self.running:
            try:
                await self._connect_and_process()
            except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
                LOG.warning("Connection failed: %s — retrying in %ds…", exc, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)
            except asyncio.CancelledError:
                raise

    async def _connect_and_process(self) -> None:
        LOG.info("Connecting to %s …", self.uri)
        async with websockets.connect(self.uri, max_size=None, open_timeout=10) as ws:
            LOG.info("Connected — waiting for camera frames (YOLO runs here, NOT on RPi)")

            # Register so the server knows we want camera frames
            await ws.send(json.dumps({
                "type": "camera_register",
                "stream": "laptop_processor",
            }))

            loop = asyncio.get_running_loop()
            cv2.namedWindow("Laptop Stream Processor", cv2.WINDOW_NORMAL)

            # Flag: True while YOLO inference is running in the thread pool.
            # Incoming frames are dropped (not queued) when this is set,
            # preventing the receive buffer from backing up.
            inference_running = False
            pending_future: asyncio.Future | None = None

            try:
                async for raw_message in ws:
                    if not self.running:
                        break

                    # ── Check if previous inference finished ──────────────
                    if pending_future is not None and pending_future.done():
                        try:
                            annotated, stop_reason = pending_future.result()
                        except Exception as exc:
                            LOG.debug("Inference error: %s", exc)
                            annotated, stop_reason = None, None

                        inference_running = False
                        pending_future = None

                        if annotated is not None:
                            if stop_reason:
                                await self._send_stop(ws, stop_reason)
                                cv2.putText(
                                    annotated, f"STOP: {stop_reason}",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2,
                                )
                            cv2.imshow("Laptop Stream Processor", annotated)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False
                                break

                    # ── Parse message ─────────────────────────────────────
                    try:
                        data = json.loads(raw_message)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") != "camera_frame":
                        continue

                    b64_data = data.get("data")
                    if not b64_data:
                        continue

                    # ── Drop frame if YOLO is still busy ─────────────────
                    if inference_running:
                        LOG.debug("Dropping frame – inference still running")
                        continue

                    # ── Offload decode + YOLO to thread pool ──────────────
                    inference_running = True
                    pending_future = loop.run_in_executor(
                        self._executor, decode_and_infer, self.model, b64_data
                    )

            except websockets.exceptions.ConnectionClosed as exc:
                LOG.warning("Connection closed: %s", exc)
                raise
            finally:
                cv2.destroyAllWindows()
                if pending_future is not None:
                    pending_future.cancel()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop Stream Processor — YOLO on laptop, stream from RPi")
    parser.add_argument("--host", required=True, help="IP address of the RPi")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path (default: yolov8n.pt)")
    args = parser.parse_args()

    processor = StreamProcessor(args)
    try:
        asyncio.run(processor.run())
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")


if __name__ == "__main__":
    main()
