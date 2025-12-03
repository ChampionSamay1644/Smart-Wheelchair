#!/usr/bin/env python3
"""Lightweight camera streamer that publishes JPEG frames to the
existing wheelchair control WebSocket server.

The script captures frames from the Raspberry Pi camera (preferring
picamera2 when available, falling back to OpenCV's VideoCapture) and sends
base64-encoded JPEG frames over the same WebSocket connection used for
voice recognition and command control. The wheelchair Flutter app receives
`camera_frame` messages and renders the live preview without any on-device
object detection.

Usage example:
    python3 rpi_stream_detect_ws.py --host 192.168.1.10 --port 8765 \
        --width 320 --height 240 --stream-fps 15
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
from contextlib import suppress

import cv2
import websockets

LOG = logging.getLogger("rpi_camera_stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream camera frames over the wheelchair WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="IP/hostname of the control WebSocket server")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port of the control server")
    parser.add_argument("--stream-name", default="front_camera", help="Identifier for this camera stream")
    parser.add_argument("--width", type=int, default=320, help="Captured frame width")
    parser.add_argument("--height", type=int, default=240, help="Captured frame height")
    parser.add_argument("--stream-fps", type=float, default=15.0, help="Target streaming FPS")
    parser.add_argument(
        "--device",
        default="0",
        help="Camera device index or V4L2 path (e.g. 0 or /dev/video0)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality (1-100)")
    parser.add_argument("--reconnect-delay", type=float, default=2.5, help="Seconds to wait before reconnecting")
    return parser.parse_args()


class CameraCapture(threading.Thread):
    """Continuously capture frames on a background thread."""

    def __init__(self, width: int, height: int, target_fps: float, device: str | int) -> None:
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.interval = 1.0 / max(1.0, target_fps)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.use_picamera2 = False
        self.use_picamera_legacy = False
        self.picam2 = None
        self.picamera = None
        self._legacy_stream = None
        self._legacy_buffer = None
        self.cap = None
        self.device = device
        self._failure_count = 0
        self._last_failure_log = 0.0

        try:
            from picamera2 import Picamera2  # type: ignore
        except ModuleNotFoundError as exc:
            self._ensure_picamera2_path()
            try:
                from picamera2 import Picamera2  # type: ignore
            except Exception as retry_exc:  # pragma: no cover - optional dependency
                LOG.warning("picamera2 unavailable, attempting legacy picamera: %s", retry_exc)
                if not self._init_legacy_picamera(target_fps):
                    self._init_opencv_capture()
                return
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("picamera2 unavailable, attempting legacy picamera: %s", exc)
            if not self._init_legacy_picamera(target_fps):
                self._init_opencv_capture()
            return

        self.picam2 = Picamera2()
        video_config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)},
            buffer_count=2,
        )
        self.picam2.configure(video_config)
        self.picam2.start()
        time.sleep(0.3)
        try:
            target_rate = max(1.0, min(60.0, target_fps))
            self.picam2.set_controls({"FrameRate": target_rate})
        except Exception:  # pragma: no cover - best effort tuning
            LOG.debug("Unable to apply requested frame rate to picamera2")
        self.use_picamera2 = True
        LOG.info(
            "Using picamera2 for capture at %dx%d (target %.1f FPS)",
            width,
            height,
            target_fps,
        )

    def _resolve_device(self):
        if isinstance(self.device, str) and self.device.isdigit():
            return int(self.device)
        return self.device

    def _ensure_picamera2_path(self) -> None:
        """Expose system Picamera2 modules to virtual environments."""
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        candidate_paths = [
            "/usr/lib/python3/dist-packages",
            f"/usr/lib/python{python_version}/dist-packages",
            "/usr/local/lib/python3/dist-packages",
            f"/usr/local/lib/python{python_version}/dist-packages",
            f"/usr/lib/aarch64-linux-gnu/python{python_version}/dist-packages",
            f"/usr/lib/arm-linux-gnueabihf/python{python_version}/dist-packages",
            os.environ.get("LIBCAMERA_PYTHON_PATH"),
        ]

        try:  # pragma: no cover - depends on system configuration
            import sysconfig

            sysconfig_paths = sysconfig.get_paths()
            candidate_paths.extend(
                sysconfig_paths.get(key)
                for key in ("platlib", "platstdlib", "purelib", "stdlib")
            )
        except Exception:
            pass

        for path in candidate_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)

    def _init_opencv_capture(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        device = self._resolve_device()
        backends = [cv2.CAP_V4L2, cv2.CAP_V4L, cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(device, backend)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOG.debug("OpenCV backend %s failed to init: %s", backend, exc)
                continue

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 1.0 / self.interval)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap

            backend_name = {
                cv2.CAP_V4L2: "CAP_V4L2",
                cv2.CAP_V4L: "CAP_V4L",
                cv2.CAP_ANY: "CAP_ANY",
            }.get(backend, str(backend))

            LOG.info("Using OpenCV capture on %s with backend %s", device, backend_name)
            self._failure_count = 0
            self._last_failure_log = 0.0
            return True

        LOG.error(
            "Unable to open camera device %s via OpenCV; check cabling or enable the legacy V4L2 driver",
            device,
        )
        self.cap = None
        return False

    def _init_legacy_picamera(self, target_fps: float) -> bool:
        try:
            from picamera import PiCamera  # type: ignore
            from picamera.array import PiRGBArray  # type: ignore

            camera = PiCamera()
            camera.resolution = (self.width, self.height)
            camera.framerate = target_fps
            time.sleep(0.5)
            buffer = PiRGBArray(camera, size=(self.width, self.height))
            buffer.truncate(0)
            buffer.seek(0)
            self.picamera = camera
            self._legacy_buffer = buffer
            self._legacy_stream = camera.capture_continuous(
                buffer, format="bgr", use_video_port=True
            )
            self.use_picamera_legacy = True
            self._failure_count = 0
            self._last_failure_log = 0.0
            LOG.info("Using legacy picamera module for capture")
            return True
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("Legacy picamera unavailable, falling back to OpenCV capture: %s", exc)
            self.use_picamera_legacy = False
            self.picamera = None
            self._legacy_stream = None
            self._legacy_buffer = None
            return False

    def run(self) -> None:
        while self.running:
            start = time.time()
            frame = None
            if self.use_picamera2 and self.picam2 is not None:
                try:
                    arr = self.picam2.capture_array("main")
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    LOG.error("picamera2 capture failed: %s", exc)
                    time.sleep(0.05)
                    continue
                if arr.size == 0:
                    LOG.debug("picamera2 returned empty frame; retrying")
                    time.sleep(0.02)
                    continue
                if arr.ndim == 3:
                    if arr.shape[2] == 3:
                        frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    elif arr.shape[2] == 4:
                        frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    else:
                        frame = arr
                else:
                    frame = arr
            elif self.use_picamera_legacy and self._legacy_stream is not None:
                try:
                    stream = next(self._legacy_stream)
                except StopIteration:
                    LOG.error("Legacy picamera stream terminated; switching to OpenCV fallback")
                    self.use_picamera_legacy = False
                    self._legacy_stream = None
                    self._legacy_buffer = None
                    if self.picamera is not None:
                        try:
                            self.picamera.close()
                        except Exception:
                            pass
                        self.picamera = None
                    continue
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    LOG.error("Legacy picamera capture failed: %s", exc)
                    self.use_picamera_legacy = False
                    self._legacy_stream = None
                    self._legacy_buffer = None
                    if self.picamera is not None:
                        try:
                            self.picamera.close()
                        except Exception:
                            pass
                        self.picamera = None
                    continue
                frame = stream.array
                stream.truncate(0)
                stream.seek(0)
            else:
                if self.cap is None:
                    if not self._init_opencv_capture():
                        time.sleep(0.5)
                        continue
                success, grabbed = self.cap.read()
                if not success or grabbed is None:
                    self._failure_count += 1
                    now = time.time()
                    if now - self._last_failure_log > 1.0:
                        LOG.warning(
                            "Camera read failed (attempt %d); retrying",
                            self._failure_count,
                        )
                        self._last_failure_log = now
                    if self._failure_count and self._failure_count % 20 == 0:
                        LOG.info("Reinitializing OpenCV capture after repeated failures")
                        self._init_opencv_capture()
                    time.sleep(0.1)
                    continue
                self._failure_count = 0
                frame = grabbed

            with self.lock:
                self.frame = frame

            elapsed = time.time() - start
            sleep_for = self.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.use_picamera2 and self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self.picam2 = None
        if self.use_picamera_legacy and self.picamera is not None:
            try:
                self.picamera.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self.picamera = None
            self._legacy_stream = None
            self._legacy_buffer = None
            self.use_picamera_legacy = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self.cap = None


def encode_jpeg(frame, quality: int) -> bytes:
    quality = int(max(1, min(100, quality)))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


async def drain_messages(ws: websockets.WebSocketClientProtocol) -> None:
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                LOG.debug("Received non-JSON text message from server")
                continue
            msg_type = data.get("type")
            if msg_type in {"connection", "camera_registered"}:
                LOG.info("Server: %s", data.get("message", msg_type))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        LOG.debug("WebSocket reader stopped: %s", exc)


async def stream_frames(args: argparse.Namespace, capture: CameraCapture) -> None:
    uri = f"ws://{args.host}:{args.port}"
    interval = 1.0 / max(1.0, args.stream_fps)
    loop = asyncio.get_running_loop()

    while True:
        try:
            LOG.info("Connecting to %s", uri)
            async with websockets.connect(uri, max_size=None, ping_interval=20, ping_timeout=20) as ws:
                LOG.info("Camera stream connected; registering as '%s'", args.stream_name)
                register_payload = {
                    "type": "camera_register",
                    "stream": args.stream_name,
                    "width": args.width,
                    "height": args.height,
                }
                await ws.send(json.dumps(register_payload))

                reader = asyncio.create_task(drain_messages(ws))
                try:
                    while True:
                        frame = capture.get_frame()
                        if frame is None:
                            await asyncio.sleep(0.02)
                            continue

                        jpeg_bytes = await loop.run_in_executor(None, encode_jpeg, frame, args.jpeg_quality)
                        frame_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                        payload = {
                            "type": "camera_frame",
                            "stream": args.stream_name,
                            "format": "jpeg",
                            "timestamp": time.time(),
                            "data": frame_b64,
                        }
                        await ws.send(json.dumps(payload))
                        await asyncio.sleep(interval)
                finally:
                    reader.cancel()
                    with suppress(Exception):
                        await reader
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Camera stream error: %s", exc)
            await asyncio.sleep(args.reconnect_delay)


async def main_async(args: argparse.Namespace, capture: CameraCapture) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal(signum):
        LOG.info("Signal %s received, shutting down camera stream", signum)
        capture.stop()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal, sig)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(sig, lambda _sig, _frame: handle_signal(_sig))

    stream_task = asyncio.create_task(stream_frames(args, capture))

    await stop_event.wait()
    stream_task.cancel()
    with suppress(asyncio.CancelledError):
        await stream_task


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    capture = CameraCapture(args.width, args.height, args.stream_fps, args.device)
    capture.start()

    try:
        asyncio.run(main_async(args, capture))
        return 0
    except KeyboardInterrupt:
        LOG.info("Keyboard interrupt received, stopping camera stream")
        return 0
    except Exception as exc:  # pragma: no cover - top-level guard
        LOG.exception("Fatal error: %s", exc)
        return 1
    finally:
        capture.stop()
        capture.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
