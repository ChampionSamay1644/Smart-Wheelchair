#!/usr/bin/env python3
"""Bluetooth transport for Smart Wheelchair control.

This module exposes a Bluetooth RFCOMM server that mirrors the voice/websocket
command pipeline so the Flutter client can drive the motors when Wi-Fi is
unavailable. Messages are newline-delimited JSON payloads. Two logical command
modes are supported:

- Manual button presses (``{"mode": "manual", "command": "forward"}``)
- Joystick streaming updates (``{"mode": "joystick", "x": 0.1, "y": 0.9}``)

Telemetry and acknowledgements are emitted back to the client using the same
framing (one JSON object per line).

The implementation runs the blocking RFCOMM socket stack inside a background
thread and hands off decoded messages to the asyncio command dispatcher running
in the main event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

try:  # pragma: no cover - imported only on Raspberry Pi with PyBluez installed
    import bluetooth  # type: ignore[import]

    from bluetooth import (
        BluetoothError,
        BluetoothSocket,
        RFCOMM,
        SERIAL_PORT_CLASS,
        SERIAL_PORT_PROFILE,
        advertise_service,
        stop_advertising,
    )  # type: ignore[import]

    BLUETOOTH_AVAILABLE = True
except ImportError:  # pragma: no cover - development environments w/out PyBluez
    bluetooth = None
    BluetoothSocket = object  # type: ignore
    BluetoothError = Exception  # type: ignore
    BLUETOOTH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - type check helpers only
    from bluetooth import BluetoothSocket as _BluetoothSocket


class BluetoothController:
    """Thin RFCOMM server that forwards commands to the asyncio dispatcher."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        dispatcher,
        event_callback,
        device_name: str = "SmartWheelchair",
        channel: int = 3,
        service_uuid: str = "94f39d29-7d6d-437d-973b-fba39e49d4ee",
        handshake_timeout: float = 8.0,
    ) -> None:
        self.loop = loop
        self._dispatcher = dispatcher
        self._event_callback = event_callback
        self.device_name = device_name
        self.channel = channel
        self.service_uuid = service_uuid
        self.handshake_timeout = handshake_timeout

        self._server_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tx_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._client_socket: Optional[Any] = None
        self._client_info: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    def start(self) -> bool:
        if not BLUETOOTH_AVAILABLE:
            logger.warning("PyBluez is not available; Bluetooth transport disabled")
            return False

        if self.is_running:
            return True

        self._stop_event.clear()
        self._server_thread = threading.Thread(
            target=self._run_server, name="bluetooth-rfcomm", daemon=True
        )
        self._server_thread.start()
        logger.info(
            "Bluetooth RFCOMM server thread started (channel=%s, name=%s)",
            self.channel,
            self.device_name,
        )
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except Exception:
                pass
            self._client_socket = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        self._server_thread = None
        logger.info("Bluetooth RFCOMM server stopped")

    def publish_event(self, payload: Dict[str, Any]) -> None:
        if not self._client_socket:
            return
        self._tx_queue.put(payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_server(self) -> None:  # pragma: no cover - runs in background
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                server_sock = BluetoothSocket(RFCOMM)
                server_sock.bind(("", self.channel))
                server_sock.listen(1)

                advertise_service(
                    server_sock,
                    self.device_name,
                    service_id=self.service_uuid,
                    service_classes=[self.service_uuid, SERIAL_PORT_CLASS],
                    profiles=[SERIAL_PORT_PROFILE],
                )

                logger.info("Bluetooth RFCOMM listening for connections…")
                client_sock, client_info = server_sock.accept()
                logger.info("Bluetooth client connected from %s", client_info)
                self._client_socket = client_sock
                self._client_info = f"{client_info[0]}:{client_info[1]}"

                self._notify_status("connected")
                self._handle_client(client_sock)
            except BluetoothError as exc:
                if self._stop_event.is_set():
                    break
                logger.error("Bluetooth socket error: %s", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.exception("Unexpected Bluetooth server error: %s", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
            finally:
                try:
                    stop_advertising(server_sock)
                except Exception:
                    pass
                try:
                    server_sock.close()
                except Exception:
                    pass
                if self._client_socket is not None:
                    try:
                        self._client_socket.close()
                    except Exception:
                        pass
                    self._client_socket = None
                self._notify_status("disconnected")

        logger.debug("Bluetooth server loop exited")

    def _notify_status(self, status: str) -> None:
        if self._event_callback is None:
            return
        event = {
            "type": "bluetooth_status",
            "status": status,
            "client": self._client_info,
            "timestamp": time.time(),
        }
        try:
            asyncio.run_coroutine_threadsafe(
                self._event_callback(event), self.loop
            )
        except RuntimeError:
            # Event loop already closed
            pass
        if status == "disconnected":
            try:
                asyncio.run_coroutine_threadsafe(
                    self._dispatcher.submit_command(
                        source="bluetooth",
                        mode="manual",
                        payload={"command": "stop", "reason": "bluetooth_disconnect"},
                    ),
                    self.loop,
                )
            except RuntimeError:
                pass

    def _handle_client(self, client_sock: Any) -> None:  # pragma: no cover
        client_sock.settimeout(0.5)
        buffer = bytearray()
        backoff = 0.05
        try:
            self._queue_event(
                {
                    "type": "telemetry",
                    "state": "connected",
                    "transport": "bluetooth",
                    "message": "Bluetooth link established",
                }
            )
            while not self._stop_event.is_set():
                self._flush_tx(client_sock)
                try:
                    chunk = client_sock.recv(1024)
                except BluetoothError as exc:
                    if self._stop_event.is_set():
                        break
                    errno = getattr(exc, "errno", None)
                    if errno in {0, 11, 110} or "timed out" in str(exc).lower():
                        continue
                    logger.warning("Bluetooth recv error: %s", exc)
                    break
                if not chunk:
                    logger.info("Bluetooth client closed connection")
                    break
                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b"\n")
                    if newline == -1:
                        break
                    frame = buffer[:newline].decode("utf-8", errors="ignore").strip()
                    del buffer[: newline + 1]
                    if not frame:
                        continue
                    self._dispatch_frame(frame)
                backoff = 0.05
            self._queue_event(
                {
                    "type": "telemetry",
                    "state": "disconnected",
                    "transport": "bluetooth",
                    "message": "Bluetooth link closed",
                }
            )
        finally:
            try:
                client_sock.close()
            except Exception:
                pass
            self._client_socket = None
            self._client_info = None

    def _flush_tx(self, client_sock: Any) -> None:  # pragma: no cover
        try:
            while True:
                payload = self._tx_queue.get_nowait()
                data = json.dumps(payload, separators=(",", ":")) + "\n"
                client_sock.send(data.encode("utf-8"))
        except queue.Empty:
            return
        except BluetoothError as exc:
            logger.warning("Bluetooth send failed: %s", exc)

    def _dispatch_frame(self, frame: str) -> None:  # pragma: no cover
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            logger.warning("Discarding malformed Bluetooth frame: %s", frame)
            return

        mode = payload.get("mode")
        if mode not in {"manual", "joystick"}:
            logger.warning("Unsupported Bluetooth payload: %s", payload)
            return

        coro = self._dispatcher.submit_command(
            source="bluetooth",
            mode=mode,
            payload=payload,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            future.result(timeout=self.handshake_timeout)
        except Exception as exc:
            logger.error("Failed to process Bluetooth command: %s", exc)
            self._queue_event(
                {
                    "type": "command_ack",
                    "state": "error",
                    "source": "bluetooth",
                    "mode": mode,
                    "message": f"Command processing failed: {exc}",
                    "timestamp": time.time(),
                }
            )

    def _queue_event(self, payload: Dict[str, Any]) -> None:
        if self._client_socket is None:
            return
        self._tx_queue.put(payload)


__all__ = ["BluetoothController", "BLUETOOTH_AVAILABLE"]
