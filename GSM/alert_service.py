"""Emergency alert service for notifying an ESP + GSM module stack.

This module prepares health and location payloads when an emergency stop is
triggered on the Raspberry Pi controller. The data is pushed to an ESP device
that is responsible for formatting and relaying the alert via an attached GSM
module.
"""
from __future__ import annotations

import http.client
import json
import logging
import socket
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


@dataclass
class ESPConfig:
    """Connection metadata for the ESP listener."""

    host: str
    port: int = 80
    endpoint: str = "/emergency"
    use_https: bool = False
    timeout: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ESPConfig":
        if not isinstance(data, dict):
            raise ValueError("ESP configuration must be a dictionary")
        host = data.get("host")
        if not host:
            raise ValueError("ESP configuration requires a 'host' entry")
        return cls(
            host=str(host),
            port=int(data.get("port", 80)),
            endpoint=str(data.get("endpoint", "/emergency")),
            use_https=bool(data.get("use_https", False)),
            timeout=float(data.get("timeout_seconds", 5.0)),
        )


@dataclass
class GSMConfig:
    """Static metadata for composing outbound SMS payloads."""

    sender: str
    receiver: str
    template: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GSMConfig":
        if not isinstance(data, dict):
            raise ValueError("GSM configuration must be a dictionary")
        sender = data.get("sender_number")
        receiver = data.get("receiver_number")
        template = data.get(
            "message_template",
            "Emergency stop triggered. Location: {latitude}, {longitude}. "
            "Heart rate: {heart_rate_bpm} bpm. SpO2: {spo2_percent}%.",
        )
        if not sender or not receiver:
            raise ValueError("GSM configuration requires sender_number and receiver_number")
        return cls(
            sender=str(sender),
            receiver=str(receiver),
            template=str(template),
        )


@dataclass
class WiFiConfig:
    ssid: str
    password: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WiFiConfig":
        if not isinstance(data, dict):
            raise ValueError("WiFi configuration must be a dictionary")
        ssid = data.get("ssid")
        password = data.get("password")
        if not ssid or not password:
            raise ValueError("WiFi configuration requires 'ssid' and 'password'")
        return cls(ssid=str(ssid), password=str(password))


def _load_config() -> Dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        LOGGER.warning("config.json not found at %s", CONFIG_PATH)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse %s: %s", CONFIG_PATH, exc)
    return {}


def _ensure_wifi_connection(cfg: WiFiConfig) -> None:
    """Attempt to connect to the configured Wi-Fi network using nmcli.

    This call is best-effort and will be skipped when nmcli is unavailable or
    when the system reports that it is already connected.
    """

    try:
        subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], check=False, capture_output=True)
    except FileNotFoundError:
        LOGGER.debug("nmcli not installed; skipping Wi-Fi connectivity check")
        return

    try:
        # Check current connection state
        result = subprocess.run(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split(":", 1)
                if len(parts) == 2 and parts[0] == "yes" and parts[1] == cfg.ssid:
                    LOGGER.debug("Already connected to Wi-Fi '%s'", cfg.ssid)
                    return
    except Exception as exc:
        LOGGER.debug("Wi-Fi status check failed: %s", exc)

    LOGGER.info("Attempting to connect to Wi-Fi SSID '%s'", cfg.ssid)
    try:
        subprocess.run(
            ["nmcli", "dev", "wifi", "connect", cfg.ssid, "password", cfg.password],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except Exception as exc:
        LOGGER.warning("Failed to issue Wi-Fi connect command: %s", exc)


def _sanitize_health(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "heart_rate_bpm": _safe_float(raw.get("heart_rate_bpm")),
        "spo2_percent": _safe_float(raw.get("spo2_percent")),
        "timestamp": raw.get("timestamp"),
        "ir_value": raw.get("ir_value"),
        "red_value": raw.get("red_value"),
    }


def _sanitize_gps(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        "latitude": _safe_float(raw.get("latitude")),
        "longitude": _safe_float(raw.get("longitude")),
        "altitude_m": _safe_float(raw.get("altitude_m")),
        "num_sats": raw.get("num_sats"),
        "fix_quality": raw.get("fix_quality"),
        "horizontal_dilution": _safe_float(raw.get("horizontal_dilution")),
        "raw_sentence": raw.get("raw_sentence"),
        "timestamp": raw.get("timestamp"),
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class EmergencyAlertService:
    """Handles outbound emergency notifications."""

    def __init__(self) -> None:
        config = _load_config()
        self._wifi_cfg = None
        wifi_section = config.get("network", {}).get("wifi")
        if wifi_section:
            try:
                self._wifi_cfg = WiFiConfig.from_dict(wifi_section)
            except ValueError as exc:
                LOGGER.warning("Invalid Wi-Fi configuration: %s", exc)

        try:
            self._esp_cfg = ESPConfig.from_dict(config.get("esp", {}))
        except ValueError as exc:
            LOGGER.error("ESP configuration missing or invalid: %s", exc)
            self._esp_cfg = None

        try:
            self._gsm_cfg = GSMConfig.from_dict(config.get("gsm", {}))
        except ValueError as exc:
            LOGGER.error("GSM configuration missing or invalid: %s", exc)
            self._gsm_cfg = None

    @property
    def available(self) -> bool:
        return self._esp_cfg is not None and self._gsm_cfg is not None

    def notify_emergency(
        self,
        *,
        source: str,
        reason: str,
        health: Optional[Dict[str, Any]],
        gps: Optional[Dict[str, Any]],
    ) -> bool:
        """Send the emergency payload to the ESP.

        Returns True on success, False otherwise.
        """

        if not self.available:
            LOGGER.warning("Emergency alert service not available; missing configuration")
            return False

        if self._wifi_cfg is not None:
            _ensure_wifi_connection(self._wifi_cfg)

        payload = self._build_payload(source=source, reason=reason, health=health, gps=gps)

        try:
            self._dispatch(payload)
        except Exception as exc:
            LOGGER.error("Failed to dispatch emergency payload: %s", exc)
            return False
        return True

    def _build_payload(
        self,
        *,
        source: str,
        reason: str,
        health: Optional[Dict[str, Any]],
        gps: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        gsm_cfg = self._gsm_cfg
        assert gsm_cfg is not None  # guarded by available()

        sanitized_health = _sanitize_health(health)
        sanitized_gps = _sanitize_gps(gps)

        message = gsm_cfg.template.format(
            latitude=self._fmt_coord(sanitized_gps.get("latitude")),
            longitude=self._fmt_coord(sanitized_gps.get("longitude")),
            altitude_m=self._fmt_number(sanitized_gps.get("altitude_m")),
            heart_rate_bpm=self._fmt_number(sanitized_health.get("heart_rate_bpm")),
            spo2_percent=self._fmt_number(sanitized_health.get("spo2_percent")),
            gps_timestamp=sanitized_gps.get("timestamp"),
            health_timestamp=sanitized_health.get("timestamp"),
            source=source,
            reason=reason,
        )

        payload = {
            "event": "emergency_stop",
            "source": source,
            "reason": reason,
            "sender_number": gsm_cfg.sender,
            "receiver_number": gsm_cfg.receiver,
            "message": message,
            "health": sanitized_health,
            "gps": sanitized_gps,
        }
        return payload

    def _dispatch(self, payload: Dict[str, Any]) -> None:
        esp_cfg = self._esp_cfg
        assert esp_cfg is not None  # guarded by available()

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }

        LOGGER.info("Sending emergency notification to ESP at %s:%s", esp_cfg.host, esp_cfg.port)
        conn: Optional[http.client.HTTPConnection] = None
        try:
            if esp_cfg.use_https:
                conn = http.client.HTTPSConnection(
                    esp_cfg.host,
                    esp_cfg.port,
                    timeout=esp_cfg.timeout,
                    context=ssl.create_default_context(),
                )
            else:
                conn = http.client.HTTPConnection(
                    esp_cfg.host,
                    esp_cfg.port,
                    timeout=esp_cfg.timeout,
                )
            conn.request("POST", esp_cfg.endpoint, body=body, headers=headers)
            response = conn.getresponse()
            response_body = response.read()  # noqa: F841 (helpful during debugging)
            if response.status >= 400:
                raise RuntimeError(f"ESP responded with HTTP {response.status}")
        except socket.timeout as exc:
            raise RuntimeError("Timeout while contacting ESP") from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def _fmt_coord(value: Optional[float]) -> str:
        if value is None:
            return "unknown"
        return f"{value:.6f}"

    @staticmethod
    def _fmt_number(value: Optional[float]) -> str:
        if value is None:
            return "unknown"
        return f"{value:.1f}"
