"""Continuous Raspberry Pi sensor monitoring utilities.

This module centralizes the collection of environmental, health, and
navigation data from:
  * DHT11 temperature/humidity sensor wired to GPIO 4 (physical pin 7)
  * MAX30100 pulse oximeter wired to I2C (SDA=GPIO 2, SCL=GPIO 3)
  * Neo-6M GPS receiver wired to the primary UART (RX=GPIO 15, TX=GPIO 14)

Usage pattern
-------------
>>> import sensor_monitor
>>> sensor_monitor.start_monitoring()
>>> time.sleep(2)
>>> sensor_monitor.get_latest_dht()
{'temperature_c': 24.1, 'humidity_percent': 42.0, 'timestamp': 1734134066.12}

Each polling loop runs in its own background thread so that client code can
fetch the freshest reading on demand without blocking.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import board
    import adafruit_dht
except ImportError:  # pragma: no cover - hardware-specific import guard
    board = None  # type: ignore[assignment]
    adafruit_dht = None  # type: ignore[assignment]
    logger.warning("adafruit_dht library not available; DHT11 monitoring disabled")

try:
    import max30100
except ImportError:  # pragma: no cover - hardware-specific import guard
    max30100 = None  # type: ignore[assignment]
    logger.warning("max30100 library not available; MAX30100 monitoring disabled")

try:
    import serial
except ImportError:  # pragma: no cover - hardware-specific import guard
    serial = None  # type: ignore[assignment]
    logger.warning("pyserial library not available; GPS monitoring disabled")

try:
    import pynmea2
except ImportError:  # pragma: no cover - optional parser fallback
    pynmea2 = None  # type: ignore[assignment]
    logger.warning("pynmea2 library not available; GPS parsing may be limited")


@dataclass
class DHTReading:
    """Container for the latest DHT11 reading."""

    temperature_c: Optional[float]
    humidity_percent: Optional[float]
    timestamp: float

    def as_dict(self) -> Dict[str, Optional[float]]:
        """Return the reading as a simple dictionary output."""
        return {
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "timestamp": self.timestamp,
        }


@dataclass
class HealthReading:
    """Container for the latest MAX30100 reading."""

    heart_rate_bpm: Optional[float]
    spo2_percent: Optional[float]
    ir_value: Optional[int]
    red_value: Optional[int]
    timestamp: float

    def as_dict(self) -> Dict[str, Optional[float]]:
        """Return the reading as a simple dictionary output."""
        return {
            "heart_rate_bpm": self.heart_rate_bpm,
            "spo2_percent": self.spo2_percent,
            "ir_value": self.ir_value,
            "red_value": self.red_value,
            "timestamp": self.timestamp,
        }


@dataclass
class GPSReading:
    """Container for the latest GPS fix."""

    latitude: Optional[float]
    longitude: Optional[float]
    altitude_m: Optional[float]
    num_sats: Optional[int]
    fix_quality: Optional[int]
    horizontal_dilution: Optional[float]
    raw_sentence: Optional[str]
    timestamp: float

    def as_dict(self) -> Dict[str, Optional[float]]:
        """Return the reading as a simple dictionary output."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "num_sats": self.num_sats,
            "fix_quality": self.fix_quality,
            "horizontal_dilution": self.horizontal_dilution,
            "raw_sentence": self.raw_sentence,
            "timestamp": self.timestamp,
        }


class SensorMonitor:
    """Spawns threads that continuously collect sensor data."""

    def __init__(
        self,
        *,
        dht_pin: Optional[Any] = None,
        gps_port: str = "/dev/serial0",
        gps_baudrate: int = 9600,
        dht_interval: float = 2.0,
        health_interval: float = 1.0,
        gps_interval: float = 0.2,
    ) -> None:
        # DHT11 configuration: expects GPIO4 (physical pin 7)
        self._dht_pin = dht_pin if dht_pin is not None else getattr(board, "D4", None)
        self._dht_interval = max(0.5, dht_interval)
        self._dht_sensor = None
        self._dht_reading: Optional[DHTReading] = None
        self._dht_lock = threading.Lock()

        # MAX30100 configuration: expects SDA=GPIO2, SCL=GPIO3
        self._health_interval = max(0.5, health_interval)
        self._health_sensor = None
        self._health_reading: Optional[HealthReading] = None
        self._health_lock = threading.Lock()

        # Neo-6M GPS configuration: expects UART (TX=GPIO14, RX=GPIO15)
        self._gps_port = gps_port
        self._gps_baudrate = gps_baudrate
        self._gps_interval = max(0.1, gps_interval)
        self._gps_serial = None
        self._gps_reading: Optional[GPSReading] = None
        self._gps_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}

    # Public API -----------------------------------------------------------------

    def start(self) -> None:
        """Spin up worker threads.

        Input: none (uses configured sensor interfaces).
        Output: none (threads run in background and update cached readings).
        """
        if any(t.is_alive() for t in self._threads.values()):
            logger.debug("Sensor monitor already running")
            return

        self._stop_event.clear()

        if adafruit_dht and self._dht_pin is not None:
            self._ensure_dht_sensor()
            self._start_thread("dht", target=self._poll_dht)
        else:
            logger.warning("DHT11 sensor disabled; missing library or pin mapping")

        if max30100:
            self._ensure_health_sensor()
            self._start_thread("health", target=self._poll_health)
        else:
            logger.warning("MAX30100 sensor disabled; missing library")

        if serial:
            self._start_thread("gps", target=self._poll_gps)
        else:
            logger.warning("GPS monitoring disabled; missing pyserial")

    def stop(self, *, join_timeout: float = 1.0) -> None:
        """Signal all worker threads to shut down.

        Input: optional join timeout in seconds.
        Output: none (threads are stopped, cached readings remain available).
        """
        self._stop_event.set()
        for name, thread in list(self._threads.items()):
            if thread.is_alive():
                thread.join(timeout=join_timeout)
            self._threads.pop(name, None)

        if self._gps_serial is not None:
            try:
                self._gps_serial.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Swallowing exception while closing GPS serial", exc_info=True)
            self._gps_serial = None

    def get_dht_reading(self) -> Optional[Dict[str, Optional[float]]]:
        """Fetch the latest cached DHT11 reading.

        Input: none. Output: dictionary with keys temperature_c, humidity_percent, timestamp.
        Returns None when no successful sample has been collected yet.
        """
        with self._dht_lock:
            return self._dht_reading.as_dict() if self._dht_reading else None

    def get_health_reading(self) -> Optional[Dict[str, Optional[float]]]:
        """Fetch the latest cached MAX30100 reading.

        Input: none. Output: dictionary with SpO2, heart rate, raw IR/RED values, timestamp.
        Returns None when no successful sample has been collected yet.
        """
        with self._health_lock:
            return self._health_reading.as_dict() if self._health_reading else None

    def get_gps_reading(self) -> Optional[Dict[str, Optional[float]]]:
        """Fetch the latest cached GPS fix.

        Input: none. Output: dictionary with position, fix metadata, and timestamp.
        Returns None when no successful sentence has been parsed yet.
        """
        with self._gps_lock:
            return self._gps_reading.as_dict() if self._gps_reading else None

    # Internal helpers -----------------------------------------------------------

    def _start_thread(self, name: str, *, target) -> None:
        thread = threading.Thread(target=target, name=f"sensor-{name}", daemon=True)
        self._threads[name] = thread
        thread.start()
        logger.info("Started %s polling thread", name)

    def _ensure_dht_sensor(self) -> None:
        if self._dht_sensor is None and adafruit_dht:
            try:
                self._dht_sensor = adafruit_dht.DHT11(self._dht_pin)  # type: ignore[arg-type]
                logger.info("Initialized DHT11 on pin %s", self._dht_pin)
            except Exception:  # pragma: no cover - hardware errors
                logger.exception("Failed to initialize DHT11 sensor")
                self._dht_sensor = None

    def _ensure_health_sensor(self) -> None:
        if self._health_sensor is None and max30100:
            try:
                sensor = max30100.MAX30100()
                sensor.enable_spo2()
                self._health_sensor = sensor
                logger.info("Initialized MAX30100 sensor in SpO2 mode")
            except Exception:  # pragma: no cover - hardware errors
                logger.exception("Failed to initialize MAX30100 sensor")
                self._health_sensor = None

    def _get_gps_serial(self):
        if self._gps_serial is None and serial:
            try:
                self._gps_serial = serial.Serial(
                    self._gps_port,
                    self._gps_baudrate,
                    timeout=1,
                )
                logger.info("Opened GPS serial port %s at %d baud", self._gps_port, self._gps_baudrate)
            except Exception:  # pragma: no cover - hardware errors
                logger.exception("Failed to open GPS serial port")
                self._gps_serial = None
        return self._gps_serial

    # Polling loops --------------------------------------------------------------

    def _poll_dht(self) -> None:
        while not self._stop_event.is_set():
            if self._dht_sensor is None:
                self._ensure_dht_sensor()
                time.sleep(self._dht_interval)
                continue

            try:
                temperature = float(self._dht_sensor.temperature)
                humidity = float(self._dht_sensor.humidity)
                reading = DHTReading(
                    temperature_c=temperature,
                    humidity_percent=humidity,
                    timestamp=time.time(),
                )
                with self._dht_lock:
                    self._dht_reading = reading
            except RuntimeError as exc:  # common transient error from DHT sensors
                logger.debug("Transient DHT read error: %s", exc)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Unexpected DHT read failure")
                self._dht_sensor = None
            time.sleep(self._dht_interval)

    def _poll_health(self) -> None:
        while not self._stop_event.is_set():
            if self._health_sensor is None:
                self._ensure_health_sensor()
                time.sleep(self._health_interval)
                continue

            sensor = self._health_sensor
            try:
                sensor.read_sensor()
                ir_value = getattr(sensor, "ir", None)
                red_value = getattr(sensor, "red", None)
                heart_rate = getattr(sensor, "heart_rate", None)
                spo2 = getattr(sensor, "spO2", None)
                reading = HealthReading(
                    heart_rate_bpm=float(heart_rate) if heart_rate is not None else None,
                    spo2_percent=float(spo2) if spo2 is not None else None,
                    ir_value=int(ir_value) if ir_value is not None else None,
                    red_value=int(red_value) if red_value is not None else None,
                    timestamp=time.time(),
                )
                with self._health_lock:
                    self._health_reading = reading
            except Exception:  # pragma: no cover - hardware faults
                logger.exception("Unexpected MAX30100 read failure")
                self._health_sensor = None
            time.sleep(self._health_interval)

    def _poll_gps(self) -> None:
        while not self._stop_event.is_set():
            ser = self._get_gps_serial()
            if ser is None:
                time.sleep(2.0)
                continue

            try:
                raw_bytes = ser.readline()
                if not raw_bytes:
                    time.sleep(self._gps_interval)
                    continue
                sentence = raw_bytes.decode("ascii", errors="ignore").strip()
            except Exception:  # pragma: no cover - serial issues
                logger.exception("Unexpected GPS serial failure")
                if self._gps_serial is not None:
                    try:
                        self._gps_serial.close()
                    except Exception:
                        logger.debug("Silencing GPS close failure", exc_info=True)
                self._gps_serial = None
                time.sleep(2.0)
                continue

            if not sentence.startswith("$"):
                time.sleep(self._gps_interval)
                continue

            latitude = None
            longitude = None
            altitude = None
            num_sats = None
            fix_quality = None
            dilution = None

            if pynmea2:
                try:
                    message = pynmea2.parse(sentence)
                    latitude = getattr(message, "latitude", None)
                    longitude = getattr(message, "longitude", None)
                    altitude = getattr(message, "altitude", None)
                    num_sats = getattr(message, "num_sats", None)
                    fix_quality = getattr(message, "gps_qual", None)
                    dilution = getattr(message, "horizontal_dil", None)

                    # Convert to float when values are present and not empty strings.
                    latitude = float(latitude) if latitude not in (None, "") else None
                    longitude = float(longitude) if longitude not in (None, "") else None
                    altitude = float(altitude) if altitude not in (None, "") else None
                    num_sats = int(num_sats) if num_sats not in (None, "") else None
                    fix_quality = int(fix_quality) if fix_quality not in (None, "") else None
                    dilution = float(dilution) if dilution not in (None, "") else None
                except pynmea2.ParseError:
                    logger.debug("Failed to parse NMEA sentence: %s", sentence)
                    time.sleep(self._gps_interval)
                    continue
            else:
                logger.debug("Raw GPS sentence captured (parser unavailable): %s", sentence)

            reading = GPSReading(
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude,
                num_sats=num_sats,
                fix_quality=fix_quality,
                horizontal_dilution=dilution,
                raw_sentence=sentence,
                timestamp=time.time(),
            )
            with self._gps_lock:
                self._gps_reading = reading
            time.sleep(self._gps_interval)


# Module-level singleton helpers -------------------------------------------------
_MONITOR = SensorMonitor()


def start_monitoring() -> None:
    """Start background polling for all sensors."""
    _MONITOR.start()


def stop_monitoring(join_timeout: float = 1.0) -> None:
    """Stop background polling for all sensors."""
    _MONITOR.stop(join_timeout=join_timeout)


def get_latest_dht() -> Optional[Dict[str, Optional[float]]]:
    """Return the current DHT11 reading as a dictionary or None."""
    return _MONITOR.get_dht_reading()


def get_latest_health() -> Optional[Dict[str, Optional[float]]]:
    """Return the current MAX30100 reading as a dictionary or None."""
    return _MONITOR.get_health_reading()


def get_latest_gps() -> Optional[Dict[str, Optional[float]]]:
    """Return the current GPS reading as a dictionary or None."""
    return _MONITOR.get_gps_reading()


__all__ = [
    "start_monitoring",
    "stop_monitoring",
    "get_latest_dht",
    "get_latest_health",
    "get_latest_gps",
    "SensorMonitor",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_monitoring()
    try:
        while True:
            time.sleep(5)
            logger.info("DHT: %s", get_latest_dht())
            logger.info("Health: %s", get_latest_health())
            logger.info("GPS: %s", get_latest_gps())
    except KeyboardInterrupt:
        logger.info("Stopping sensor monitor")
    finally:
        stop_monitoring()
