#!/usr/bin/env python3
"""
Raspberry Pi Wheelchair Control WebSocket Server

This server receives audio chunks from the Flutter app via WebSocket,
processes them using the wheelchair_control.py voice recognition system,
and controls the wheelchair motors via GPIO pins.

Features:
- WebSocket server for receiving audio streams
- Voice command processing
- Motor control via GPIO
- Command feedback to the app

Author: GitHub Copilot
Date: November 2024
"""

import asyncio
import websockets
import json
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import time
import sys
import io
import re
from typing import Optional, Tuple
from bluetooth_controller import BluetoothController, BLUETOOTH_AVAILABLE

# GPIO control (only on Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. Motor control disabled.")

# Import wheelchair control helpers
from wheelchair_control import (
    process_command_with_whisper_tiny,
    preprocess_audio_for_verification,
    compute_enhanced_embedding,
    cosine_similarity,
    SIMILARITY_THRESHOLD,
    SPEAKER_VARIANT_SUPPORT_WINDOW,
    SPEAKER_VARIANT_SUPPORT_STEP,
    SPEAKER_VARIANT_SUPPORT_MAX,
    SPEAKER_MIN_SCORE_GAP,
    SPEAKER_SCORE_GAIN,
    SPEAKER_SCORE_OFFSET,
    SPEAKER_SCORE_AVG_WEIGHT,
    SPEAKER_SCORE_AVG_BIAS,
    SPEAKER_SOLO_THRESHOLD_RELAX,
    load_local_stt_pipeline,
    load_speaker_recognizer,
    ENROLLMENT_PHRASES,
)

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Configuration
WEBSOCKET_PORT = CONFIG['raspberry_pi']['websocket_port']
SAMPLE_RATE = CONFIG['voice_processing']['sample_rate']
TEMP_DIR = Path("./temp_ws_audio")
TEMP_DIR.mkdir(exist_ok=True)

COMMAND_TIMEOUT_SECONDS = float(CONFIG['raspberry_pi'].get('command_timeout_seconds', 1.5))
JOYSTICK_TIMEOUT_SECONDS = float(CONFIG['raspberry_pi'].get('joystick_timeout_seconds', 0.2))
JOYSTICK_DEAD_ZONE = float(CONFIG['raspberry_pi'].get('joystick_dead_zone', 0.15))
JOYSTICK_STOP_GRACE_UPDATES = int(CONFIG['raspberry_pi'].get('joystick_stop_grace_updates', 2))
JOYSTICK_STOP_CONFIRMATION = int(CONFIG['raspberry_pi'].get('joystick_stop_confirmation', 2))
JOYSTICK_STOP_BURST_COUNT = int(CONFIG['raspberry_pi'].get('joystick_stop_burst_count', 6))
SPEAKER_VERIFICATION_ENABLED = bool(CONFIG['voice_processing'].get('speaker_verification_enabled', False))

BT_CONFIG = CONFIG.get('bluetooth', {})
BT_ENABLED = bool(BT_CONFIG.get('enabled', False))
BT_DEVICE_NAME = BT_CONFIG.get('device_name', 'SmartWheelchair')
BT_CHANNEL = int(BT_CONFIG.get('channel', 3))
BT_SERVICE_UUID = BT_CONFIG.get('service_uuid', '94f39d29-7d6d-437d-973b-fba39e49d4ee')

VOICE_DB_PROCESSED_DIR = Path("./voice_db_processed")
VOICE_DB_EMBEDDINGS_DIR = Path("./voice_db_embeddings")
VOICE_DB_PROCESSED_DIR.mkdir(exist_ok=True)
VOICE_DB_EMBEDDINGS_DIR.mkdir(exist_ok=True)

# GPIO Pin Configuration
MOTOR_PINS = CONFIG['raspberry_pi']['gpio_pins']
M_L_IN1 = MOTOR_PINS['motor_left_in1']
M_L_IN2 = MOTOR_PINS['motor_left_in2']
M_R_IN1 = MOTOR_PINS['motor_right_in1']
M_R_IN2 = MOTOR_PINS['motor_right_in2']

# =============================================================================
# MOTOR CONTROL CLASS
# =============================================================================

class MotorController:
    """Controls wheelchair motors via GPIO pins."""
    
    def __init__(self):
        self.initialized = False
        self._current_command = None
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup motor pins
                for pin in [M_L_IN1, M_L_IN2, M_R_IN1, M_R_IN2]:
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                
                self.initialized = True
                print(f"✓ GPIO initialized - Motor control enabled")
                print(f"  Left motor: IN1={M_L_IN1}, IN2={M_L_IN2}")
                print(f"  Right motor: IN1={M_R_IN1}, IN2={M_R_IN2}")
            except Exception as e:
                print(f"✗ Failed to initialize GPIO: {e}")
                self.initialized = False
        else:
            print("✗ GPIO not available - Motor control disabled")
    
    def _set_motors(self, l1: bool, l2: bool, r1: bool, r2: bool):
        """Set motor pin states."""
        if not self.initialized:
            return
        
        GPIO.output(M_L_IN1, GPIO.HIGH if l1 else GPIO.LOW)
        GPIO.output(M_L_IN2, GPIO.HIGH if l2 else GPIO.LOW)
        GPIO.output(M_R_IN1, GPIO.HIGH if r1 else GPIO.LOW)
        GPIO.output(M_R_IN2, GPIO.HIGH if r2 else GPIO.LOW)

    def _apply_state(self, command: str, *, l1: bool, l2: bool, r1: bool, r2: bool) -> bool:
        """Apply a motor command only when it represents a state change."""
        if self._current_command == command:
            return False

        self._set_motors(l1, l2, r1, r2)
        self._current_command = command
        print(f"Motors: {command.upper()}")
        return True
    
    def stop(self) -> bool:
        """Stop all motors."""
        return self._apply_state("stop", l1=False, l2=False, r1=False, r2=False)
    
    def forward(self) -> bool:
        """Move forward."""
        return self._apply_state("forward", l1=True, l2=False, r1=True, r2=False)
    
    def backward(self) -> bool:
        """Move backward."""
        return self._apply_state("backward", l1=False, l2=True, r1=False, r2=True)
    
    def left(self) -> bool:
        """Turn left."""
        return self._apply_state("left", l1=False, l2=True, r1=True, r2=False)
    
    def right(self) -> bool:
        """Turn right."""
        return self._apply_state("right", l1=True, l2=False, r1=False, r2=True)
    
    def execute_command(self, command: str) -> dict:
        """Execute a motor command and describe the outcome."""
        alias_map = {
            "rotate_left": "left",
            "spin_left": "left",
            "turn_left": "left",
            "circle_left": "left",
            "veer_left": "left",
            "rotate_right": "right",
            "spin_right": "right",
            "turn_right": "right",
            "circle_right": "right",
            "veer_right": "right",
        }

        normalized = alias_map.get(command, command)

        command_map = {
            "forward": self.forward,
            "backward": self.backward,
            "left": self.left,
            "right": self.right,
            "stop": self.stop,
        }

        action = command_map.get(normalized)
        if action is None:
            print(f"Unknown command: {command}")
            return {"known": False, "changed": False, "command": normalized}

        changed = action()
        return {"known": True, "changed": changed, "command": normalized}
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self.initialized:
            self.stop()
            GPIO.cleanup()
            print("GPIO cleaned up")


class CommandDispatcher:
    """Single-threaded command queue feeding the motor controller."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        motor_controller: MotorController,
        event_callback,
        manual_timeout: float,
        joystick_timeout: float,
        joystick_dead_zone: float,
        joystick_stop_grace: int = JOYSTICK_STOP_GRACE_UPDATES,
        joystick_stop_confirmation: int = JOYSTICK_STOP_CONFIRMATION,
        joystick_stop_burst: int = JOYSTICK_STOP_BURST_COUNT,
    ) -> None:
        self.loop = loop
        self.motor_controller = motor_controller
        self.event_callback = event_callback
        self.manual_timeout = max(0.0, manual_timeout)
        self.joystick_timeout = max(0.0, joystick_timeout)
        self.joystick_dead_zone = max(0.0, min(1.0, joystick_dead_zone))
        self.joystick_stop_grace = max(0, int(joystick_stop_grace))
        self.joystick_stop_confirmation = max(1, int(joystick_stop_confirmation))
        self.joystick_stop_burst = max(1, int(joystick_stop_burst))

        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = loop.create_task(self._worker())
        self._manual_timer = None
        self._joystick_timer = None
        self._joystick_idle_frames = 0
        self._last_joystick_command = "stop"
        self._last_joystick_vector = {"x": 0.0, "y": 0.0, "magnitude": 0.0}
        self._joystick_stop_streak = 0
        self._joystick_stop_burst_remaining = 0

    async def submit_command(self, *, source: str, mode: str, payload: dict):
        future = self.loop.create_future()
        await self._queue.put((source, mode, payload, future))
        return await future

    async def emergency_stop(self, *, source: str, reason: str = "emergency"):
        payload = {"command": "stop", "reason": reason}
        return await self.submit_command(source=source, mode="manual", payload=payload)

    async def shutdown(self):
        if self._manual_timer:
            self._manual_timer.cancel()
        if self._joystick_timer:
            self._joystick_timer.cancel()
        await self._queue.put((None, None, None, None))
        if self._worker_task:
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None

    async def _worker(self):
        while True:
            source, mode, payload, future = await self._queue.get()
            if source is None:
                if future is not None and not future.done():
                    future.set_result(None)
                break

            try:
                if mode == "manual":
                    event = self._handle_manual(source, payload)
                elif mode == "joystick":
                    event = self._handle_joystick(source, payload)
                else:
                    event = {
                        "type": "command_ack",
                        "state": "error",
                        "source": source,
                        "mode": mode,
                        "message": f"Unsupported mode '{mode}'",
                        "timestamp": time.time(),
                    }

                if future is not None and not future.done():
                    future.set_result(event)

                if event and self.event_callback is not None:
                    await self.event_callback(event)

            except Exception as exc:
                error_event = {
                    "type": "command_ack",
                    "state": "error",
                    "source": source,
                    "mode": mode,
                    "message": f"Command processing failed: {exc}",
                    "timestamp": time.time(),
                }
                if future is not None and not future.done():
                    future.set_exception(exc)
                if self.event_callback is not None:
                    await self.event_callback(error_event)
            finally:
                self._queue.task_done()

    def _handle_manual(self, source: str, payload: dict):
        command = str(payload.get("command", "")).strip().lower()
        result = {"known": False, "changed": False, "command": command}

        if not command:
            state = "error"
            message = "Missing manual command"
        else:
            result = self.motor_controller.execute_command(command)
            normalized = result.get("command", command)

            if not result.get("known", False):
                state = "error"
                message = f"Unknown command '{command}'"
            elif not result.get("changed", False):
                if normalized != "stop":
                    self._schedule_manual_timeout(source)
                else:
                    self._cancel_manual_timeout()
                    self._cancel_joystick_timeout()
                return None
            else:
                if normalized == "stop":
                    self._cancel_manual_timeout()
                    self._cancel_joystick_timeout()
                    self._last_joystick_command = "stop"
                    self._joystick_idle_frames = 0
                    state = "stopped"
                    message = "Motors stopped"
                else:
                    self._schedule_manual_timeout(source)
                    state = "moving"
                    message = f"Command '{normalized}' executed"
            command = normalized

        event = {
            "type": "command_ack",
            "state": state,
            "source": source,
            "mode": "manual",
            "command": command,
            "timestamp": time.time(),
            "success": result.get("known", False) and result.get("changed", False),
            "details": {k: v for k, v in payload.items() if k != "command"},
            "message": message,
        }
        return event

    def _handle_joystick(self, source: str, payload: dict):
        def _extract_axis(candidates, fallback):
            for key in candidates:
                if key in payload:
                    raw = payload.get(key)
                    try:
                        return float(raw), True
                    except (TypeError, ValueError):
                        return fallback, False
            return fallback, False

        last_vector = self._last_joystick_vector
        x, x_valid = _extract_axis(("x", "horizontal", "axis_x", "leftX", "lx"), last_vector["x"])
        y, y_valid = _extract_axis(("y", "vertical", "axis_y", "leftY", "ly"), last_vector["y"])

        has_force_stop_flag = "force_stop" in payload
        force_stop = bool(payload.get("force_stop", False))

        magnitude = None
        if "magnitude" in payload:
            try:
                magnitude = float(payload.get("magnitude"))
            except (TypeError, ValueError):
                magnitude = None
        if magnitude is None:
            magnitude = min(1.0, (x ** 2 + y ** 2) ** 0.5)

        magnitude = max(0.0, min(1.0, magnitude))
        self._last_joystick_vector = {"x": x, "y": y, "magnitude": magnitude}

        axis_keys_present = any(
            key in payload
            for key in ("x", "horizontal", "axis_x", "leftX", "lx", "y", "vertical", "axis_y", "leftY", "ly")
        )

        if not axis_keys_present and "magnitude" not in payload:
            # Ignore frames that only carry metadata and no positional update.
            return None

        if axis_keys_present and not (x_valid or y_valid):
            return {
                "type": "command_ack",
                "state": "error",
                "source": source,
                "mode": "joystick",
                "command": "stop",
                "timestamp": time.time(),
                "success": False,
                "message": "Invalid joystick payload",
            }

        if magnitude < self.joystick_dead_zone:
            if force_stop:
                self._joystick_idle_frames = 0
                self._joystick_stop_streak = 0
                self._joystick_stop_burst_remaining = 0
                changed = self.motor_controller.stop()
                self._last_joystick_command = "stop"
                self._cancel_joystick_timeout()
                state = "stopped"
                command = "stop"
                message = (
                    "Joystick released; motors stopped"
                    if changed
                    else "Joystick already stopped"
                )
            elif has_force_stop_flag:
                self._joystick_idle_frames = 0
                self._joystick_stop_streak = 0
                self._schedule_joystick_timeout()
                return None
            else:
                if self._last_joystick_command != "stop":
                    self._joystick_idle_frames += 1
                    if self._joystick_idle_frames <= self.joystick_stop_grace:
                        self._schedule_joystick_timeout()
                        self._joystick_stop_streak = 0
                        return None
                else:
                    self._joystick_idle_frames = 0

                self._joystick_stop_streak += 1
                self._joystick_stop_streak = min(self._joystick_stop_streak, self.joystick_stop_confirmation)
                self._schedule_joystick_timeout()

                if self._joystick_stop_streak < self.joystick_stop_confirmation:
                    return None

                emit_event = False
                changed = False
                if self._last_joystick_command != "stop":
                    changed = self.motor_controller.stop()
                    self._last_joystick_command = "stop"
                    self._cancel_joystick_timeout()
                    self._joystick_stop_burst_remaining = max(0, self.joystick_stop_burst - 1)
                    emit_event = changed or self._joystick_stop_burst_remaining > 0
                else:
                    if self._joystick_stop_burst_remaining > 0:
                        self._joystick_stop_burst_remaining -= 1
                        emit_event = True
                    else:
                        emit_event = False

                if not emit_event:
                    return None

                state = "stopped"
                command = "stop"
                message = (
                    "Joystick centered; motors stopped"
                    if self._joystick_stop_burst_remaining == 0
                    else "Joystick centered; confirming stop"
                )
        else:
            self._joystick_idle_frames = 0
            self._joystick_stop_streak = 0
            self._joystick_stop_burst_remaining = 0
            if abs(y) >= abs(x):
                command = "forward" if y >= 0 else "backward"
            else:
                command = "right" if x >= 0 else "left"

            result = self.motor_controller.execute_command(command)
            self._last_joystick_command = command
            self._schedule_joystick_timeout()
            if not result.get("known", False):
                state = "error"
                message = f"Unknown joystick direction '{command}'"
            elif not result.get("changed", False):
                self._schedule_joystick_timeout()
                return None
            else:
                command = result.get("command", command)
                state = "moving"
                message = f"Joystick command '{command}' executed"
                self._schedule_joystick_timeout()

        event = {
            "type": "command_ack",
            "state": state,
            "source": source,
            "mode": "joystick",
            "command": command,
            "timestamp": time.time(),
            "success": state != "error",
            "details": {
                "x": x,
                "y": y,
                "magnitude": magnitude,
            },
            "message": message,
        }
        if command == "stop":
            event["details"]["stop_streak"] = self._joystick_stop_streak
            if self.joystick_stop_burst > 1:
                event["details"]["stop_burst_remaining"] = self._joystick_stop_burst_remaining
        return event

    def _schedule_manual_timeout(self, source: str):
        if self.manual_timeout <= 0:
            return
        if source in {"bluetooth", "websocket", "voice"}:  # clients send explicit stop
            return
        if self._manual_timer:
            self._manual_timer.cancel()
        self._manual_timer = self.loop.call_later(
            self.manual_timeout,
            lambda: asyncio.create_task(
                self.submit_command(
                    source="safety",
                    mode="manual",
                    payload={"command": "stop", "reason": "manual_timeout"},
                )
            ),
        )

    def _schedule_joystick_timeout(self):
        if self.joystick_timeout <= 0:
            return
        if self._joystick_timer:
            self._joystick_timer.cancel()
        self._joystick_timer = self.loop.call_later(
            self.joystick_timeout,
            lambda: asyncio.create_task(
                self.submit_command(
                    source="safety",
                    mode="manual",
                    payload={"command": "stop", "reason": "joystick_timeout"},
                )
            ),
        )

    def _cancel_manual_timeout(self):
        if self._manual_timer:
            self._manual_timer.cancel()
            self._manual_timer = None

    def _cancel_joystick_timeout(self):
        if self._joystick_timer:
            self._joystick_timer.cancel()
            self._joystick_timer = None

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

class AudioProcessor:
    """Processes audio chunks and assembles them into complete audio files."""
    
    def __init__(self):
        self.audio_chunks = []
        self.session_id = None
    
    def start_session(self, session_id: str):
        """Start a new audio recording session."""
        self.session_id = session_id
        self.audio_chunks = []
        print(f"Started audio session: {session_id}")
    
    def add_chunk(self, chunk_data: bytes):
        """Add an audio chunk to the current session."""
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_array = np.frombuffer(chunk_data, dtype=np.int16)
        self.audio_chunks.append(audio_array)
    
    def finalize_session(self) -> Optional[Path]:
        """Finalize the session and save the complete audio file."""
        if not self.audio_chunks:
            print("No audio chunks received")
            return None
        
        # Concatenate all chunks
        complete_audio = np.concatenate(self.audio_chunks)
        
        # Save to WAV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = TEMP_DIR / f"voice_cmd_{timestamp}_{self.session_id}.wav"
        wav.write(str(audio_path), SAMPLE_RATE, complete_audio)
        
        print(f"Saved audio file: {audio_path} ({len(complete_audio)} samples)")
        
        # Reset session
        self.audio_chunks = []
        self.session_id = None
        
        return audio_path
    
    def clear_session(self):
        """Clear the current session without saving."""
        self.audio_chunks = []
        self.session_id = None

# =============================================================================
# WEBSOCKET SERVER
# =============================================================================

class WheelchairWebSocketServer:
    """WebSocket server for wheelchair control."""
    
    def __init__(self):
        self.motor_controller = MotorController()
        self.audio_processor = AudioProcessor()
        self.connected_clients = set()
        self.voice_encoder = None
        self.voice_embeddings_cache = {}
        self.voice_embeddings_mtime = {}
        self._voice_lock = asyncio.Lock()
        self.enrollment_progress = {}
        self._warm_stt_pipeline()
        self.dispatcher: Optional[CommandDispatcher] = None
        self.bluetooth_controller: Optional[BluetoothController] = None
        self.camera_clients = {}
        self._last_command_signature = {}

    async def _broadcast_event(self, event: dict):
        if not event:
            return

        if event.get("type") == "command_ack":
            mode_key = event.get("mode", "*")
            signature = (event.get("command"), event.get("state"), bool(event.get("success")))
            if self._last_command_signature.get(mode_key) == signature:
                return
            self._last_command_signature[mode_key] = signature

        if self.connected_clients:
            message = json.dumps(event)
            disconnected = []
            for ws in list(self.connected_clients):
                try:
                    await ws.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(ws)
                except Exception as exc:
                    print(f"WebSocket broadcast error: {exc}")
                    disconnected.append(ws)
            for ws in disconnected:
                self.connected_clients.discard(ws)

        if self.bluetooth_controller is not None:
            self.bluetooth_controller.publish_event(event)

    async def _broadcast_camera_frame(self, payload: dict, *, exclude=None):
        if not payload:
            return

        exclude = exclude or set()
        if self.connected_clients:
            message = json.dumps(payload)
            disconnected = []
            for ws in list(self.connected_clients):
                if ws in exclude:
                    continue
                try:
                    await ws.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(ws)
                except Exception as exc:
                    print(f"WebSocket broadcast error: {exc}")
                    disconnected.append(ws)
            for ws in disconnected:
                self.connected_clients.discard(ws)
                self.camera_clients.pop(ws, None)

    def _warm_stt_pipeline(self):
        """Warm Whisper tiny pipeline so the first command is instant."""
        try:
            load_local_stt_pipeline()
            print("STT pipeline loaded and ready for commands")
        except Exception as exc:
            print(f"Warning: STT warm-up failed: {exc}")

    def _get_voice_encoder(self):
        """Lazily initialize and return the shared speaker recognizer."""
        if self.voice_encoder is None:
            try:
                self.voice_encoder = load_speaker_recognizer()
                print("Speaker recognizer loaded for speaker identification")
            except Exception as exc:
                print(f"Failed to load speaker recognizer: {exc}")
                self.voice_encoder = None
        return self.voice_encoder

    def _load_voice_embedding(self, name: str, path: Path):
        """Load a stored voice embedding with simple caching."""
        mtime = path.stat().st_mtime
        cached = self.voice_embeddings_cache.get(name)
        cached_mtime = self.voice_embeddings_mtime.get(name)
        if cached is not None and cached_mtime == mtime:
            return cached
        embedding = np.load(path)
        self.voice_embeddings_cache[name] = embedding
        self.voice_embeddings_mtime[name] = mtime
        return embedding

    def _load_voice_embedding_bank(self, name: str):
        """Load optional multi-sample embedding and partial banks if present."""
        bank_path = VOICE_DB_EMBEDDINGS_DIR / f"{name}_samples.npz"
        if not bank_path.exists():
            return None, None
        try:
            with np.load(bank_path) as data:
                embeddings = data.get("embeddings")
                partials = data.get("partials")
                emb_stack = embeddings.astype(np.float32) if embeddings is not None and embeddings.size else None
                part_stack = partials.astype(np.float32) if partials is not None and partials.size else None
                return emb_stack, part_stack
        except Exception as exc:
            print(f"Failed to load embedding bank for {name}: {exc}")
            return None, None

    def _prune_voice_cache(self, active_profiles):
        """Discard cache entries for profiles that were removed from disk."""
        stale = set(self.voice_embeddings_cache.keys()) - set(active_profiles)
        for name in stale:
            self.voice_embeddings_cache.pop(name, None)
            self.voice_embeddings_mtime.pop(name, None)

    async def identify_speaker_from_audio(self, audio_path: Path):
        """Identify the speaker from a recorded audio sample."""
        profile_paths = list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))
        if not profile_paths:
            return None, 0.0, False

        grouped_profiles = {}
        for emb_path in profile_paths:
            base_name = re.sub(r"_(\d+)$", "", emb_path.stem)
            grouped_profiles.setdefault(base_name, []).append(emb_path)

        if not grouped_profiles:
            return None, 0.0, False

        profile_genders = {}
        for base_name in grouped_profiles:
            gender_file = VOICE_DB_PROCESSED_DIR / f"{base_name}_gender.txt"
            gender = "unknown"
            if gender_file.exists():
                try:
                    gender = gender_file.read_text().strip().lower()
                except Exception:
                    gender = "unknown"
            profile_genders[base_name] = gender

        try:
            processed_audio = preprocess_audio_for_verification(str(audio_path))
        except Exception as exc:
            print(f"Speaker preprocessing failed: {exc}")
            return None, 0.0, False

        if processed_audio is None or len(processed_audio) == 0:
            print("Speaker preprocessing produced empty audio")
            return None, 0.0, False

        async with self._voice_lock:
            encoder = self._get_voice_encoder()
            if encoder is None:
                return None, 0.0, False

            try:
                probe_embedding, probe_partials = compute_enhanced_embedding(encoder, processed_audio)
            except Exception as exc:
                print(f"Failed to generate voice embedding: {exc}")
                return None, 0.0, False

            if probe_embedding is None:
                print("Speaker embedding unavailable (empty audio)")
                return None, 0.0, False

            profile_stats = []
            active_variants = []

            for base_name, variant_paths in grouped_profiles.items():
                variant_details = []
                candidate_scores = []

                for variant_path in variant_paths:
                    variant_name = variant_path.stem
                    try:
                        stored_emb = self._load_voice_embedding(variant_name, variant_path)
                    except Exception as exc:
                        print(f"Failed to load embedding for {variant_name}: {exc}")
                        continue

                    if stored_emb.ndim > 1:
                        stored_vector = stored_emb.reshape(-1, stored_emb.shape[-1]).mean(axis=0)
                    else:
                        stored_vector = stored_emb

                    if stored_vector.shape != probe_embedding.shape:
                        print(
                            f"Embedding dimension mismatch for profile {variant_name}. "
                            "Please re-enroll this voice with the new recognizer."
                        )
                        continue

                    sim = cosine_similarity(probe_embedding, stored_vector)
                    variant_details.append((variant_name, float(sim)))
                    candidate_scores.append(float(sim))
                    active_variants.append(variant_name)

                if not candidate_scores:
                    continue

                bank_embeddings, bank_partials = self._load_voice_embedding_bank(base_name)
                if bank_embeddings is not None and bank_embeddings.size:
                    if bank_embeddings.shape[-1] != probe_embedding.shape[0]:
                        print(
                            f"Sample bank dimension mismatch for {base_name}; "
                            "please re-enroll to regenerate embeddings."
                        )
                        bank_embeddings = None
                    else:
                        bank_embeddings = bank_embeddings.reshape(-1, probe_embedding.shape[0])
                        bank_norms = np.linalg.norm(bank_embeddings, axis=1, keepdims=True) + 1e-8
                        bank_normed = bank_embeddings / bank_norms
                        sims = bank_normed @ probe_embedding
                        if sims.size:
                            best_sample = float(np.max(sims))
                            top_k = min(3, sims.size)
                            avg_top = float(np.mean(np.sort(sims)[-top_k:]))
                            fused_score = (0.55 * best_sample) + (0.45 * avg_top)
                            variant_details.append((f"{base_name}_bank_top", best_sample))
                            variant_details.append((f"{base_name}_bank_avg", avg_top))
                            variant_details.append((f"{base_name}_bank_fused", fused_score))
                            candidate_scores.extend([best_sample, avg_top, fused_score])

                if (
                    bank_partials is not None
                    and probe_partials is not None
                    and bank_partials.size
                    and probe_partials.size
                ):
                    if bank_partials.shape[-1] != probe_partials.shape[1]:
                        print(
                            f"Partial embedding dimension mismatch for {base_name}; "
                            "re-enrollment recommended."
                        )
                    else:
                        bank_partials = bank_partials.reshape(-1, probe_partials.shape[1])
                        bank_partial_norms = np.linalg.norm(bank_partials, axis=1, keepdims=True) + 1e-8
                        probe_partial_norms = np.linalg.norm(probe_partials, axis=1, keepdims=True) + 1e-8
                        bank_partial_normed = bank_partials / bank_partial_norms
                        probe_partial_normed = probe_partials / probe_partial_norms
                        partial_sims = bank_partial_normed @ probe_partial_normed.T
                        if partial_sims.size:
                            best_partial = float(np.max(partial_sims))
                            flat = np.sort(partial_sims.reshape(-1))
                            top_n = min(6, flat.size)
                            avg_partial = float(np.mean(flat[-top_n:]))
                            fused_partial = (0.6 * best_partial) + (0.4 * avg_partial)
                            variant_details.append((f"{base_name}_partial_top", best_partial))
                            variant_details.append((f"{base_name}_partial_avg", avg_partial))
                            variant_details.append((f"{base_name}_partial_fused", fused_partial))
                            candidate_scores.extend([best_partial, avg_partial, fused_partial])

                candidate_scores = [float(score) for score in candidate_scores if np.isfinite(score)]
                if not candidate_scores:
                    continue

                candidate_scores.sort(reverse=True)
                best_raw = candidate_scores[0]
                avg_raw = float(sum(candidate_scores) / len(candidate_scores))
                support_count = sum(
                    1 for score in candidate_scores
                    if (best_raw - score) <= SPEAKER_VARIANT_SUPPORT_WINDOW
                )
                bonus = 0.0
                if support_count > 1:
                    bonus = min(
                        SPEAKER_VARIANT_SUPPORT_MAX,
                        (support_count - 1) * SPEAKER_VARIANT_SUPPORT_STEP,
                    )
                avg_contrib = (avg_raw * SPEAKER_SCORE_AVG_WEIGHT) + SPEAKER_SCORE_AVG_BIAS
                gain_path = (best_raw * SPEAKER_SCORE_GAIN) + SPEAKER_SCORE_OFFSET
                avg_path = min(0.999, (best_raw * 0.55) + avg_contrib)
                boost_candidates = [best_raw, gain_path, avg_path]
                boosted_core = max(boost_candidates)
                applied_gain = max(0.0, boosted_core - best_raw)
                final_score = float(min(0.999, boosted_core + bonus))

                top_variants = ", ".join(
                    f"{label}:{score:.3f}"
                    for label, score in sorted(variant_details, key=lambda item: item[1], reverse=True)[:3]
                )

                profile_stats.append(
                    {
                        "name": base_name,
                        "score": final_score,
                        "best_raw": best_raw,
                        "avg_raw": avg_raw,
                        "support_count": support_count,
                        "bonus": bonus,
                        "boost": applied_gain,
                        "variants": variant_details,
                        "top_summary": top_variants,
                        "gender": profile_genders.get(base_name, "unknown"),
                        "best_variant": max(variant_details, key=lambda item: item[1])[0],
                    }
                )

            self._prune_voice_cache(active_variants)

        if not profile_stats:
            print("No valid voice profiles found.")
            return None, 0.0, False

        profile_stats.sort(key=lambda entry: entry["score"], reverse=True)

        print(f"Comparing against {len(profile_stats)} enrolled speaker(s)...")
        for entry in profile_stats:
            note_parts = []
            if entry["bonus"] > 0:
                note_parts.append(f"+{entry['bonus']:.3f} bonus")
            if entry["boost"] > 1e-4:
                note_parts.append(f"+{entry['boost']:.3f} gain")
            detail_suffix = f" ({', '.join(note_parts)})" if note_parts else ""
            print(
                f"  - {entry['name']}: best {entry['best_raw']:.3f} via {entry['best_variant']}{detail_suffix}; "
                f"final={entry['score']:.3f}; avg={entry['avg_raw']:.3f}; variants[{len(entry['variants'])}]={entry['top_summary']} "
                f"[Gender: {entry['gender']}]"
            )

        best_entry = profile_stats[0]
        enrolled_count = len(profile_stats)
        effective_threshold = SIMILARITY_THRESHOLD
        if enrolled_count == 1:
            relaxed = max(0.40, SIMILARITY_THRESHOLD - SPEAKER_SOLO_THRESHOLD_RELAX)
            if relaxed < effective_threshold:
                effective_threshold = relaxed
                print(
                    f"Adjusting verification threshold to {effective_threshold:.3f} "
                    "(single enrolled speaker)"
                )
        elif best_entry["support_count"] >= 3:
            relaxed = max(0.45, SIMILARITY_THRESHOLD - (SPEAKER_SOLO_THRESHOLD_RELAX * 0.5))
            if relaxed < effective_threshold:
                effective_threshold = relaxed
                print(
                    f"Adjusting verification threshold to {effective_threshold:.3f} "
                    "(strong multi-segment agreement)"
                )

        confusion_warning = ""
        if len(profile_stats) > 1:
            runner = profile_stats[1]
            score_diff = best_entry["score"] - runner["score"]
            if score_diff < SPEAKER_MIN_SCORE_GAP:
                confusion_warning = (
                    f" (Warning: Close match with {runner['name']}: "
                    f"{runner['score']:.3f}, diff: {score_diff:.3f})"
                )

        print(
            f"\nBest match: {best_entry['name']} "
            f"(similarity {best_entry['score']:.3f}, gender: {best_entry['gender']})"
            f"{confusion_warning}"
        )

        if best_entry["score"] >= effective_threshold:
            if len(profile_stats) > 1 and (
                best_entry["score"] - profile_stats[1]["score"]
            ) < SPEAKER_MIN_SCORE_GAP:
                print(
                    "Authentication rejected: score gap "
                    f"{(best_entry['score'] - profile_stats[1]['score']):.3f} < "
                    f"{SPEAKER_MIN_SCORE_GAP:.3f}"
                )
                return None, best_entry["score"], False
            print(
                f"Authentication successful! ({best_entry['score']:.3f} >= {effective_threshold:.3f})"
            )
            return best_entry["name"], best_entry["score"], True

        print(
            f"Authentication failed. Score {best_entry['score']:.3f} "
            f"below threshold {effective_threshold:.3f}"
        )
        return None, best_entry["score"], False
    
    async def handle_client(self, websocket):
        """Handle a client connection."""
        remote = websocket.remote_address or ("unknown", "?")
        client_id = f"{remote[0]}:{remote[1]}"
        self.connected_clients.add(websocket)
        print(f"\n✓ Client connected: {client_id}")
        print(f"  Total clients: {len(self.connected_clients)}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "message": "Wheelchair control server ready"
            }))
            
            async for message in websocket:
                try:
                    # Check if message is JSON (control message) or binary (audio data)
                    if isinstance(message, str):
                        await self.handle_control_message(websocket, message)
                    elif isinstance(message, bytes):
                        await self.handle_audio_chunk(websocket, message)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed as exc:
            print(f"✗ Client disconnected: {client_id} ({exc.code} - {exc.reason})")
        finally:
            self.connected_clients.discard(websocket)
            stream_name = self.camera_clients.pop(websocket, None)
            if stream_name:
                await self._broadcast_event({
                    "type": "camera_stream_status",
                    "stream": stream_name,
                    "state": "offline",
                    "timestamp": time.time(),
                })
            print(f"  Total clients: {len(self.connected_clients)}")
    
    async def handle_control_message(self, websocket, message: str):
        """Handle control messages (JSON)."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'start_recording':
                # Start a new recording session
                session_id = data.get('session_id', str(time.time()))
                self.audio_processor.start_session(session_id)
                
                await websocket.send(json.dumps({
                    "type": "recording_started",
                    "session_id": session_id
                }))
            
            elif msg_type == 'stop_recording':
                # Finalize recording and process the command
                await self.process_voice_command(websocket)
            
            elif msg_type == 'start_voice_enrollment':
                # Start voice enrollment session
                speaker_name = data.get('speaker_name', 'user')
                gender = data.get('gender', 'unknown')
                total_samples = max(1, int(data.get('total_samples', 3)))
                sample_index = max(1, int(data.get('sample_index', 1)))
                session_id = data.get('session_id', str(time.time()))
                self.audio_processor.start_session(session_id)

                normalized_name = speaker_name.strip().lower()
                prompt_text = data.get('prompt')

                progress = self.enrollment_progress.setdefault(
                    normalized_name,
                    {
                        "gender": gender,
                        "total": total_samples,
                        "embeddings": [],
                        "raw_paths": [],
                        "processed_paths": [],
                        "received": 0,
                        "created": time.time(),
                    },
                )

                progress["gender"] = gender
                progress["total"] = max(progress.get("total", 1), total_samples)
                
                next_prompt = prompt_text or ENROLLMENT_PHRASES[
                    progress["received"] % len(ENROLLMENT_PHRASES)
                ]

                current_index = progress["received"] + 1

                await websocket.send(json.dumps({
                    "type": "enrollment_started",
                    "session_id": session_id,
                    "speaker_name": normalized_name,
                    "total_samples": total_samples,
                    "sample_index": current_index,
                    "recommended_prompt": next_prompt,
                }))
            
            elif msg_type == 'stop_voice_enrollment':
                # Process voice enrollment
                speaker_name = data.get('speaker_name', 'user')
                gender = data.get('gender', 'unknown')
                sample_index = max(1, int(data.get('sample_index', 1)))
                total_samples = max(1, int(data.get('total_samples', 3)))
                finalize_now = bool(data.get('finalize', False))
                await self.process_voice_enrollment(
                    websocket,
                    speaker_name,
                    gender,
                    sample_index=sample_index,
                    total_samples=total_samples,
                    force_finalize=finalize_now,
                )

            elif msg_type == 'finish_voice_enrollment':
                speaker_name = data.get('speaker_name', 'user')
                await self.finalize_enrollment(websocket, speaker_name)
            
            elif msg_type == 'cancel_recording':
                # Cancel the current recording
                self.audio_processor.clear_session()
                await websocket.send(json.dumps({
                    "type": "recording_cancelled"
                }))
            
            elif msg_type == 'emergency_stop':
                # Emergency stop
                if self.dispatcher is not None:
                    await self.dispatcher.emergency_stop(
                        source="websocket", reason="remote_emergency"
                    )
                else:
                    self.motor_controller.stop()
                await websocket.send(json.dumps({
                    "type": "emergency_stop_executed"
                }))
            
            elif msg_type == 'ping':
                # Respond to ping
                await websocket.send(json.dumps({
                    "type": "pong"
                }))
            
            elif msg_type == 'check_voice_profile':
                speaker_name = data.get('speaker_name')
                await self.send_voice_profile_status(websocket, speaker_name)

            elif msg_type == 'camera_register':
                stream_name = data.get('stream', 'camera')
                self.camera_clients[websocket] = stream_name
                response = {
                    "type": "camera_registered",
                    "stream": stream_name,
                    "message": "Camera stream registered",
                }
                width = data.get('width')
                height = data.get('height')
                if width is not None and height is not None:
                    response["resolution"] = {"width": width, "height": height}
                await websocket.send(json.dumps(response))
                await self._broadcast_event({
                    "type": "camera_stream_status",
                    "stream": stream_name,
                    "state": "online",
                    "timestamp": time.time(),
                })

            elif msg_type == 'camera_unregister':
                stream_name = self.camera_clients.pop(websocket, None) or data.get('stream')
                await websocket.send(json.dumps({
                    "type": "camera_unregistered",
                    "stream": stream_name,
                }))
                if stream_name:
                    await self._broadcast_event({
                        "type": "camera_stream_status",
                        "stream": stream_name,
                        "state": "offline",
                        "timestamp": time.time(),
                    })

            elif msg_type == 'camera_frame':
                frame_data = data.get('data')
                if not isinstance(frame_data, str):
                    return
                stream_name = self.camera_clients.get(websocket) or data.get('stream', 'camera')
                payload = {
                    "type": "camera_frame",
                    "stream": stream_name,
                    "format": data.get('format', 'jpeg'),
                    "timestamp": data.get('timestamp', time.time()),
                    "data": frame_data,
                }
                await self._broadcast_camera_frame(payload, exclude={websocket})

            elif msg_type == 'manual_control':
                mode = data.get('mode', 'manual')
                if self.dispatcher is None:
                    raise RuntimeError("Command dispatcher not ready")

                if mode == 'joystick':
                    event = await self.dispatcher.submit_command(
                        source="websocket", mode="joystick", payload=data
                    )
                else:
                    event = await self.dispatcher.submit_command(
                        source="websocket", mode="manual", payload=data
                    )

                if event:
                    response = {"type": "manual_control_ack"}
                    response.update(event)
                    await websocket.send(json.dumps(response))
            
            else:
                print(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
    
    async def handle_audio_chunk(self, websocket, chunk_data: bytes):
        """Handle binary audio chunk."""
        self.audio_processor.add_chunk(chunk_data)
        # Optional: send acknowledgment for each chunk
        # await websocket.send(json.dumps({"type": "chunk_received"}))
    
    async def send_voice_profile_status(self, websocket, speaker_name: Optional[str]):
        """Send voice profile availability information to the client."""
        normalized = (speaker_name or "").strip().lower()
        profiles = sorted(
            [path.stem for path in VOICE_DB_EMBEDDINGS_DIR.glob("*.npy")]
        )
        if normalized:
            exists = (VOICE_DB_EMBEDDINGS_DIR / f"{normalized}.npy").exists()
        else:
            exists = len(profiles) > 0
        await websocket.send(json.dumps({
            "type": "voice_profile_status",
            "speaker_name": normalized,
            "exists": bool(exists),
            "available_profiles": profiles,
        }))
    
    async def process_voice_command(self, websocket):
        """Process the recorded voice command."""
        print("\n" + "="*50)
        print("Processing voice command...")
        print("="*50)
        
        # Send processing status
        await websocket.send(json.dumps({
            "type": "processing",
            "message": "Processing your command..."
        }))
        
        # Finalize audio and get the file path
        audio_path = self.audio_processor.finalize_session()
        
        if audio_path is None:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "No audio data received"
            }))
            return
        
        try:
            # Process the command using wheelchair_control
            command, confidence, transcription = process_command_with_whisper_tiny(
                audio_path=str(audio_path),
                detect_lang=True,
                fast_mode=True,
            )

            if SPEAKER_VERIFICATION_ENABLED:
                speaker_name, speaker_score, speaker_verified = await self.identify_speaker_from_audio(audio_path)

                if speaker_name:
                    print(
                        f"Speaker match: {speaker_name} (score: {speaker_score:.2f}, verified: {speaker_verified})"
                    )
                else:
                    print(f"Speaker match: unknown (score: {speaker_score:.2f})")
            else:
                speaker_name = None
                speaker_score = 1.0
                speaker_verified = True
                print("Speaker verification disabled; executing command without identity check")
            
            if transcription:
                print(f"Transcription: {transcription}")

            if command:
                print(f"\n✓ Command recognized: '{command}' (confidence: {confidence:.2f})")

                if SPEAKER_VERIFICATION_ENABLED and not speaker_verified:
                    print(
                        "Speaker verification below threshold "
                        f"({speaker_score:.2f} < {SIMILARITY_THRESHOLD:.2f}) — blocking motors."
                    )
                    await websocket.send(json.dumps({
                        "type": "command_recognized",
                        "command": command,
                        "transcription": transcription,
                        "confidence": float(confidence),
                        "executed": False,
                        "message": (
                            "Authentication failed. Command blocked until an "
                            "authorized voice is detected."
                        ),
                        "speaker": {
                            "name": speaker_name,
                            "score": float(speaker_score),
                            "verified": bool(speaker_verified),
                            "required_threshold": float(SIMILARITY_THRESHOLD),
                        },
                    }))
                    return

                if self.dispatcher is not None:
                    event = await self.dispatcher.submit_command(
                        source="voice",
                        mode="manual",
                        payload={
                            "command": command,
                            "confidence": float(confidence),
                            "transcription": transcription,
                            "speaker": speaker_name,
                        },
                    )
                    executed = event.get("state") in {"moving", "stopped"}
                    message = event.get("message")
                else:
                    result = self.motor_controller.execute_command(command)
                    normalized = result.get("command", command)
                    executed = result.get("known", False) and result.get("changed", False)
                    if not result.get("known", False):
                        message = f"Unknown command '{command}'"
                    elif executed:
                        message = f"Command '{normalized}' executed"
                    else:
                        message = f"Command '{normalized}' already active"

                # Send feedback to the app
                speaker_payload = {
                    "name": speaker_name,
                    "score": float(speaker_score),
                    "verified": bool(speaker_verified),
                }
                if not SPEAKER_VERIFICATION_ENABLED:
                    speaker_payload["verification"] = "disabled"

                await websocket.send(json.dumps({
                    "type": "command_recognized",
                    "command": command,
                    "transcription": transcription,
                    "confidence": float(confidence),
                    "executed": bool(executed),
                    "message": message,
                    "speaker": speaker_payload,
                }))

            else:
                print(f"\n✗ Command not recognized (confidence: {confidence:.2f})")
                if SPEAKER_VERIFICATION_ENABLED:
                    print(
                        "Speaker match during failure: "
                        f"{speaker_name or 'unknown'} (score: {speaker_score:.2f}, verified: {speaker_verified})"
                    )

                failure_speaker_payload = {
                    "name": speaker_name,
                    "score": float(speaker_score),
                    "verified": bool(speaker_verified),
                }
                if not SPEAKER_VERIFICATION_ENABLED:
                    failure_speaker_payload["verification"] = "disabled"

                await websocket.send(json.dumps({
                    "type": "command_not_recognized",
                    "transcription": transcription,
                    "confidence": float(confidence),
                    "message": "Could not recognize the command. Please try again.",
                    "speaker": failure_speaker_payload,
                }))
        
        except Exception as e:
            print(f"\n✗ Error processing command: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Processing error: {str(e)}"
            }))
        
        finally:
            # Cleanup: delete the audio file
            try:
                audio_path.unlink()
            except:
                pass
        
        print("="*50 + "\n")
    
    async def process_voice_enrollment(
        self,
        websocket,
        speaker_name: str,
        gender: str,
        *,
        sample_index: int = 1,
        total_samples: int = 3,
        force_finalize: bool = False,
    ):
        """Process voice enrollment from audio chunks with multi-sample support."""
        normalized_name = speaker_name.strip().lower()
        print("\n" + "=" * 50)
        print(
            f"Processing voice enrollment for: {normalized_name} "
            f"(sample {sample_index}/{total_samples})"
        )
        print("=" * 50)

        await websocket.send(json.dumps({
            "type": "enrollment_processing",
            "message": "Processing voice enrollment..."
        }))

        audio_path = self.audio_processor.finalize_session()
        if audio_path is None:
            await websocket.send(json.dumps({
                "type": "enrollment_error",
                "message": "No audio data received"
            }))
            return

        progress = self.enrollment_progress.setdefault(
            normalized_name,
            {
                "gender": gender,
                "total": max(1, int(total_samples)),
                "embeddings": [],
                "partials": [],
                "raw_paths": [],
                "processed_paths": [],
                "received": 0,
                "created": time.time(),
            },
        )

        progress["gender"] = gender
        progress["total"] = max(progress.get("total", 3), int(total_samples))
        progress.setdefault("partials", [])

        try:
            encoder = self._get_voice_encoder()
            if encoder is None:
                raise RuntimeError("Speaker recognizer unavailable")

            sample_number = progress["received"] + 1

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            raw_path = VOICE_DB_PROCESSED_DIR / f"{normalized_name}_raw_{sample_number}_{timestamp}.wav"
            processed_path = VOICE_DB_PROCESSED_DIR / f"{normalized_name}_processed_{sample_number}_{timestamp}.wav"

            import shutil
            shutil.copy(audio_path, raw_path)

            raw_wav = preprocess_audio_for_verification(str(raw_path))
            wav.write(
                processed_path,
                SAMPLE_RATE,
                (np.clip(raw_wav, -1.0, 1.0) * 32767).astype(np.int16),
            )

            embedding, partials = compute_enhanced_embedding(encoder, raw_wav)
            if embedding is None:
                raise RuntimeError("Failed to compute embedding for enrollment sample")

            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            if partials is not None and partials.size:
                partials = partials.astype(np.float32)
            else:
                partials = np.empty((0, embedding.shape[0]), dtype=np.float32)

            progress["embeddings"].append(embedding.astype(np.float32))
            progress["partials"].append(partials)
            progress["raw_paths"].append(raw_path)
            progress["processed_paths"].append(processed_path)
            progress["received"] = sample_number

            remaining = max(0, progress["total"] - progress["received"])
            should_finalize = force_finalize or remaining == 0

            if should_finalize:
                await self.finalize_enrollment(websocket, normalized_name)
            else:
                next_prompt = ENROLLMENT_PHRASES[
                    progress["received"] % len(ENROLLMENT_PHRASES)
                ]
                await websocket.send(json.dumps({
                    "type": "enrollment_sample_received",
                    "speaker_name": normalized_name,
                    "samples_recorded": progress["received"],
                    "total_samples": progress["total"],
                    "recommended_prompt": next_prompt,
                    "message": (
                        f"Sample {progress['received']} saved. "
                        f"{remaining} more recommended for best results."
                    ),
                }))

        except Exception as exc:
            print(f"✗ Error processing enrollment sample: {exc}")
            await websocket.send(json.dumps({
                "type": "enrollment_error",
                "message": f"Error processing enrollment sample: {exc}",
            }))
        finally:
            try:
                audio_path.unlink()
            except Exception:
                pass

        print("=" * 50 + "\n")

    async def finalize_enrollment(self, websocket, speaker_name: str):
        normalized_name = speaker_name.strip().lower()
        progress = self.enrollment_progress.get(normalized_name)

        if not progress or not progress.get("embeddings"):
            await websocket.send(json.dumps({
                "type": "enrollment_error",
                "message": "No enrollment samples captured yet."
            }))
            return

        try:
            embeddings = np.stack(progress["embeddings"])
            if embeddings.shape[0] < progress.get("total", embeddings.shape[0]):
                print(
                    f"Finalizing voice profile for {normalized_name} with "
                    f"{embeddings.shape[0]} sample(s) (recommended {progress.get('total')})."
                )
            mean_embedding = embeddings.mean(axis=0)
            mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

            partials_list = [p for p in progress.get("partials", []) if p is not None and p.size]
            if partials_list:
                partial_bank = np.concatenate(partials_list, axis=0)
            else:
                partial_bank = np.empty((0, embeddings.shape[1]), dtype=np.float32)

            embedding_path = VOICE_DB_EMBEDDINGS_DIR / f"{normalized_name}.npy"
            np.save(embedding_path, mean_embedding.astype(np.float32))

            sample_bank_path = VOICE_DB_EMBEDDINGS_DIR / f"{normalized_name}_samples.npz"
            np.savez(
                sample_bank_path,
                embeddings=embeddings.astype(np.float32),
                partials=partial_bank.astype(np.float32),
            )

            # Refresh in-memory caches to use the latest embedding immediately
            self.voice_embeddings_cache[normalized_name] = mean_embedding.astype(np.float32)
            self.voice_embeddings_mtime[normalized_name] = embedding_path.stat().st_mtime

            with open(VOICE_DB_PROCESSED_DIR / f"{normalized_name}_gender.txt", "w") as fh:
                fh.write(progress.get("gender", "unknown"))

            print(
                f"✓ Voice profile created for {normalized_name} "
                f"using {len(progress['embeddings'])} samples"
            )

            await websocket.send(json.dumps({
                "type": "enrollment_success",
                "speaker_name": normalized_name,
                "samples_recorded": len(progress["embeddings"]),
                "total_samples": progress.get("total", len(progress["embeddings"])),
                "message": (
                    f"Voice profile created successfully with "
                    f"{len(progress['embeddings'])} samples"
                ),
            }))

            await self.send_voice_profile_status(websocket, normalized_name)

        except Exception as exc:
            print(f"✗ Error finalizing voice profile: {exc}")
            await websocket.send(json.dumps({
                "type": "enrollment_error",
                "message": f"Error finalizing voice profile: {exc}",
            }))
            return
        finally:
            # keep stored wav files but clear session state
            self.enrollment_progress.pop(normalized_name, None)
            self.audio_processor.clear_session()

        print("=" * 50 + "\n")
    
    async def shutdown(self):
        if self.dispatcher is not None:
            await self.dispatcher.shutdown()
            self.dispatcher = None

    def cleanup(self):
        """Cleanup resources."""
        if self.bluetooth_controller is not None:
            self.bluetooth_controller.stop()
            self.bluetooth_controller = None
        self.motor_controller.cleanup()
    
    async def start(self):
        """Start the WebSocket server."""
        print("\n" + "="*60)
        print("🦽 SMART WHEELCHAIR CONTROL SERVER")
        print("="*60)
        print(f"WebSocket server starting on port {WEBSOCKET_PORT}...")

        loop = asyncio.get_running_loop()
        self.dispatcher = CommandDispatcher(
            loop=loop,
            motor_controller=self.motor_controller,
            event_callback=self._broadcast_event,
            manual_timeout=COMMAND_TIMEOUT_SECONDS,
            joystick_timeout=JOYSTICK_TIMEOUT_SECONDS,
            joystick_dead_zone=JOYSTICK_DEAD_ZONE,
            joystick_stop_grace=JOYSTICK_STOP_GRACE_UPDATES,
            joystick_stop_confirmation=JOYSTICK_STOP_CONFIRMATION,
            joystick_stop_burst=JOYSTICK_STOP_BURST_COUNT,
        )

        if BT_ENABLED:
            if not BLUETOOTH_AVAILABLE:
                print("Warning: PyBluez not installed. Bluetooth transport disabled.")
            else:
                controller = BluetoothController(
                    loop=loop,
                    dispatcher=self.dispatcher,
                    event_callback=self._broadcast_event,
                    device_name=BT_DEVICE_NAME,
                    channel=BT_CHANNEL,
                    service_uuid=BT_SERVICE_UUID,
                )
                if controller.start():
                    self.bluetooth_controller = controller
                    print(
                        f"✓ Bluetooth RFCOMM listening as '{BT_DEVICE_NAME}' on channel {BT_CHANNEL}"
                    )
                else:
                    print("✗ Failed to start Bluetooth controller")
        else:
            print("Bluetooth transport disabled in configuration")
        
        async with websockets.serve(
            self.handle_client, 
            "0.0.0.0", 
            WEBSOCKET_PORT,
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=60    # Wait 60 seconds for pong
        ):
            print(f"✓ Server running on ws://0.0.0.0:{WEBSOCKET_PORT}")
            print(f"✓ Waiting for connections from the Flutter app...")
            print("="*60 + "\n")
            
            await asyncio.Future()  # Run forever

# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    server = WheelchairWebSocketServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    finally:
        await server.shutdown()
        server.cleanup()
        print("Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
