#!/usr/bin/env python3
"""
Smart Wheelchair WebSocket Server
Receives streaming audio from Flutter app, processes commands using wheelchair_control.py

⚠️  SAFETY WARNING ⚠️
By default, this server will NOT actuate motors. To enable motor control:
1. Set ENABLE_ACTUATION=true
2. Set ACTUATION_CONFIRM=true
Both must be set explicitly to enable physical wheelchair movement.

Author: GitHub Copilot
Date: November 22, 2025
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import wave
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Server configuration
HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8765"))
API_SECRET = os.getenv("API_SECRET", None)

# ⚠️  SAFETY: Motor actuation disabled by default
ENABLE_ACTUATION = os.getenv("ENABLE_ACTUATION", "false").lower() == "true"
ACTUATION_CONFIRM = os.getenv("ACTUATION_CONFIRM", "false").lower() == "true"
MOTORS_ENABLED = ENABLE_ACTUATION and ACTUATION_CONFIRM

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio
AUDIO_CHUNK_SIZE = 4096

# Paths (works in Docker and local)
TEMP_DIR = Path("/app/temp_ws_audio") if Path("/app").exists() else Path("./temp_ws_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Wheelchair control script paths (in Docker, everything is in /app)
WHEELCHAIR_WRAPPER_PATH = Path("/app/wheelchair_control_wrapper.py") if Path("/app/wheelchair_control_wrapper.py").exists() else Path("./wheelchair_control_wrapper.py")
WHEELCHAIR_CONTROL_PATH = Path("/app/wheelchair_control.py") if Path("/app/wheelchair_control.py").exists() else Path("./wheelchair_control.py")

# Session management
active_sessions: Dict[str, dict] = {}

# =============================================================================
# SAFETY CHECKS
# =============================================================================

def check_safety_configuration():
    """Display safety status on startup"""
    logger.info("=" * 70)
    logger.info("SAFETY CONFIGURATION STATUS")
    logger.info("=" * 70)
    logger.info(f"ENABLE_ACTUATION: {ENABLE_ACTUATION}")
    logger.info(f"ACTUATION_CONFIRM: {ACTUATION_CONFIRM}")
    logger.info(f"MOTORS_ENABLED: {MOTORS_ENABLED}")
    
    if MOTORS_ENABLED:
        logger.warning("⚠️  WARNING: MOTOR ACTUATION IS ENABLED!")
        logger.warning("⚠️  Physical wheelchair movement is ALLOWED!")
        logger.warning("⚠️  Ensure safety protocols are in place!")
    else:
        logger.info("✓ SAFE MODE: Motor actuation is DISABLED (simulation only)")
        logger.info("  To enable actuation, set both:")
        logger.info("    ENABLE_ACTUATION=true")
        logger.info("    ACTUATION_CONFIRM=true")
    
    logger.info("=" * 70)
    
    if not API_SECRET:
        logger.error("=" * 70)
        logger.error("SECURITY ERROR: API_SECRET environment variable not set!")
        logger.error("Server will not start without authentication configured.")
        logger.error("Set API_SECRET=your_secret_key in environment")
        logger.error("=" * 70)
        sys.exit(1)
    else:
        logger.info("✓ API_SECRET is configured")
        logger.info("=" * 70)

# =============================================================================
# WHEELCHAIR CONTROL INTEGRATION
# =============================================================================

def get_wheelchair_control_path() -> Path:
    """Determine which wheelchair_control wrapper to use"""
    # Prefer wrapper script for subprocess execution
    if WHEELCHAIR_WRAPPER_PATH.exists():
        return WHEELCHAIR_WRAPPER_PATH
    elif WHEELCHAIR_CONTROL_PATH.exists():
        return WHEELCHAIR_CONTROL_PATH
    else:
        raise FileNotFoundError(
            f"wheelchair_control scripts not found at {WHEELCHAIR_WRAPPER_PATH} or {WHEELCHAIR_CONTROL_PATH}"
        )

async def process_audio_with_wheelchair_control(audio_path: Path, session_id: str) -> dict:
    """
    Process audio file using wheelchair_control.py
    
    Returns dict with structure:
    {
        "success": bool,
        "command": str,
        "language": str,
        "confidence": float,
        "action_taken": str,
        "error": str (optional)
    }
    """
    try:
        control_script = get_wheelchair_control_path()
        
        # Build command to run wheelchair_control.py as subprocess
        # This assumes we'll add a --process flag to wheelchair_control.py
        cmd = [
            sys.executable,
            str(control_script),
            "--process",
            str(audio_path),
            "--json-output"
        ]
        
        if not MOTORS_ENABLED:
            cmd.append("--simulate-only")
        
        logger.info(f"[{session_id}] Running: {' '.join(cmd)}")
        
        # Run subprocess with timeout
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": "Processing timeout (30s)",
                "command": None,
                "language": None,
                "confidence": 0.0,
                "action_taken": "timeout"
            }
        
        # Parse JSON output from wheelchair_control.py
        if process.returncode == 0:
            try:
                result = json.loads(stdout.decode('utf-8'))
                logger.info(f"[{session_id}] Command processed: {result.get('command')}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"[{session_id}] Failed to parse JSON output: {e}")
                logger.error(f"[{session_id}] stdout: {stdout.decode('utf-8')}")
                logger.error(f"[{session_id}] stderr: {stderr.decode('utf-8')}")
                return {
                    "success": False,
                    "error": f"Invalid JSON output: {e}",
                    "command": None,
                    "language": None,
                    "confidence": 0.0,
                    "action_taken": "parse_error"
                }
        else:
            error_msg = stderr.decode('utf-8')
            logger.error(f"[{session_id}] Processing failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "command": None,
                "language": None,
                "confidence": 0.0,
                "action_taken": "process_error"
            }
            
    except FileNotFoundError:
        logger.error(f"[{session_id}] wheelchair_control.py not found")
        return {
            "success": False,
            "error": "wheelchair_control.py not found",
            "command": None,
            "language": None,
            "confidence": 0.0,
            "action_taken": "not_found"
        }
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "command": None,
            "language": None,
            "confidence": 0.0,
            "action_taken": "exception"
        }

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def save_audio_chunks_to_wav(chunks: list, session_id: str) -> Path:
    """Save audio chunks to WAV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = TEMP_DIR / f"audio_{session_id}_{timestamp}.wav"
    
    try:
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            
            # Concatenate all chunks
            audio_data = b''.join(chunks)
            wav_file.writeframes(audio_data)
        
        logger.info(f"[{session_id}] Saved {len(audio_data)} bytes to {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"[{session_id}] Failed to save audio: {e}", exc_info=True)
        raise

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def create_session(session_id: str, user_agent: str = None) -> dict:
    """Create a new session"""
    session = {
        "id": session_id,
        "created_at": datetime.now().isoformat(),
        "user_agent": user_agent,
        "audio_chunks": [],
        "state": "ready",  # ready, recording, processing
        "commands_processed": 0
    }
    active_sessions[session_id] = session
    logger.info(f"[{session_id}] Session created")
    return session

def get_session(session_id: str) -> Optional[dict]:
    """Get session by ID"""
    return active_sessions.get(session_id)

def cleanup_session(session_id: str):
    """Clean up session resources"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        # Clean up audio chunks
        session["audio_chunks"].clear()
        del active_sessions[session_id]
        logger.info(f"[{session_id}] Session cleaned up")

# =============================================================================
# WEBSOCKET HANDLERS
# =============================================================================

async def authenticate_connection(websocket: WebSocketServerProtocol) -> bool:
    """
    Authenticate WebSocket connection using token parameter
    
    TODO: For production:
    - Use Authorization header instead of query parameter
    - Implement TLS (wss://) 
    - Add rate limiting
    - Add proper session tokens with expiry
    """
    try:
        # Get path from websocket.request (websockets 12.0+ API)
        path = websocket.request.path if hasattr(websocket, 'request') else websocket.path
        
        # Parse query parameters from path
        if '?' in path:
            query_string = path.split('?', 1)[1]
            params = dict(qc.split('=') for qc in query_string.split('&') if '=' in qc)
            token = params.get('token')
            
            if token == API_SECRET:
                logger.info(f"Client authenticated from {websocket.remote_address}")
                return True
            else:
                logger.warning(f"Authentication failed from {websocket.remote_address}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "authentication_failed",
                    "message": "Invalid token"
                }))
                return False
        else:
            logger.warning(f"No token provided from {websocket.remote_address}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": "authentication_required",
                "message": "Token required in query parameter: ?token=YOUR_TOKEN"
            }))
            return False
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        return False

async def handle_hello_message(websocket: WebSocketServerProtocol, data: dict) -> Optional[str]:
    """Handle initial hello message and create session"""
    session_id = data.get("session_id")
    
    if not session_id:
        await websocket.send(json.dumps({
            "type": "error",
            "error": "missing_session_id",
            "message": "session_id required in hello message"
        }))
        return None
    
    # Create session
    user_agent = data.get("user_agent", "unknown")
    session = create_session(session_id, user_agent)
    
    # Send welcome response
    await websocket.send(json.dumps({
        "type": "welcome",
        "session_id": session_id,
        "motors_enabled": MOTORS_ENABLED,
        "server_time": datetime.now().isoformat(),
        "message": "Connected to Smart Wheelchair WebSocket Server"
    }))
    
    return session_id

async def handle_audio_start(websocket: WebSocketServerProtocol, session_id: str, data: dict):
    """Handle start of audio streaming"""
    session = get_session(session_id)
    if not session:
        await websocket.send(json.dumps({
            "type": "error",
            "error": "invalid_session",
            "message": "Session not found"
        }))
        return
    
    session["state"] = "recording"
    session["audio_chunks"].clear()
    
    logger.info(f"[{session_id}] Audio recording started")
    
    await websocket.send(json.dumps({
        "type": "audio_start_ack",
        "session_id": session_id,
        "message": "Ready to receive audio data"
    }))

async def handle_audio_chunk(websocket: WebSocketServerProtocol, session_id: str, audio_data: bytes):
    """Handle incoming audio chunk"""
    session = get_session(session_id)
    if not session:
        logger.warning(f"[{session_id}] Audio chunk received for invalid session")
        return
    
    if session["state"] != "recording":
        logger.warning(f"[{session_id}] Audio chunk received in wrong state: {session['state']}")
        return
    
    session["audio_chunks"].append(audio_data)
    
    # Send periodic acknowledgments (every 10 chunks)
    if len(session["audio_chunks"]) % 10 == 0:
        await websocket.send(json.dumps({
            "type": "audio_progress",
            "session_id": session_id,
            "chunks_received": len(session["audio_chunks"])
        }))

async def handle_audio_end(websocket: WebSocketServerProtocol, session_id: str, data: dict):
    """Handle end of audio streaming and process command"""
    session = get_session(session_id)
    if not session:
        await websocket.send(json.dumps({
            "type": "error",
            "error": "invalid_session",
            "message": "Session not found"
        }))
        return
    
    session["state"] = "processing"
    
    try:
        # Save audio to file
        if not session["audio_chunks"]:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "no_audio_data",
                "message": "No audio data received"
            }))
            session["state"] = "ready"
            return
        
        wav_path = save_audio_chunks_to_wav(session["audio_chunks"], session_id)
        
        # Send processing status
        await websocket.send(json.dumps({
            "type": "processing",
            "session_id": session_id,
            "message": "Processing audio command..."
        }))
        
        # Process with wheelchair_control.py
        result = await process_audio_with_wheelchair_control(wav_path, session_id)
        
        # Send result back to client
        await websocket.send(json.dumps({
            "type": "command_result",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "motors_enabled": MOTORS_ENABLED,
            **result
        }))
        
        session["commands_processed"] += 1
        session["state"] = "ready"
        
        # Clean up audio file (optional, keep for debugging)
        # wav_path.unlink()
        
    except Exception as e:
        logger.error(f"[{session_id}] Error processing audio: {e}", exc_info=True)
        await websocket.send(json.dumps({
            "type": "error",
            "error": "processing_failed",
            "message": str(e),
            "session_id": session_id
        }))
        session["state"] = "ready"

async def handle_voice_enrollment(websocket: WebSocketServerProtocol, data: dict):
    """Handle voice enrollment request"""
    user_name = data.get("user_name", "unknown")
    audio_chunks = []
    
    logger.info(f"Starting voice enrollment for: {user_name}")
    
    # Collect audio chunks
    async for message in websocket:
        if isinstance(message, bytes):
            audio_chunks.append(message)
        else:
            msg_data = json.loads(message)
            if msg_data.get("type") == "enroll_complete":
                break
    
    try:
        # Save enrollment audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        enrollment_path = TEMP_DIR / f"enrollment_{user_name}_{timestamp}.wav"
        
        with wave.open(str(enrollment_path), 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            audio_data = b''.join(audio_chunks)
            wav_file.writeframes(audio_data)
        
        logger.info(f"Saved enrollment audio: {enrollment_path}")
        
        # Call wheelchair_control.py to process enrollment
        cmd = [
            sys.executable,
            str(WHEELCHAIR_WRAPPER_PATH),
            "--enroll-voice",
            str(enrollment_path),
            "--user-name",
            user_name,
            "--json-output"
        ]
        
        logger.info(f"Running enrollment: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
        
        if process.returncode == 0:
            result = json.loads(stdout.decode('utf-8'))
            if result.get("success"):
                await websocket.send(json.dumps({
                    "type": "enroll_success",
                    "user_name": user_name,
                    "message": f"Voice enrolled successfully for {user_name}"
                }))
                logger.info(f"Voice enrollment successful for: {user_name}")
            else:
                await websocket.send(json.dumps({
                    "type": "enroll_error",
                    "message": result.get("error", "Enrollment failed")
                }))
        else:
            error_msg = stderr.decode('utf-8')
            logger.error(f"Enrollment failed: {error_msg}")
            await websocket.send(json.dumps({
                "type": "enroll_error",
                "message": error_msg
            }))
            
    except Exception as e:
        logger.error(f"Enrollment error: {e}", exc_info=True)
        await websocket.send(json.dumps({
            "type": "enroll_error",
            "message": str(e)
        }))

async def handle_client(websocket: WebSocketServerProtocol):
    """Main WebSocket client handler (websockets 12.0+ signature)"""
    session_id = None
    
    try:
        # Authenticate connection
        if not await authenticate_connection(websocket):
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Handle messages
        async for message in websocket:
            try:
                # Handle binary audio data
                if isinstance(message, bytes):
                    if session_id:
                        await handle_audio_chunk(websocket, session_id, message)
                    else:
                        logger.warning("Received audio data before hello message")
                    continue
                
                # Handle JSON text messages
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "hello":
                    session_id = await handle_hello_message(websocket, data)
                    
                elif msg_type == "enroll_voice":
                    await handle_voice_enrollment(websocket, data)
                    return  # Close connection after enrollment
                    
                elif msg_type == "audio_start":
                    if session_id:
                        await handle_audio_start(websocket, session_id, data)
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "no_session",
                            "message": "Send hello message first"
                        }))
                
                elif msg_type == "audio_end":
                    if session_id:
                        await handle_audio_end(websocket, session_id, data)
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": "no_session",
                            "message": "Send hello message first"
                        }))
                
                elif msg_type == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                else:
                    logger.warning(f"[{session_id}] Unknown message type: {msg_type}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "unknown_message_type",
                        "message": f"Unknown message type: {msg_type}"
                    }))
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "invalid_json",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": "internal_error",
                    "message": str(e)
                }))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{session_id}] Connection closed")
    except Exception as e:
        logger.error(f"[{session_id}] Connection error: {e}", exc_info=True)
    finally:
        if session_id:
            cleanup_session(session_id)

# =============================================================================
# SERVER STARTUP
# =============================================================================

async def main():
    """Start WebSocket server"""
    check_safety_configuration()
    
    logger.info(f"Starting WebSocket server on {HOST}:{PORT}")
    logger.info("Waiting for connections...")
    logger.info("Press Ctrl+C to stop")
    
    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
