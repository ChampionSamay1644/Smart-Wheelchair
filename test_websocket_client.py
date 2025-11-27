#!/usr/bin/env python3
"""
WebSocket Test Client for Smart Wheelchair Server
Simulates the Flutter app's WebSocket audio streaming

Usage:
    python test_websocket_client.py <audio_file.wav>
    python test_websocket_client.py --record 5  # Record 5 seconds
"""

import asyncio
import json
import sys
import uuid
import wave
from pathlib import Path
import argparse

try:
    import websockets
except ImportError:
    print("Error: websockets not installed. Install with: pip install websockets")
    sys.exit(1)

try:
    import sounddevice as sd
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    print("Warning: sounddevice not available. Recording disabled.")
    print("Install with: pip install sounddevice numpy")
    RECORDING_AVAILABLE = False


class WheelchairWebSocketClient:
    def __init__(self, server_url, api_token):
        self.server_url = server_url
        self.api_token = api_token
        self.session_id = str(uuid.uuid4())
        self.websocket = None
        
    async def connect(self):
        """Connect to WebSocket server"""
        uri = f"{self.server_url}?token={self.api_token}"
        print(f"Connecting to {self.server_url}...")
        
        try:
            self.websocket = await websockets.connect(uri)
            print(f"Connected! Session ID: {self.session_id}")
            
            # Send hello message
            hello_msg = {
                "type": "hello",
                "session_id": self.session_id,
                "user_agent": "Python Test Client v1.0"
            }
            await self.websocket.send(json.dumps(hello_msg))
            
            # Wait for welcome
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "welcome":
                print(f"✓ Welcome message received")
                print(f"  Motors enabled: {data.get('motors_enabled')}")
                print(f"  Message: {data.get('message')}")
                return True
            else:
                print(f"Unexpected response: {data}")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    async def send_audio_file(self, audio_path):
        """Send audio file to server for processing"""
        if not self.websocket:
            print("Error: Not connected to server")
            return
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return
        
        print(f"\nSending audio file: {audio_path}")
        
        try:
            # Read WAV file
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                print(f"  Sample rate: {sample_rate} Hz")
                print(f"  Channels: {channels}")
                print(f"  Sample width: {sample_width} bytes")
                print(f"  Duration: {n_frames / sample_rate:.2f} seconds")
                
                if sample_rate != 16000:
                    print(f"  Warning: Expected 16kHz, got {sample_rate}Hz")
                
                # Send audio_start
                start_msg = {
                    "type": "audio_start",
                    "session_id": self.session_id
                }
                await self.websocket.send(json.dumps(start_msg))
                print("  Sent audio_start")
                
                # Wait for ack
                response = await self.websocket.recv()
                data = json.loads(response)
                if data.get("type") == "audio_start_ack":
                    print("  ✓ Server ready to receive audio")
                
                # Stream audio in chunks
                chunk_size = 4096
                total_bytes = 0
                chunk_count = 0
                
                while True:
                    audio_data = wav_file.readframes(chunk_size)
                    if not audio_data:
                        break
                    
                    await self.websocket.send(audio_data)
                    total_bytes += len(audio_data)
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:
                        print(f"  Sent {chunk_count} chunks ({total_bytes} bytes)...")
                
                print(f"  ✓ Sent {chunk_count} chunks ({total_bytes} bytes total)")
                
                # Send audio_end
                end_msg = {
                    "type": "audio_end",
                    "session_id": self.session_id
                }
                await self.websocket.send(json.dumps(end_msg))
                print("  Sent audio_end")
                
                # Wait for result
                print("\nWaiting for processing result...")
                while True:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    data = json.loads(response)
                    msg_type = data.get("type")
                    
                    if msg_type == "processing":
                        print(f"  {data.get('message')}")
                    
                    elif msg_type == "audio_progress":
                        chunks = data.get("chunks_received")
                        print(f"  Progress: {chunks} chunks received by server")
                    
                    elif msg_type == "command_result":
                        print("\n" + "="*60)
                        print("COMMAND RESULT")
                        print("="*60)
                        
                        if data.get("success"):
                            print(f"✓ Success!")
                            print(f"  Command: {data.get('command')}")
                            print(f"  Language: {data.get('language')}")
                            print(f"  Confidence: {data.get('confidence', 0) * 100:.1f}%")
                            print(f"  Action taken: {data.get('action_taken')}")
                            print(f"  Motors enabled: {data.get('motors_enabled')}")
                        else:
                            print(f"✗ Failed")
                            print(f"  Error: {data.get('error')}")
                        
                        print("="*60)
                        break
                    
                    elif msg_type == "error":
                        print(f"\n✗ Error: {data.get('message')}")
                        break
                
        except asyncio.TimeoutError:
            print("\n✗ Timeout waiting for response")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print("\nConnection closed")


def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    if not RECORDING_AVAILABLE:
        print("Error: Recording not available. Install sounddevice and numpy.")
        return None
    
    print(f"\nRecording {duration} seconds of audio...")
    print("Speak now!")
    
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        print("Recording complete")
        
        # Save to temporary file
        temp_path = Path("temp_recording.wav")
        with wave.open(str(temp_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        print(f"Saved to {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"Recording error: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Test client for Smart Wheelchair WebSocket Server"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to WAV audio file to send"
    )
    parser.add_argument(
        "--record",
        type=int,
        metavar="SECONDS",
        help="Record audio for specified seconds instead of using file"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)"
    )
    parser.add_argument(
        "--token",
        default="changeme_generate_secure_token",
        help="API token for authentication"
    )
    
    args = parser.parse_args()
    
    # Determine audio source
    audio_path = None
    if args.record:
        audio_path = record_audio(duration=args.record)
        if not audio_path:
            sys.exit(1)
    elif args.audio_file:
        audio_path = args.audio_file
    else:
        parser.print_help()
        print("\nError: Specify either audio file or --record option")
        sys.exit(1)
    
    # Create client and connect
    client = WheelchairWebSocketClient(args.server, args.token)
    
    if await client.connect():
        await client.send_audio_file(audio_path)
        await client.close()
    else:
        print("Failed to connect to server")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
