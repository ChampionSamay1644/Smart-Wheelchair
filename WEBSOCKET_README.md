# Smart Wheelchair WebSocket Streaming Audio System

Complete end-to-end implementation for streaming audio from Android Flutter app to Python WebSocket server for real-time wheelchair command processing.

## 🔒 Safety & Security Notice

### Motor Actuation Control

**⚠️  BY DEFAULT, MOTORS ARE DISABLED ⚠️**

This system is configured for **simulation mode only** by default. Physical wheelchair motors will NOT actuate unless explicitly enabled.

To enable motor actuation, you MUST:
1. Set `ENABLE_ACTUATION=true`
2. Set `ACTUATION_CONFIRM=true`
3. Both environment variables must be set to enable physical movement

**Always test in simulation mode first!**

### Authentication

- Uses `API_SECRET` token for authentication
- **Development**: Token passed as query parameter (`?token=XXX`)
- **TODO for Production**:
  - Migrate to `Authorization` header
  - Implement TLS (wss://)
  - Add proper session tokens with expiry
  - Implement rate limiting

## 📋 System Architecture

```
┌─────────────────────┐         WebSocket          ┌──────────────────────┐
│  Flutter Android    │◄──────(ws:// or wss://)────►│  Python WebSocket    │
│  App                │         Audio Stream        │  Server (Docker)     │
│                     │         JSON Messages       │                      │
│  - Record Audio     │                             │  - Receive Stream    │
│  - Stream PCM 16bit │                             │  - Save to WAV       │
│  - Session Mgmt     │                             │  - Process Command   │
│  - Display Results  │                             │  - Return JSON       │
└─────────────────────┘                             └──────────────────────┘
                                                              │
                                                              ▼
                                                    ┌──────────────────────┐
                                                    │ wheelchair_control.py│
                                                    │ (Mounted from host)  │
                                                    │                      │
                                                    │  - Speech Recognition│
                                                    │  - Command Matching  │
                                                    │  - Motor Control     │
                                                    └──────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Server**: Linux Mint (x86_64) or Raspberry Pi (ARM/ARM64)
- **Docker**: 20.10+ with buildx support
- **Flutter**: 3.0+
- **Python**: 3.11+

### 1. Server Setup (Docker)

#### Generate API Secret

```bash
# Generate a secure random token
openssl rand -hex 32
# Save this token - you'll need it for both server and client
```

#### Configure Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
# REQUIRED: API Authentication Token
API_SECRET=your_generated_token_here

# SAFETY: Motor actuation (both must be true to enable)
ENABLE_ACTUATION=false
ACTUATION_CONFIRM=false

# Server configuration
WS_HOST=0.0.0.0
WS_PORT=8765
EOF
```

#### Build Docker Image (x86_64)

```bash
# Standard build for x86_64 (Linux Mint)
docker build -f Dockerfile.websocket -t wheelchair-websocket:latest .
```

#### Build for Multiple Architectures (Including Raspberry Pi)

```bash
# Setup buildx (one-time setup)
docker buildx create --name wheelchair-builder --use
docker buildx inspect --bootstrap

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -f Dockerfile.websocket \
  -t wheelchair-websocket:latest \
  --load \
  .

# For Raspberry Pi specifically
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.websocket \
  -t wheelchair-websocket:rpi \
  --load \
  .
```

#### Start Server with Docker Compose

```bash
# Start the server
docker-compose -f docker-compose.websocket.yml up -d

# View logs
docker-compose -f docker-compose.websocket.yml logs -f

# Stop server
docker-compose -f docker-compose.websocket.yml down
```

#### Alternative: Run without Docker Compose

```bash
docker run -d \
  --name wheelchair-websocket \
  -p 8765:8765 \
  -v $(pwd)/wheelchair_control.py:/mnt/data/wheelchair_control.py:ro \
  -v $(pwd)/system.py:/mnt/data/system.py:ro \
  -v $(pwd)/multi_voice_tts.py:/mnt/data/multi_voice_tts.py:ro \
  -v $(pwd)/models:/mnt/data/models:ro \
  -v $(pwd)/optimized_models:/mnt/data/optimized_models:ro \
  -v $(pwd)/temp_ws_audio:/app/temp_ws_audio \
  -e API_SECRET=your_token_here \
  -e ENABLE_ACTUATION=false \
  -e ACTUATION_CONFIRM=false \
  wheelchair-websocket:latest
```

### 2. Test Server with Python Client

```bash
# Install test client dependencies
pip install websockets sounddevice numpy

# Test with recording (record 5 seconds)
python test_websocket_client.py --record 5 --server ws://localhost:8765 --token your_token_here

# Test with audio file
python test_websocket_client.py path/to/audio.wav --server ws://localhost:8765 --token your_token_here

# Test remote server
python test_websocket_client.py --record 5 --server ws://192.168.1.100:8765 --token your_token_here
```

### 3. Flutter App Setup

#### Update Configuration

Edit `smart_wheelchair_app/lib/voice_control_page.dart`:

```dart
// Line 35-36: Update these values
static const String _wsServerUrl = 'ws://YOUR_SERVER_IP:8765';
static const String _apiToken = 'your_generated_token_here';
```

#### Install Dependencies

```bash
cd smart_wheelchair_app
flutter pub get
```

#### Run on Android

```bash
# Connect Android device via USB or use emulator
flutter run
```

## 📱 Flutter App Usage

1. **Launch App**: Open Smart Wheelchair app
2. **Check Connection**: Green indicator shows connected status
3. **Blue Badge**: "SIMULATION MODE" appears if motors disabled (safe)
4. **Record Command**:
   - Tap microphone button
   - Speak command (e.g., "Move forward", "Turn left")
   - Tap again to stop recording
5. **View Result**: Command and confidence displayed
6. **Emergency Stop**: Red button immediately stops recording and sends stop signal

## 🔧 WebSocket Protocol

### Message Types

#### Client → Server

**1. Hello (Initial Connection)**
```json
{
  "type": "hello",
  "session_id": "uuid-v4-string",
  "user_agent": "Flutter/Android Smart Wheelchair App v1.0"
}
```

**2. Audio Start**
```json
{
  "type": "audio_start",
  "session_id": "uuid-v4-string"
}
```

**3. Audio Chunks (Binary)**
- Raw PCM 16-bit audio data
- 16kHz sample rate
- Mono channel
- Sent as binary WebSocket frames

**4. Audio End**
```json
{
  "type": "audio_end",
  "session_id": "uuid-v4-string"
}
```

**5. Ping**
```json
{
  "type": "ping"
}
```

#### Server → Client

**1. Welcome**
```json
{
  "type": "welcome",
  "session_id": "uuid-v4-string",
  "motors_enabled": false,
  "server_time": "2025-11-22T10:30:00",
  "message": "Connected to Smart Wheelchair WebSocket Server"
}
```

**2. Audio Start Acknowledgment**
```json
{
  "type": "audio_start_ack",
  "session_id": "uuid-v4-string",
  "message": "Ready to receive audio data"
}
```

**3. Audio Progress**
```json
{
  "type": "audio_progress",
  "session_id": "uuid-v4-string",
  "chunks_received": 42
}
```

**4. Processing**
```json
{
  "type": "processing",
  "session_id": "uuid-v4-string",
  "message": "Processing audio command..."
}
```

**5. Command Result**
```json
{
  "type": "command_result",
  "session_id": "uuid-v4-string",
  "timestamp": "2025-11-22T10:30:05",
  "motors_enabled": false,
  "success": true,
  "command": "forward",
  "language": "en",
  "confidence": 0.95,
  "action_taken": "simulated"
}
```

**6. Error**
```json
{
  "type": "error",
  "error": "error_code",
  "message": "Human readable error message",
  "session_id": "uuid-v4-string"
}
```

**7. Pong**
```json
{
  "type": "pong",
  "timestamp": "2025-11-22T10:30:00"
}
```

## 🐛 Troubleshooting

### Server Issues

**Server won't start**
```bash
# Check if API_SECRET is set
docker-compose -f docker-compose.websocket.yml config

# Check logs
docker-compose -f docker-compose.websocket.yml logs

# Verify port availability
netstat -tuln | grep 8765
```

**Connection refused**
```bash
# Check firewall
sudo ufw status
sudo ufw allow 8765/tcp

# Check if server is listening
docker exec wheelchair-websocket-server netstat -tuln | grep 8765
```

**wheelchair_control.py not found**
```bash
# Verify file exists
ls -la wheelchair_control.py

# Check Docker mount
docker exec wheelchair-websocket-server ls -la /mnt/data/
```

### Flutter App Issues

**"Connection failed"**
- Verify server IP address in `voice_control_page.dart`
- Check server is running: `docker ps`
- Test connectivity: `ping YOUR_SERVER_IP`
- Check token matches server's API_SECRET

**"Microphone permission required"**
- Android: Go to Settings → Apps → Smart Wheelchair → Permissions → Enable Microphone

**Audio not recording**
```bash
# Check Android permissions in manifest
grep -A 5 "RECORD_AUDIO" smart_wheelchair_app/android/app/src/main/AndroidManifest.xml

# Rebuild app
cd smart_wheelchair_app
flutter clean
flutter pub get
flutter run
```

### Processing Issues

**"Processing timeout"**
- Check server has sufficient resources
- Verify ML models are mounted correctly
- Check `docker stats` for resource usage

**"Invalid command" / Low confidence**
- Speak clearly and close to microphone
- Reduce background noise
- Try supported commands (see Commands section)

## 📝 Supported Commands

The system recognizes multilingual commands:

### English
- **Movement**: forward, backward, left, right
- **Rotation**: rotate left, rotate right
- **Control**: start, stop, faster, slower

### Hindi (हिंदी)
- **Movement**: आगे (aage), पीछे (peeche), बाएं (baaye), दाएं (daaye)
- **Control**: रुको (ruko), शुरू (shuru)

### Marathi (मराठी)
- **Movement**: पुढे (pudhe), मागे (mage), डावीकडे (davikade), उजवीकडे (ujavikade)

...and many more languages (Spanish, French, German, Chinese, etc.)

See `wheelchair_control.py` for complete list.

## 🔐 Production Deployment Checklist

- [ ] Generate strong API_SECRET (32+ random bytes)
- [ ] Enable TLS with valid SSL certificate
- [ ] Update client to use `wss://` instead of `ws://`
- [ ] Move token from query parameter to Authorization header
- [ ] Implement rate limiting (e.g., nginx + limit_req)
- [ ] Add DDoS protection
- [ ] Set up proper secrets management (Docker secrets, Vault)
- [ ] Implement session token expiry
- [ ] Add authentication logging and monitoring
- [ ] Configure firewall to restrict access
- [ ] Set up automatic container restarts
- [ ] Implement health checks and alerting
- [ ] Only enable ACTUATION after thorough testing

## 🛠️ Development

### Modify Server

```bash
# Edit server code
vim websocket_server.py

# Rebuild and restart
docker-compose -f docker-compose.websocket.yml down
docker build -f Dockerfile.websocket -t wheelchair-websocket:latest .
docker-compose -f docker-compose.websocket.yml up -d
```

### Modify Flutter App

```bash
cd smart_wheelchair_app

# Edit Dart code
vim lib/voice_control_page.dart

# Hot reload (if app is running)
# Press 'r' in terminal running `flutter run`

# Or rebuild
flutter run
```

### Enable Motor Actuation (CAREFUL!)

```bash
# Edit .env file
vim .env

# Set both to true:
ENABLE_ACTUATION=true
ACTUATION_CONFIRM=true

# Restart container
docker-compose -f docker-compose.websocket.yml restart
```

## 📊 Performance Considerations

- **Latency**: Expect 1-3 second processing time (network + ML inference)
- **Audio Quality**: Higher sample rates increase bandwidth but improve recognition
- **Chunk Size**: 4KB chunks balance latency and throughput
- **Resource Usage**: ML models require 2GB+ RAM, consider GPU for faster processing

## 📚 File Structure

```
Smart-Wheelchair/
├── websocket_server.py              # WebSocket server implementation
├── wheelchair_control.py            # Voice command processing logic
├── system.py                        # Shared utilities
├── multi_voice_tts.py              # Text-to-speech
├── Dockerfile.websocket            # Docker image definition
├── docker-compose.websocket.yml    # Docker Compose config
├── requirements_websocket.txt      # Python dependencies
├── test_websocket_client.py        # Python test client
├── WEBSOCKET_README.md             # This file
├── .env                            # Environment variables (create this)
├── models/                         # ML models directory
├── optimized_models/               # Optimized ML models
└── smart_wheelchair_app/           # Flutter app
    ├── lib/
    │   └── voice_control_page.dart # Voice control UI + WebSocket client
    ├── android/
    │   └── app/src/main/
    │       └── AndroidManifest.xml # Permissions
    └── pubspec.yaml                # Flutter dependencies
```

## 🤝 Contributing

When contributing, ensure:
1. Safety checks remain in place
2. Default to motors disabled
3. Test in simulation mode first
4. Document all changes
5. Follow existing code style

## 📄 License

See project LICENSE file.

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Test with Python client first
4. Open GitHub issue with logs and details

---

**Remember: Safety First! Always test in simulation mode before enabling motor actuation.**
