# WebSocket System Testing Guide

This guide walks you through testing the complete WebSocket streaming audio system.

## Prerequisites

- Docker installed and running
- Python 3.11+ with pip
- Flutter SDK (for app testing)
- Android device or emulator (for app testing)

## Phase 1: Server Setup and Testing

### Step 1: Start the Server

```bash
# Quick start (recommended)
./start_websocket_server.sh

# Or manual start
cp .env.example .env
# Edit .env and set API_SECRET
docker build -f Dockerfile.websocket -t wheelchair-websocket:latest .
docker-compose -f docker-compose.websocket.yml up -d
```

### Step 2: Verify Server is Running

```bash
# Check container status
docker ps | grep wheelchair-websocket

# View logs
docker-compose -f docker-compose.websocket.yml logs -f

# Expected output:
# - "SAFETY CONFIGURATION STATUS"
# - "SAFE MODE: Motor actuation is DISABLED"
# - "✓ API_SECRET is configured"
# - "Starting WebSocket server on 0.0.0.0:8765"
```

### Step 3: Test with Python Client

```bash
# Install test dependencies
pip install websockets sounddevice numpy

# Test 1: Record and send audio
python test_websocket_client.py --record 5

# Say a command like "move forward" or "stop"

# Test 2: Use existing audio file
python test_websocket_client.py path/to/test.wav

# Test 3: Test with custom server
python test_websocket_client.py --record 5 \
  --server ws://192.168.1.100:8765 \
  --token your_api_secret_here
```

### Expected Test Results

**Successful connection:**
```
Connecting to ws://localhost:8765...
Connected! Session ID: abc123-...
✓ Welcome message received
  Motors enabled: False
  Message: Connected to Smart Wheelchair WebSocket Server
```

**Successful processing:**
```
Sending audio file: temp_recording.wav
  Sample rate: 16000 Hz
  Channels: 1
  ...
  ✓ Sent 42 chunks (172032 bytes total)

====================================================================
COMMAND RESULT
====================================================================
✓ Success!
  Command: forward
  Language: en
  Confidence: 95.0%
  Action taken: simulated
  Motors enabled: False
====================================================================
```

## Phase 2: Network Testing

### Step 1: Find Server IP

```bash
# On server machine
hostname -I
# Or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

### Step 2: Test from Another Machine

```bash
# From a different computer on the same network
python test_websocket_client.py --record 5 \
  --server ws://SERVER_IP:8765 \
  --token your_api_secret_here
```

### Step 3: Check Firewall

```bash
# If connection fails, check firewall
sudo ufw status
sudo ufw allow 8765/tcp

# Or for testing, temporarily disable
sudo ufw disable
```

## Phase 3: Flutter App Testing

### Step 1: Update App Configuration

Edit `smart_wheelchair_app/lib/voice_control_page.dart`:

```dart
// Line 35-36
static const String _wsServerUrl = 'ws://YOUR_SERVER_IP:8765';
static const String _apiToken = 'your_api_secret_from_env_file';
```

### Step 2: Install Dependencies

```bash
cd smart_wheelchair_app
flutter pub get
```

### Step 3: Run on Android

```bash
# Connect device via USB or start emulator
adb devices

# Run app
flutter run

# Or build APK for installation
flutter build apk
# APK will be at: build/app/outputs/flutter-apk/app-release.apk
```

### Step 4: Test App Functionality

1. **Launch app** - Open Smart Wheelchair app
2. **Check connection indicator** - Should show green dot and "Connected"
3. **Verify safety badge** - Should show "SIMULATION MODE (Motors Disabled)"
4. **Test recording**:
   - Tap microphone button (should turn red)
   - Say "move forward" clearly
   - Tap again to stop
   - Wait for processing
5. **Check result** - Should display command and confidence percentage
6. **Test different commands**:
   - "stop"
   - "turn left"
   - "turn right"
   - "move backward"

### Expected App Behavior

**On successful connection:**
- Green connection indicator in top-right
- "Connected" status message
- Blue "SIMULATION MODE" badge (if motors disabled)
- Microphone button is blue and clickable

**During recording:**
- Microphone button turns red with glow effect
- "Listening..." text displays
- Recording indicator active

**During processing:**
- "Processing..." text displays
- Microphone button disabled

**After command processed:**
- Command displays in "Last Command" box
- Server message shows command and confidence
- Ready for next command

## Phase 4: Error Testing

### Test 1: Invalid Token

```bash
python test_websocket_client.py --record 5 --token invalid_token

# Expected: "Authentication failed" error
```

### Test 2: Connection Timeout

```bash
# Stop server
docker-compose -f docker-compose.websocket.yml down

# Try to connect
python test_websocket_client.py --record 5

# Expected: "Connection error" or timeout
```

### Test 3: No Audio Data

Modify test client to send empty audio - should receive "no_audio_data" error.

### Test 4: Malformed JSON

Test client sends invalid JSON - should receive "invalid_json" error.

## Phase 5: Performance Testing

### Test 1: Latency Measurement

```bash
# Time the entire process
time python test_websocket_client.py --record 3

# Typical results:
# - Connection: < 1 second
# - Audio upload: < 1 second
# - Processing: 2-5 seconds
# - Total: 3-7 seconds
```

### Test 2: Multiple Concurrent Connections

```bash
# Terminal 1
python test_websocket_client.py --record 5 &

# Terminal 2
python test_websocket_client.py --record 5 &

# Both should work simultaneously
```

### Test 3: Stress Test

```bash
# Send 10 commands in sequence
for i in {1..10}; do
  echo "Command $i"
  python test_websocket_client.py --record 3
  sleep 1
done
```

### Test 4: Monitor Resources

```bash
# Watch container resource usage
docker stats wheelchair-websocket-server

# Check for:
# - CPU usage (should spike during processing)
# - Memory usage (watch for leaks)
# - Network I/O
```

## Phase 6: Safety Testing

### Test 1: Verify Motors Disabled by Default

```bash
# Check logs for safety configuration
docker-compose -f docker-compose.websocket.yml logs | grep "SAFETY"

# Should show:
# - ENABLE_ACTUATION: False
# - ACTUATION_CONFIRM: False
# - MOTORS_ENABLED: False
# - "SAFE MODE: Motor actuation is DISABLED"
```

### Test 2: Emergency Stop

In Flutter app:
1. Start recording
2. Tap red emergency stop button
3. Verify recording stops immediately

## Troubleshooting Common Issues

### Issue: "API_SECRET not set"

**Solution:**
```bash
# Edit .env file
vim .env

# Add:
API_SECRET=your_generated_token_here

# Restart server
docker-compose -f docker-compose.websocket.yml restart
```

### Issue: "Connection refused"

**Possible causes:**
1. Server not running - Check with `docker ps`
2. Wrong IP address - Verify with `hostname -I`
3. Firewall blocking - Check with `sudo ufw status`
4. Wrong port - Verify 8765 is exposed

**Solutions:**
```bash
# Restart server
docker-compose -f docker-compose.websocket.yml restart

# Check port binding
docker port wheelchair-websocket-server

# Test local connection first
python test_websocket_client.py --server ws://localhost:8765 --record 3
```

### Issue: "wheelchair_control.py not found"

**Solution:**
```bash
# Verify files exist
ls -la wheelchair_control.py
ls -la wheelchair_control_wrapper.py

# Check Docker mounts
docker exec wheelchair-websocket-server ls -la /mnt/data/

# Rebuild if needed
docker-compose -f docker-compose.websocket.yml down
docker build -f Dockerfile.websocket -t wheelchair-websocket:latest .
docker-compose -f docker-compose.websocket.yml up -d
```

### Issue: "Microphone permission denied" (Android)

**Solution:**
1. Go to Settings → Apps → Smart Wheelchair
2. Tap Permissions
3. Enable Microphone
4. Restart app

### Issue: Low command accuracy

**Causes:**
1. Background noise
2. Poor microphone quality
3. Speaking too fast/unclear
4. Unsupported language/command

**Solutions:**
1. Record in quiet environment
2. Speak clearly and at normal pace
3. Use supported commands (see README)
4. Try different recording duration (3-5 seconds optimal)

## Success Criteria

✅ **Server**
- Starts without errors
- Shows safety configuration
- Accepts connections
- Processes audio files
- Returns valid JSON results

✅ **Python Test Client**
- Connects successfully
- Records audio
- Streams to server
- Receives command results
- Handles errors gracefully

✅ **Flutter App**
- Connects to server
- Shows connection status
- Records audio
- Streams in real-time
- Displays results
- Handles errors
- Emergency stop works

✅ **Safety**
- Motors disabled by default
- Safety warnings visible
- Simulation mode active
- Emergency stop functional

## Next Steps After Successful Testing

1. **Production deployment**:
   - Generate strong API_SECRET
   - Set up TLS/SSL for wss://
   - Configure nginx reverse proxy
   - Implement rate limiting
   - Set up monitoring

2. **Motor actuation** (CAREFUL!):
   - Thoroughly test all commands in simulation
   - Have emergency stop accessible
   - Set ENABLE_ACTUATION=true
   - Set ACTUATION_CONFIRM=true
   - Test with wheelchair not moving anything first
   - Gradually test actual movement

3. **Optimization**:
   - Tune audio chunk size
   - Optimize ML model loading
   - Add caching where appropriate
   - Consider GPU acceleration

---

**Remember: Always test in safe, controlled environments before enabling motor actuation!**
