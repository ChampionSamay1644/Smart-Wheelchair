# Flutter App Setup Instructions

## Prerequisites

- Flutter SDK 3.0 or higher
- Android SDK
- Android device or emulator

## Installation Steps

### 1. Navigate to the Flutter app directory

```bash
cd smart_wheelchair_app
```

### 2. Install dependencies

```bash
# Get all Flutter packages
flutter pub get

# This will install:
# - web_socket_channel: ^2.4.0
# - uuid: ^4.5.1
# - record: ^5.1.2
# Plus all existing dependencies
```

### 3. Verify installation

```bash
# Check for errors
flutter doctor

# Analyze code
flutter analyze
```

### 4. Configure server connection

Edit `lib/voice_control_page.dart`:

```dart
// Lines 35-36
static const String _wsServerUrl = 'ws://YOUR_SERVER_IP:8765';
static const String _apiToken = 'YOUR_API_SECRET_FROM_ENV_FILE';
```

**Find your server IP:**
```bash
# On the server machine
hostname -I
# Or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**Get your API token:**
```bash
# On the server machine
cat .env | grep API_SECRET
```

### 5. Build and run

#### Option A: Run on connected device

```bash
# Check connected devices
flutter devices

# Run app
flutter run

# Or run in release mode for better performance
flutter run --release
```

#### Option B: Build APK for installation

```bash
# Build release APK
flutter build apk --release

# APK location:
# build/app/outputs/flutter-apk/app-release.apk

# Install on device
adb install build/app/outputs/flutter-apk/app-release.apk
```

#### Option C: Build for specific architecture

```bash
# For ARM64 devices (most modern Android phones)
flutter build apk --target-platform android-arm64

# For ARMv7 devices (older phones)
flutter build apk --target-platform android-arm

# For x86 emulator
flutter build apk --target-platform android-x86
```

## Troubleshooting

### Issue: "Target of URI doesn't exist" errors

These errors appear before running `flutter pub get`. They are normal and will be resolved after installing dependencies.

**Solution:**
```bash
cd smart_wheelchair_app
flutter pub get
```

### Issue: "package:record not found"

**Solution:**
```bash
# Clean build cache
flutter clean

# Reinstall dependencies
flutter pub get

# Rebuild
flutter run
```

### Issue: Permission errors on Android

**Solution:**
The app will request microphone permission on first use. If denied:

1. Go to Android Settings
2. Apps → Smart Wheelchair App
3. Permissions
4. Enable "Microphone"
5. Restart app

### Issue: "Connection failed" in app

**Possible causes:**

1. **Wrong server IP**
   - Verify IP in `voice_control_page.dart` matches server
   - Test connectivity: `ping YOUR_SERVER_IP`

2. **Wrong API token**
   - Verify token in app matches `.env` file on server
   - Check server logs for "Authentication failed" messages

3. **Server not running**
   - Check: `docker ps | grep wheelchair-websocket`
   - Start: `./start_websocket_server.sh`

4. **Firewall blocking**
   - On server: `sudo ufw allow 8765/tcp`
   - Test from computer: `telnet SERVER_IP 8765`

5. **Network issue**
   - Ensure device and server on same network
   - Try using server's WiFi IP instead of ethernet IP

### Issue: Audio quality is poor

**Solutions:**

1. **Reduce background noise**
   - Record in quiet environment
   - Speak close to microphone

2. **Adjust recording duration**
   - Optimal: 3-5 seconds
   - Too short: May miss command
   - Too long: More noise captured

3. **Check microphone**
   - Test with phone's voice recorder
   - Ensure microphone not blocked

### Issue: Build errors after code changes

**Solution:**
```bash
# Full clean and rebuild
flutter clean
rm -rf build/
flutter pub get
flutter run
```

## Development Tips

### Hot Reload

When app is running with `flutter run`:
- Press `r` for hot reload (applies code changes)
- Press `R` for hot restart (full restart)
- Press `q` to quit

### Debugging

```bash
# Run with verbose logging
flutter run -v

# View device logs
flutter logs

# Or use adb directly
adb logcat | grep flutter
```

### VS Code Integration

If using VS Code:

1. Install Flutter extension
2. Open `smart_wheelchair_app` folder
3. Press F5 to debug
4. Set breakpoints in Dart code
5. Use Debug Console for variable inspection

### Android Studio Integration

If using Android Studio:

1. Open `smart_wheelchair_app/android` folder
2. Let Gradle sync
3. Run → Edit Configurations → Flutter
4. Set entry point: `lib/main.dart`
5. Click Run button

## Testing the App

### 1. Connection Test

1. Launch app
2. Check connection indicator (top-right)
   - Green = Connected
   - Red = Disconnected
3. If red, check server and network

### 2. Recording Test

1. Tap microphone button
2. Should turn red with glow effect
3. Say "stop" or "move forward"
4. Tap again to stop
5. Wait for processing
6. Check "Last Command" displays result

### 3. Command Recognition Test

Test various commands:
- "move forward"
- "stop"
- "turn left"
- "turn right"
- "move backward"
- "start"

### 4. Multilingual Test (if supported)

Hindi commands:
- "आगे" (aage - forward)
- "रुको" (ruko - stop)
- "बाएं" (baaye - left)
- "दाएं" (daaye - right)

### 5. Emergency Stop Test

1. Start recording
2. Press red emergency button (bottom-right)
3. Verify recording stops immediately

## Performance Optimization

### For production builds:

```bash
# Build with optimizations
flutter build apk --release --shrink --split-per-abi

# This creates separate APKs for each architecture:
# - app-armeabi-v7a-release.apk (32-bit ARM)
# - app-arm64-v8a-release.apk (64-bit ARM)
# - app-x86_64-release.apk (64-bit x86)

# Install appropriate one for your device
```

### Reduce APK size:

```bash
# Enable R8 optimization
# In android/app/build.gradle, ensure:
buildTypes {
    release {
        minifyEnabled true
        shrinkResources true
    }
}
```

## Next Steps

After successful setup:

1. Test all features following `TESTING_GUIDE.md`
2. Configure server IP for your network
3. Test in various network conditions
4. Test with different background noise levels
5. Gather feedback on command recognition accuracy

## Getting Help

If you encounter issues:

1. Check this document's troubleshooting section
2. Review `WEBSOCKET_README.md` for server issues
3. Run `flutter doctor` and fix any issues
4. Check server logs: `docker-compose logs -f`
5. Test with Python client first to isolate issues

---

**Remember:** The app requires an active connection to the WebSocket server to function. Always ensure the server is running and accessible before testing the app.
