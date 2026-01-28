import 'dart:async';
import 'dart:convert';
import 'dart:io' show Platform;
import 'dart:math' as math;
// `Uint8List` is provided via `package:flutter/foundation.dart` so explicit import not required

import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:permission_handler/permission_handler.dart';

/// Singleton wrapper around [FlutterBluetoothSerial] that speaks the wheelchair
/// RFCOMM protocol (newline-delimited JSON payloads).
class WheelchairBluetoothService {
  WheelchairBluetoothService._internal();

  static final WheelchairBluetoothService _instance =
      WheelchairBluetoothService._internal();

  factory WheelchairBluetoothService() => _instance;

  final FlutterBluetoothSerial _bluetooth = FlutterBluetoothSerial.instance;
  final StreamController<Map<String, dynamic>> _telemetryController =
      StreamController<Map<String, dynamic>>.broadcast();
  static const double _stopMagnitudeThreshold = 0.12;
  static const Duration _directionRefreshInterval = Duration(milliseconds: 140);
  static const int _stopBurstCount = 6;
  static const Duration _stopBurstInterval = Duration(milliseconds: 120);

  BluetoothConnection? _connection;
  StreamSubscription<Uint8List>? _inputSubscription;
  String? _connectedAddress;
  bool _isConnecting = false;
  String _partialLine = '';
  bool _permissionsGranted = false;
  String? _lastJoystickDirection;
  String? _lastJoystickLabel;
  DateTime? _lastJoystickDispatch;
  Timer? _stopBurstTimer;
  int _stopBurstRemaining = 0;

  /// Stream of parsed telemetry/ack messages from the Raspberry Pi.
  Stream<Map<String, dynamic>> get telemetryStream =>
      _telemetryController.stream;

  /// Synchronous flag indicating whether a link is active.
  bool get isConnected => _connection?.isConnected ?? false;

  /// True while a connection attempt is in progress.
  bool get isConnecting => _isConnecting;

  /// MAC address of the currently connected device, if any.
  String? get connectedAddress => _connectedAddress;

  /// Obtain the current adapter state.
  Future<BluetoothState> getState() async {
    if (kIsWeb) return BluetoothState.UNKNOWN;
    return _bluetooth.state;
  }

  /// Listen to adapter state updates.
  Stream<BluetoothState> onAdapterStateChanged() {
    if (kIsWeb) return const Stream.empty();
    return _bluetooth.onStateChanged();
  }

  /// List bonded/paired devices. Manual pairing must happen in the OS
  /// settings before establishing RFCOMM connections.
  Future<List<BluetoothDevice>> listBondedDevices() async {
    await ensurePermissions();
    final devices = await _bluetooth.getBondedDevices();
    devices.sort((a, b) => (a.name ?? '').compareTo(b.name ?? ''));
    return devices;
  }

  Future<bool> connect(String address) async {
    await ensurePermissions();
    if (isConnected && _connectedAddress == address) {
      return true;
    }
    if (_isConnecting) {
      return false;
    }

    _isConnecting = true;
    try {
      final connection = await BluetoothConnection.toAddress(address);
      _connection = connection;
      _connectedAddress = address;
      _partialLine = '';
      _lastJoystickDirection = null;
      _lastJoystickLabel = null;
      _lastJoystickDispatch = null;

      _inputSubscription = connection.input?.listen(
        _handleIncoming,
        onDone: _handleDisconnect,
        onError: (Object error, StackTrace stackTrace) {
          _handleDisconnect();
          debugPrint('Bluetooth stream error: $error');
        },
        cancelOnError: true,
      );

      _telemetryController.add({
        'type': 'bluetooth_status',
        'status': 'connected',
        'address': address,
        'timestamp': DateTime.now().toIso8601String(),
      });
      return true;
    } catch (error) {
      debugPrint('Failed to open Bluetooth connection: $error');
      await disconnect();
      return false;
    } finally {
      _isConnecting = false;
    }
  }

  Future<void> disconnect() async {
    _isConnecting = false;
    _partialLine = '';
    _connectedAddress = null;
    _lastJoystickDirection = null;
    _lastJoystickLabel = null;
    _lastJoystickDispatch = null;
    _cancelStopBurst();

    await _inputSubscription?.cancel();
    _inputSubscription = null;

    if (_connection != null) {
      try {
        await _connection?.close();
      } catch (_) {
        // Ignore socket close failures.
      }
      _connection = null;
    }

    _telemetryController.add({
      'type': 'bluetooth_status',
      'status': 'disconnected',
      'timestamp': DateTime.now().toIso8601String(),
    });
  }

  Future<void> sendManualCommand(
    String command, {
    Map<String, dynamic>? metadata,
  }) async {
    final payload = <String, dynamic>{'mode': 'manual', 'command': command};
    if (metadata != null) {
      payload.addAll(metadata);
    }
    await _sendPayload(payload);
  }

  Future<void> sendJoystickUpdate({
    required double x,
    required double y,
    required double magnitude,
    bool forceStop = false,
  }) async {
    if (forceStop) {
      await _sendStopBurst();
      return;
    }

    _cancelStopBurst();

    final clampedX = x.clamp(-1.0, 1.0);
    final clampedY = y.clamp(-1.0, 1.0);
    final normalizedMagnitude = math.min(
      1.0,
      math.sqrt((clampedX * clampedX) + (clampedY * clampedY)),
    );

    final direction = _classifyJoystickDirection(
      x: clampedX,
      y: clampedY,
      magnitude: normalizedMagnitude,
    );
    if (direction == null) {
      _lastJoystickDirection = null;
      _lastJoystickLabel = 'idle';
      return;
    }

    final label = direction;
    final lastDispatch = _lastJoystickDispatch;
    final now = DateTime.now();
    if (_lastJoystickLabel == label &&
        lastDispatch != null &&
        now.difference(lastDispatch) < _directionRefreshInterval) {
      return;
    }

    await _sendDirectionPayload(
      x: clampedX,
      y: clampedY,
      magnitude: normalizedMagnitude,
      direction: label,
      forceStop: false,
    );

    _lastJoystickLabel = label;
    _lastJoystickDirection = direction;
  }

  Future<void> _sendStopBurst() async {
    _cancelStopBurst();
    await _sendDirectionPayload(
      x: 0.0,
      y: 0.0,
      magnitude: 0.0,
      direction: 'stop',
      forceStop: true,
    );
    _lastJoystickDirection = null;
    _lastJoystickLabel = 'stop';
    _stopBurstRemaining = _stopBurstCount - 1;
    if (_stopBurstRemaining > 0) {
      _stopBurstTimer = Timer.periodic(_stopBurstInterval, (timer) {
        if (_stopBurstRemaining <= 0 ||
            _connection == null ||
            !_connection!.isConnected) {
          timer.cancel();
          _stopBurstRemaining = 0;
          return;
        }
        if (_lastJoystickDirection != null) {
          timer.cancel();
          _stopBurstRemaining = 0;
          return;
        }
        _lastJoystickDispatch = DateTime.now();
        unawaited(
          _sendPayload(<String, dynamic>{
            'mode': 'joystick',
            'x': 0.0,
            'y': 0.0,
            'magnitude': 0.0,
            'direction': 'stop',
            'force_stop': true,
          }).catchError(
            (Object error) =>
                debugPrint('Failed to send stop burst frame: $error'),
          ),
        );
        _stopBurstRemaining -= 1;
        if (_stopBurstRemaining <= 0) {
          timer.cancel();
        }
      });
    }
  }

  void _cancelStopBurst() {
    _stopBurstTimer?.cancel();
    _stopBurstTimer = null;
    _stopBurstRemaining = 0;
  }

  Future<void> _sendDirectionPayload({
    required double x,
    required double y,
    required double magnitude,
    required String direction,
    required bool forceStop,
  }) async {
    _lastJoystickDispatch = DateTime.now();
    await _sendPayload(<String, dynamic>{
      'mode': 'joystick',
      'x': x,
      'y': y,
      'magnitude': magnitude,
      'direction': direction,
      'force_stop': forceStop,
    });
  }

  Future<void> _sendPayload(Map<String, dynamic> payload) async {
    final connection = _connection;
    if (connection == null || !connection.isConnected) {
      throw StateError('Bluetooth connection not established');
    }

    try {
      final frame = '${jsonEncode(payload)}\n';
      connection.output.add(Uint8List.fromList(frame.codeUnits));
      await connection.output.allSent;
    } catch (error) {
      debugPrint('Failed to send Bluetooth payload: $error');
      rethrow;
    }
  }

  String? _classifyJoystickDirection({
    required double x,
    required double y,
    required double magnitude,
  }) {
    final effectiveMagnitude = math.max(
      magnitude,
      math.min(1.0, math.sqrt((x * x) + (y * y))),
    );

    if (effectiveMagnitude < _stopMagnitudeThreshold) {
      return null;
    }

    if (y >= 0 && y.abs() >= x.abs()) {
      return 'forward';
    }
    if (y < 0 && y.abs() >= x.abs()) {
      return 'backward';
    }
    if (x >= 0) {
      return 'right';
    }
    return 'left';
  }

  void _handleIncoming(Uint8List data) {
    _partialLine += utf8.decode(data, allowMalformed: true);
    while (true) {
      final newlineIndex = _partialLine.indexOf('\n');
      if (newlineIndex < 0) {
        break;
      }
      final line = _partialLine.substring(0, newlineIndex).trim();
      _partialLine = _partialLine.substring(newlineIndex + 1);
      if (line.isEmpty) {
        continue;
      }
      try {
        final message = jsonDecode(line) as Map<String, dynamic>;
        _telemetryController.add(message);
      } catch (error) {
        debugPrint('Discarding malformed Bluetooth frame: $line');
      }
    }
  }

  void _handleDisconnect() {
    if (_connection == null) {
      return;
    }
    unawaited(disconnect());
  }

  Future<void> ensurePermissions() async {
    if (!Platform.isAndroid || _permissionsGranted) {
      return;
    }

    final permissions = <Permission>[
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      Permission.bluetoothAdvertise,
      Permission.locationWhenInUse,
    ];

    final denied = <Permission>[];
    for (final permission in permissions) {
      final status = await permission.status;
      if (status.isGranted || status.isLimited) {
        continue;
      }
      
      // Request the permission
      final result = await permission.request();
      if (!result.isGranted && !result.isLimited) {
        // Some permissions might not be available on older/newer versions, 
        // they might return permanentlyDenied or restricted.
        // We only add to denied if it's strictly necessary and denied.
        if (permission == Permission.locationWhenInUse) {
           // On some Android 12+ devices with neverForLocation, 
           // location might not be strictly required for BT. We'll be lenient.
           continue; 
        }
        denied.add(permission);
      }
    }

    if (denied.isNotEmpty) {
      final labels = denied
          .map((permission) => permission.toString().split('.').last)
          .join(', ');
      throw BluetoothPermissionException(
        'Bluetooth permissions denied: $labels',
      );
    }

    _permissionsGranted = true;
  }
}

class BluetoothPermissionException implements Exception {
  BluetoothPermissionException(this.message);

  final String message;

  @override
  String toString() => message;
}
