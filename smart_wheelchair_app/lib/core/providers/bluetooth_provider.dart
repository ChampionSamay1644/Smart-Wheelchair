import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../services/bluetooth_service.dart';

class BluetoothProvider extends ChangeNotifier {
  BluetoothProvider() {
    _init();
  }

  final WheelchairBluetoothService _service = WheelchairBluetoothService();
  StreamSubscription<Map<String, dynamic>>? _telemetrySubscription;
  StreamSubscription<BluetoothState>? _adapterSubscription;
  SharedPreferences? _prefs;

  List<BluetoothDevice> _bondedDevices = <BluetoothDevice>[];
  BluetoothState _adapterState = BluetoothState.UNKNOWN;
  String? _activeCommand;
  String _movementState = 'idle';
  String? _lastError;
  DateTime? _lastUpdate;

  List<BluetoothDevice> get bondedDevices => _bondedDevices;
  BluetoothState get adapterState => _adapterState;
  bool get isAdapterOn => _adapterState == BluetoothState.STATE_ON;
  bool get isConnected => _service.isConnected;
  bool get isConnecting => _service.isConnecting;
  String? get connectedAddress => _service.connectedAddress;
  String? get activeCommand => _activeCommand;
  String get movementState => _movementState;
  String? get lastError => _lastError;
  DateTime? get lastUpdate => _lastUpdate;

  Future<void> _init() async {
    if (kIsWeb) {
      _lastError = 'Bluetooth is only available on mobile devices.';
      _adapterState = BluetoothState.UNKNOWN;
      notifyListeners();
      return;
    }
    _adapterState = await _service.getState();
    _prefs = await SharedPreferences.getInstance();
    try {
      await _service.ensurePermissions();
      _lastError = null;
    } on BluetoothPermissionException catch (error) {
      _lastError = error.message;
      notifyListeners();
      return;
    }
    _adapterSubscription = _service.onAdapterStateChanged().listen((
      BluetoothState state,
    ) {
      _adapterState = state;
      notifyListeners();
    });

    _telemetrySubscription = _service.telemetryStream.listen(
      _handleTelemetry,
      onError: (Object error) {
        _lastError = error.toString();
        notifyListeners();
      },
    );

    await refreshBondedDevices();

    final cachedAddress = _prefs?.getString('last_bluetooth_address');
    if (cachedAddress != null &&
        cachedAddress.isNotEmpty &&
        _adapterState == BluetoothState.STATE_ON) {
      unawaited(connectToDevice(cachedAddress));
    }
  }

  Future<void> refreshBondedDevices() async {
    try {
      await _service.ensurePermissions();
      _bondedDevices = await _service.listBondedDevices();
      _lastError = null;
    } catch (e) {
      _lastError = e.toString();
      _bondedDevices = <BluetoothDevice>[];
    }
    notifyListeners();
  }

  Future<bool> connectToDevice(String address) async {
    try {
      await _service.ensurePermissions();
    } on BluetoothPermissionException catch (error) {
      _lastError = error.message;
      notifyListeners();
      return false;
    }

    final success = await _service.connect(address);
    if (!success) {
      _lastError = 'Failed to connect to $address';
    } else {
      _lastError = null;
      await _prefs?.setString('last_bluetooth_address', address);
    }
    notifyListeners();
    return success;
  }

  Future<void> disconnect() async {
    await _service.disconnect();
    _activeCommand = null;
    _movementState = 'idle';
    notifyListeners();
  }

  Future<void> sendManualCommand(String command) async {
    await _service.sendManualCommand(command);
  }

  Future<void> sendJoystickUpdate(
    double x,
    double y, {
    bool forceStop = false,
  }) async {
    final magnitude = math.min(1.0, math.sqrt(x * x + y * y));
    await _service.sendJoystickUpdate(
      x: x,
      y: y,
      magnitude: magnitude,
      forceStop: forceStop,
    );
  }

  void _handleTelemetry(Map<String, dynamic> message) {
    _lastUpdate = DateTime.now();
    final type = message['type'];

    if (type == 'command_ack') {
      final state = message['state']?.toString();
      _movementState = state ?? 'idle';
      if (state == 'moving') {
        _activeCommand = message['command']?.toString();
      } else if (state == 'stopped') {
        _activeCommand = null;
      }
    } else if (type == 'bluetooth_status') {
      if (message['status'] == 'disconnected') {
        _activeCommand = null;
        _movementState = 'idle';
      }
    }

    notifyListeners();
  }

  @override
  void dispose() {
    _telemetrySubscription?.cancel();
    _adapterSubscription?.cancel();
    super.dispose();
  }
}
