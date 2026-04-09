import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../services/bluetooth_service.dart';
import '../../services/api_service.dart';
import 'api_provider.dart';

class BluetoothProvider extends ChangeNotifier {
  BluetoothProvider(this._apiProvider) {
    _init();
  }

  final ApiProvider _apiProvider;
  final ApiService _apiService = ApiService();
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

  void _broadcastCommand(Map<String, dynamic> commandData) {
    if (_apiProvider.selectedDeviceId != null) {
      final payload = {
        'deviceId': _apiProvider.selectedDeviceId,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'motorStatus': commandData,
      };
      
      _apiService.uploadData(payload).then((_) {
        debugPrint('☁️ Command Sync: Uploaded ${commandData['lastCommand']} to cloud');
      }).catchError((Object e) {
        debugPrint('Cloud command upload failed: $e');
      });
    }
  }

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
    _broadcastCommand({
      'lastCommand': command,
      'mode': 'manual',
    });
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

    // Only sync to cloud if it's a significant change or stop
    // (Actual debouncing is handled in WheelchairBluetoothService, but we'll sync the result here)
    if (forceStop) {
      _broadcastCommand({'lastCommand': 'stop', 'mode': 'joystick'});
    } else {
      // We can look at the service's classification if we expose it, or just sync periodically
      // For now, let's sync the raw intent
      _broadcastCommand({
        'lastCommand': 'moving', 
        'mode': 'joystick',
        'x': x,
        'y': y
      });
    }
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
