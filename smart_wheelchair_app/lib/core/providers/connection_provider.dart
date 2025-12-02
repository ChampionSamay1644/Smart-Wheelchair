import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../services/wheelchair_websocket_service.dart';

class ConnectionProvider extends ChangeNotifier {
  ConnectionProvider() {
    _init();
  }

  final WheelchairWebSocketService _wsService = WheelchairWebSocketService();
  SharedPreferences? _prefs;
  StreamSubscription<Map<String, dynamic>>? _messageSubscription;

  String _ipAddress = '';
  int _port = 8765;
  bool _isConnecting = false;
  bool _initialized = false;
  String? _lastError;

  bool get isInitialized => _initialized;
  bool get isConnected => _wsService.isConnected;
  bool get isConnecting => _isConnecting;
  String? get lastError => _lastError;
  String get ipAddress => _ipAddress;
  int get port => _port;
  bool get hasValidConfig => _ipAddress.isNotEmpty;

  Stream<Map<String, dynamic>> get messageStream => _wsService.messageStream;

  Future<void> _init() async {
    _prefs = await SharedPreferences.getInstance();
    _ipAddress = _prefs?.getString('raspberry_pi_ip')?.trim() ?? '';
    _port = _prefs?.getInt('websocket_port') ?? 8765;
    _subscribeToMessages();
    _initialized = true;
    notifyListeners();

    if (_ipAddress.isNotEmpty) {
      await connect(ip: _ipAddress, port: _port, autoAttempt: true);
    }
  }

  Future<bool> connect({
    required String ip,
    required int port,
    bool autoAttempt = false,
  }) async {
    final trimmedIp = ip.trim();
    if (trimmedIp.isEmpty) {
      _lastError = 'Enter a valid IP address';
      notifyListeners();
      return false;
    }

    _ipAddress = trimmedIp;
    _port = port;
    _isConnecting = true;
    _lastError = null;
    notifyListeners();

    final success = await _wsService.connect(
      trimmedIp,
      port,
      autoReconnect: true,
    );

    _isConnecting = false;
    if (success) {
      await _prefs?.setString('raspberry_pi_ip', _ipAddress);
      await _prefs?.setInt('websocket_port', _port);
      _lastError = null;
      _subscribeToMessages();
      notifyListeners();
    } else {
      if (!autoAttempt) {
        _lastError = 'Failed to connect to $trimmedIp:$port';
      }
      notifyListeners();
    }

    return success;
  }

  Future<void> disconnect({bool userInitiated = false}) async {
    _wsService.setAutoReconnect(false);
    await _wsService.disconnect(permanent: true);
    _isConnecting = false;
    if (userInitiated) {
      _lastError = null;
    }
    notifyListeners();
  }

  void _subscribeToMessages() {
    _messageSubscription?.cancel();
    _messageSubscription = _wsService.messageStream.listen(
      (message) {
        final type = message['type'] as String?;
        switch (type) {
          case 'connection':
            _isConnecting = false;
            _lastError = null;
            notifyListeners();
            break;
          case 'disconnected':
            _isConnecting = true;
            _lastError = message['message']?.toString();
            notifyListeners();
            break;
          case 'connection_failed':
            _isConnecting = false;
            _lastError = message['message']?.toString();
            notifyListeners();
            break;
        }
      },
      onError: (error) {
        _isConnecting = false;
        _lastError = error.toString();
        notifyListeners();
      },
    );
  }

  @override
  void dispose() {
    _messageSubscription?.cancel();
    super.dispose();
  }
}
