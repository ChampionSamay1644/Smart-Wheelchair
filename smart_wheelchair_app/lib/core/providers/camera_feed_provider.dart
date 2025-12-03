import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../../services/wheelchair_websocket_service.dart';

class CameraFeedProvider extends ChangeNotifier {
  CameraFeedProvider() {
    _subscription = WheelchairWebSocketService().messageStream.listen(
      _handleMessage,
      onError: (Object error, StackTrace stackTrace) {
        _handleError(error, stackTrace);
      },
    );
  }

  StreamSubscription<Map<String, dynamic>>? _subscription;
  Uint8List? _latestFrame;
  String? _streamName;
  DateTime? _lastUpdated;
  String? _error;

  Uint8List? get latestFrame => _latestFrame;
  String? get streamName => _streamName;
  DateTime? get lastUpdated => _lastUpdated;
  String? get error => _error;
  bool get hasFrame => _latestFrame != null;

  void _handleMessage(Map<String, dynamic> message) {
    final type = message['type'] as String?;
    if (type == null) {
      return;
    }

    switch (type) {
      case 'camera_frame':
        final data = message['data'];
        if (data is! String || data.isEmpty) {
          return;
        }
        try {
          final decoded = base64Decode(data);
          _latestFrame = decoded;
          _streamName = message['stream']?.toString();
          _lastUpdated = DateTime.now();
          _error = null;
          notifyListeners();
        } catch (e) {
          _error = 'Failed to decode camera frame';
          notifyListeners();
        }
        break;
      case 'disconnected':
      case 'reconnect_exhausted':
      case 'camera_unregistered':
        _latestFrame = null;
        _lastUpdated = null;
        _error = message['message']?.toString();
        notifyListeners();
        break;
      case 'connection':
      case 'reconnecting':
        _error = null;
        break;
    }
  }

  void _handleError(Object error, [StackTrace? stackTrace]) {
    _error = error.toString();
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    super.dispose();
  }
}
