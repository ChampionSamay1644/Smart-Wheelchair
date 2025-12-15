import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// WebSocket service for communicating with the Raspberry Pi wheelchair control server
class WheelchairWebSocketService {
  WheelchairWebSocketService._internal();

  static final WheelchairWebSocketService _instance =
      WheelchairWebSocketService._internal();

  factory WheelchairWebSocketService() => _instance;

  // WebSocket connection
  WebSocketChannel? _channel;
  final StreamController<Map<String, dynamic>> _messageController =
      StreamController<Map<String, dynamic>>.broadcast();

  // Connection state
  bool _isConnected = false;
  bool _isConnecting = false;
  String? _websocketUrl;
  String? _lastIp;
  int? _lastPort;

  // Auto reconnect handling
  bool _shouldAutoReconnect = true;
  bool _intentionalDisconnect = false;
  Timer? _reconnectTimer;
  final Duration _reconnectDelay = const Duration(seconds: 5);
  final int _maxReconnectAttempts = 2;
  int _reconnectAttempts = 0;

  // Session management
  String? _currentSessionId;

  /// Get the message stream
  Stream<Map<String, dynamic>> get messageStream => _messageController.stream;

  /// Check if connected
  bool get isConnected => _isConnected;

  /// Check if a connection attempt is in progress
  bool get isConnecting => _isConnecting;

  /// Get current session ID
  String? get currentSessionId => _currentSessionId;

  /// Connect to the WebSocket server
  Future<bool> connect(
    String ipAddress,
    int port, {
    bool autoReconnect = true,
    bool resetAttempts = true,
  }) async {
    if (_isConnected) {
      debugPrint('Already connected to WebSocket server');
      return true;
    }
    if (_isConnecting) {
      debugPrint('Connection attempt already in progress');
      return false;
    }

    if (resetAttempts) {
      _reconnectAttempts = 0;
    }
    _shouldAutoReconnect = autoReconnect;
    _intentionalDisconnect = false;
    _isConnecting = true;
    _lastIp = ipAddress;
    _lastPort = port;
    _reconnectTimer?.cancel();

    try {
      _websocketUrl = 'ws://$ipAddress:$port';
      debugPrint('Connecting to $_websocketUrl...');

      // Create WebSocket connection
      _channel = WebSocketChannel.connect(Uri.parse(_websocketUrl!));

      // Listen to incoming messages
      _channel!.stream.listen(
        (message) {
          try {
            final data = jsonDecode(message as String) as Map<String, dynamic>;
            debugPrint('Received message: ${data['type']}');
            _messageController.add(data);
          } catch (e) {
            debugPrint('Error parsing message: $e');
          }
        },
        onError: (error) {
          debugPrint('WebSocket error: $error');
          _emitDisconnected('Connection error: $error');
        },
        onDone: () {
          debugPrint('WebSocket connection closed');
          _emitDisconnected('Connection closed');
        },
        cancelOnError: false,
      );

      _isConnected = true;
      _isConnecting = false;
      _reconnectAttempts = 0;
      _reconnectTimer?.cancel();
      debugPrint('✓ Connected to wheelchair control server');
      return true;
    } catch (e) {
      debugPrint('Failed to connect: $e');
      _isConnected = false;
      _isConnecting = false;
      _messageController.add({
        'type': 'connection_failed',
        'message': 'Failed to connect: $e',
      });
      _scheduleReconnect();
      return false;
    }
  }

  void _emitDisconnected(String reason) {
    if (_isConnected) {
      _messageController.add({'type': 'disconnected', 'message': reason});
    }

    _isConnected = false;
    _isConnecting = false;

    if (_intentionalDisconnect) {
      return;
    }

    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (!_shouldAutoReconnect) {
      return;
    }
    if (_lastIp == null || _lastPort == null) {
      return;
    }
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      _shouldAutoReconnect = false;
      _isConnecting = false;
      _messageController.add({
        'type': 'reconnect_exhausted',
        'message':
            'Unable to connect after $_maxReconnectAttempts attempts. Please check the IP address and try again.',
      });
      return;
    }

    _reconnectAttempts += 1;
    _isConnecting = true;
    _messageController.add({
      'type': 'reconnecting',
      'attempt': _reconnectAttempts,
      'max_attempts': _maxReconnectAttempts,
      'message': 'Attempting reconnect ($_reconnectAttempts/$_maxReconnectAttempts)...',
    });
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(_reconnectDelay, () async {
      if (_intentionalDisconnect) {
        return;
      }
      await connect(
        _lastIp!,
        _lastPort!,
        autoReconnect: _shouldAutoReconnect,
        resetAttempts: false,
      );
    });
  }

  /// Disconnect from the WebSocket server
  Future<void> disconnect({bool permanent = true}) async {
    _intentionalDisconnect = permanent;
    if (permanent) {
      _shouldAutoReconnect = false;
    }
    _reconnectTimer?.cancel();
    _reconnectAttempts = 0;

    if (_channel != null) {
      await _channel!.sink.close();
      _channel = null;
    }

    _isConnected = false;
    _isConnecting = false;
    _currentSessionId = null;
    debugPrint('Disconnected from WebSocket server');
  }

  /// Enable or disable automatic reconnection without closing the connection.
  void setAutoReconnect(bool enabled) {
    _shouldAutoReconnect = enabled;
    if (!enabled) {
      _reconnectTimer?.cancel();
      _reconnectAttempts = 0;
    }
  }

  /// Send a control message
  void _sendMessage(Map<String, dynamic> message) {
    if (!_isConnected || _channel == null) {
      debugPrint('Cannot send message: Not connected');
      return;
    }

    try {
      final jsonMessage = jsonEncode(message);
      _channel!.sink.add(jsonMessage);
      debugPrint('Sent message: ${message['type']}');
    } catch (e) {
      debugPrint('Error sending message: $e');
    }
  }

  /// Start a recording session
  void startRecording() {
    _currentSessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _sendMessage({'type': 'start_recording', 'session_id': _currentSessionId});
  }

  /// Send an audio chunk
  void sendAudioChunk(Uint8List audioData) {
    if (!_isConnected || _channel == null) {
      debugPrint('Cannot send audio: Not connected');
      return;
    }

    try {
      // Send binary data directly
      _channel!.sink.add(audioData);
      // debugPrint('Sent audio chunk: ${audioData.length} bytes');
    } catch (e) {
      debugPrint('Error sending audio chunk: $e');
    }
  }

  /// Stop recording and process the command
  void stopRecording() {
    _sendMessage({'type': 'stop_recording'});
    _currentSessionId = null;
  }

  /// Cancel the current recording
  void cancelRecording() {
    _sendMessage({'type': 'cancel_recording'});
    _currentSessionId = null;
  }

  /// Send emergency stop command
  void emergencyStop() {
    _sendMessage({'type': 'emergency_stop'});
  }

  /// Send ping to check connection
  void ping() {
    _sendMessage({'type': 'ping'});
  }

  /// Request voice profile status from the server.
  void checkVoiceProfileStatus({String? speakerName}) {
    final payload = <String, dynamic>{'type': 'check_voice_profile'};
    if (speakerName != null && speakerName.isNotEmpty) {
      payload['speaker_name'] = speakerName;
    }
    _sendMessage(payload);
  }

  /// Start voice enrollment
  void startVoiceEnrollment(
    String speakerName,
    String gender, {
    int sampleIndex = 1,
    int totalSamples = 3,
    String? prompt,
  }) {
    _currentSessionId = DateTime.now().millisecondsSinceEpoch.toString();
    _sendMessage({
      'type': 'start_voice_enrollment',
      'session_id': _currentSessionId,
      'speaker_name': speakerName,
      'gender': gender,
      'sample_index': sampleIndex,
      'total_samples': totalSamples,
      if (prompt != null) 'prompt': prompt,
    });
  }

  /// Stop voice enrollment and process
  void stopVoiceEnrollment(
    String speakerName,
    String gender, {
    int sampleIndex = 1,
    int totalSamples = 3,
    bool finalize = false,
  }) {
    _sendMessage({
      'type': 'stop_voice_enrollment',
      'speaker_name': speakerName,
      'gender': gender,
      'sample_index': sampleIndex,
      'total_samples': totalSamples,
      'finalize': finalize,
    });
    _currentSessionId = null;
  }

  /// Finalize enrollment with collected samples
  void finishVoiceEnrollment(String speakerName) {
    _sendMessage({
      'type': 'finish_voice_enrollment',
      'speaker_name': speakerName,
    });
  }
}
