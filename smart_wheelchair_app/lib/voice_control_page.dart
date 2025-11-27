// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:uuid/uuid.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

class VoiceControlPage extends StatefulWidget {
  @override
  State<VoiceControlPage> createState() => VoiceControlPageState();
}

class VoiceControlPageState extends State<VoiceControlPage> {
  // WebSocket connection
  WebSocketChannel? _channel;
  String? _sessionId;
  bool _isConnected = false;
  
  // Audio recording
  FlutterSoundRecorder? _audioRecorder;
  bool _isListening = false;
  bool _isProcessing = false;
  StreamController<Uint8List>? _audioStreamController;
  
  // Command state
  String _lastCommand = 'No command yet';
  String _connectionStatus = 'Disconnected';
  String _serverMessage = '';
  bool _motorsEnabled = false;
  
  // Configuration - UPDATE THESE VALUES
  static const String _wsServerUrl = 'ws://192.168.0.105:8765'; // Change to your server IP
  static const String _apiToken = '31a8c2806fb17e6e8a156d99f617d1f7f2dade9caab97bea3b567042b75097dd'; // Change to match server API_SECRET

  @override
  void initState() {
    super.initState();
    _initializeRecorder();
    _connectToServer();
  }

  @override
  void dispose() {
    _disconnectFromServer();
    _audioRecorder?.closeRecorder();
    super.dispose();
  }

  Future<void> _initializeRecorder() async {
    _audioRecorder = FlutterSoundRecorder();
    await _audioRecorder!.openRecorder();
  }

  // ============================================================================
  // WebSocket Connection Management
  // ============================================================================

  Future<void> _connectToServer() async {
    try {
      setState(() {
        _connectionStatus = 'Connecting...';
      });

      // Generate session ID
      _sessionId = const Uuid().v4();
      
      // TODO: For production, use Authorization header and wss://
      // Connect to WebSocket server with token authentication
      final uri = Uri.parse('$_wsServerUrl?token=$_apiToken');
      _channel = WebSocketChannel.connect(uri);

      // Listen to server messages
      _channel!.stream.listen(
        (message) {
          try {
            _handleServerMessage(message);
          } catch (e) {
            print('Error handling message: $e');
          }
        },
        onError: (error) {
          print('WebSocket error: $error');
          if (mounted) {
            setState(() {
              _connectionStatus = 'Error: $error';
              _isConnected = false;
            });
          }
        },
        onDone: () {
          print('WebSocket connection closed');
          if (mounted) {
            setState(() {
              _connectionStatus = 'Disconnected';
              _isConnected = false;
            });
          }
        },
        cancelOnError: false, // Don't cancel stream on error
      );

      // Send hello message
      final helloMessage = jsonEncode({
        'type': 'hello',
        'session_id': _sessionId,
        'user_agent': 'Flutter/Android Smart Wheelchair App v1.0',
      });
      
      _channel!.sink.add(helloMessage);
      
      print('Connected to server with session: $_sessionId');
      
    } catch (e) {
      print('Connection error: $e');
      if (mounted) {
        setState(() {
          _connectionStatus = 'Connection failed: $e';
          _isConnected = false;
        });
      }
    }
  }

  void _disconnectFromServer() {
    print('Disconnecting from server...');
    _audioStreamController?.close();
    _channel?.sink.close();
    if (mounted) {
      setState(() {
        _isConnected = false;
        _connectionStatus = 'Disconnected';
      });
    }
  }

  void _handleServerMessage(dynamic message) {
    if (!mounted) return;
    
    try {
      final data = jsonDecode(message);
      final type = data['type'];

      print('Received message type: $type');

      switch (type) {
        case 'welcome':
          if (mounted) {
            setState(() {
              _isConnected = true;
              _connectionStatus = 'Connected';
              _motorsEnabled = data['motors_enabled'] ?? false;
              _serverMessage = data['message'] ?? '';
            });
          }
          print('Successfully connected! Motors enabled: $_motorsEnabled');
          break;

        case 'audio_start_ack':
          print('Server ready to receive audio');
          break;

        case 'audio_progress':
          final chunks = data['chunks_received'];
          print('Audio progress: $chunks chunks received');
          break;

        case 'processing':
          if (mounted) {
            setState(() {
              _isProcessing = true;
              _serverMessage = data['message'] ?? 'Processing...';
            });
          }
          break;

        case 'command_result':
          if (mounted) {
            setState(() {
              _isProcessing = false;
              
              if (data['success'] == true) {
                final command = data['command'] ?? 'Unknown';
                final language = data['language'] ?? '';
                final confidence = data['confidence'] ?? 0.0;
                final action = data['action_taken'] ?? '';
                
                _lastCommand = command;
                _serverMessage = 'Command: $command (${(confidence * 100).toStringAsFixed(1)}%)';
                
                if (!_motorsEnabled && action == 'simulated') {
                  _serverMessage += '\n(Simulated - motors disabled)';
                }
                
                print('Command result: $command [$language] - confidence: $confidence');
              } else {
                _serverMessage = 'Error: ${data['error'] ?? 'Unknown error'}';
                print('Command processing failed: ${data['error']}');
              }
            });
          }
          break;

        case 'error':
          if (mounted) {
            setState(() {
              _isProcessing = false;
              _serverMessage = 'Error: ${data['message'] ?? data['error']}';
            });
          }
          print('Server error: ${data['error']} - ${data['message']}');
          break;

        case 'pong':
          print('Pong received');
          break;

        default:
          print('Unknown message type: $type');
      }
    } catch (e) {
      print('Error handling server message: $e');
    }
  }

  // ============================================================================
  // Audio Recording & Streaming
  // ============================================================================

  Future<void> _startVoiceRecording() async {
    if (!_isConnected) {
      setState(() {
        _serverMessage = 'Not connected to server';
      });
      return;
    }

    // Check if voice is enrolled
    final prefs = await SharedPreferences.getInstance();
    final isEnrolled = prefs.getBool('voice_enrolled') ?? false;
    
    if (!isEnrolled) {
      if (mounted) {
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Voice Not Enrolled'),
            content: const Text('Please enroll your voice in Settings before using voice commands.'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.pushNamed(context, '/settings');
                },
                child: const Text('Go to Settings'),
              ),
            ],
          ),
        );
      }
      return;
    }

    try {
      // Check and request permission
      var status = await Permission.microphone.request();
      if (status.isGranted) {
        setState(() {
          _isListening = true;
          _serverMessage = 'Recording...';
        });

        // Send audio_start message
        final startMessage = jsonEncode({
          'type': 'audio_start',
          'session_id': _sessionId,
        });
        _channel!.sink.add(startMessage);

        // Create stream controller
        _audioStreamController = StreamController<Uint8List>();
        
        // Listen to stream and send to WebSocket
        _audioStreamController!.stream.listen((data) {
          if (_isListening && _isConnected) {
            _channel!.sink.add(data);
          }
        });

        // Start recording to stream
        await _audioRecorder!.startRecorder(
          toStream: _audioStreamController!.sink,
          codec: Codec.pcm16,
          sampleRate: 16000,
          numChannels: 1,
        );

        print('Started voice recording and streaming');
      } else {
        print('Microphone permission denied');
        setState(() {
          _serverMessage = 'Microphone permission required';
        });
      }
    } catch (e) {
      print('Error starting recording: $e');
      setState(() {
        _isListening = false;
        _serverMessage = 'Recording error: $e';
      });
    }
  }

  Future<void> _stopVoiceRecording() async {
    try {
      setState(() {
        _isListening = false;
        _serverMessage = 'Processing...';
      });

      // Stop recording
      await _audioRecorder?.stopRecorder();
      
      // Close stream controller
      await _audioStreamController?.close();
      _audioStreamController = null;

      // Send audio_end message
      if (_isConnected && _sessionId != null) {
        final endMessage = jsonEncode({
          'type': 'audio_end',
          'session_id': _sessionId,
        });
        _channel!.sink.add(endMessage);
      }

      print('Stopped voice recording');
    } catch (e) {
      print('Error stopping recording: $e');
      setState(() {
        _serverMessage = 'Stop error: $e';
      });
    }
  }

  void _toggleVoiceRecording() {
    if (_isListening) {
      _stopVoiceRecording();
    } else {
      _startVoiceRecording();
    }
  }

  void _emergencyStop() {
    print('EMERGENCY STOP ACTIVATED');
    
    // Stop recording
    if (_isListening) {
      _stopVoiceRecording();
    }
    
    // Send emergency stop command
    if (_isConnected) {
      // You can implement a direct emergency stop message here
      setState(() {
        _serverMessage = 'EMERGENCY STOP';
      });
    }
  }

  // ============================================================================
  // UI
  // ============================================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Voice Control'),
        backgroundColor: Theme.of(context).primaryColor,
        actions: [
          // Connection status indicator
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Center(
              child: Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isConnected ? Colors.green : Colors.red,
                ),
              ),
            ),
          ),
        ],
      ),
      body: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Connection status
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _isConnected 
                    ? Colors.green.withAlpha((0.1 * 255).round())
                    : Colors.orange.withAlpha((0.1 * 255).round()),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _isConnected ? Icons.cloud_done : Icons.cloud_off,
                    color: _isConnected ? Colors.green : Colors.orange,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _connectionStatus,
                    style: TextStyle(
                      color: _isConnected ? Colors.green : Colors.orange,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Motors status warning
            if (!_motorsEnabled && _isConnected)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blue.withAlpha((0.1 * 255).round()),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.blue),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.info_outline, color: Colors.blue),
                    SizedBox(width: 8),
                    Text(
                      'SIMULATION MODE (Motors Disabled)',
                      style: TextStyle(
                        color: Colors.blue,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            
            const SizedBox(height: 32),
            
            // Microphone button
            GestureDetector(
              onTap: _isConnected && !_isProcessing ? _toggleVoiceRecording : null,
              child: Container(
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isListening
                      ? Colors.red.withAlpha((0.2 * 255).round())
                      : _isConnected
                          ? Colors.blue.withAlpha((0.1 * 255).round())
                          : Colors.grey.withAlpha((0.1 * 255).round()),
                  boxShadow: _isListening
                      ? [
                          BoxShadow(
                            color: Colors.red.withAlpha((0.5 * 255).round()),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ]
                      : null,
                ),
                child: Icon(
                  _isListening ? Icons.mic : Icons.mic_none,
                  color: _isListening
                      ? Colors.red
                      : _isConnected
                          ? Colors.blue
                          : Colors.grey,
                  size: 64,
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Status text
            Text(
              _isProcessing
                  ? 'Processing...'
                  : _isListening
                      ? 'Listening...'
                      : _isConnected
                          ? 'Tap microphone to speak'
                          : 'Connecting to server...',
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            
            const SizedBox(height: 32),
            
            // Server message
            if (_serverMessage.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.withAlpha((0.1 * 255).round()),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _serverMessage,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                  ),
                ),
              ),
            
            const SizedBox(height: 16),
            
            // Last command display
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.withAlpha((0.1 * 255).round()),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  const Text(
                    'Last Command:',
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _lastCommand,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Reconnect button
            if (!_isConnected)
              ElevatedButton.icon(
                onPressed: _connectToServer,
                icon: const Icon(Icons.refresh),
                label: const Text('Reconnect'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                ),
              ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: _emergencyStop,
        child: const Icon(Icons.warning),
      ),
    );
  }
}
