// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

class VoiceEnrollmentPage extends StatefulWidget {
  @override
  State<VoiceEnrollmentPage> createState() => _VoiceEnrollmentPageState();
}

class _VoiceEnrollmentPageState extends State<VoiceEnrollmentPage> {
  final TextEditingController _nameController = TextEditingController();
  FlutterSoundRecorder? _audioRecorder;
  WebSocketChannel? _channel;
  bool _isRecording = false;
  bool _isUploading = false;
  StreamController<Uint8List>? _audioStreamController;
  List<Uint8List> _audioChunks = [];
  String _statusMessage = 'Enter your name and record your voice';
  
  // Configuration - MATCH websocket_server.py
  static const String _wsServerUrl = 'ws://192.168.0.105:8765';
  static const String _apiToken = '31a8c2806fb17e6e8a156d99f617d1f7f2dade9caab97bea3b567042b75097dd';

  @override
  void initState() {
    super.initState();
    _initializeRecorder();
    _loadUserName();
  }

  @override
  void dispose() {
    _audioRecorder?.closeRecorder();
    _channel?.sink.close();
    _audioStreamController?.close();
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _loadUserName() async {
    final prefs = await SharedPreferences.getInstance();
    final savedName = prefs.getString('user_voice_name');
    if (savedName != null && mounted) {
      setState(() {
        _nameController.text = savedName;
      });
    }
  }

  Future<void> _initializeRecorder() async {
    _audioRecorder = FlutterSoundRecorder();
    
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      if (mounted) {
        setState(() {
          _statusMessage = 'Microphone permission denied';
        });
      }
      return;
    }

    try {
      await _audioRecorder!.openRecorder();
      print('Voice enrollment recorder initialized');
    } catch (e) {
      print('Failed to initialize recorder: $e');
      if (mounted) {
        setState(() {
          _statusMessage = 'Failed to initialize recorder: $e';
        });
      }
    }
  }

  Future<void> _startRecording() async {
    if (_nameController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter your name first')),
      );
      return;
    }

    _audioChunks.clear();
    _audioStreamController = StreamController<Uint8List>();
    
    _audioStreamController!.stream.listen(
      (buffer) {
        _audioChunks.add(buffer);
      },
      cancelOnError: false,
    );

    await _audioRecorder!.startRecorder(
      toStream: _audioStreamController!.sink,
      codec: Codec.pcm16,
      numChannels: 1,
      sampleRate: 16000,
    );

    if (mounted) {
      setState(() {
        _isRecording = true;
        _statusMessage = 'Recording... Say: "Hello, my name is ${_nameController.text}"';
      });
    }
  }

  Future<void> _stopRecording() async {
    await _audioRecorder!.stopRecorder();
    await _audioStreamController?.close();

    if (mounted) {
      setState(() {
        _isRecording = false;
        _statusMessage = 'Recording complete. Uploading...';
      });
    }

    await _uploadVoiceSample();
  }

  Future<void> _uploadVoiceSample() async {
    if (_audioChunks.isEmpty) {
      if (mounted) {
        setState(() {
          _statusMessage = 'No audio recorded';
        });
      }
      return;
    }

    setState(() {
      _isUploading = true;
    });

    try {
      // Connect to WebSocket
      final uri = Uri.parse('$_wsServerUrl?token=$_apiToken');
      _channel = WebSocketChannel.connect(uri);

      // Send enrollment request
      _channel!.sink.add(jsonEncode({
        'type': 'enroll_voice',
        'user_name': _nameController.text.trim(),
        'audio_length': _audioChunks.length,
      }));

      // Send audio chunks
      for (var chunk in _audioChunks) {
        _channel!.sink.add(chunk);
      }

      // Send completion message
      _channel!.sink.add(jsonEncode({
        'type': 'enroll_complete',
      }));

      // Listen for response
      _channel!.stream.listen((message) {
        final data = jsonDecode(message);
        if (data['type'] == 'enroll_success') {
          _saveUserName();
          if (mounted) {
            setState(() {
              _statusMessage = 'Voice enrolled successfully!';
              _isUploading = false;
            });
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Voice enrolled for ${_nameController.text}'),
                backgroundColor: Colors.green,
              ),
            );
          }
        } else if (data['type'] == 'enroll_error') {
          if (mounted) {
            setState(() {
              _statusMessage = 'Error: ${data['message']}';
              _isUploading = false;
            });
          }
        }
      });

    } catch (e) {
      print('Upload error: $e');
      if (mounted) {
        setState(() {
          _statusMessage = 'Upload failed: $e';
          _isUploading = false;
        });
      }
    }
  }

  Future<void> _saveUserName() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_voice_name', _nameController.text.trim());
    await prefs.setBool('voice_enrolled', true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Voice Enrollment'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 20),
            const Icon(
              Icons.record_voice_over,
              size: 80,
              color: Colors.blue,
            ),
            const SizedBox(height: 24),
            const Text(
              'Enroll Your Voice',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'Record your voice to enable authenticated voice commands',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Your Name',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.person),
              ),
              enabled: !_isRecording && !_isUploading,
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 13),
              ),
            ),
            const SizedBox(height: 24),
            if (_isUploading)
              const CircularProgressIndicator()
            else
              ElevatedButton.icon(
                onPressed: _isRecording ? _stopRecording : _startRecording,
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  backgroundColor: _isRecording ? Colors.red : Colors.blue,
                  foregroundColor: Colors.white,
                ),
              ),
            const SizedBox(height: 16),
            const Text(
              'Tip: Say "Hello, my name is [Your Name]" clearly for 2-3 seconds',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, fontStyle: FontStyle.italic, color: Colors.grey),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
