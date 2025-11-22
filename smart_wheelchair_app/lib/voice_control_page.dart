// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

class VoiceControlPage extends StatefulWidget {
  @override
  State<VoiceControlPage> createState() => VoiceControlPageState();
}

class VoiceControlPageState extends State<VoiceControlPage> {
  bool _isListening = false;
  String _lastCommand = 'No command yet';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Voice Control'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isListening
                    ? Colors.red.withAlpha((0.2 * 255).round())
                    : Colors.blue.withAlpha((0.1 * 255).round()),
              ),
              child: IconButton(
                iconSize: 64,
                icon: Icon(
                  _isListening ? Icons.mic : Icons.mic_none,
                  color: _isListening ? Colors.red : Colors.blue,
                ),
                onPressed: () {
                  setState(() {
                    _isListening = !_isListening;
                    if (_isListening) {
                      print('Voice Recognition Started');
                      // Simulate receiving a command after 2 seconds
                      Future.delayed(const Duration(seconds: 2), () {
                        if (_isListening) {
                          // Check if still listening
                          setState(() {
                            _lastCommand = 'Move Forward';
                            print('Command received: $_lastCommand');
                          });
                        }
                      });
                    } else {
                      print('Voice Recognition Stopped');
                    }
                  });
                },
              ),
            ),
            const SizedBox(height: 32),
            Text(
              _isListening ? 'Listening...' : 'Tap microphone to start',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 32),
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
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: () {
          print('EMERGENCY STOP ACTIVATED');
        },
        child: Icon(Icons.warning),
      ),
    );
  }
}
