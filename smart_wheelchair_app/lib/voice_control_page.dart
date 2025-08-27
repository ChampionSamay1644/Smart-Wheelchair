// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

class VoiceControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Voice Control')),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          ElevatedButton(
            onPressed: () {
              print('Voice Recognition Started');
            },
            child: Text('Start/Stop Voice Recognition'),
          ),
          SizedBox(height: 20),
          Text('Recognized Command: Forward'), // Placeholder text
        ],
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
