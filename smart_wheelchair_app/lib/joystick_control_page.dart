// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

class JoystickControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Joystick Control')),
      body: Center(child: Text('Joystick Control Page')),
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
