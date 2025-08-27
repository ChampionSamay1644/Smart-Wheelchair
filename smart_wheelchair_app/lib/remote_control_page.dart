// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

class RemoteControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Remote Control')),
      body: ListView(
        children: [
          ListTile(
            title: Text('Go to Kitchen'),
            onTap: () {
              print('Selected: Go to Kitchen');
            },
          ),
          ListTile(
            title: Text('Go to Bedroom'),
            onTap: () {
              print('Selected: Go to Bedroom');
            },
          ),
          ListTile(
            title: Text('Go to Living Room'),
            onTap: () {
              print('Selected: Go to Living Room');
            },
          ),
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
