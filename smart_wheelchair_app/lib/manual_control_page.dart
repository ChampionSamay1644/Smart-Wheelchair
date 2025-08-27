// ignore_for_file: avoid_print

import 'package:flutter/material.dart';

// ignore: use_key_in_widget_constructors
class ManualControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Manual Control')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    print('Move Up');
                  },
                  child: Icon(Icons.arrow_upward),
                ),
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    print('Move Left');
                  },
                  child: Icon(Icons.arrow_back),
                ),
                SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () {
                    print('Move Right');
                  },
                  child: Icon(Icons.arrow_forward),
                ),
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    print('Move Down');
                  },
                  child: Icon(Icons.arrow_downward),
                ),
              ],
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
