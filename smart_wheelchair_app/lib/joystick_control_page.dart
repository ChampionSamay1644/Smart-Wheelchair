// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:geolocator/geolocator.dart';
import 'core/widgets/hold_button.dart';
import 'core/providers/notifications_provider.dart';
import 'core/providers/movement_log_provider.dart';

class JoystickControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Drive Control'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildDirectionButton(
                context,
                onPressed: () async {
                  final movementLog = context.read<MovementLogProvider>();
                  double? lat, lon;
                  try {
                    final p = await Geolocator.getCurrentPosition();
                    lat = p.latitude;
                    lon = p.longitude;
                  } catch (_) {}
                  try {
                    movementLog.addEntry(
                      'drive',
                      'forward',
                      lat: lat,
                      lon: lon,
                    );
                  } catch (_) {}
                  print('Drive Forward');
                },
                icon: Icons.arrow_upward,
                label: 'Forward',
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildDirectionButton(
                    context,
                    onPressed: () async {
                      final movementLog = context.read<MovementLogProvider>();
                      double? lat, lon;
                      try {
                        final p = await Geolocator.getCurrentPosition();
                        lat = p.latitude;
                        lon = p.longitude;
                      } catch (_) {}
                      try {
                        movementLog.addEntry(
                          'drive',
                          'left',
                          lat: lat,
                          lon: lon,
                        );
                      } catch (_) {}
                      print('Drive Left');
                    },
                    icon: Icons.arrow_back,
                    label: 'Left',
                  ),
                  const SizedBox(width: 80),
                  _buildDirectionButton(
                    context,
                    onPressed: () async {
                      final movementLog = context.read<MovementLogProvider>();
                      double? lat, lon;
                      try {
                        final p = await Geolocator.getCurrentPosition();
                        lat = p.latitude;
                        lon = p.longitude;
                      } catch (_) {}
                      try {
                        movementLog.addEntry(
                          'drive',
                          'right',
                          lat: lat,
                          lon: lon,
                        );
                      } catch (_) {}
                      print('Drive Right');
                    },
                    icon: Icons.arrow_forward,
                    label: 'Right',
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildDirectionButton(
                context,
                onPressed: () async {
                  final movementLog = context.read<MovementLogProvider>();
                  double? lat, lon;
                  try {
                    final p = await Geolocator.getCurrentPosition();
                    lat = p.latitude;
                    lon = p.longitude;
                  } catch (_) {}
                  try {
                    movementLog.addEntry(
                      'drive',
                      'backward',
                      lat: lat,
                      lon: lon,
                    );
                  } catch (_) {}
                  print('Drive Backward');
                },
                icon: Icons.arrow_downward,
                label: 'Backward',
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: SizedBox(
        height: 64,
        width: 64,
        child: FloatingActionButton(
          backgroundColor: Colors.red,
          onPressed: null,
          child: HoldButton(
            holdDuration: const Duration(seconds: 2),
            onHold: () {
              try {
                context.read<NotificationsProvider>().addEvent(
                  'Emergency',
                  'Emergency triggered from Drive Control',
                );
              } catch (_) {}
              print('EMERGENCY STOP ACTIVATED');
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('EMERGENCY STOP ACTIVATED'),
                  backgroundColor: Colors.red,
                  duration: Duration(seconds: 2),
                ),
              );
            },
            child: const Icon(Icons.warning),
          ),
        ),
      ),
    );
  }

  Widget _buildDirectionButton(
    BuildContext context, {
    required VoidCallback onPressed,
    required IconData icon,
    required String label,
  }) {
    return Column(
      children: [
        Container(
          width: 80,
          height: 80,
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.purple,
                Colors.purple.withAlpha((0.8 * 255).round()),
              ],
            ),
            borderRadius: BorderRadius.circular(40),
            boxShadow: [
              BoxShadow(
                color: Colors.purple.withAlpha((0.3 * 255).round()),
                blurRadius: 8,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              onTap: onPressed,
              borderRadius: BorderRadius.circular(40),
              child: Icon(icon, size: 36, color: Colors.white),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
