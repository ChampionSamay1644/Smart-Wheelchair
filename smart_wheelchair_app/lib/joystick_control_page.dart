// ignore_for_file: use_key_in_widget_constructors

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:geolocator/geolocator.dart';
import 'core/widgets/hold_button.dart';
import 'core/providers/notifications_provider.dart';
import 'core/providers/movement_log_provider.dart';

class JoystickControlPage extends StatefulWidget {
  const JoystickControlPage({super.key});

  @override
  State<JoystickControlPage> createState() => _JoystickControlPageState();
}

class _JoystickControlPageState extends State<JoystickControlPage> {
  Future<Position?> _safeGetPosition() async {
    try {
      final p = await Geolocator.getCurrentPosition().timeout(
        const Duration(seconds: 10),
      );
      if (!mounted) return null;
      return p;
    } on TimeoutException catch (_) {
      debugPrint('Location timeout');
      return null;
    } catch (e) {
      debugPrint('Location error: $e');
      return null;
    }
  }

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
                  final messenger = ScaffoldMessenger.of(context);
                  double? lat, lon;
                  final p = await _safeGetPosition();
                  if (!mounted) return;
                  if (p == null) {
                    messenger.showSnackBar(
                      const SnackBar(
                        content: Text(
                          'Could not get location. Please check GPS.',
                        ),
                        duration: Duration(seconds: 3),
                      ),
                    );
                  } else {
                    lat = p.latitude;
                    lon = p.longitude;
                  }
                  try {
                    await movementLog.addEntry(
                      'drive',
                      'forward',
                      lat: lat,
                      lon: lon,
                    );
                  } catch (e) {
                    debugPrint('Movement log error: $e');
                    if (!mounted) return;
                    messenger.showSnackBar(
                      const SnackBar(
                        content: Text('Could not log movement.'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  }
                  debugPrint('Drive Forward');
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
                      final messenger = ScaffoldMessenger.of(context);
                      double? lat, lon;
                      final p = await _safeGetPosition();
                      if (!mounted) return;
                      if (p == null) {
                        messenger.showSnackBar(
                          const SnackBar(
                            content: Text(
                              'Could not get location. Please check GPS.',
                            ),
                            duration: Duration(seconds: 3),
                          ),
                        );
                      } else {
                        lat = p.latitude;
                        lon = p.longitude;
                      }
                      try {
                        await movementLog.addEntry(
                          'drive',
                          'left',
                          lat: lat,
                          lon: lon,
                        );
                      } catch (e) {
                        debugPrint('Movement log error: $e');
                        if (!mounted) return;
                        messenger.showSnackBar(
                          const SnackBar(
                            content: Text('Could not log movement.'),
                            duration: Duration(seconds: 2),
                          ),
                        );
                      }
                      debugPrint('Drive Left');
                    },
                    icon: Icons.arrow_back,
                    label: 'Left',
                  ),
                  const SizedBox(width: 80),
                  _buildDirectionButton(
                    context,
                    onPressed: () async {
                      final movementLog = context.read<MovementLogProvider>();
                      final messenger = ScaffoldMessenger.of(context);
                      double? lat, lon;
                      final p = await _safeGetPosition();
                      if (!mounted) return;
                      if (p == null) {
                        messenger.showSnackBar(
                          const SnackBar(
                            content: Text(
                              'Could not get location. Please check GPS.',
                            ),
                            duration: Duration(seconds: 3),
                          ),
                        );
                      } else {
                        lat = p.latitude;
                        lon = p.longitude;
                      }
                      try {
                        await movementLog.addEntry(
                          'drive',
                          'right',
                          lat: lat,
                          lon: lon,
                        );
                      } catch (e) {
                        debugPrint('Movement log error: $e');
                        if (!mounted) return;
                        messenger.showSnackBar(
                          const SnackBar(
                            content: Text('Could not log movement.'),
                            duration: Duration(seconds: 2),
                          ),
                        );
                      }
                      debugPrint('Drive Right');
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
                  final messenger = ScaffoldMessenger.of(context);
                  double? lat, lon;
                  final p = await _safeGetPosition();
                  if (!mounted) return;
                  if (p == null) {
                    messenger.showSnackBar(
                      const SnackBar(
                        content: Text(
                          'Could not get location. Please check GPS.',
                        ),
                        duration: Duration(seconds: 3),
                      ),
                    );
                  } else {
                    lat = p.latitude;
                    lon = p.longitude;
                  }
                  try {
                    await movementLog.addEntry(
                      'drive',
                      'backward',
                      lat: lat,
                      lon: lon,
                    );
                  } catch (e) {
                    debugPrint('Movement log error: $e');
                    if (!mounted) return;
                    messenger.showSnackBar(
                      const SnackBar(
                        content: Text('Could not log movement.'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  }
                  debugPrint('Drive Backward');
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
              debugPrint('EMERGENCY STOP ACTIVATED');
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
