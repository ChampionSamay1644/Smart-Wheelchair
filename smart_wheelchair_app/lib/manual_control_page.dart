// ignore_for_file: avoid_print
// ignore: use_key_in_widget_constructors
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:geolocator/geolocator.dart';
import 'core/widgets/hold_button.dart';
import 'core/providers/notifications_provider.dart';
import 'core/providers/movement_log_provider.dart';

class ManualControlPage extends StatefulWidget {
  const ManualControlPage({super.key});

  @override
  State<ManualControlPage> createState() => _ManualControlPageState();
}

class _ManualControlPageState extends State<ManualControlPage> {
  Future<Position?> _safeGetPosition(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final p = await Geolocator.getCurrentPosition().timeout(
        const Duration(seconds: 10),
      );
      if (!mounted) return null;
      return p;
    } on TimeoutException catch (_) {
      if (!mounted) return null;
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Could not get location. Please check GPS.'),
          duration: Duration(seconds: 3),
        ),
      );
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
        title: const Text('Manual Control'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.withAlpha(30), Colors.white],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildDirectionButton(
                onPressed: () async {
                  // Log movement and try to include current GPS coordinates
                  final movementLog = context.read<MovementLogProvider>();
                  final messenger = ScaffoldMessenger.of(context);
                  double? lat, lon;
                  final p = await _safeGetPosition(context);
                  if (p != null) {
                    lat = p.latitude;
                    lon = p.longitude;
                  }
                  try {
                    await movementLog.addEntry(
                      'manual',
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
                  debugPrint('Move Up');
                },
                icon: Icons.arrow_upward,
                label: 'Forward',
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildDirectionButton(
                    onPressed: () async {
                      final movementLog = context.read<MovementLogProvider>();
                      final messenger = ScaffoldMessenger.of(context);
                      double? lat, lon;
                      final p = await _safeGetPosition(context);
                      if (p != null) {
                        lat = p.latitude;
                        lon = p.longitude;
                      }
                      try {
                        await movementLog.addEntry(
                          'manual',
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
                      debugPrint('Move Left');
                    },
                    icon: Icons.arrow_back,
                    label: 'Left',
                  ),
                  const SizedBox(width: 100),
                  _buildDirectionButton(
                    onPressed: () async {
                      final movementLog = context.read<MovementLogProvider>();
                      final messenger = ScaffoldMessenger.of(context);
                      double? lat, lon;
                      final p = await _safeGetPosition(context);
                      if (p != null) {
                        lat = p.latitude;
                        lon = p.longitude;
                      }
                      try {
                        await movementLog.addEntry(
                          'manual',
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
                      debugPrint('Move Right');
                    },
                    icon: Icons.arrow_forward,
                    label: 'Right',
                  ),
                ],
              ),
              const SizedBox(height: 16),
              _buildDirectionButton(
                onPressed: () async {
                  final movementLog = context.read<MovementLogProvider>();
                  final messenger = ScaffoldMessenger.of(context);
                  double? lat, lon;
                  final p = await _safeGetPosition(context);
                  if (p != null) {
                    lat = p.latitude;
                    lon = p.longitude;
                  }
                  try {
                    await movementLog.addEntry(
                      'manual',
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
                  debugPrint('Move Down');
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
                  'Emergency triggered from Manual Control',
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

  Widget _buildDirectionButton({
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
              colors: [Colors.blue, Colors.blue.withAlpha((0.8 * 255).round())],
            ),
            borderRadius: BorderRadius.circular(40),
            boxShadow: [
              BoxShadow(
                color: Colors.blue.withAlpha((0.3 * 255).round()),
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
