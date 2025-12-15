import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/providers/bluetooth_provider.dart';
import 'services/emergency_stop_service.dart';

class JoystickControlPage extends StatefulWidget {
  const JoystickControlPage({super.key});

  @override
  State<JoystickControlPage> createState() => _JoystickControlPageState();
}

class _JoystickControlPageState extends State<JoystickControlPage> {
  static const double _radius = 120;

  Offset _currentOffset = Offset.zero;
  Timer? _streamTimer;
  bool _touchActive = false;
  bool _isSending = false;

  @override
  void dispose() {
    _streamTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Joystick Control'),
        backgroundColor: Theme.of(context).primaryColor,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_bluetooth),
            tooltip: 'Bluetooth settings',
            onPressed: () =>
                Navigator.pushNamed(context, '/bluetooth_connection'),
          ),
        ],
      ),
      body: Consumer<BluetoothProvider>(
        builder: (context, provider, _) {
          final isConnected = provider.isConnected;
          final statusText = isConnected
              ? provider.movementState == 'moving'
                    ? 'Streaming motion via joystick'
                    : 'Bluetooth connected'
              : 'Connect to wheelchair over Bluetooth to enable joystick control';

          return Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Card(
                  child: ListTile(
                    leading: Icon(
                      Icons.sports_esports,
                      color: isConnected ? Colors.blue : Colors.grey,
                    ),
                    title: Text(statusText),
                    subtitle: isConnected
                        ? Text(
                            'Active command: ${provider.activeCommand ?? 'none'}',
                          )
                        : null,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              GestureDetector(
                onPanStart: isConnected ? _handlePanStart : null,
                onPanUpdate: isConnected ? _handlePanUpdate : null,
                onPanEnd: isConnected ? (details) => _handlePanEnd() : null,
                child: Container(
                  width: _radius * 2,
                  height: _radius * 2,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.grey.shade200,
                    boxShadow: const [
                      BoxShadow(
                        color: Colors.black12,
                        blurRadius: 12,
                        offset: Offset(0, 6),
                      ),
                    ],
                  ),
                  child: Center(
                    child: Transform.translate(
                      offset: _currentOffset,
                      child: Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          color: isConnected ? Colors.blueAccent : Colors.grey,
                          shape: BoxShape.circle,
                          boxShadow: const [
                            BoxShadow(
                              color: Colors.black26,
                              blurRadius: 8,
                              offset: Offset(0, 4),
                            ),
                          ],
                        ),
                        child: const Icon(
                          Icons.control_camera,
                          color: Colors.white,
                          size: 36,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              Text(
                isConnected
                    ? _buildVectorSummary()
                    : 'Bluetooth joystick disabled',
                style: TextStyle(
                  color: isConnected ? Colors.black87 : Colors.redAccent,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: () async {
          final messenger = ScaffoldMessenger.of(context);
          await EmergencyStopService.trigger();
          if (!mounted) return;
          messenger.showSnackBar(
            const SnackBar(content: Text('Emergency stop sent')),
          );
        },
        child: const Icon(Icons.warning),
      ),
    );
  }

  void _handlePanStart(DragStartDetails details) {
    _touchActive = true;
    _updateOffset(details.localPosition);
    _startStreaming();
    _sendUpdate();
  }

  void _handlePanUpdate(DragUpdateDetails details) {
    _updateOffset(details.localPosition);
    _sendUpdate();
  }

  void _handlePanEnd() {
    _touchActive = false;
    setState(() {
      _currentOffset = Offset.zero;
    });
    _stopStreaming();
    _sendUpdate(forceStop: true);
  }

  void _updateOffset(Offset localPosition) {
    final center = Offset(_radius, _radius);
    final delta = localPosition - center;
    final distance = math.min(delta.distance, _radius);
    final direction = delta.distance == 0
        ? Offset.zero
        : Offset(delta.dx / delta.distance, delta.dy / delta.distance);
    setState(() {
      _currentOffset = direction * distance;
    });
  }

  void _startStreaming() {
    _streamTimer?.cancel();
    _streamTimer = Timer.periodic(
      const Duration(milliseconds: 50),
      (_) => _sendUpdate(),
    );
  }

  void _stopStreaming() {
    _streamTimer?.cancel();
    _streamTimer = null;
  }

  Future<void> _sendUpdate({bool forceStop = false}) async {
    if (_isSending) {
      return;
    }
    _isSending = true;
    final provider = context.read<BluetoothProvider>();
    if (!provider.isConnected) {
      _isSending = false;
      return;
    }

    final shouldRequestStop = forceStop || !_touchActive;
    final normalized = shouldRequestStop
        ? Offset.zero
        : Offset(
            (_currentOffset.dx / _radius).clamp(-1.0, 1.0),
            (-_currentOffset.dy / _radius).clamp(-1.0, 1.0),
          );

    try {
      await provider.sendJoystickUpdate(
        normalized.dx,
        normalized.dy,
        forceStop: shouldRequestStop,
      );
    } catch (e) {
      if (mounted) {
        final messenger = ScaffoldMessenger.of(context);
        messenger.showSnackBar(
          SnackBar(content: Text('Failed to stream joystick data: $e')),
        );
      }
      _stopStreaming();
    } finally {
      _isSending = false;
    }
  }

  String _buildVectorSummary() {
    final x = (_currentOffset.dx / _radius).clamp(-1.0, 1.0);
    final y = (-_currentOffset.dy / _radius).clamp(-1.0, 1.0);
    final magnitude = math.min(1.0, math.sqrt(x * x + y * y));
    return 'Vector: x=${x.toStringAsFixed(2)} y=${y.toStringAsFixed(2)} | mag=${magnitude.toStringAsFixed(2)}';
  }
}
