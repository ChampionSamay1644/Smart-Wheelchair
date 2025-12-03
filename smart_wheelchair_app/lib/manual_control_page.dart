import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'core/providers/bluetooth_provider.dart';
import 'services/emergency_stop_service.dart';

class ManualControlPage extends StatelessWidget {
  const ManualControlPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Manual Control'),
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
      body: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.withAlpha(30), Colors.white],
          ),
        ),
        child: Consumer<BluetoothProvider>(
          builder: (context, provider, _) {
            final isConnected = provider.isConnected;
            final movementState = provider.movementState;
            final activeCommand = provider.activeCommand;

            return Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _StatusBanner(
                  isConnected: isConnected,
                  movementState: movementState,
                  activeCommand: activeCommand,
                ),
                const SizedBox(height: 24),
                _buildDirectionButton(
                  context,
                  icon: Icons.arrow_upward,
                  label: 'Forward',
                  command: 'forward',
                  enabled: isConnected && activeCommand != 'forward',
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildDirectionButton(
                      context,
                      icon: Icons.arrow_back,
                      label: 'Left',
                      command: 'left',
                      enabled: isConnected && activeCommand != 'left',
                    ),
                    const SizedBox(width: 80),
                    _buildDirectionButton(
                      context,
                      icon: Icons.arrow_forward,
                      label: 'Right',
                      command: 'right',
                      enabled: isConnected && activeCommand != 'right',
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _buildDirectionButton(
                  context,
                  icon: Icons.arrow_downward,
                  label: 'Backward',
                  command: 'backward',
                  enabled: isConnected && activeCommand != 'backward',
                ),
                const SizedBox(height: 32),
                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isConnected ? Colors.orange : Colors.grey,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 32,
                      vertical: 16,
                    ),
                  ),
                  onPressed: isConnected
                      ? () => _sendCommand(context, 'stop')
                      : null,
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('Stop'),
                ),
              ],
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: () async {
          await EmergencyStopService.trigger();
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Emergency stop sent')),
          );
        },
        child: const Icon(Icons.warning),
      ),
    );
  }

  Widget _buildDirectionButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required String command,
    required bool enabled,
  }) {
    return Opacity(
      opacity: enabled ? 1.0 : 0.5,
      child: SizedBox(
        width: 100,
        child: Column(
          children: [
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(22),
                shape: const CircleBorder(),
              ),
              onPressed: enabled ? () => _sendCommand(context, command) : null,
              child: Icon(icon, size: 36),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _sendCommand(BuildContext context, String command) async {
    final provider = context.read<BluetoothProvider>();
    if (!provider.isConnected) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Connect to the wheelchair over Bluetooth first.'),
        ),
      );
      return;
    }

    try {
      await provider.sendManualCommand(command);
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to send command: $e')));
    }
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({
    required this.isConnected,
    required this.movementState,
    required this.activeCommand,
  });

  final bool isConnected;
  final String movementState;
  final String? activeCommand;

  @override
  Widget build(BuildContext context) {
    final color = isConnected ? Colors.green : Colors.red;
    final status = isConnected ? 'Connected' : 'Disconnected';
    final movement = movementState == 'moving'
        ? 'Moving ${activeCommand ?? ''}'.trim()
        : 'Idle';

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      child: ListTile(
        leading: Icon(Icons.bluetooth, color: color, size: 32),
        title: Text(status),
        subtitle: Text(movement),
        trailing: activeCommand != null
            ? Chip(
                label: Text(activeCommand!.toUpperCase()),
                backgroundColor: Colors.blue.withOpacity(0.1),
              )
            : null,
      ),
    );
  }
}
