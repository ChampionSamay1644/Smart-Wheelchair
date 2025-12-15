// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

import 'services/emergency_stop_service.dart';
import 'core/localization.dart';

class RemoteControlPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(tr(context, 'remote_control')),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildDestinationCard(
            context,
            tr(context, 'kitchen'),
            Icons.kitchen,
            Colors.orange,
            () => print('Selected: Go to Kitchen'),
          ),
          const SizedBox(height: 12),
          _buildDestinationCard(
            context,
            tr(context, 'bedroom'),
            Icons.bedroom_parent,
            Colors.blue,
            () => print('Selected: Go to Bedroom'),
          ),
          const SizedBox(height: 12),
          _buildDestinationCard(
            context,
            tr(context, 'living_room'),
            Icons.living,
            Colors.green,
            () => print('Selected: Go to Living Room'),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: () async {
          final messenger = ScaffoldMessenger.of(context);
          final message = tr(context, 'emergency_stop_sent');
          await EmergencyStopService.trigger();
          messenger.showSnackBar(SnackBar(content: Text(message)));
        },
        child: const Icon(Icons.warning),
      ),
    );
  }

  Widget _buildDestinationCard(
    BuildContext context,
    String destination,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: color.withAlpha(30),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 32),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  '${tr(context, 'go_to')} $destination',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Icon(Icons.arrow_forward_ios, color: Colors.grey[400]),
            ],
          ),
        ),
      ),
    );
  }
}
