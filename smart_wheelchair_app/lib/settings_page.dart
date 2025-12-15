// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

import 'widgets/connection_dialog.dart';
import 'core/localization.dart';

class SettingsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(tr(context, 'settings')),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildSection(tr(context, 'wheelchair_configuration'), [
            _buildSettingItem(
              context,
              tr(context, 'speed_control'),
              Icons.speed,
              'Adjust maximum speed limit',
              onTap: () {
                print('Speed settings tapped');
              },
            ),
            _buildSettingItem(
              context,
              tr(context, 'sensitivity'),
              Icons.tune,
              'Adjust control sensitivity',
              onTap: () {
                print('Sensitivity settings tapped');
              },
            ),
          ]),
          const SizedBox(height: 20),
          _buildSection(tr(context, 'user_preferences'), [
            _buildSettingItem(
              context,
              tr(context, 'emergency_contacts'),
              Icons.emergency,
              'Add or edit emergency contacts',
              onTap: () {
                print('Emergency contacts tapped');
              },
            ),
            _buildSettingItem(
              context,
              tr(context, 'voice_commands'),
              Icons.record_voice_over,
              'Customize voice commands',
              onTap: () {
                print('Voice commands settings tapped');
                Navigator.pushNamed(context, '/voice_control');
              },
            ),
          ]),
          const SizedBox(height: 20),
          _buildSection(tr(context, 'system'), [
            _buildSettingItem(
              context,
              tr(context, 'device_info'),
              Icons.info,
              'View system information',
              onTap: () {
                print('Device info tapped');
              },
            ),
            _buildSettingItem(
              context,
              tr(context, 'bluetooth_control'),
              Icons.bluetooth,
              'Pair and monitor wheelchair connection',
              onTap: () {
                Navigator.pushNamed(context, '/bluetooth_connection');
              },
            ),
            _buildSettingItem(
              context,
              tr(context, 'connection_status'),
              Icons.bluetooth,
              'Check device connectivity',
              onTap: () {
                ConnectionDialog.show(context);
              },
            ),
            _buildSettingItem(
              context,
              tr(context, 'battery'),
              Icons.battery_full,
              'View battery status',
              onTap: () {
                showModalBottomSheet(
                  context: context,
                  builder: (context) => Container(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        ListTile(
                          leading: const Icon(
                            Icons.battery_full,
                            color: Colors.green,
                          ),
                          title: Text(tr(context, 'battery_level')),
                          subtitle: Text(tr(context, 'battery_estimate')),
                        ),
                        const LinearProgressIndicator(
                          value: 0.75,
                          backgroundColor: Colors.grey,
                          valueColor: AlwaysStoppedAnimation<Color>(
                            Colors.green,
                          ),
                        ),
                        const SizedBox(height: 16),
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: Text(tr(context, 'close')),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ]),
        ],
      ),
    );
  }

  Widget _buildSection(String title, List<Widget> items) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
          child: Text(
            title,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.blue,
            ),
          ),
        ),
        Card(
          elevation: 2,
          margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 8),
          child: Column(children: items),
        ),
      ],
    );
  }

  Widget _buildSettingItem(
    BuildContext context,
    String title,
    IconData icon,
    String subtitle, {
    required VoidCallback onTap,
  }) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.blue.withAlpha((0.1 * 255).round()),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(icon, color: Colors.blue),
      ),
      title: Text(title),
      subtitle: Text(subtitle),
      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
      onTap: onTap,
    );
  }
}
