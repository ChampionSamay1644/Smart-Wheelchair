// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';

import 'widgets/connection_dialog.dart';
import 'core/localization.dart';
import 'package:provider/provider.dart';
import 'core/providers/api_provider.dart';
import 'core/providers/auth_provider.dart';
import 'core/providers/connection_provider.dart';
import 'core/enums.dart';

class SettingsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final userRole = context.read<AuthProvider>().userRole;
    final isPatient = userRole == UserRole.patient;

    return Scaffold(
      appBar: AppBar(
        title: Text(tr(context, 'settings')),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (isPatient) ...[
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
          ],
          _buildSection(tr(context, 'user_preferences'), [
            _buildSettingItem(
              context,
              tr(context, 'emergency_contacts'),
              Icons.emergency,
              'Add or edit emergency contacts',
              onTap: () {
                Navigator.pushNamed(context, '/emergency_contacts');
              },
            ),
            if (isPatient)
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
            if (isPatient) ...[
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
            ],
            _buildSettingItem(
              context,
              tr(context, 'battery'),
              Icons.battery_full,
              'View battery status',
              onTap: () {
                // ... battery logic
              },
            ),
          ]),
          const SizedBox(height: 20),
          _buildCloudSettings(context),
        ],
      ),
    );
  }

  Widget _buildCloudSettings(BuildContext context) {
    return Consumer<ApiProvider>(
      builder: (context, apiProvider, _) {
        return _buildSection('Cloud Integration', [
          _buildSettingItem(
            context,
            'Vercel API URL',
            Icons.cloud,
            apiProvider.apiUrl ?? 'Not configured',
            onTap: () => _showApiUrlDialog(context, apiProvider),
          ),
          _buildSettingItem(
            context,
            'Selected Wheelchair',
            Icons.accessible,
            apiProvider.selectedDeviceId ?? 'None selected',
            onTap: () => _showDeviceSelectionDialog(context, apiProvider),
          ),
          SwitchListTile(
            title: const Text('Local Mock Mode'),
            subtitle: const Text('Bypass API and use simulated data for testing UI'),
            secondary: const Icon(Icons.bug_report, color: Colors.orange),
            value: apiProvider.mockMode,
            onChanged: (value) {
              apiProvider.toggleMockMode(value);
              if (value) {
                context.read<ConnectionProvider>().disconnect(userInitiated: true);
              }
            },
          ),
        ]);
      },
    );
  }

  void _showApiUrlDialog(BuildContext context, ApiProvider provider) {
    final controller = TextEditingController(text: provider.apiUrl);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Vercel API URL'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'https://your-api.vercel.app',
            helperText: 'Enter the base URL of your Vercel API',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              provider.updateApiUrl(controller.text);
              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  void _showDeviceSelectionDialog(BuildContext context, ApiProvider provider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Select Wheelchair'),
        content: SizedBox(
          width: double.maxFinite,
          child: provider.devices.isEmpty
              ? const Text('No devices found. Check your API URL.')
              : ListView.builder(
                  shrinkWrap: true,
                  itemCount: provider.devices.length,
                  itemBuilder: (context, index) {
                    final device = provider.devices[index];
                    final deviceId = device['deviceId'];
                    final isOnline = device['online'] ?? false;
                    return ListTile(
                      title: Text(deviceId),
                      subtitle: Text(isOnline ? 'Online' : 'Offline'),
                      trailing: provider.selectedDeviceId == deviceId
                          ? const Icon(Icons.check, color: Colors.green)
                          : null,
                      onTap: () {
                        provider.selectDevice(deviceId);
                        Navigator.pop(context);
                      },
                    );
                  },
                ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              provider.refreshDevices();
            },
            child: const Text('Refresh'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
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
