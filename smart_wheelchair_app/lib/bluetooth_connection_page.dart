import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:provider/provider.dart';

import 'core/providers/bluetooth_provider.dart';

class BluetoothConnectionPage extends StatelessWidget {
  const BluetoothConnectionPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bluetooth Control'),
        backgroundColor: Theme.of(context).primaryColor,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh paired devices',
            onPressed: () =>
                context.read<BluetoothProvider>().refreshBondedDevices(),
          ),
        ],
      ),
      body: Consumer<BluetoothProvider>(
        builder: (context, provider, _) {
          final isAdapterOn = provider.isAdapterOn;
          final isConnected = provider.isConnected;
          final devices = provider.bondedDevices;
          final subtitle = isConnected
              ? 'Connected to ${provider.connectedAddress}'
              : isAdapterOn
              ? 'Select a paired device to connect'
              : 'Enable Bluetooth to proceed';

          final tiles = <Widget>[
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Card(
                child: ListTile(
                  leading: Icon(
                    isConnected
                        ? Icons.bluetooth_connected
                        : isAdapterOn
                        ? Icons.bluetooth
                        : Icons.bluetooth_disabled,
                    color: isConnected
                        ? Colors.lightBlueAccent
                        : isAdapterOn
                        ? Colors.blue
                        : Colors.redAccent,
                    size: 32,
                  ),
                  title: Text(
                    isConnected
                        ? 'Bluetooth Connected'
                        : isAdapterOn
                        ? 'Bluetooth Ready'
                        : 'Bluetooth Disabled',
                  ),
                  subtitle: Text(subtitle),
                  trailing: isAdapterOn
                      ? ElevatedButton(
                          onPressed: isConnected
                              ? () => provider.disconnect()
                              : null,
                          child: const Text('Disconnect'),
                        )
                      : ElevatedButton(
                          onPressed: () async {
                            await FlutterBluetoothSerial.instance
                                .requestEnable();
                          },
                          child: const Text('Enable'),
                        ),
                ),
              ),
            ),
          ];

          if (provider.lastError != null) {
            tiles.add(
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                child: Text(
                  provider.lastError!,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
            );
          }

          if (devices.isEmpty) {
            tiles.add(
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 32,
                ),
                child: Text(
                  isAdapterOn
                      ? 'No paired devices found. Pair your wheelchair from the system settings and pull to refresh.'
                      : 'Bluetooth is disabled. Turn it on to discover your wheelchair.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
              ),
            );
          } else {
            for (final device in devices) {
              final isDeviceConnected =
                  provider.connectedAddress == device.address;
              tiles.add(
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: ListTile(
                    leading: Icon(
                      isDeviceConnected
                          ? Icons.radio_button_checked
                          : Icons.radio_button_unchecked,
                      color: isDeviceConnected
                          ? Colors.lightGreen
                          : Colors.grey,
                    ),
                    title: Text(device.name ?? 'Unknown device'),
                    subtitle: Text(device.address),
                    trailing: ElevatedButton(
                      onPressed: provider.isConnecting
                          ? null
                          : isDeviceConnected
                          ? () => provider.disconnect()
                          : () => provider.connectToDevice(device.address),
                      child: provider.isConnecting && !isDeviceConnected
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : Text(isDeviceConnected ? 'Disconnect' : 'Connect'),
                    ),
                  ),
                ),
              );
            }
          }

          tiles.add(const SizedBox(height: 32));

          return RefreshIndicator(
            onRefresh: provider.refreshBondedDevices,
            child: ListView(
              physics: const AlwaysScrollableScrollPhysics(),
              children: tiles,
            ),
          );
        },
      ),
    );
  }
}
