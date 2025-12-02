import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../core/providers/connection_provider.dart';

class ConnectionDialog {
  static Future<void> show(
    BuildContext context, {
    bool barrierDismissible = true,
    String? errorMessage,
  }) async {
    final navigator = Navigator.of(context);
    await showDialog<void>(
      context: context,
      barrierDismissible: barrierDismissible,
      builder: (dialogContext) {
        return ChangeNotifierProvider.value(
          value: context.read<ConnectionProvider>(),
          child: _ConnectionDialogContent(
            initialError: errorMessage,
            allowCancel: barrierDismissible,
            onConnected: () {
              if (navigator.canPop()) {
                navigator.pop();
              }
            },
          ),
        );
      },
    );
  }
}

class _ConnectionDialogContent extends StatefulWidget {
  const _ConnectionDialogContent({
    this.initialError,
    required this.allowCancel,
    required this.onConnected,
  });

  final String? initialError;
  final bool allowCancel;
  final VoidCallback onConnected;

  @override
  State<_ConnectionDialogContent> createState() =>
      _ConnectionDialogContentState();
}

class _ConnectionDialogContentState extends State<_ConnectionDialogContent> {
  late final TextEditingController _ipController;
  late final TextEditingController _portController;
  String? _localError;

  @override
  void initState() {
    super.initState();
    final provider = context.read<ConnectionProvider>();
    _ipController = TextEditingController(text: provider.ipAddress);
    _portController = TextEditingController(text: provider.port.toString());
    _localError = widget.initialError;
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ConnectionProvider>(
      builder: (context, provider, _) {
        final isConnecting = provider.isConnecting;
        final isConnected = provider.isConnected;
        final lastError = provider.lastError;
        final effectiveError = _localError ?? lastError;

        return AlertDialog(
          title: const Text('Connect to Wheelchair'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _ipController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Raspberry Pi IP address',
                  hintText: 'e.g. 192.168.0.25',
                ),
                enabled: !isConnecting,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _portController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'WebSocket port',
                  hintText: '8765',
                ),
                enabled: !isConnecting,
              ),
              if (effectiveError != null) ...[
                const SizedBox(height: 12),
                Text(effectiveError, style: const TextStyle(color: Colors.red)),
              ],
              if (isConnected) ...[
                const SizedBox(height: 12),
                Text(
                  'Connected to ${provider.ipAddress}:${provider.port}',
                  style: const TextStyle(color: Colors.green),
                ),
              ],
            ],
          ),
          actions: [
            if (widget.allowCancel)
              TextButton(
                onPressed: isConnecting
                    ? null
                    : () {
                        Navigator.of(context).pop();
                      },
                child: const Text('Cancel'),
              ),
            if (isConnected)
              TextButton(
                onPressed: isConnecting
                    ? null
                    : () async {
                        await provider.disconnect(userInitiated: true);
                        setState(() {
                          _localError = null;
                        });
                      },
                child: const Text('Disconnect'),
              ),
            ElevatedButton(
              onPressed: isConnecting
                  ? null
                  : () async {
                      final parsedPort =
                          int.tryParse(_portController.text.trim()) ?? 8765;
                      final success = await provider.connect(
                        ip: _ipController.text,
                        port: parsedPort,
                      );
                      if (success) {
                        widget.onConnected();
                      } else {
                        setState(() {
                          _localError = provider.lastError;
                        });
                      }
                    },
              child: isConnecting
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Text(isConnected ? 'Reconnect' : 'Connect'),
            ),
          ],
        );
      },
    );
  }
}
