import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../core/services/pairing_service.dart';

class GeneratePairCodePage extends StatefulWidget {
  const GeneratePairCodePage({super.key});

  @override
  State<GeneratePairCodePage> createState() => _GeneratePairCodePageState();
}

class _GeneratePairCodePageState extends State<GeneratePairCodePage> {
  String? _code;
  bool _loading = false;
  String? _error;

  Future<void> _generate() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final code = await PairingService.generatePairCode();
      setState(() => _code = code);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Generate Pair Code')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: _loading ? null : _generate,
              child: _loading ? const CircularProgressIndicator() : const Text('Generate Code'),
            ),
            const SizedBox(height: 24),
            if (_code != null)
              Column(
                children: [
                  SelectableText('Code: $_code', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: () {
                      if (_code != null) {
                        Clipboard.setData(ClipboardData(text: _code!));
                        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Copied to clipboard')));
                      }
                    },
                    child: const Text('Copy Code'),
                  ),
                ],
              ),
            if (_error != null) ...[
              const SizedBox(height: 16),
              Text('Error: $_error', style: const TextStyle(color: Colors.red)),
            ],
          ],
        ),
      ),
    );
  }
}
