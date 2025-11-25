import 'package:flutter/material.dart';
import '../../core/services/pairing_service.dart';

class ClaimPairCodePage extends StatefulWidget {
  const ClaimPairCodePage({super.key});

  @override
  State<ClaimPairCodePage> createState() => _ClaimPairCodePageState();
}

class _ClaimPairCodePageState extends State<ClaimPairCodePage> {
  final _controller = TextEditingController();
  bool _loading = false;
  String? _error;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _claim() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      await PairingService.claimPairCode(_controller.text.trim());
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Pairing successful')));
      Navigator.pop(context);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isValid = _controller.text.trim().length >= 6;
    return Scaffold(
      appBar: AppBar(title: const Text('Claim Pair Code')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(labelText: 'Enter code'),
              textCapitalization: TextCapitalization.characters,
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _loading || !isValid ? null : _claim,
              child: _loading ? const CircularProgressIndicator() : const Text('Claim'),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(_error!, style: const TextStyle(color: Colors.red)),
            ]
          ],
        ),
      ),
    );
  }
}
