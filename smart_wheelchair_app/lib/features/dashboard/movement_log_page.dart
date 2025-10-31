import 'package:flutter/material.dart';
import '../../core/services/firebase_service.dart';
import '../../core/services/auth_service.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MovementLogPage extends StatelessWidget {
  const MovementLogPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Movement Log')),
      body: const _MovementLogBody(),
    );
  }
}

class _MovementLogBody extends StatefulWidget {
  const _MovementLogBody();

  @override
  State<_MovementLogBody> createState() => _MovementLogBodyState();
}

class _MovementLogBodyState extends State<_MovementLogBody> {
  String? _patientId;

  @override
  void initState() {
    super.initState();
    _initPatientId();
  }

  Future<void> _initPatientId() async {
    final user = await AuthService().currentUser;
    if (!mounted) return;
    if (user == null) return;

    if (user.userType == 'patient') {
      setState(() => _patientId = user.uid);
    } else if (user.userType == 'guardian') {
      setState(
        () => _patientId = user.patientIds?.isNotEmpty == true
            ? user.patientIds!.first
            : null,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_patientId == null) {
      return const Center(child: Text('No patient linked'));
    }

    return StreamBuilder<QuerySnapshot<Map<String, dynamic>>>(
      stream: FirebaseService.firestore
          .collection('users')
          .doc(_patientId)
          .collection('movementLogs')
          .orderBy('timestamp', descending: true)
          .snapshots(),
      builder: (context, snap) {
        if (snap.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        final docs = snap.data?.docs ?? [];
        if (docs.isEmpty) return const Center(child: Text('No movements'));

        return ListView.builder(
          padding: const EdgeInsets.all(8),
          itemCount: docs.length,
          itemBuilder: (context, index) {
            final d = docs[index].data();
            final action = d['action'] ?? '';
            final mode = d['mode'] ?? '';
            final ts = (d['timestamp'] as Timestamp?)?.toDate();
            final lat = d['lat']?.toString();
            final lon = d['lon']?.toString();

            return Card(
              margin: const EdgeInsets.symmetric(vertical: 4),
              child: ListTile(
                leading: const Icon(Icons.directions),
                title: Text(action),
                subtitle: Text(mode + (ts != null ? ' • ${ts.toLocal()}' : '')),
                trailing: (lat != null && lon != null)
                    ? Text('${lat.substring(0, 6)}, ${lon.substring(0, 6)}')
                    : null,
              ),
            );
          },
        );
      },
    );
  }
}
