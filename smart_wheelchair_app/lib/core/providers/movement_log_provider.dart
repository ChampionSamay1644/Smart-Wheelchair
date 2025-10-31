import 'package:flutter/foundation.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../services/firebase_service.dart';
import '../services/auth_service.dart';

class MovementEntry {
  final DateTime time;
  final String mode; // manual, voice, remote
  final String action; // forward/back/left/right/other
  final double? latitude;
  final double? longitude;

  MovementEntry({
    required this.time,
    required this.mode,
    required this.action,
    this.latitude,
    this.longitude,
  });
}

class MovementLogProvider extends ChangeNotifier {
  final List<MovementEntry> _entries = [];

  List<MovementEntry> get entries => List.unmodifiable(_entries.reversed);

  /// Logs a movement locally and to Firestore under users/{patientId}/movementLogs
  Future<void> addEntry(
    String mode,
    String action, {
    double? lat,
    double? lon,
    String? forPatientId,
  }) async {
    final now = DateTime.now();
    _entries.add(
      MovementEntry(
        time: now,
        mode: mode,
        action: action,
        latitude: lat,
        longitude: lon,
      ),
    );
    notifyListeners();

    try {
      final currentUser = await AuthService().currentUser;
      final targetId = forPatientId ?? currentUser?.uid;
      if (targetId == null) return;

      await FirebaseService.firestore
          .collection('users')
          .doc(targetId)
          .collection('movementLogs')
          .add({
            'action': action,
            'mode': mode,
            if (lat != null) 'lat': lat,
            if (lon != null) 'lon': lon,
            'timestamp': FieldValue.serverTimestamp(),
          });
    } catch (_) {
      // Swallow Firestore errors; local log remains
    }
  }

  void clear() {
    _entries.clear();
    notifyListeners();
  }
}
