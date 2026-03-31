import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/foundation.dart';

class SyncService {
  final FirebaseDatabase _db = FirebaseDatabase.instance;
  
  /// Upload status (For Patients)
  Future<void> updatePatientStatus({
    required String uid,
    required double heartRate,
    required double spo2,
    required double lat,
    required double lng,
    bool isEmergency = false,
  }) async {
    try {
      final ref = _db.ref('users/$uid/status');
      await ref.set({
        'heartRate': heartRate,
        'spo2': spo2,
        'latitude': lat,
        'longitude': lng,
        'isEmergency': isEmergency,
        'lastUpdate': ServerValue.timestamp,
      });
    } catch (e) {
      debugPrint('❌ SYNC ERROR: Could not update status: $e');
    }
  }

  /// Listen to status (For Guardians)
  Stream<Map<String, dynamic>?> listenToPatientStatus(String patientUid) {
    return _db.ref('users/$patientUid/status').onValue.map((event) {
      final data = event.snapshot.value as Map?;
      return data?.cast<String, dynamic>();
    });
  }

  /// Trigger emergency alert
  Future<void> triggerEmergency(String uid, String message) async {
    final ref = _db.ref('alerts/$uid').push();
    await ref.set({
      'message': message,
      'timestamp': ServerValue.timestamp,
      'resolved': false,
    });
    
    // Also update current status
    await _db.ref('users/$uid/status').update({'isEmergency': true});
  }
}
