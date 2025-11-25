import 'dart:math';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'firebase_service.dart';

class PairingError implements Exception {
  final String message;
  PairingError(this.message);
  @override
  String toString() => 'PairingError: $message';
}

class PairingService {
  static final _firestore = FirebaseService.firestore;
  static final _auth = FirebaseService.auth;

  /// Generate a short alphanumeric code (6 chars by default) and store it
  /// under collection `pair_codes` with doc id equal to the code (uppercase).
  static Future<String> generatePairCode({Duration validFor = const Duration(days: 7)}) async {
    final user = _auth.currentUser;
    if (user == null) throw PairingError('User not authenticated');

    final code = _generateCode(6);
    final ref = _firestore.collection('pair_codes').doc(code);

    // Ensure uniqueness (low collision chance) — try once, fail safe if exists
    final exists = (await ref.get()).exists;
    if (exists) throw PairingError('Try again');

    await ref.set({
      'code': code,
      'ownerUid': user.uid,
      'ownerRole': (await _firestore.collection('users').doc(user.uid).get()).data()?['userType'] ?? 'unknown',
      'createdAt': FieldValue.serverTimestamp(),
      'expiresAt': Timestamp.fromDate(DateTime.now().add(validFor)),
      'used': false,
      'claimedBy': null,
      'claimedAt': null,
    });

    return code;
  }

  /// Claim a pair code and atomically update both user docs to reflect pairing.
  static Future<void> claimPairCode(String code) async {
    final claimer = _auth.currentUser;
    if (claimer == null) throw PairingError('User not authenticated');

    final pairRef = _firestore.collection('pair_codes').doc(code.toUpperCase());

    await _firestore.runTransaction((tx) async {
      final pairSnap = await tx.get(pairRef);
      if (!pairSnap.exists) throw PairingError('Invalid code');

      final data = pairSnap.data()!;
      final used = data['used'] as bool? ?? false;
      final expires = data['expiresAt'] as Timestamp?;
      final ownerUid = data['ownerUid'] as String?;
      final ownerRole = (data['ownerRole'] as String?) ?? 'unknown';

      if (used) throw PairingError('Code already used');
      if (expires != null && expires.toDate().isBefore(DateTime.now())) throw PairingError('Code expired');
      if (ownerUid == null) throw PairingError('Invalid code owner');
      if (ownerUid == claimer.uid) throw PairingError('Cannot claim your own code');

      final ownerRef = _firestore.collection('users').doc(ownerUid);
      final claimerRef = _firestore.collection('users').doc(claimer.uid);

      final ownerSnap = await tx.get(ownerRef);
      final claimerSnap = await tx.get(claimerRef);

      if (!ownerSnap.exists || !claimerSnap.exists) throw PairingError('User documents not found');

  final claimerData = claimerSnap.data()!;

  final claimerType = (claimerData['userType'] as String?) ?? 'unknown';

      // Determine pairing direction based on ownerRole
      if (ownerRole == 'guardian' && claimerType == 'patient') {
        // owner is guardian, claimer is patient -> set patient.guardianId and add patient to guardian.patientIds
        tx.update(ownerRef, {
          'patientIds': FieldValue.arrayUnion([claimer.uid])
        });
        tx.update(claimerRef, {
          'guardianId': ownerUid,
        });
      } else if (ownerRole == 'patient' && claimerType == 'guardian') {
        // owner is patient, claimer is guardian -> add patient to guardian.patientIds and set patient's guardianId
        tx.update(claimerRef, {
          'patientIds': FieldValue.arrayUnion([ownerUid])
        });
        tx.update(ownerRef, {
          'guardianId': claimer.uid,
        });
      } else {
  throw PairingError('Role mismatch: cannot pair $ownerRole with $claimerType');
      }

      // mark pair code used
      tx.update(pairRef, {
        'used': true,
        'claimedBy': claimer.uid,
        'claimedAt': FieldValue.serverTimestamp(),
      });
    });
  }

  static String _generateCode(int length) {
    const chars = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'; // avoid confusing chars
    final rand = Random.secure();
    return List.generate(length, (_) => chars[rand.nextInt(chars.length)]).join();
  }
}
