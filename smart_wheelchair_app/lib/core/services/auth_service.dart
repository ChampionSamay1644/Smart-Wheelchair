import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../enums.dart';
import '../models/user_model.dart';

class AuthError implements Exception {
  final String message;
  AuthError(this.message);
}

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Future<SmartWheelchairUser?> get currentUser async {
    final user = _auth.currentUser;
    if (user == null) return null;

    final doc = await _firestore.collection('users').doc(user.uid).get();
    return doc.exists
        ? SmartWheelchairUser.fromMap({'uid': user.uid, ...doc.data()!})
        : null;
  }

  Future<SmartWheelchairUser> signUp({
    required String email,
    required String password,
    required String name,
    required UserRole role,
    String? phoneNumber,
    String? deviceId,
  }) async {
    try {
      final userCredential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      final user = SmartWheelchairUser(
        uid: userCredential.user!.uid,
        email: email,
        name: name,
        userType: role == UserRole.patient ? 'patient' : 'guardian',
        patientIds: role == UserRole.guardian ? [] : null,
      );

      await _firestore.collection('users').doc(user.uid).set(user.toMap());
      return user;
    } on FirebaseAuthException catch (e) {
      throw AuthError(_getErrorMessage(e.code));
    }
  }

  Future<SmartWheelchairUser> login(String email, String password) async {
    try {
      final userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      final doc = await _firestore
          .collection('users')
          .doc(userCredential.user!.uid)
          .get();

      if (!doc.exists) {
        throw AuthError('User data not found');
      }

      return SmartWheelchairUser.fromMap({
        'uid': userCredential.user!.uid,
        ...doc.data()!,
      });
    } on FirebaseAuthException catch (e) {
      throw AuthError(_getErrorMessage(e.code));
    }
  }

  Future<void> logout() async {
    await _auth.signOut();
  }

  Future<String> createInviteCode(SmartWheelchairUser guardian) async {
    if (guardian.userType != 'guardian') {
      throw AuthError('Only guardians can create invite codes');
    }

    final code = _generateInviteCode();
    await _firestore.collection('invites').doc(code).set({
      'guardianId': guardian.uid,
      'guardianName': guardian.name,
      'guardianEmail': guardian.email,
      'expiresAt': Timestamp.fromDate(
        DateTime.now().add(const Duration(days: 7)),
      ),
      'used': false,
      'createdAt': FieldValue.serverTimestamp(),
    });

    return code;
  }

  Future<void> acceptInvite(String inviteCode, String patientId) async {
    final batch = _firestore.batch();

    // Get and validate invite
    final inviteDoc = await _firestore
        .collection('invites')
        .doc(inviteCode)
        .get();
    if (!inviteDoc.exists) {
      throw AuthError('Invalid invite code');
    }

    final invite = inviteDoc.data()!;
    if (invite['used'] == true ||
        (invite['expiresAt'] as Timestamp).toDate().isBefore(DateTime.now())) {
      throw AuthError('Invite code is expired or already used');
    }

    // Update patient's guardian
    final patientRef = _firestore.collection('users').doc(patientId);
    batch.update(patientRef, {'guardianId': invite['guardianId']});

    // Update guardian's patients list
    final guardianRef = _firestore
        .collection('users')
        .doc(invite['guardianId']);
    batch.update(guardianRef, {
      'patientIds': FieldValue.arrayUnion([patientId]),
    });

    // Mark invite as used
    final inviteRef = _firestore.collection('invites').doc(inviteCode);
    batch.update(inviteRef, {'used': true});

    await batch.commit();
  }

  String _generateInviteCode() {
    // Generate a 6-character code
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    final random = DateTime.now().millisecondsSinceEpoch;
    final code = List.generate(6, (index) {
      return chars[((random >> (index * 5)) + index) % chars.length];
    }).join();
    return code;
  }

  String _getErrorMessage(String code) {
    switch (code) {
      case 'weak-password':
        return 'The password provided is too weak.';
      case 'email-already-in-use':
        return 'An account already exists for that email.';
      case 'invalid-email':
        return 'The email address is not valid.';
      case 'user-disabled':
        return 'This account has been disabled.';
      case 'user-not-found':
        return 'No user found for that email.';
      case 'wrong-password':
        return 'Wrong password provided.';
      case 'operation-not-allowed':
        return 'Email/password accounts are not enabled.';
      default:
        return 'An error occurred. Please try again.';
    }
  }
}
