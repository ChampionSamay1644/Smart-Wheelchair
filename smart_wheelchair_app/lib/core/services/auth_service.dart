import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import '../enums.dart';

class AuthUser {
  final String id;
  final String name;
  final String email;
  final UserRole role;
  final bool isNew;

  AuthUser({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    this.isNew = false,
  });
}

class AuthService {
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;

  static final Map<String, String> _hardcodedNames = {
    'patient@test.com': 'John Patient',
    'guardian@test.com': 'Mary Guardian',
  };

  /// Real Firebase Authentication
  Future<AuthUser?> login(String email, String password, {UserRole role = UserRole.patient}) async {
    debugPrint('🔐 AUTH SECURITY CHECK: Email="$email"');
    
    try {
      // 1. Authenticate with Firebase
      UserCredential credential;
      bool isNewUser = false;
      try {
        credential = await _firebaseAuth.signInWithEmailAndPassword(
          email: email,
          password: password,
        );
      } on FirebaseAuthException catch (e) {
        // Modern Firebase returns 'invalid-credential' instead of 'user-not-found'
        // to prevent email enumeration. We'll try to create the account if login fails.
        if (e.code == 'user-not-found' || e.code == 'invalid-credential' || e.code == 'wrong-password') {
          debugPrint('🔄 AUTH FALLBACK: Login failed (${e.code}), checking if user exists...');
          try {
            credential = await _firebaseAuth.createUserWithEmailAndPassword(
              email: email,
              password: password,
            );
            isNewUser = true;
            debugPrint('🆕 AUTH AUTO-REGISTERED: New account created for $email');
          } on FirebaseAuthException catch (createError) {
            if (createError.code == 'email-already-in-use') {
              // If registration fails because user exists, then the initial login 
              // failure was definitely due to a wrong password.
              debugPrint('❌ AUTH ERROR: Account exists but password was incorrect.');
              throw FirebaseAuthException(code: 'wrong-password', message: 'Incorrect password for this account.');
            }
            rethrow;
          }
        } else {
          rethrow;
        }
      }

      final user = credential.user;
      if (user == null) return null;

      debugPrint('✅ AUTH SUCCESS: Firebase verified ID=${user.uid}');

      // Return user with mapped role
      return AuthUser(
        id: user.uid,
        name: _hardcodedNames[email] ?? email.split('@')[0],
        email: email,
        role: role,
        isNew: isNewUser,
      );
    } catch (e) {
      debugPrint('❌ AUTH FAILED: $e');
      return null;
    }
  }

  Future<void> logout() async {
    await _firebaseAuth.signOut();
  }
}
