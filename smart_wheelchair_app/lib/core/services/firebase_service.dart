import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter/foundation.dart';

/// Core Firebase service that initializes and provides access to Firebase instances
class FirebaseService {
  static final FirebaseAuth _auth = FirebaseAuth.instance;
  static final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  static final FirebaseStorage _storage = FirebaseStorage.instance;
  static final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  static final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;

  static FirebaseAuth get auth => _auth;
  static FirebaseFirestore get firestore => _firestore;
  static FirebaseStorage get storage => _storage;
  static FirebaseMessaging get messaging => _messaging;
  static FirebaseAnalytics get analytics => _analytics;

  /// Initialize Firebase and configure messaging permissions
  static Future<FirebaseService> initialize() async {
    // Request notification permissions
    if (!kIsWeb) {
      final settings = await _messaging.requestPermission(
        alert: true,
        announcement: false,
        badge: true,
        carPlay: false,
        criticalAlert: true,
        provisional: false,
        sound: true,
      );

      debugPrint('User granted permission: ${settings.authorizationStatus}');

      // Get FCM token for this device
      final token = await _messaging.getToken();
      debugPrint('FCM Token: $token');

      // Listen for token refresh
      _messaging.onTokenRefresh.listen((token) {
        // TODO: Save the new token to Firestore for the current user
        debugPrint('FCM Token refreshed: $token');
        if (_auth.currentUser != null) {
          _firestore.collection('users').doc(_auth.currentUser!.uid).update({
            'fcmToken': token,
          });
        }
      });
    }

    // Configure how to handle messages when app is in foreground
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      debugPrint('Got a message whilst in the foreground!');
      debugPrint('Message data: ${message.data}');

      if (message.notification != null) {
        debugPrint(
          'Message also contained a notification: ${message.notification}',
        );
        // TODO: Show the notification using the fading panel
      }
    });

    return FirebaseService();
  }
}
