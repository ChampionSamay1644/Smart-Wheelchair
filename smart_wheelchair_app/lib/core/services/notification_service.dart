import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import '../models/user_model.dart';
import '../providers/notifications_provider.dart';

class NotificationService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  final NotificationsProvider _notificationsProvider;
  SmartWheelchairUser? _currentUser;

  NotificationService(this._notificationsProvider);

  Future<void> initialize() async {
    // Request permission for notifications
    await _messaging.requestPermission(alert: true, badge: true, sound: true);

    // Handle FCM messages
    FirebaseMessaging.onMessage.listen(_handleForegroundMessage);
    FirebaseMessaging.onMessageOpenedApp.listen(_handleMessageOpenedApp);
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

    // Get the token and update it in Firestore
    final token = await _messaging.getToken();
    if (token != null && _currentUser != null) {
      await _updateFCMToken(token);
    }

    // Listen for token refreshes
    _messaging.onTokenRefresh.listen((token) {
      if (_currentUser != null) {
        _updateFCMToken(token);
      }
    });
  }

  void setUser(SmartWheelchairUser? user) {
    _currentUser = user;
    if (user != null) {
      _messaging.getToken().then((token) {
        if (token != null) {
          _updateFCMToken(token);
        }
      });
    }
  }

  Future<void> _updateFCMToken(String token) async {
    if (_currentUser == null) return;

    await _firestore.collection('users').doc(_currentUser!.uid).update({
      'fcmToken': token,
    });
  }

  Future<void> _handleForegroundMessage(RemoteMessage message) async {
    // Show the notification using our fading panel
    _notificationsProvider.addEvent(
      message.notification?.title ?? 'New Alert',
      message.notification?.body ?? '',
    );

    // The actual system notification will be handled by the OS
    // since we're receiving FCM notifications
  }

  void _handleMessageOpenedApp(RemoteMessage message) {
    //Navigate to appropriate screen based on message data
    // We'll implement this when we add navigation
  }

  Future<void> sendPatientAlert({
    required String patientId,
    required String guardianId,
    required String title,
    required String body,
    required String type,
    Map<String, dynamic>? additionalData,
  }) async {
    // Create the alert in Firestore
    await _firestore.collection('alerts').add({
      'patientId': patientId,
      'guardianId': guardianId,
      'title': title,
      'body': body,
      'type': type,
      'data': additionalData,
      'createdAt': FieldValue.serverTimestamp(),
      'read': false,
    });

    // The actual FCM message will be sent by a Cloud Function
    // that watches the alerts collection
  }
}

// This function must be declared outside the class and marked as top-level
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // No need to show notifications here as the system will do it automatically
  // Just log for debugging
  debugPrint('Handling a background message: ${message.messageId}');
}
