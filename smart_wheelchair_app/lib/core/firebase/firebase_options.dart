import 'package:firebase_core/firebase_core.dart';

// TODO: Replace with your Firebase configuration
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform => const FirebaseOptions(
    apiKey: 'YOUR-API-KEY',
    appId: 'YOUR-APP-ID',
    messagingSenderId: 'YOUR-SENDER-ID',
    projectId: 'YOUR-PROJECT-ID',
    authDomain: 'YOUR-AUTH-DOMAIN',
    storageBucket: 'YOUR-STORAGE-BUCKET',
  );
}
