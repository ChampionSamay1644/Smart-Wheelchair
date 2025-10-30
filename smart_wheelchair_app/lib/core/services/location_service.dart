import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import '../models/location_model.dart';
import '../models/user_model.dart';

class LocationService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  StreamSubscription<Position>? _positionSubscription;
  SmartWheelchairUser? _currentUser;
  bool _isTracking = false;

  // Stream controller for live location updates (for map view)
  final _locationController = StreamController<LocationUpdate>.broadcast();
  Stream<LocationUpdate> get locationStream => _locationController.stream;

  void setUser(SmartWheelchairUser? user) {
    _currentUser = user;
    if (user == null) {
      stopTracking();
    }
  }

  Future<bool> checkLocationPermission() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return false;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return false;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      return false;
    }

    return true;
  }

  Future<void> startTracking() async {
    if (_isTracking || _currentUser == null) return;

    final hasPermission = await checkLocationPermission();
    if (!hasPermission) {
      throw Exception('Location permission not granted');
    }

    _isTracking = true;

    // Configure for high accuracy and frequent updates
    const locationSettings = LocationSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 0, // Update on any movement
      timeLimit: Duration(seconds: 1), // Limit to prevent excessive updates
    );

    _positionSubscription =
        Geolocator.getPositionStream(locationSettings: locationSettings).listen(
          (Position position) async {
            final update = LocationUpdate(
              patientId: _currentUser!.uid,
              position: LatLng(position.latitude, position.longitude),
              speed: position.speed,
              heading: position.heading,
              accuracy: position.accuracy,
              timestamp: DateTime.now(),
            );

            // Update Firestore (only if patient)
            if (_currentUser!.userType == 'patient') {
              await _firestore
                  .collection('locations')
                  .doc(_currentUser!.uid)
                  .set(update.toFirestore());
            }

            _locationController.add(update);
          },
        );
  }

  void stopTracking() {
    _positionSubscription?.cancel();
    _positionSubscription = null;
    _isTracking = false;
  }

  Stream<LocationUpdate?> watchPatientLocation(String patientId) {
    return _firestore.collection('locations').doc(patientId).snapshots().map((
      snapshot,
    ) {
      if (!snapshot.exists) return null;
      return LocationUpdate.fromFirestore(snapshot);
    });
  }

  Future<LocationUpdate?> getLastKnownLocation(String patientId) async {
    final doc = await _firestore.collection('locations').doc(patientId).get();

    if (!doc.exists) return null;
    return LocationUpdate.fromFirestore(doc);
  }

  Future<void> dispose() async {
    stopTracking();
    await _locationController.close();
  }
}
