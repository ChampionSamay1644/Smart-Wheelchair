import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/location_model.dart';
import '../models/user_model.dart';
import '../services/location_service.dart';

class LocationProvider extends ChangeNotifier {
  final LocationService _locationService;
  SmartWheelchairUser? _currentUser;

  final Map<String, LocationUpdate?> _patientLocations = {};
  final List<StreamSubscription> _locationSubscriptions = [];
  LocationUpdate? _currentLocation;
  bool _isTracking = false;

  LocationProvider(this._locationService);

  LocationUpdate? get currentLocation => _currentLocation;
  bool get isTracking => _isTracking;
  Map<String, LocationUpdate?> get patientLocations => _patientLocations;

  void setUser(SmartWheelchairUser? user) {
    _currentUser = user;
    _locationService.setUser(user);

    // Clear existing subscriptions and locations
    _cancelSubscriptions();
    _patientLocations.clear();

    if (user != null) {
      if (user.userType == 'guardian') {
        // Subscribe to all patient locations
        for (final patientId in user.patientIds ?? []) {
          _subscribeToPatientLocation(patientId);
        }
      }
    }
    notifyListeners();
  }

  Future<void> startTracking() async {
    if (_currentUser == null) return;

    try {
      await _locationService.startTracking();
      _isTracking = true;
      notifyListeners();
    } catch (e) {
      _isTracking = false;
      notifyListeners();
      rethrow;
    }
  }

  void stopTracking() {
    _locationService.stopTracking();
    _isTracking = false;
    notifyListeners();
  }

  void _subscribeToPatientLocation(String patientId) {
    final subscription = _locationService
        .watchPatientLocation(patientId)
        .listen((location) {
          _patientLocations[patientId] = location;
          notifyListeners();
        });

    _locationSubscriptions.add(subscription);
  }

  Future<void> _cancelSubscriptions() async {
    for (final subscription in _locationSubscriptions) {
      await subscription.cancel();
    }
    _locationSubscriptions.clear();
  }

  @override
  void dispose() {
    _cancelSubscriptions();
    super.dispose();
  }
}
