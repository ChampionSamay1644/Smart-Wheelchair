import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import '../enums.dart';
import 'connection_provider.dart';
import 'api_provider.dart';
import '../../services/api_service.dart';

class NavigationState {
  final LatLng? currentPosition;
  final LatLng? destination;
  final List<LatLng> routePoints;
  final bool isNavigating;
  final bool isPaused;
  final String? destinationAddress;
  final double? distance; // in km
  final double? estimatedTime; // in minutes
  final List<Map<String, dynamic>> searchResults;
  final bool isSearching;

  NavigationState({
    this.currentPosition,
    this.destination,
    this.routePoints = const [],
    this.isNavigating = false,
    this.isPaused = false,
    this.destinationAddress,
    this.distance,
    this.estimatedTime,
    this.searchResults = const [],
    this.isSearching = false,
  });

  NavigationState copyWith({
    LatLng? currentPosition,
    LatLng? destination,
    List<LatLng>? routePoints,
    bool? isNavigating,
    bool? isPaused,
    String? destinationAddress,
    double? distance,
    double? estimatedTime,
    List<Map<String, dynamic>>? searchResults,
    bool? isSearching,
  }) {
    return NavigationState(
      currentPosition: currentPosition ?? this.currentPosition,
      destination: destination ?? this.destination,
      routePoints: routePoints ?? this.routePoints,
      isNavigating: isNavigating ?? this.isNavigating,
      isPaused: isPaused ?? this.isPaused,
      destinationAddress: destinationAddress ?? this.destinationAddress,
      distance: distance ?? this.distance,
      estimatedTime: estimatedTime ?? this.estimatedTime,
      searchResults: searchResults ?? this.searchResults,
      isSearching: isSearching ?? this.isSearching,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'currentPosition': currentPosition != null 
          ? {'lat': currentPosition!.latitude, 'lng': currentPosition!.longitude} 
          : null,
      'destination': destination != null 
          ? {'lat': destination!.latitude, 'lng': destination!.longitude} 
          : null,
      'isNavigating': isNavigating,
      'isPaused': isPaused,
      'destinationAddress': destinationAddress,
      'distance': distance,
      'estimatedTime': estimatedTime,
      'routePoints': routePoints.map((p) => {'lat': p.latitude, 'lng': p.longitude}).toList(),
    };
  }
}

class OutdoorNavigationProvider extends ChangeNotifier {
  final ConnectionProvider _connectionProvider;
  final ApiProvider _apiProvider;
  final ApiService _apiService = ApiService();
  UserRole? _userRole;
  
  NavigationState _state = NavigationState();
  NavigationState get state => _state;

  StreamSubscription? _msgSub;
  StreamSubscription<Position>? _positionSub;
  Timer? _cloudPollTimer;
  String _lastQuery = '';
  bool get lastQueryNotEmpty => _lastQuery.trim().isNotEmpty;

  OutdoorNavigationProvider(this._connectionProvider, this._apiProvider) {
    _initSync();
  }

  void setUserRole(UserRole role) {
    _userRole = role;
    _cloudPollTimer?.cancel();
    _positionSub?.cancel();
    
    if (_userRole == UserRole.guardian) {
      _startCloudPolling();
    } else if (_userRole == UserRole.patient) {
      _startLocationTracking();
    }
    notifyListeners();
  }

  void _initSync() {
    _msgSub = _connectionProvider.messageStream.listen((msg) {
      if (msg['type'] == 'navigation_update' && _userRole == UserRole.guardian) {
        if (msg['data'] != null) {
          _handleRemoteUpdate(msg['data']);
        }
      }
    });
  }

  void _startCloudPolling() {
    _cloudPollTimer?.cancel();
    _cloudPollTimer = Timer.periodic(const Duration(seconds: 3), (timer) async {
      if (_userRole != UserRole.guardian || _apiProvider.selectedDeviceId == null) return;
      
      try {
        final data = await _apiService.fetchLatestSensorData(_apiProvider.selectedDeviceId!);
        if (data.containsKey('navigation')) {
          debugPrint('☁️ Map Sync: Received location from cloud');
          _handleRemoteUpdate(data['navigation']);
        }
      } catch (e) {
        debugPrint('Cloud navigation poll failed: $e');
      }
    });
  }

  void _handleRemoteUpdate(Map<String, dynamic> data) {
    LatLng? current;
    LatLng? dest;
    
    if (data['currentPosition'] != null) {
      current = LatLng(data['currentPosition']['lat'], data['currentPosition']['lng']);
    }
    if (data['destination'] != null) {
      dest = LatLng(data['destination']['lat'], data['destination']['lng']);
    }

    List<LatLng> pts = [];
    if (data['routePoints'] != null) {
      pts = (data['routePoints'] as List).map<LatLng>((p) => LatLng(p['lat'], p['lng'])).toList();
    }

    _state = _state.copyWith(
      currentPosition: current,
      destination: dest,
      isNavigating: data['isNavigating'],
      isPaused: data['isPaused'],
      destinationAddress: data['destinationAddress'],
      distance: data['distance']?.toDouble(),
      estimatedTime: data['estimatedTime']?.toDouble(),
      routePoints: pts,
    );
    notifyListeners();
  }

  Future<void> _startLocationTracking() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        debugPrint('📍 GPS: Permission denied');
        return;
      }

      // Get initial position
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.best),
      );
      updatePosition(LatLng(pos.latitude, pos.longitude));

      // Subscribe to stream
      _positionSub?.cancel();
      _positionSub = Geolocator.getPositionStream(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.best,
          distanceFilter: 5,
        ),
      ).listen((pos) {
        updatePosition(LatLng(pos.latitude, pos.longitude));
      });
      debugPrint('📍 GPS: Tracking started');
    } catch (e) {
      debugPrint('📍 GPS Error: $e');
    }
  }

  Future<void> searchLocation(String query) async {
    _lastQuery = query;
    if (query.trim().isEmpty) {
      _state = _state.copyWith(searchResults: [], isSearching: false);
      notifyListeners();
      return;
    }
    
    _state = _state.copyWith(isSearching: true, searchResults: []);
    notifyListeners();

    final Map<String, String> params = {
      'q': query,
      'format': 'json',
      'addressdetails': '1',
      'limit': '10',
      'countrycodes': 'in',
    };

    if (_state.currentPosition != null) {
      final lat = _state.currentPosition!.latitude;
      final lon = _state.currentPosition!.longitude;
      params['viewbox'] = '${lon - 0.2},${lat + 0.2},${lon + 0.2},${lat - 0.2}';
      params['bounded'] = '0';
    }

    final uri = Uri.parse('https://nominatim.openstreetmap.org/search').replace(queryParameters: params);

    try {
      final res = await http.get(
        uri,
        headers: {
          'User-Agent': 'SmartWheelchairApp-v1-Nishal',
        },
      ).timeout(const Duration(seconds: 8));

      if (res.statusCode == 200) {
        final List data = jsonDecode(res.body);
        _state = _state.copyWith(
          searchResults: data.cast<Map<String, dynamic>>(),
          isSearching: false,
        );
      } else {
        await _searchWithPhoton(query);
      }
    } catch (e) {
      await _searchWithPhoton(query);
    }
    notifyListeners();
  }

  Future<void> _searchWithPhoton(String query) async {
    final uri = Uri.parse('https://photon.komoot.io/api/').replace(queryParameters: {
      'q': query,
      'limit': '5',
      'lat': _state.currentPosition?.latitude.toString(),
      'lon': _state.currentPosition?.longitude.toString(),
      'lang': 'en',
    });

    try {
      final res = await http.get(uri).timeout(const Duration(seconds: 8));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final List features = data['features'] ?? [];
        
        final mappedResults = features.map<Map<String, dynamic>>((f) {
          final props = f['properties'] ?? {};
          final coords = f['geometry']['coordinates'] as List;
          
          String name = props['name'] ?? '';
          String city = props['city'] ?? props['district'] ?? '';
          String country = props['country'] ?? '';
          String displayName = [name, city, country].where((s) => s.isNotEmpty).join(', ');

          return {
            'display_name': displayName.isEmpty ? 'Unknown Location' : displayName,
            'lat': coords[1].toString(),
            'lon': coords[0].toString(),
          };
        }).toList();

        _state = _state.copyWith(
          searchResults: mappedResults,
          isSearching: false,
        );
      } else {
        _state = _state.copyWith(isSearching: false);
      }
    } catch (e) {
      _state = _state.copyWith(isSearching: false);
    }
  }

  void clearSearchResults() {
    _state = _state.copyWith(searchResults: []);
    notifyListeners();
  }

  Future<void> selectSearchResult(Map<String, dynamic> item) async {
    final lat = double.parse(item['lat']);
    final lon = double.parse(item['lon']);
    final address = item['display_name'];
    
    _state = _state.copyWith(
      destination: LatLng(lat, lon),
      destinationAddress: address,
      searchResults: [],
    );
    notifyListeners();
    await fetchRoute();
  }

  Future<void> fetchRoute() async {
    if (_state.currentPosition == null || _state.destination == null) return;

    final from = '${_state.currentPosition!.longitude},${_state.currentPosition!.latitude}';
    final to = '${_state.destination!.longitude},${_state.destination!.latitude}';
    final uri = Uri.parse(
      'https://router.project-osrm.org/route/v1/driving/$from;$to',
    ).replace(queryParameters: {'overview': 'full', 'geometries': 'geojson'});

    try {
      final res = await http.get(uri);
      if (res.statusCode == 200) {
        final Map data = jsonDecode(res.body);
        if (data['routes'] != null && (data['routes'] as List).isNotEmpty) {
          final route = data['routes'][0];
          final coords = route['geometry']['coordinates'] as List;
          final distance = (route['distance'] as num).toDouble() / 1000.0; // km
          final duration = (route['duration'] as num).toDouble() / 60.0; // minutes

          final pts = coords.map<LatLng>((c) {
            final lon = (c[0] as num).toDouble();
            final lat = (c[1] as num).toDouble();
            return LatLng(lat, lon);
          }).toList();
          
          _state = _state.copyWith(
            routePoints: pts,
            distance: distance,
            estimatedTime: duration,
          );
          _broadcastState();
          notifyListeners();
        }
      }
    } catch (e) {
      debugPrint('Routing error: $e');
    }
  }

  void updatePosition(LatLng position) {
    debugPrint('📍 GPS Update: ${position.latitude}, ${position.longitude}');
    _state = _state.copyWith(currentPosition: position);
    _broadcastState();
    notifyListeners();
  }

  void setDestination(LatLng? destination, {String? address, List<LatLng>? route, double? dist, double? time}) {
    _state = _state.copyWith(
      destination: destination,
      destinationAddress: address,
      routePoints: route ?? [],
      distance: dist,
      estimatedTime: time,
    );
    _broadcastState();
    notifyListeners();
  }

  void setNavigating(bool navigating) {
    _state = _state.copyWith(isNavigating: navigating);
    _broadcastState();
    notifyListeners();
  }

  void setPaused(bool paused) {
    _state = _state.copyWith(isPaused: paused);
    _broadcastState();
    notifyListeners();
  }

  void _broadcastState() {
    // Bluetooth/Local Sync
    if (_userRole == UserRole.patient && _connectionProvider.isConnected) {
      _connectionProvider.sendMessage({
        'type': 'navigation_update',
        'data': _state.toJson(),
      });
    }

    // Cloud Sync
    if (_userRole == UserRole.patient && _apiProvider.selectedDeviceId != null) {
      debugPrint('☁️ Map Sync: Attempting upload for ${_apiProvider.selectedDeviceId}');
      _apiService.uploadData({
        'deviceId': _apiProvider.selectedDeviceId,
        'timestamp': DateTime.now().millisecondsSinceEpoch,
        'navigation': _state.toJson(),
      }).then((_) => debugPrint('☁️ Map Sync: Uploaded location to cloud ✅'))
        .catchError((e) => debugPrint('☁️ Map Sync: Cloud upload FAILED: $e ❌'));
    }
  }

  @override
  void dispose() {
    _msgSub?.cancel();
    _cloudPollTimer?.cancel();
    _positionSub?.cancel();
    super.dispose();
  }
}
