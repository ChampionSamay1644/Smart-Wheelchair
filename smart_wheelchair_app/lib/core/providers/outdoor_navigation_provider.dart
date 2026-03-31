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
  final bool isMapPageOpen;
  final String? currentInstruction;
  final double? nextTurnDistance;
  final List<Map<String, dynamic>> routeSteps;
  final int currentStepIndex;
  final double? totalRemainingDistance; // in meters

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
    this.isMapPageOpen = false,
    this.currentInstruction,
    this.nextTurnDistance,
    this.routeSteps = const [],
    this.currentStepIndex = 0,
    this.totalRemainingDistance,
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
    bool? isMapPageOpen,
    String? currentInstruction,
    double? nextTurnDistance,
    List<Map<String, dynamic>>? routeSteps,
    int? currentStepIndex,
    double? totalRemainingDistance,
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
      isMapPageOpen: isMapPageOpen ?? this.isMapPageOpen,
      currentInstruction: currentInstruction ?? this.currentInstruction,
      nextTurnDistance: nextTurnDistance ?? this.nextTurnDistance,
      routeSteps: routeSteps ?? this.routeSteps,
      currentStepIndex: currentStepIndex ?? this.currentStepIndex,
      totalRemainingDistance: totalRemainingDistance ?? this.totalRemainingDistance,
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
      'isMapPageOpen': isMapPageOpen,
      'currentInstruction': currentInstruction,
      'nextTurnDistance': nextTurnDistance,
      'totalRemainingDistance': totalRemainingDistance,
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
      isMapPageOpen: data['isMapPageOpen'] ?? false,
      currentInstruction: data['currentInstruction'],
      nextTurnDistance: data['nextTurnDistance']?.toDouble(),
      // We don't necessarily need to sync full route steps for simple tracking, but can
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
      currentStepIndex: 0,
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
    ).replace(queryParameters: {'overview': 'full', 'geometries': 'geojson', 'steps': 'true'});

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
          
          List<Map<String, dynamic>> parsedSteps = [];
          if (route['legs'] != null && (route['legs'] as List).isNotEmpty) {
            final stepsInfo = route['legs'][0]['steps'];
            if (stepsInfo != null) {
              parsedSteps = (stepsInfo as List).cast<Map<String, dynamic>>();
            }
          }
          
          String? initialInstruction;
          double? initialTurnDistance;
          if (parsedSteps.isNotEmpty) {
             final maneuver = parsedSteps[0]['maneuver'];
             final type = maneuver['type'] as String?;
             final modifier = maneuver['modifier'] as String?;
             final name = parsedSteps[0]['name'] as String?;
             
             if (type != null) {
               initialInstruction = '$type ${modifier ?? ""} ${name?.isNotEmpty == true ? "on $name" : ""}';
             }
             initialTurnDistance = (parsedSteps[0]['distance'] as num?)?.toDouble();
          }

          _state = _state.copyWith(
            routePoints: pts,
            distance: distance,
            estimatedTime: duration,
            routeSteps: parsedSteps,
            currentInstruction: initialInstruction ?? 'Head to destination',
            nextTurnDistance: initialTurnDistance,
            currentStepIndex: 0,
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
    
    String? currentInstruction = _state.currentInstruction;
    double? nextTurnDistance = _state.nextTurnDistance;
    List<Map<String, dynamic>> routeSteps = List.from(_state.routeSteps);
    int currentStepIndex = _state.currentStepIndex;

    if (_state.isNavigating && routeSteps.isNotEmpty && currentStepIndex < routeSteps.length) {
        final maneuver = routeSteps[currentStepIndex]['maneuver'];
        final loc = maneuver['location'] as List; // lon, lat
        final stepPos = LatLng((loc[1] as num).toDouble(), (loc[0] as num).toDouble());
        final distToStep = const Distance().as(LengthUnit.Meter, position, stepPos);
        
        if (distToStep < 20.0) {
           final nextIndex = currentStepIndex + 1;
           if (nextIndex < routeSteps.length) {
             final nextManeuver = routeSteps[nextIndex]['maneuver'];
             final type = nextManeuver['type'] as String?;
             final modifier = nextManeuver['modifier'] as String?;
             final name = routeSteps[nextIndex]['name'] as String?;
             
             if (type != null) {
               currentInstruction = '$type ${modifier ?? ""} ${name?.isNotEmpty == true ? "on $name" : ""}';
             }
             nextTurnDistance = (routeSteps[nextIndex]['distance'] as num?)?.toDouble();
             currentStepIndex = nextIndex;
           } else {
             currentInstruction = 'Arrived at destination';
             nextTurnDistance = 0;
           }
        } else {
          nextTurnDistance = distToStep;
        }

        // Calculate total remaining distance (current step to destination)
        double total = distToStep;
        if (currentStepIndex + 1 < routeSteps.length) {
          for (int i = currentStepIndex + 1; i < routeSteps.length; i++) {
            total += (routeSteps[i]['distance'] as num?)?.toDouble() ?? 0.0;
          }
        }
        
        // Jitter Filtering: Only update if change > 3m or very close (< 10m)
        final oldDist = _state.nextTurnDistance ?? 999.0;
        final currentDist = nextTurnDistance ?? 0.0;
        final delta = (currentDist - oldDist).abs();
        
        if (delta < 3.0 && currentDist > 10.0 && currentStepIndex == _state.currentStepIndex) {
           // Skip update to prevent jitter while stationary
           return;
        }

        _state = _state.copyWith(
          currentPosition: position,
          currentInstruction: currentInstruction,
          nextTurnDistance: nextTurnDistance,
          totalRemainingDistance: total,
          currentStepIndex: currentStepIndex,
        );
        _broadcastState();
        notifyListeners();
        return;
    }

    _state = _state.copyWith(
      currentPosition: position,
      currentInstruction: currentInstruction,
      nextTurnDistance: nextTurnDistance,
      currentStepIndex: currentStepIndex,
    );
    _broadcastState();
    notifyListeners();
  }

  void setMapPageOpen(bool isOpen) {
    _state = _state.copyWith(isMapPageOpen: isOpen);
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
