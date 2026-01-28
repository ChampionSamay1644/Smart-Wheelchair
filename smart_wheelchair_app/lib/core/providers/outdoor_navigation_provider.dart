import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;
import '../enums.dart';
import 'connection_provider.dart';

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
  UserRole? _userRole;
  
  NavigationState _state = NavigationState();
  NavigationState get state => _state;

  StreamSubscription? _msgSub;
  String _lastQuery = '';
  bool get lastQueryNotEmpty => _lastQuery.trim().isNotEmpty;

  OutdoorNavigationProvider(this._connectionProvider) {
    _initSync();
  }

  void setUserRole(UserRole role) {
    _userRole = role;
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
      'countrycodes': 'in', // Restrict search to India for accuracy
    };

    // Prioritize results near current position if available
    if (_state.currentPosition != null) {
      final lat = _state.currentPosition!.latitude;
      final lon = _state.currentPosition!.longitude;
      params['viewbox'] = '${lon - 0.2},${lat + 0.2},${lon + 0.2},${lat - 0.2}';
      params['bounded'] = '0'; // Prioritize but don't strictly limit
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
        debugPrint('Nominatim 403/Error: ${res.statusCode}. Falling back to Photon...');
        await _searchWithPhoton(query);
      }
    } catch (e) {
      debugPrint('Nominatim failed: $e. Falling back to Photon...');
      await _searchWithPhoton(query);
    }
    notifyListeners();
  }

  /// Photon (Komoot) fallback - often faster and more lenient with rate limits
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
          
          // Build a display name similar to Nominatim
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
      debugPrint('Photon fallback also failed: $e');
      _state = _state.copyWith(
        isSearching: false,
        searchResults: [
          {'display_name': 'Search failed. Please check connection.'}
        ],
      );
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
    if (_userRole == UserRole.patient && _connectionProvider.isConnected) {
      _connectionProvider.sendMessage({
        'type': 'navigation_update',
        'data': _state.toJson(),
      });
    }
  }

  @override
  void dispose() {
    _msgSub?.cancel();
    super.dispose();
  }
}
