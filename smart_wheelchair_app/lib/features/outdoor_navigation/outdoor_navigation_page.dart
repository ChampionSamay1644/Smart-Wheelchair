// Outdoor navigation page: search -> geocode (Nominatim) -> route (OSRM) -> simulate navigation
// Minimal MVP implementation

// ignore_for_file: avoid_print
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;

import 'map_widget.dart';
import 'search_bar_widget.dart';
import 'navigation_controls_widget.dart';

class OutdoorNavigationPage extends StatefulWidget {
  final bool remoteView; // if true, this is guardian viewing patient location

  const OutdoorNavigationPage({super.key, this.remoteView = false});

  @override
  State<OutdoorNavigationPage> createState() => _OutdoorNavigationPageState();
}

class _OutdoorNavigationPageState extends State<OutdoorNavigationPage> {
  LatLng? _currentPosition;
  LatLng? _destination;
  List<LatLng> _routePoints = [];
  double _currentSpeed = 0.0; // m/s
  StreamSubscription<Position>? _posSub;

  bool _isNavigating = false;
  bool _isPaused = false;

  final LatLng _fallbackPosition = LatLng(28.7041, 77.1025);

  @override
  void initState() {
    super.initState();
    _initLocation();
  }

  Future<void> _initLocation() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        setState(() {
          _currentPosition = _fallbackPosition;
        });
        return;
      }

      // Optimize location settings for faster updates and better accuracy
      const locationSettings = LocationSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0, // Update on any movement
        timeLimit: Duration(seconds: 10), // Allow longer time to get a GPS fix
      );
      _posSub = Geolocator.getPositionStream(locationSettings: locationSettings)
          .listen((pos) {
            if (!mounted) return;
            setState(() {
              _currentPosition = LatLng(pos.latitude, pos.longitude);
              _currentSpeed = pos.speed; // meters/sec
            });
          });
    } catch (e) {
      debugPrint('Failed to get location: $e');
      setState(() => _currentPosition = _fallbackPosition);
    }
  }

  @override
  void dispose() {
    _posSub?.cancel();
    // No timers to cancel; location subscription handled above
    super.dispose();
  }

  Future<void> _onSearch(String query) async {
    if (query.trim().isEmpty) return;
    final uri = Uri.parse(
      'https://nominatim.openstreetmap.org/search',
    ).replace(queryParameters: {'q': query, 'format': 'json', 'limit': '1'});

    try {
      final res = await http.get(
        uri,
        headers: {'User-Agent': 'SmartWheelchair/1.0'},
      );
      if (res.statusCode == 200) {
        final List data = jsonDecode(res.body);
        if (data.isNotEmpty) {
          final item = data.first;
          final lat = double.parse(item['lat']);
          final lon = double.parse(item['lon']);
          setState(() {
            _destination = LatLng(lat, lon);
          });
          await _fetchRoute();
        } else {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No results from geocoding')),
          );
        }
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Geocoding failed')));
      }
    } catch (e) {
      debugPrint('Geocoding error: $e');
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Geocoding error')));
    }
  }

  Future<void> _fetchRoute() async {
    if (_currentPosition == null || _destination == null) return;

    final from = '${_currentPosition!.longitude},${_currentPosition!.latitude}';
    final to = '${_destination!.longitude},${_destination!.latitude}';
    final uri = Uri.parse(
      'https://router.project-osrm.org/route/v1/driving/$from;$to',
    ).replace(queryParameters: {'overview': 'full', 'geometries': 'geojson'});

    try {
      final res = await http.get(uri);
      if (res.statusCode == 200) {
        final Map data = jsonDecode(res.body);
        if (data['routes'] != null && (data['routes'] as List).isNotEmpty) {
          final coords = data['routes'][0]['geometry']['coordinates'] as List;
          final pts = coords.map<LatLng>((c) {
            final lon = (c[0] as num).toDouble();
            final lat = (c[1] as num).toDouble();
            return LatLng(lat, lon);
          }).toList();
          setState(() => _routePoints = pts);
        } else {
          if (!mounted) return;
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(const SnackBar(content: Text('No route found')));
        }
      } else {
        debugPrint('Routing failed: ${res.statusCode} ${res.body}');
        if (!mounted) return;
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Routing service error')));
      }
    } catch (e) {
      debugPrint('Routing error: $e');
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Routing error')));
    }
  }

  void _startNavigation() {
    if (_routePoints.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('No route to navigate')));
      return;
    }

    setState(() {
      _isNavigating = true;
      _isPaused = false;
    });

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Navigation started')));
  }

  // Simulation removed: navigation relies on real position updates from GPS

  void _stopNavigation() {
    // stop navigation: rely on location updates from GPS
    setState(() {
      _isNavigating = false;
      _isPaused = false;
      _routePoints = [];
      _destination = null;
      _currentSpeed = 0.0;
    });

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Navigation stopped')));
  }

  void _togglePauseResume() {
    if (!_isNavigating) return;
    setState(() => _isPaused = !_isPaused);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(_isPaused ? 'Navigation paused' : 'Navigation resumed'),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: widget.remoteView
          ? null
          : AppBar(title: const Text('Outdoor Navigation')),
      body: Stack(
        children: [
          MapWidget(
            currentPosition: _currentPosition ?? _fallbackPosition,
            routePoints: widget.remoteView ? [] : _routePoints,
            destination: widget.remoteView ? null : _destination,
            isGuardianView: widget.remoteView,
            patientSpeed: widget.remoteView ? _currentSpeed : null,
          ),
          if (!widget.remoteView)
            Positioned(
              left: 0,
              right: 0,
              top: 0,
              child: SearchBarWidget(onSearch: _onSearch),
            ),
          if (widget.remoteView)
            Positioned(
              right: 12,
              top: 12,
              child: Card(
                color: const Color.fromRGBO(255, 255, 255, 0.9),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  child: Column(
                    children: [
                      const Text(
                        'Patient Speed',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text('${(_currentSpeed * 3.6).toStringAsFixed(1)} km/h'),
                    ],
                  ),
                ),
              ),
            ),
          if (!widget.remoteView)
            Positioned(
              left: 0,
              right: 0,
              bottom: 8,
              child: NavigationControlsWidget(
                isNavigating: _isNavigating,
                isPaused: _isPaused,
                onStart: _startNavigation,
                onStop: _stopNavigation,
                onPauseResume: _togglePauseResume,
              ),
            ),
        ],
      ),
    );
  }
}
