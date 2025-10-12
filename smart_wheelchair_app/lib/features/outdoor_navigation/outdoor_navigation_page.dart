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
  const OutdoorNavigationPage({super.key});

  @override
  State<OutdoorNavigationPage> createState() => _OutdoorNavigationPageState();
}

class _OutdoorNavigationPageState extends State<OutdoorNavigationPage> {
  LatLng? _currentPosition;
  LatLng? _destination;
  List<LatLng> _routePoints = [];

  bool _isNavigating = false;
  Timer? _navTimer;
  int _navIndex = 0;

  final LatLng _fallbackPosition = LatLng(28.7041, 77.1025);

  @override
  void initState() {
    super.initState();
    _initLocation();
  }

  @override
  void dispose() {
    _navTimer?.cancel();
    super.dispose();
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

      // Use LocationSettings (replacement for desiredAccuracy in newer geolocator)
      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.best,
        ),
      );
      if (!mounted) return;
      setState(() {
        _currentPosition = LatLng(pos.latitude, pos.longitude);
      });
    } catch (e) {
      print('Failed to get location: $e');
      setState(() => _currentPosition = _fallbackPosition);
    }
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
      print('Geocoding error: $e');
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
        print('Routing failed: ${res.statusCode} ${res.body}');
        if (!mounted) return;
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Routing service error')));
      }
    } catch (e) {
      print('Routing error: $e');
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

    _navTimer?.cancel();
    _navIndex = 0;
    setState(() => _isNavigating = true);

    _navTimer = Timer.periodic(const Duration(seconds: 2), (t) {
      if (_navIndex >= _routePoints.length) {
        _stopNavigation();
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Arrived at destination')));
        return;
      }
      setState(() => _currentPosition = _routePoints[_navIndex]);
      _navIndex += 1;
    });
  }

  void _stopNavigation() {
    _navTimer?.cancel();
    _navTimer = null;
    setState(() => _isNavigating = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Outdoor Navigation')),
      body: Column(
        children: [
          SearchBarWidget(onSearch: _onSearch),
          Expanded(
            child: MapWidget(
              currentPosition: _currentPosition ?? _fallbackPosition,
              routePoints: _routePoints,
              destination: _destination,
            ),
          ),
          NavigationControlsWidget(
            isNavigating: _isNavigating,
            onStart: _startNavigation,
            onStop: _stopNavigation,
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
