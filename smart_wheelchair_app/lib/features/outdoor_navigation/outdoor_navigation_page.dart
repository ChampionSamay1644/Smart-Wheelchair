// Outdoor navigation page: search -> geocode (Nominatim) -> route (OSRM) -> simulate navigation
// Minimal MVP implementation

// ignore_for_file: avoid_print
import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;

import '../../core/providers/outdoor_navigation_provider.dart';
import 'map_widget.dart';
import 'search_bar_widget.dart';
import 'navigation_controls_widget.dart';

class OutdoorNavigationPage extends StatefulWidget {
  const OutdoorNavigationPage({super.key});

  @override
  State<OutdoorNavigationPage> createState() => _OutdoorNavigationPageState();
}

class _OutdoorNavigationPageState extends State<OutdoorNavigationPage> {
  StreamSubscription<Position>? _positionSub;
  final LatLng _fallbackPosition = LatLng(28.7041, 77.1025);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initLocation();
    });
  }

  @override
  void dispose() {
    _positionSub?.cancel();
    super.dispose();
  }

  Future<void> _initLocation() async {
    final navProvider = context.read<OutdoorNavigationProvider>();
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        navProvider.updatePosition(_fallbackPosition);
        return;
      }

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.best),
      );
      if (!mounted) return;
      navProvider.updatePosition(LatLng(pos.latitude, pos.longitude));
    } catch (e) {
      debugPrint('Failed to get location: $e');
      navProvider.updatePosition(_fallbackPosition);
    }
  }

  void _startNavigation() async {
    final navProvider = context.read<OutdoorNavigationProvider>();
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission != LocationPermission.always && permission != LocationPermission.whileInUse) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied || permission == LocationPermission.deniedForever) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Permission denied')));
        return;
      }

      _positionSub?.cancel();
      _positionSub = Geolocator.getPositionStream(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.best, distanceFilter: 5),
      ).listen((pos) {
        if (!mounted) return;
        navProvider.updatePosition(LatLng(pos.latitude, pos.longitude));
      });

      navProvider.setNavigating(true);
      navProvider.setPaused(false);
    } catch (e) {
      debugPrint('Failed to start navigation: $e');
    }
  }

  void _stopNavigation() {
    final navProvider = context.read<OutdoorNavigationProvider>();
    _positionSub?.cancel();
    _positionSub = null;
    navProvider.setNavigating(false);
    navProvider.setPaused(false);
  }

  void _togglePause() {
    final navProvider = context.read<OutdoorNavigationProvider>();
    if (_positionSub == null) return;
    if (navProvider.state.isPaused) {
      _positionSub!.resume();
    } else {
      _positionSub!.pause();
    }
    navProvider.setPaused(!navProvider.state.isPaused);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Outdoor Navigation')),
      body: Consumer<OutdoorNavigationProvider>(
        builder: (context, nav, _) {
          final state = nav.state;
          return Column(
            children: [
              SearchBarWidget(
                onSearch: nav.searchLocation,
                isSearching: state.isSearching,
              ),
              Expanded(
                child: Stack(
                  children: [
                    MapWidget(
                      currentPosition: state.currentPosition ?? _fallbackPosition,
                      routePoints: state.routePoints,
                      destination: state.destination,
                      onTap: (point) async {
                        nav.setDestination(point);
                        await nav.fetchRoute();
                      },
                    ),
                    if (state.searchResults.isNotEmpty)
                      Positioned(
                        top: 0,
                        left: 10,
                        right: 10,
                        child: Material(
                          elevation: 4,
                          borderRadius: BorderRadius.circular(10),
                          child: Container(
                            constraints: const BoxConstraints(maxHeight: 250),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: ListView.separated(
                              shrinkWrap: true,
                              itemCount: state.searchResults.length,
                              separatorBuilder: (context, index) => const Divider(height: 1),
                              itemBuilder: (context, index) {
                                final item = state.searchResults[index];
                                return ListTile(
                                  leading: const Icon(Icons.location_on_outlined, color: Colors.blue),
                                  title: Text(item['display_name'] ?? 'Unknown location'),
                                  onTap: () => nav.selectSearchResult(item),
                                );
                              },
                            ),
                          ),
                        ),
                      ),
                    if (!state.isSearching && state.searchResults.isEmpty && context.watch<OutdoorNavigationProvider>().lastQueryNotEmpty)
                      Positioned(
                        top: 0,
                        left: 10,
                        right: 10,
                        child: Material(
                          elevation: 2,
                          borderRadius: BorderRadius.circular(10),
                          child: Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: const Row(
                              children: [
                                Icon(Icons.search_off, color: Colors.grey),
                                SizedBox(width: 10),
                                Text('No results found. Try a different search.', style: TextStyle(color: Colors.grey)),
                              ],
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              if (state.destinationAddress != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 10,
                        offset: const Offset(0, -5),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.location_on, color: Colors.red, size: 24),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              state.destinationAddress!,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          _buildMetric(Icons.straighten, '${state.distance?.toStringAsFixed(2) ?? "0"} km', 'Distance'),
                          _buildMetric(Icons.timer, '${state.estimatedTime?.toStringAsFixed(0) ?? "0"} min', 'ETA'),
                        ],
                      ),
                    ],
                  ),
                ),
              NavigationControlsWidget(
                isNavigating: state.isNavigating,
                isPaused: state.isPaused,
                onStart: _startNavigation,
                onStop: _stopNavigation,
                onTogglePause: _togglePause,
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildMetric(IconData icon, String value, String label) {
    return Column(
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 16, color: Colors.blue),
            const SizedBox(width: 4),
            Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        Text(label, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
      ],
    );
  }
}
