// Outdoor navigation page: search -> geocode (Nominatim) -> route (OSRM) -> simulate navigation
// Minimal MVP implementation

// ignore_for_file: avoid_print
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/providers/outdoor_navigation_provider.dart';
import '../../core/providers/api_provider.dart';
import 'map_widget.dart';
import 'search_bar_widget.dart';
import 'navigation_controls_widget.dart';

class OutdoorNavigationPage extends StatefulWidget {
  const OutdoorNavigationPage({super.key});

  @override
  State<OutdoorNavigationPage> createState() => _OutdoorNavigationPageState();
}

class _OutdoorNavigationPageState extends State<OutdoorNavigationPage> {
  late OutdoorNavigationProvider _navProvider;

  @override
  void initState() {
    super.initState();
    // Ensure we mark the map UI as open after the first build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _navProvider.setMapPageOpen(true);
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Cache the provider reference safely while the widget is still active
    _navProvider = Provider.of<OutdoorNavigationProvider>(context, listen: false);
  }

  @override
  void dispose() {
    // Use microtask to avoid 'setState() called when widget tree was locked' error
    // which was causing the delay in the mini-map appearing on the dashboard.
    final provider = _navProvider;
    Future.microtask(() {
      provider.setMapPageOpen(false);
    });
    debugPrint('🎨 OutdoorNavigationPage: Disposed - Map UI marked as CLOSED');
    super.dispose();
  }



  void _startNavigation() {
    final navProvider = context.read<OutdoorNavigationProvider>();
    navProvider.setNavigating(true);
    navProvider.setPaused(false);
  }

  void _stopNavigation() {
    final navProvider = context.read<OutdoorNavigationProvider>();
    navProvider.setNavigating(false);
    navProvider.setPaused(false);
  }

  void _togglePause() {
    final navProvider = context.read<OutdoorNavigationProvider>();
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
                    if (state.currentPosition != null)
                      MapWidget(
                        currentPosition: state.currentPosition!,
                        routePoints: state.routePoints,
                        destination: state.destination,
                        onTap: (point) async {
                          nav.setDestination(point);
                          await nav.fetchRoute();
                        },
                      )
                    else
                      Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const CircularProgressIndicator(),
                            const SizedBox(height: 16),
                            Consumer<ApiProvider>(
                              builder: (context, api, _) {
                                String msg = 'Waiting for GPS signal...';
                                if (api.apiUrl == null || api.apiUrl!.isEmpty) {
                                  msg = '⚠️ API URL not configured in Settings';
                                } else if (api.selectedDeviceId == null) {
                                  msg = '⚠️ Wheelchair Device ID not set';
                                }
                                return Text(
                                  msg,
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(fontWeight: FontWeight.bold),
                                );
                              },
                            ),
                          ],
                        ),
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
                        color: Colors.black.withValues(alpha: 0.05),
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
