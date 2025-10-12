import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class MapWidget extends StatefulWidget {
  final LatLng currentPosition;
  final List<LatLng> routePoints;
  final LatLng? destination;

  const MapWidget({
    super.key,
    required this.currentPosition,
    required this.routePoints,
    this.destination,
  });

  @override
  State<MapWidget> createState() => _MapWidgetState();
}

class _MapWidgetState extends State<MapWidget> {
  final MapController _mapCtrl = MapController();
  double _zoom = 15.0;

  @override
  void didUpdateWidget(covariant MapWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    // center map when currentPosition changes
    if (oldWidget.currentPosition != widget.currentPosition) {
      _mapCtrl.move(widget.currentPosition, _zoom);
    }
  }

  @override
  Widget build(BuildContext context) {
    return FlutterMap(
      mapController: _mapCtrl,
      options: MapOptions(
        // flutter_map v8 uses initialCenter/initialZoom
        initialCenter: widget.currentPosition,
        initialZoom: _zoom,
        onPositionChanged: (pos, hasGesture) {
          // keep local zoom in sync
          _zoom = pos.zoom;
        },
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
          subdomains: const ['a', 'b', 'c'],
          userAgentPackageName: 'com.example.smart_wheelchair_app',
        ),
        if (widget.routePoints.isNotEmpty)
          PolylineLayer(
            polylines: [
              Polyline(
                points: widget.routePoints,
                color: Colors.blue,
                strokeWidth: 4.0,
              ),
            ],
          ),
        MarkerLayer(
          markers: [
            Marker(
              point: widget.currentPosition,
              width: 40,
              height: 40,
              child: const Icon(
                Icons.my_location,
                color: Colors.blue,
                size: 32,
              ),
            ),
            if (widget.destination != null)
              Marker(
                point: widget.destination!,
                width: 40,
                height: 40,
                child: const Icon(Icons.flag, color: Colors.red, size: 32),
              ),
          ],
        ),
      ],
    );
  }
}
