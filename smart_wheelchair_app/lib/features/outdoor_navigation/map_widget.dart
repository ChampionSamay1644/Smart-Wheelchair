import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class MapWidget extends StatefulWidget {
  final LatLng currentPosition;
  final List<LatLng> routePoints;
  final LatLng? destination;
  final bool isGuardianView;
  final double? patientSpeed;

  const MapWidget({
    super.key,
    required this.currentPosition,
    required this.routePoints,
    this.destination,
    this.isGuardianView = false,
    this.patientSpeed,
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
          // Use single-host OSM tile URL (avoid {s} subdomains warning)
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
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
              width: 80,
              height: 80,
              child: Column(
                children: [
                  if (widget.isGuardianView && widget.patientSpeed != null)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withAlpha(40),
                            blurRadius: 4,
                          ),
                        ],
                      ),
                      child: Text(
                        '${(widget.patientSpeed! * 3.6).toStringAsFixed(1)} km/h',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  const SizedBox(height: 4),
                  Container(
                    decoration: BoxDecoration(
                      color: widget.isGuardianView ? Colors.red : Colors.blue,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color:
                              (widget.isGuardianView ? Colors.red : Colors.blue)
                                  .withAlpha(100),
                          blurRadius: 8,
                          spreadRadius: 4,
                        ),
                      ],
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      widget.isGuardianView
                          ? Icons.wheelchair_pickup
                          : Icons.my_location,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                ],
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
