import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class MapWidget extends StatelessWidget {
  final Function(GoogleMapController) onMapCreated;
  final CameraPosition initialCameraPosition;
  final Set<Marker> markers;
  final Set<Polyline> polylines;

  const MapWidget({
    super.key,
    required this.onMapCreated,
    required this.initialCameraPosition,
    required this.markers,
    required this.polylines,
  });

  @override
  Widget build(BuildContext context) {
    return GoogleMap(
      mapType: MapType.normal,
      initialCameraPosition: initialCameraPosition,
      myLocationEnabled: true,
      myLocationButtonEnabled: false,
      zoomControlsEnabled: false,
      markers: markers,
      polylines: polylines,
      onMapCreated: onMapCreated,
    );
  }
}
