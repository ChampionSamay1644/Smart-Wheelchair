import 'dart:async';

import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter_google_places_hoc081098/flutter_google_places_hoc081098.dart';
import 'package:google_maps_webservice/places.dart';
import 'package:flutter_polyline_points/flutter_polyline_points.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

import 'widgets/search_bar_widget.dart';
import 'widgets/navigation_controls_widget.dart';

class OutdoorNavigationPage extends StatefulWidget {
  const OutdoorNavigationPage({super.key});

  @override
  State<OutdoorNavigationPage> createState() => _OutdoorNavigationPageState();
}

class _OutdoorNavigationPageState extends State<OutdoorNavigationPage> {
  // Google Maps controller
  final Completer<GoogleMapController> _controller = Completer();

  // Current location
  LatLng? _currentLocation;

  // Destination location
  LatLng? _destinationLocation;
  String _destinationName = '';

  // Map markers
  final Set<Marker> _markers = {};

  // Polylines for route
  final Map<PolylineId, Polyline> _polylines = {};
  List<LatLng> _routeCoordinates = [];

  // Navigation state
  bool _isNavigating = false;
  bool _isLoading = true;
  bool _permissionDenied = false;

  // Google Places API client
  final String _apiKey = 'YOUR_API_KEY_HERE';
  late GoogleMapsPlaces _places;

  @override
  void initState() {
    super.initState();
    _places = GoogleMapsPlaces(apiKey: _apiKey);
    _checkLocationPermission();
  }

  // Check and request location permissions
  Future<void> _checkLocationPermission() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Check if location services are enabled
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      // Location services are disabled
      setState(() {
        _isLoading = false;
        _permissionDenied = true;
      });
      return;
    }

    // Check location permission
    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      // Request permission
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        // Permission denied
        setState(() {
          _isLoading = false;
          _permissionDenied = true;
        });
        return;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      // Permission denied forever
      setState(() {
        _isLoading = false;
        _permissionDenied = true;
      });
      return;
    }

    // Permission granted, get current location
    _getCurrentLocation();
  }

  // Get current location
  Future<void> _getCurrentLocation() async {
    try {
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      setState(() {
        _currentLocation = LatLng(position.latitude, position.longitude);
        _addMarker(
          _currentLocation!,
          'Current Location',
          BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueBlue),
        );
        _isLoading = false;
      });

      // Center the map on current location
      _animateToCurrentLocation();

      // Start position updates for navigation
      if (_isNavigating) {
        _startLocationUpdates();
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      print('Error getting current location: $e');
    }
  }

  // Animate camera to current location
  Future<void> _animateToCurrentLocation() async {
    if (_currentLocation == null || !_controller.isCompleted) return;

    final GoogleMapController controller = await _controller.future;
    controller.animateCamera(
      CameraUpdate.newCameraPosition(
        CameraPosition(target: _currentLocation!, zoom: 17.0),
      ),
    );
  }

  // Add marker to the map
  void _addMarker(LatLng position, String markerId, BitmapDescriptor icon) {
    final Marker marker = Marker(
      markerId: MarkerId(markerId),
      position: position,
      icon: icon,
      infoWindow: InfoWindow(title: markerId),
    );

    setState(() {
      _markers.add(marker);
    });
  }

  // Search for a place
  Future<void> _searchPlace() async {
    try {
      Prediction? prediction = await PlacesAutocomplete.show(
        context: context,
        apiKey: _apiKey,
        mode: Mode.overlay,
        types: [],
        strictbounds: false,
        components: [
          Component(Component.country, 'in'),
        ], // Limit search to India (can be changed or removed)
        onError: (err) {
          print('Error during place search: $err');
        },
      );

      if (prediction != null) {
        // Get details of the selected place
        PlacesDetailsResponse detail = await _places.getDetailsByPlaceId(
          prediction.placeId!,
        );
        final place = detail.result;
        final lat = place.geometry?.location.lat;
        final lng = place.geometry?.location.lng;

        if (lat != null && lng != null) {
          setState(() {
            _destinationLocation = LatLng(lat, lng);
            _destinationName = place.name;
            _addMarker(
              _destinationLocation!,
              'Destination: $_destinationName',
              BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed),
            );
          });

          // Move map camera to show both points
          _showBothLocationsOnMap();

          // Get directions
          await _getDirections();
        }
      }
    } catch (e) {
      print('Error searching place: $e');
    }
  }

  // Show both current location and destination on the map
  Future<void> _showBothLocationsOnMap() async {
    if (_currentLocation == null ||
        _destinationLocation == null ||
        !_controller.isCompleted)
      return;

    final GoogleMapController controller = await _controller.future;

    final bounds = LatLngBounds(
      southwest: LatLng(
        _currentLocation!.latitude < _destinationLocation!.latitude
            ? _currentLocation!.latitude
            : _destinationLocation!.latitude,
        _currentLocation!.longitude < _destinationLocation!.longitude
            ? _currentLocation!.longitude
            : _destinationLocation!.longitude,
      ),
      northeast: LatLng(
        _currentLocation!.latitude > _destinationLocation!.latitude
            ? _currentLocation!.latitude
            : _destinationLocation!.latitude,
        _currentLocation!.longitude > _destinationLocation!.longitude
            ? _currentLocation!.longitude
            : _destinationLocation!.longitude,
      ),
    );

    controller.animateCamera(CameraUpdate.newLatLngBounds(bounds, 100));
  }

  // Get directions between current location and destination
  Future<void> _getDirections() async {
    if (_currentLocation == null || _destinationLocation == null) return;

    PolylinePoints polylinePoints = PolylinePoints();

    PolylineResult result = await polylinePoints.getRouteBetweenCoordinates(
      _apiKey,
      PointLatLng(_currentLocation!.latitude, _currentLocation!.longitude),
      PointLatLng(
        _destinationLocation!.latitude,
        _destinationLocation!.longitude,
      ),
    );

    _routeCoordinates.clear();
    if (result.points.isNotEmpty) {
      for (var point in result.points) {
        _routeCoordinates.add(LatLng(point.latitude, point.longitude));
      }
    }

    setState(() {
      _addPolyline();
    });
  }

  // Add polyline to the map
  void _addPolyline() {
    PolylineId id = const PolylineId('poly');
    Polyline polyline = Polyline(
      polylineId: id,
      color: Colors.blue,
      points: _routeCoordinates,
      width: 5,
    );

    setState(() {
      _polylines[id] = polyline;
    });
  }

  // Start navigation
  void _startNavigation() {
    if (_currentLocation == null || _destinationLocation == null) return;

    setState(() {
      _isNavigating = true;
    });

    _startLocationUpdates();
  }

  // Stop navigation
  void _stopNavigation() {
    setState(() {
      _isNavigating = false;

      // Clear route but keep destination
      _routeCoordinates.clear();
      _polylines.clear();

      // Update current location marker
      _getCurrentLocation();
    });
  }

  // Location updates stream for live navigation
  StreamSubscription<Position>? _positionStreamSubscription;

  void _startLocationUpdates() {
    // Cancel existing subscription if any
    _positionStreamSubscription?.cancel();

    // Start new subscription
    _positionStreamSubscription =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.high,
            distanceFilter: 5, // Update every 5 meters
          ),
        ).listen((Position position) {
          setState(() {
            _currentLocation = LatLng(position.latitude, position.longitude);

            // Update current location marker
            _markers.removeWhere(
              (marker) => marker.markerId.value == 'Current Location',
            );

            _addMarker(
              _currentLocation!,
              'Current Location',
              BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueBlue),
            );
          });

          // Keep map centered on current location during navigation
          if (_isNavigating) {
            _animateToCurrentLocation();
          }
        });
  }

  @override
  void dispose() {
    _positionStreamSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Outdoor Navigation',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _isLoading
          ? const Center(
              child: SpinKitDoubleBounce(color: Colors.blue, size: 50.0),
            )
          : _permissionDenied
          ? _buildPermissionDeniedView()
          : _buildMapView(),
    );
  }

  Widget _buildPermissionDeniedView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.location_off, size: 70, color: Colors.red),
            const SizedBox(height: 20),
            const Text(
              'Location permission denied',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            const Text(
              'This app needs location permission to provide navigation services. Please enable location permissions in settings.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => openAppSettings(),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                textStyle: const TextStyle(fontSize: 16),
              ),
              child: const Text('Open Settings'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMapView() {
    return Stack(
      children: [
        // Map Widget
        GoogleMap(
          mapType: MapType.normal,
          initialCameraPosition: _currentLocation != null
              ? CameraPosition(target: _currentLocation!, zoom: 17.0)
              : const CameraPosition(
                  target: LatLng(20.5937, 78.9629),
                  zoom: 5.0,
                ), // Fallback to center of India
          myLocationEnabled: true,
          myLocationButtonEnabled: false,
          zoomControlsEnabled: false,
          markers: _markers,
          polylines: Set<Polyline>.of(_polylines.values),
          onMapCreated: (GoogleMapController controller) {
            _controller.complete(controller);
          },
        ),

        // Search Bar
        Positioned(
          top: 16,
          left: 16,
          right: 16,
          child: SearchBarWidget(
            onTap: _searchPlace,
            destinationName: _destinationName,
          ),
        ),

        // Navigation Controls
        Positioned(
          bottom: 30,
          left: 16,
          right: 16,
          child: NavigationControlsWidget(
            isNavigating: _isNavigating,
            hasDestination: _destinationLocation != null,
            onStartNavigation: _startNavigation,
            onStopNavigation: _stopNavigation,
            onRecenterMap: _animateToCurrentLocation,
          ),
        ),
      ],
    );
  }
}
