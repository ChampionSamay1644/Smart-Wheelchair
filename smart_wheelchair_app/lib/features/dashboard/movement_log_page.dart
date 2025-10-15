import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

class MovementLogPage extends StatelessWidget {
  const MovementLogPage({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Movement Log'),
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Movement History'),
              Tab(text: 'Map View'),
            ],
          ),
        ),
        body: TabBarView(children: [_buildMovementHistory(), _buildMapView()]),
      ),
    );
  }

  Widget _buildMovementHistory() {
    final movements = [
      _Movement(
        time: DateTime.now().subtract(const Duration(minutes: 30)),
        type: MovementType.forward,
        duration: const Duration(seconds: 45),
        distance: 15.0,
      ),
      _Movement(
        time: DateTime.now().subtract(const Duration(hours: 1)),
        type: MovementType.turn,
        duration: const Duration(seconds: 10),
        angle: 90.0,
      ),
      _Movement(
        time: DateTime.now().subtract(const Duration(hours: 2)),
        type: MovementType.backward,
        duration: const Duration(seconds: 20),
        distance: 5.0,
      ),
      // Add more movements as needed
    ];

    return ListView.builder(
      padding: const EdgeInsets.all(8),
      itemCount: movements.length,
      itemBuilder: (context, index) {
        final movement = movements[index];
        return Card(
          margin: const EdgeInsets.symmetric(vertical: 4),
          child: ListTile(
            leading: _buildMovementIcon(movement.type),
            title: Text(_getMovementDescription(movement)),
            subtitle: Text(_formatTime(movement.time)),
            trailing: Text(_formatDuration(movement.duration)),
          ),
        );
      },
    );
  }

  Widget _buildMapView() {
    // Example coordinates for demonstration
    const center = LatLng(51.509364, -0.128928);

    return FlutterMap(
      options: const MapOptions(initialCenter: center, initialZoom: 15.0),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: 'com.example.smart_wheelchair_app',
        ),
        // Example markers and polylines could be added here
      ],
    );
  }

  Widget _buildMovementIcon(MovementType type) {
    IconData icon;
    Color color;

    switch (type) {
      case MovementType.forward:
        icon = FontAwesomeIcons.arrowUp;
        color = Colors.green;
      case MovementType.backward:
        icon = FontAwesomeIcons.arrowDown;
        color = Colors.orange;
      case MovementType.turn:
        icon = FontAwesomeIcons.arrowRotateRight;
        color = Colors.blue;
    }

    return CircleAvatar(
      backgroundColor: color.withAlpha(51), // 0.2 * 255 ≈ 51
      child: FaIcon(icon, color: color, size: 16),
    );
  }

  String _getMovementDescription(_Movement movement) {
    switch (movement.type) {
      case MovementType.forward:
        return 'Moved forward ${movement.distance?.toStringAsFixed(1)} meters';
      case MovementType.backward:
        return 'Moved backward ${movement.distance?.toStringAsFixed(1)} meters';
      case MovementType.turn:
        return 'Turned $movement.angle degrees';
    }
  }

  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';
  }

  String _formatDuration(Duration duration) {
    final seconds = duration.inSeconds;
    if (seconds < 60) {
      return '$seconds sec';
    }
    final minutes = duration.inMinutes;
    final remainingSeconds = seconds - (minutes * 60);
    return '$minutes min $remainingSeconds sec';
  }
}

enum MovementType { forward, backward, turn }

class _Movement {
  final DateTime time;
  final MovementType type;
  final Duration duration;
  final double? distance;
  final double? angle;

  _Movement({
    required this.time,
    required this.type,
    required this.duration,
    this.distance,
    this.angle,
  });
}
