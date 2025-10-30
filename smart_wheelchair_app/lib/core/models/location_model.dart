import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:latlong2/latlong.dart';

class LocationUpdate {
  final String patientId;
  final LatLng position;
  final double? speed;
  final double? heading;
  final double? accuracy;
  final DateTime timestamp;

  const LocationUpdate({
    required this.patientId,
    required this.position,
    this.speed,
    this.heading,
    this.accuracy,
    required this.timestamp,
  });

  factory LocationUpdate.fromFirestore(
    DocumentSnapshot<Map<String, dynamic>> snapshot, [
    SnapshotOptions? options,
  ]) {
    final data = snapshot.data()!;
    final geoPoint = data['position'] as GeoPoint;
    return LocationUpdate(
      patientId: snapshot.id,
      position: LatLng(geoPoint.latitude, geoPoint.longitude),
      speed: data['speed'] as double?,
      heading: data['heading'] as double?,
      accuracy: data['accuracy'] as double?,
      timestamp: (data['timestamp'] as Timestamp).toDate(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'position': GeoPoint(position.latitude, position.longitude),
      if (speed != null) 'speed': speed,
      if (heading != null) 'heading': heading,
      if (accuracy != null) 'accuracy': accuracy,
      'timestamp': Timestamp.fromDate(timestamp),
    };
  }
}
