import 'package:flutter/foundation.dart';

class MovementEntry {
  final DateTime time;
  final String mode; // manual, voice, remote
  final String action; // forward/back/left/right/other
  final double? latitude;
  final double? longitude;

  MovementEntry({
    required this.time,
    required this.mode,
    required this.action,
    this.latitude,
    this.longitude,
  });
}

class MovementLogProvider extends ChangeNotifier {
  final List<MovementEntry> _entries = [];

  List<MovementEntry> get entries => List.unmodifiable(_entries.reversed);

  void addEntry(String mode, String action, {double? lat, double? lon}) {
    _entries.add(MovementEntry(
      time: DateTime.now(),
      mode: mode,
      action: action,
      latitude: lat,
      longitude: lon,
    ));
    notifyListeners();
  }

  void clear() {
    _entries.clear();
    notifyListeners();
  }
}
