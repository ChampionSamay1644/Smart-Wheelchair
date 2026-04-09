import 'dart:async';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AlertProvider extends ChangeNotifier {
  List<dynamic> _alerts = [];
  bool _isLoading = false;
  String? _selectedDeviceId;
  StreamSubscription? _alertsSubscription;
  Set<String> _readAlertIds = {};

  AlertProvider() {
    _loadReadStatus();
  }

  List<dynamic> get alerts => _alerts;
  bool get isLoading => _isLoading;
  int get unreadCount => _alerts.where((a) => a['read'] == false).length;

  Future<void> _loadReadStatus() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList('read_alert_ids') ?? [];
    _readAlertIds = list.toSet();
  }

  void updateDeviceId(String? deviceId) {
    if (_selectedDeviceId == deviceId) return; // Don't re-attach if same device
    _selectedDeviceId = deviceId;
    _alertsSubscription?.cancel();
    _alerts = [];
    notifyListeners();

    if (deviceId != null) {
      _listenToAlerts(deviceId);
    }
  }

  void _listenToAlerts(String deviceId) {
    _isLoading = true;
    notifyListeners();

    // Use limitToLast(20) and listen to VALUE (not onChildAdded)
    // This gives a stable snapshot and updates when new alerts arrive
    _alertsSubscription = FirebaseDatabase.instance
        .ref('devices/$deviceId/alerts')
        .limitToLast(20)
        .onValue
        .listen((event) {
      final data = event.snapshot.value;

      if (data == null) {
        _alerts = [];
        _isLoading = false;
        notifyListeners();
        return;
      }

      final rawMap = data as Map;
      final alertList = rawMap.entries.map((e) {
        final val = Map<String, dynamic>.from(e.value as Map);
        final id = e.key.toString();
        return {
          'id': id,
          'title': val['title'] ?? 'Alert',
          'message': val['message'] ?? '',
          'timestamp': val['timestamp'] ?? 0,
          'read': _readAlertIds.contains(id) ? true : (val['read'] ?? false),
        };
      }).toList();

      // Sort newest first
      alertList.sort((a, b) => (b['timestamp'] as int).compareTo(a['timestamp'] as int));

      _alerts = alertList;
      _isLoading = false;
      notifyListeners();
    }, onError: (e) {
      debugPrint('AlertProvider Firebase error: $e');
      _isLoading = false;
      notifyListeners();
    });
  }

  Future<void> markAllAsRead() async {
    for (var alert in _alerts) {
      alert['read'] = true;
      _readAlertIds.add(alert['id'].toString());
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('read_alert_ids', _readAlertIds.toList());
    notifyListeners();
  }

  Future<void> clearAll() async {
    if (_selectedDeviceId == null) return;
    // Delete all alerts from Firebase
    await FirebaseDatabase.instance
        .ref('devices/$_selectedDeviceId/alerts')
        .remove();
    // Clear stored read IDs too
    _readAlertIds.clear();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('read_alert_ids');
    // _alerts will be cleared automatically by the onValue stream listener
  }

  // Manual refresh — re-trigger the stream
  void refreshAlerts() {
    if (_selectedDeviceId != null) {
      _alertsSubscription?.cancel();
      _listenToAlerts(_selectedDeviceId!);
    }
  }

  @override
  void dispose() {
    _alertsSubscription?.cancel();
    super.dispose();
  }
}
