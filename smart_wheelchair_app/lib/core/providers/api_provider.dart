import 'dart:async';
import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ApiProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();
  
  String? _apiUrl;
  List<dynamic> _devices = [];
  String? _selectedDeviceId;
  Map<String, dynamic> _latestSensorData = {};
  bool _isLoading = false;
  String? _error;
  bool _killswitchEnabled = false;
  
  Timer? _pollingTimer;
  
  ApiProvider() {
    _loadConfig();
  }
  
  String? get apiUrl => _apiUrl;
  List<dynamic> get devices => _devices;
  String? get selectedDeviceId => _selectedDeviceId;
  Map<String, dynamic> get latestSensorData => _latestSensorData;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get killswitchEnabled => _killswitchEnabled;
  bool get isPolling => _pollingTimer != null;

  Future<void> _loadConfig() async {
    final prefs = await SharedPreferences.getInstance();
    _apiUrl = prefs.getString('vercel_api_url');
    _selectedDeviceId = prefs.getString('selected_device_id');
    
    if (_apiUrl != null) {
      _apiService.setBaseUrl(_apiUrl!);
      refreshDevices();
      if (_selectedDeviceId != null) {
        startPolling();
      }
    }
    notifyListeners();
  }

  Future<void> updateApiUrl(String url) async {
    _apiUrl = url.trim();
    _apiService.setBaseUrl(_apiUrl!);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('vercel_api_url', _apiUrl!);
    
    refreshDevices();
    checkStatus();
    notifyListeners();
  }

  Future<void> selectDevice(String deviceId) async {
    _selectedDeviceId = deviceId;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('selected_device_id', deviceId);
    
    _latestSensorData = {};
    startPolling();
    notifyListeners();
  }

  Future<void> checkStatus() async {
    try {
      final status = await _apiService.fetchStatus();
      _killswitchEnabled = status['killswitch'] ?? false;
      _error = null;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }

  Future<void> refreshDevices() async {
    _isLoading = true;
    _error = null;
    notifyListeners();
    
    try {
      _devices = await _apiService.fetchDevices();
      _isLoading = false;
    } catch (e) {
      _error = e.toString();
      _isLoading = false;
    }
    notifyListeners();
  }

  Future<void> fetchLatestData() async {
    if (_selectedDeviceId == null || _apiUrl == null) return;
    
    try {
      final data = await _apiService.fetchLatestSensorData(_selectedDeviceId!);
      _latestSensorData = data;
      _error = null;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }

  void startPolling() {
    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      fetchLatestData();
      if (timer.tick % 5 == 0) { // Check status every 10 seconds
        checkStatus();
      }
    });
  }

  void stopPolling() {
    _pollingTimer?.cancel();
    _pollingTimer = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    super.dispose();
  }
}
