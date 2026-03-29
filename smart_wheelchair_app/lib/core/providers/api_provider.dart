import 'dart:async';
import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ApiProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();
  
  String? _apiUrl;
  List<dynamic> _devices = [];
  String? _selectedDeviceId;
  String? _devicePassword;
  Map<String, dynamic> _latestSensorData = {};
  bool _isLoading = false;
  String? _error;
  bool _killswitchEnabled = false;
  
  List<dynamic> _sensorHistory = [];
  
  Timer? _pollingTimer;
  
  ApiProvider() {
    _loadConfig();
  }
  
  bool get mockMode => _apiService.useMockData;

  void toggleMockMode(bool enabled) {
    _apiService.setMockMode(enabled);
    if (enabled) {
      _error = null;
      refreshDevices();
    }
    notifyListeners();
  }
  
  String? get apiUrl => _apiUrl;
  List<dynamic> get devices => _devices;
  String? get selectedDeviceId => _selectedDeviceId;
  String? get devicePassword => _devicePassword;
  Map<String, dynamic> get latestSensorData => _latestSensorData;
  List<dynamic> get sensorHistory => _sensorHistory;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get killswitchEnabled => _killswitchEnabled;
  bool get isPolling => _pollingTimer != null;

  Future<void> _loadConfig() async {
    final prefs = await SharedPreferences.getInstance();
    _apiUrl = prefs.getString('vercel_api_url') ?? "https://wheelchair-api.vercel.app";
    _selectedDeviceId = prefs.getString('selected_device_id');
    _devicePassword = prefs.getString('device_password');
    
    if (_apiUrl != null) {
      _apiService.setBaseUrl(_apiUrl!);
      _apiService.setDeviceCredentials(_selectedDeviceId, _devicePassword);
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

  Future<void> selectDevice(String deviceId, {String? password}) async {
    _selectedDeviceId = deviceId;
    _devicePassword = password;
    
    _apiService.setDeviceCredentials(deviceId, password);
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('selected_device_id', deviceId);
    if (password != null) {
      await prefs.setString('device_password', password);
    }
    
    _latestSensorData = {};
    _sensorHistory = [];
    startPolling();
    notifyListeners();
  }

  Future<void> reportPresence(String role, {String? name}) async {
    if (_selectedDeviceId == null || _apiUrl == null) return;
    try {
      await _apiService.reportSession(_selectedDeviceId!, role, name: name);
    } catch (e) {
      debugPrint('Error reporting presence: $e');
    }
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
      if (e.toString().contains('401')) {
        _error = 'Invalid Device Password';
        _latestSensorData = {}; // Clear data on auth failure
      } else {
        _error = e.toString();
      }
    }
    notifyListeners();
  }

  String _currentTimeframe = '24h';
  String get currentTimeframe => _currentTimeframe;

  void setTimeframe(String timeframe) {
    _currentTimeframe = timeframe;
    fetchHistory();
    notifyListeners();
  }

  Future<void> fetchHistory() async {
    if (_selectedDeviceId == null || _apiUrl == null) return;
    
    try {
      final history = await _apiService.fetchSensorHistory(
        _selectedDeviceId!, 
        timeframe: _currentTimeframe
      );
      _sensorHistory = history;
      _error = null;
    } catch (e) {
      if (e.toString().contains('401')) {
        _error = 'Unauthorized: Check Device Password';
        _sensorHistory = []; // Clear history on auth failure
      } else {
        _error = e.toString();
      }
    }
    notifyListeners();
  }

  void startPolling() {
    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      fetchLatestData();
      if (timer.tick % 5 == 0) { // Every 5 seconds
        fetchHistory();
      }
      if (timer.tick % 10 == 0) { // Every 10 seconds
        checkStatus();
      }
    });
  }

  void stopPolling() {
    _pollingTimer?.cancel();
    _pollingTimer = null;
    notifyListeners();
  }

  Future<void> clearSession() async {
    stopPolling();
    _selectedDeviceId = null;
    _devicePassword = null;
    _latestSensorData = {};
    _sensorHistory = [];
    _error = null;
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('selected_device_id');
    await prefs.remove('device_password');
    
    _apiService.setDeviceCredentials(null, null);
    notifyListeners();
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    super.dispose();
  }
}
