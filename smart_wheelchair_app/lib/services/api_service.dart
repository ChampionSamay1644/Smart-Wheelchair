import 'dart:convert';
import 'dart:math';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/foundation.dart';

class ApiService {
  ApiService._internal();
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;

  String? _baseUrl;
  bool _useMockData = false;
  final Map<String, List<dynamic>> _mockHistoryCache = {};
  
  String? _deviceId;
  String? _password;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString('vercel_api_url')?.trim();
    _useMockData = prefs.getBool('use_mock_data') ?? false;
  }

  void setBaseUrl(String url) {
    _baseUrl = url.trim().replaceAll(RegExp(r'/+$'), '');
  }

  void setDeviceCredentials(String? deviceId, String? password) {
    _deviceId = deviceId;
    _password = password;
  }

  Map<String, String> _getHeaders() {
    final headers = {'Content-Type': 'application/json'};
    if (_deviceId != null) headers['X-Device-Id'] = _deviceId!;
    if (_password != null) headers['X-Device-Password'] = _password!;
    return headers;
  }

  void setMockMode(bool enabled) async {
    _useMockData = enabled;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('use_mock_data', enabled);
  }

  bool get useMockData => _useMockData;
  String? get baseUrl => _baseUrl;

  Future<Map<String, dynamic>> fetchStatus() async {
    if (_useMockData) {
      return {
        'killswitch': false,
        'status': 'active',
        'message': 'System is operational (Mock Mode)',
      };
    }
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(Uri.parse('$_baseUrl/api/status'), headers: _getHeaders());
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch status: ${response.statusCode}');
    }
  }

  Future<List<dynamic>> fetchDevices() async {
    if (_useMockData) {
      return [
        {'deviceId': 'MOCK_001', 'online': true},
        {'deviceId': 'TEST_001', 'online': true},
      ];
    }
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(Uri.parse('$_baseUrl/api/devices'), headers: _getHeaders());
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['devices'] ?? [];
    } else {
      throw Exception('Failed to fetch devices: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> fetchLatestSensorData(String deviceId) async {
    if (_useMockData) {
      final data = _generateMockSensorData(deviceId);
      
      // Update persistent mock history for the 'real-time' 24h view
      if (!_mockHistoryCache.containsKey(deviceId)) {
        _mockHistoryCache[deviceId] = _generateSeededMockHistory(deviceId, timeframe: '24h');
      }
      _mockHistoryCache[deviceId]!.insert(0, data);
      if (_mockHistoryCache[deviceId]!.length > 1000) { // Keep a large enough window
        _mockHistoryCache[deviceId]!.removeLast();
      }
      
      return data;
    }
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(
      Uri.parse('$_baseUrl/api/sensors/upload').replace(
        queryParameters: {'deviceId': deviceId},
      ),
      headers: _getHeaders(),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else if (response.statusCode == 404) {
      return {}; // No data yet
    } else {
      throw Exception('Failed to fetch sensor data: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> fetchMedicalInfo(String deviceId) async {
    if (_useMockData) {
      return {
        'name': 'John Doe (Mock)',
        'age': '45',
        'bloodGroup': 'O+',
        'allergies': 'Peanuts, Penicillin',
        'conditions': 'Hypertension',
        'emergencyContact': '+1234567890',
      };
    }
    if (_baseUrl == null) throw Exception('API URL not configured');
    final response = await http.get(
      Uri.parse('$_baseUrl/api/medical-info').replace(queryParameters: {'deviceId': deviceId}),
      headers: _getHeaders(),
    );
    if (response.statusCode == 200) return jsonDecode(response.body);
    return {};
  }

  Future<void> updateMedicalInfo(String deviceId, Map<String, dynamic> info) async {
    if (_useMockData) return;
    if (_baseUrl == null) throw Exception('API URL not configured');
    final response = await http.post(
      Uri.parse('$_baseUrl/api/medical-info'),
      headers: _getHeaders(),
      body: jsonEncode({
        'deviceId': deviceId,
        'info': info,
      }),
    );
    if (response.statusCode != 200) throw Exception('Failed to update medical info');
  }

  Future<void> sendHealthAlertEmail({
    required String targetEmail,
    required String type,
    required String patientName,
    required String vitalInfo,
  }) async {
    if (_baseUrl == null) return;
    debugPrint('📨 [API Service] Triggering Health Alert Email -> $targetEmail');
    final response = await http.post(
      Uri.parse('$_baseUrl/api/notifications/email'),
      headers: _getHeaders(),
      body: jsonEncode({
        'type': type,
        'targetEmail': targetEmail,
        'patientName': patientName,
        'vitalInfo': vitalInfo,
      }),
    );
    if (response.statusCode != 200) {
      debugPrint('❌ [API Service] Email trigger failed: ${response.statusCode} - ${response.body}');
    } else {
      debugPrint('✅ [API Service] Email trigger success');
    }
  }

  Future<List<dynamic>> fetchSensorHistory(String deviceId, {String timeframe = '24h'}) async {
    // The user explicitly requested Weeks and Months to be heavily populated with DISTINCT static generated data
    if (timeframe == '7d' || timeframe == '30d' || timeframe == '4m' || timeframe == 'week' || timeframe == 'month') {
      return _generateSeededMockHistory(deviceId, timeframe: timeframe);
    }

    // However, 24H data MUST be purely real data from the API simulator! No mocks for 24h.
    if (_useMockData) return _generateSeededMockHistory(deviceId, timeframe: '24h'); // Fallback if mock mode toggled
    
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(
      Uri.parse('$_baseUrl/api/sensors/upload').replace(
        queryParameters: {
           'deviceId': deviceId,
           'history': 'true',
           'timeframe': timeframe,
        },
      ),
      headers: _getHeaders(),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch sensor history: ${response.statusCode}');
    }
  }

  Future<void> reportSession(String deviceId, String role, {String? name}) async {
    if (_useMockData) return;
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.post(
      Uri.parse('$_baseUrl/api/sessions'),
      headers: _getHeaders(),
      body: jsonEncode({
        'deviceId': deviceId,
        'role': role,
        'name': name,
      }),
    );
    
    if (response.statusCode != 200) {
      throw Exception('Failed to report session: ${response.statusCode}');
    }
  }

  Future<void> uploadData(Map<String, dynamic> data) async {
    if (_useMockData) return;
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.post(
      Uri.parse('$_baseUrl/api/sensors/upload'),
      headers: _getHeaders(),
      body: jsonEncode(data),
    );
    
    if (response.statusCode != 200) {
      throw Exception('Failed to upload data: ${response.statusCode}');
    }
  }

  Future<void> sendNotificationEmail({
    required String type,
    required String targetEmail,
    String? contextEmail,
    String? userName,
    String? patientName,
  }) async {
    if (_baseUrl == null) {
      debugPrint('⚠️ Email Notification skipped: No Base URL configured.');
      return;
    }

    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/notifications/email'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'type': type,
          'targetEmail': targetEmail,
          'contextEmail': contextEmail,
          'userName': userName,
          'patientName': patientName,
        }),
      );

      // Check if response is JSON (avoiding DOCTYPE errors)
      if (response.headers['content-type']?.contains('application/json') == true) {
        final data = jsonDecode(response.body);
        if (response.statusCode == 200) {
          debugPrint('📧 EMAIL TRIGGERED: $type to $targetEmail. Server status: ${data['serverStatus']}');
        } else {
          debugPrint('⚠️ EMAIL ERROR [${response.statusCode}]: ${data['error']}');
        }
      } else {
        // We got HTML or something else (e.g. Vercel error page)
        debugPrint('❌ EMAIL CRITICAL ERROR [${response.statusCode}]: Received non-JSON response.');
        if (response.body.contains('<!DOCTYPE html>')) {
          debugPrint('   Detected HTML (DOCTYPE) - check if API route exists at: $_baseUrl/api/notifications/email');
        } else {
          debugPrint('   Raw Response: ${response.body.length > 100 ? response.body.substring(0, 100) : response.body}');
        }
      }
    } catch (e) {
      debugPrint('❌ EMAIL EXCEPTION: $e');
    }
  }



  Future<void> syncUserToDatabase(String uid, Map<String, dynamic> userData) async {
    if (_useMockData) return;
    if (_baseUrl == null) throw Exception('API URL not configured');

    await http.post(
      Uri.parse('$_baseUrl/api/users/sync'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'uid': uid,
        'userData': userData,
      }),
    );
  }

  Future<Map<String, dynamic>> loginWithDevice(String deviceId, String password, String role, {String? email}) async {
    if (_useMockData) {
      return {
        'success': true, 
        'role': role, 
        'deviceId': deviceId,
        'user': {
          'id': deviceId,
          'name': role == 'patient' ? 'Patient' : 'Guardian',
          'email': email ?? 'test@example.com',
          'role': role
        }
      };
    }
    if (_baseUrl == null) throw Exception('API URL not configured');

    final response = await http.post(
      Uri.parse('$_baseUrl/api/auth/device'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'deviceId': deviceId,
        'password': password,
        'role': role,
        'email': email,
      }),
    );

    final data = jsonDecode(response.body);
    if (response.statusCode == 200) {
      setDeviceCredentials(deviceId, password);
      return data;
    } else {
      throw Exception(data['error'] ?? 'Authentication failed');
    }
  }

  Future<bool> checkDeviceExists(String deviceId) async {
    if (_useMockData) return true;
    if (_baseUrl == null) throw Exception('API URL not configured');

    final response = await http.get(
      Uri.parse('$_baseUrl/api/sensors/upload').replace(
        queryParameters: {'deviceId': deviceId, 'checkOnly': 'true'},
      ),
      headers: _getHeaders(),
    );
    
    return response.statusCode == 200;
  }



  // Alerts API
  Future<List<dynamic>> fetchAlerts(String deviceId) async {
    if (_useMockData) {
      return [
        {
          'id': '1',
          'title': 'Emergency',
          'message': 'Patient triggered emergency stop',
          'timestamp': DateTime.now().subtract(const Duration(hours: 1)).millisecondsSinceEpoch,
          'read': false,
        },
        {
          'id': '2',
          'title': 'High Pulse',
          'message': 'Pulse detected at 110 BPM (Attention Required)',
          'timestamp': DateTime.now().subtract(const Duration(minutes: 5)).millisecondsSinceEpoch,
          'read': false,
        }
      ];
    }
    if (_baseUrl == null) throw Exception('API URL not configured');

    final response = await http.get(
      Uri.parse('$_baseUrl/api/alerts').replace(
        queryParameters: {'deviceId': deviceId},
      ),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      return [];
    }
  }

  Future<void> createAlert(String deviceId, String title, String message) async {
    if (_useMockData) return;
    if (_baseUrl == null) throw Exception('API URL not configured');

    await http.post(
      Uri.parse('$_baseUrl/api/alerts'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'deviceId': deviceId,
        'title': title,
        'message': message,
      }),
    );
  }

  Map<String, dynamic> _generateMockSensorData(String deviceId, {int? timestamp, double basePulse = 75.0}) {
    final now = timestamp ?? DateTime.now().millisecondsSinceEpoch;
    final rand = Random(now);
    
    // Pulse follows a slow trend + small random noise
    final pulseVariation = (sin(now / 1000000) * 10) + (rand.nextDouble() * 5);
    final currentPulse = basePulse + pulseVariation;

    return {
      'deviceId': deviceId,
      'timestamp': now,
      'dht11': {
        'temperature': 24.0 + (rand.nextDouble() * 4),
        'humidity': 55.0 + (rand.nextDouble() * 10),
      },
      'ultrasonic': {
        'front': 40.0 + (rand.nextDouble() * 200),
      },
      'max30100': {
        'pulse': currentPulse,
        'spo2': 96.0 + (rand.nextDouble() * 3),
      },
      'motorStatus': {
        'lastCommand': ['F', 'B', 'L', 'R', 'S'][rand.nextInt(5)],
        'mode': 'MANUAL',
      }
    };
  }

  List<dynamic> _generateSeededMockHistory(String deviceId, {String timeframe = '24h'}) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final List<dynamic> history = [];
    
    int points;
    int intervalMs;
    double basePulse;
    double baseSpO2;
    double baseTemp;

    // Use a strict seed based on the timeframe string so Week 1 and Week 2 always look different 
    // but remain statically reliable when viewed by the user.
    final seedHash = timeframe.hashCode ^ DateTime.now().year; 
    final rand = Random(seedHash);

    switch (timeframe) {
      case '7d':
      case 'week':
        points = 24 * 7; // Every hour for a week
        intervalMs = 3600000;
        basePulse = 72.0 + rand.nextInt(10);
        baseSpO2 = 96.0 + rand.nextDouble();
        baseTemp = 24.5 + rand.nextDouble();
        break;
      case '30d':
      case 'month':
        points = 30 * 4; // 4 points per day
        intervalMs = 21600000;
        basePulse = 68.0 + rand.nextInt(15);
        baseSpO2 = 98.0;
        baseTemp = 23.0 + rand.nextDouble() * 2;
        break;
      case '4m':
        points = 120; // 1 point per day
        intervalMs = 86400000;
        basePulse = 75.0 + rand.nextInt(5);
        baseSpO2 = 95.0 + rand.nextDouble() * 3;
        baseTemp = 26.0 + rand.nextDouble();
        break;
      default: // 24h
        points = 144; // every 10 mins
        intervalMs = 600000;
        basePulse = 75.0;
        baseSpO2 = 97.0;
        baseTemp = 24.0;
        break;
    }

    for (int i = 0; i < points; i++) {
      final timestamp = now - ((points - i) * intervalMs);
      
      // Add complex mathematical waves to ensure non-repeating curves
      final pulseVariation = sin(i * 0.1) * 8 + cos(i * 0.05) * 4 + rand.nextInt(3);
      final spVariation = cos(i * 0.2) * 1.5 + rand.nextDouble();
      final tempVariation = sin(i * 0.05) * 1.5 + cos(i * 0.01) * 0.5;

      history.add({
        'deviceId': deviceId,
        'timestamp': timestamp,
        'dht11': {
          'temperature': double.parse((baseTemp + tempVariation).toStringAsFixed(1)),
          'humidity': 55.0 + rand.nextInt(5),
        },
        'ultrasonic': {
          'front': 40.0 + rand.nextInt(100),
        },
        'max30100': {
          'pulse': double.parse((basePulse + pulseVariation).toStringAsFixed(1)),
          'spo2': double.parse((baseSpO2 + spVariation).clamp(90.0, 100.0).toStringAsFixed(1)),
        },
        'motorStatus': {
          'lastCommand': 'S',
          'mode': 'MANUAL',
        }
      });
    }

    return history.reversed.toList();
  }
}
