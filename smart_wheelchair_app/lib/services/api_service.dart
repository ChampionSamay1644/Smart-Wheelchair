import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/foundation.dart';

class ApiService {
  ApiService._internal();
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;

  String? _baseUrl;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString('vercel_api_url')?.trim();
  }

  void setBaseUrl(String url) {
    _baseUrl = url.trim().replaceAll(RegExp(r'/+$'), '');
  }

  String? get baseUrl => _baseUrl;

  Future<Map<String, dynamic>> fetchStatus() async {
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(Uri.parse('$_baseUrl/api/status'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch status: ${response.statusCode}');
    }
  }

  Future<List<dynamic>> fetchDevices() async {
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(Uri.parse('$_baseUrl/api/devices'));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['devices'] ?? [];
    } else {
      throw Exception('Failed to fetch devices: ${response.statusCode}');
    }
  }

  Future<Map<String, dynamic>> fetchLatestSensorData(String deviceId) async {
    if (_baseUrl == null) throw Exception('API URL not configured');
    
    final response = await http.get(
      Uri.parse('$_baseUrl/api/sensors/upload').replace(
        queryParameters: {'deviceId': deviceId},
      ),
    );
    
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else if (response.statusCode == 404) {
      return {}; // No data yet
    } else {
      throw Exception('Failed to fetch sensor data: ${response.statusCode}');
    }
  }
}
