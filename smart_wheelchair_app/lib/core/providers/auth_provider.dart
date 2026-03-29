import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/auth_service.dart';
import '../services/user_cache_service.dart';
import '../../services/api_service.dart';
import '../enums.dart';

class AuthProvider extends ChangeNotifier {
  final AuthService _authService;
  final ApiService _apiService;
  AuthUser? _currentUser;
  bool _loading = false;
  bool _initialized = false;

  AuthProvider() 
    : _authService = AuthService(),
      _apiService = ApiService() {
    _loadUser();
  }

  AuthUser? get currentUser => _currentUser;
  bool get isAuthenticated => _currentUser != null;
  bool get isLoading => _loading;
  bool get isInitialized => _initialized;
  UserRole? get userRole => _currentUser?.role;

  Future<void> _loadUser() async {
    final prefs = await SharedPreferences.getInstance();
    final userJson = prefs.getString('auth_user');
    
    debugPrint('🔍 AUTH: Checking for persistent session... Found: ${userJson != null}');
    
    if (userJson != null) {
      try {
        final Map<String, dynamic> userMap = jsonDecode(userJson);
        final roleStr = userMap['role'];
        
        // Robust role parsing
        UserRole? role;
        try {
          role = UserRole.values.firstWhere((e) => e.toString() == roleStr);
        } catch (_) {
          // Fallback if the string doesn't include the enum name (legacy/mismatch)
          if (roleStr?.contains('patient') == true) role = UserRole.patient;
          if (roleStr?.contains('guardian') == true) role = UserRole.guardian;
        }

        if (role != null) {
          _currentUser = AuthUser(
            id: userMap['id'] ?? 'unknown',
            name: userMap['name'] ?? 'Generic User',
            email: userMap['email'] ?? '',
            role: role,
            isNew: false, // Session restore is never a 'new' user
          );
          debugPrint('✅ AUTH: Session restored for ${_currentUser?.name} as $role');
        } else {
          debugPrint('⚠️ AUTH: Failed to parse role from session: $roleStr');
        }
      } catch (e) {
        debugPrint('❌ AUTH: Error decoding user session: $e');
      }
    }
    _initialized = true;
    notifyListeners();
  }

  Future<bool> login(String deviceId, String password, {UserRole role = UserRole.patient}) async {
    _loading = true;
    notifyListeners();

    try {
      final response = await _apiService.loginWithDevice(deviceId, password, role.toString().split('.').last);
      
      final user = AuthUser(
        id: response['user']['id'],
        name: response['user']['name'],
        email: response['user']['email'],
        role: role,
        isNew: false
      );

      _currentUser = user;
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('auth_user', jsonEncode({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role.toString(),
      }));

      // Cache for "Welcome Back" UX
      await UserCacheService.saveProfile(CachedProfile(
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
        lastActive: DateTime.now(),
      ));

      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('Device Login Failed: $e');
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> logout() async {
    _loading = true;
    notifyListeners();

    await _authService.logout();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_user');
    
    _currentUser = null;
    _loading = false;
    notifyListeners();
  }
}
