import 'package:flutter/material.dart';
import '../services/auth_service.dart';

class AuthProvider extends ChangeNotifier {
  final AuthService _authService;
  AuthUser? _currentUser;
  bool _loading = false;

  AuthProvider() : _authService = AuthService();

  AuthUser? get currentUser => _currentUser;
  bool get isAuthenticated => _currentUser != null;
  bool get isLoading => _loading;

  Future<bool> login(String email, String password) async {
    _loading = true;
    notifyListeners();

    try {
      final user = await _authService.login(email, password);
      _currentUser = user;
      _loading = false;
      notifyListeners();
      return user != null;
    } catch (e) {
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> logout() async {
    _loading = true;
    notifyListeners();

    await _authService.logout();
    _currentUser = null;
    _loading = false;
    notifyListeners();
  }
}
