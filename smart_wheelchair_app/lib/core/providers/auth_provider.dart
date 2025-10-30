import 'package:flutter/foundation.dart';
import '../enums.dart';
import '../services/auth_service.dart';
import '../models/user_model.dart';

class AuthProvider extends ChangeNotifier {
  final AuthService _authService;
  SmartWheelchairUser? _currentUser;
  bool _loading = false;
  String? _error;

  AuthProvider({required AuthService authService})
    : _authService = authService {
    // initialize current user if already signed in
    _authService.currentUser
        .then((user) {
          _currentUser = user;
          notifyListeners();
        })
        .catchError((_) {});
  }

  SmartWheelchairUser? get currentUser => _currentUser;
  bool get isAuthenticated => _currentUser != null;
  bool get isLoading => _loading;
  String? get error => _error;

  Future<bool> signUp({
    required String email,
    required String password,
    required String name,
    required UserRole role,
    String? phoneNumber,
    String? deviceId,
  }) async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final user = await _authService.signUp(
        email: email,
        password: password,
        name: name,
        role: role,
        phoneNumber: phoneNumber,
        deviceId: deviceId,
      );
      _currentUser = user;
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> login(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final user = await _authService.login(email, password);
      _currentUser = user;
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<String?> createInviteCode() async {
    if (_currentUser == null || _currentUser!.userType != 'guardian') {
      _error = 'Only guardians can create invite codes';
      notifyListeners();
      return null;
    }

    try {
      final code = await _authService.createInviteCode(_currentUser!);
      return code;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return null;
    }
  }

  Future<bool> acceptInvite(String inviteCode) async {
    if (_currentUser == null || _currentUser!.userType != 'patient') {
      _error = 'Only patients can accept invite codes';
      notifyListeners();
      return false;
    }

    _loading = true;
    _error = null;
    notifyListeners();

    try {
      await _authService.acceptInvite(inviteCode, _currentUser!.uid);
      // Refresh user data
      _currentUser = await _authService.currentUser;
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> logout() async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      await _authService.logout();
      _currentUser = null;
      _loading = false;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
    }
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}
