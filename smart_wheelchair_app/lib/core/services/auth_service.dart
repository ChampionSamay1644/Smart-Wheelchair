import '../enums.dart';

class AuthUser {
  final String id;
  final String name;
  final UserRole role;

  AuthUser({required this.id, required this.name, required this.role});
}

class AuthService {
  // Dummy users for testing
  static final Map<String, AuthUser> _users = {
    'patient@test.com': AuthUser(
      id: '1',
      name: 'John Patient',
      role: UserRole.patient,
    ),
    'guardian@test.com': AuthUser(
      id: '2',
      name: 'Mary Guardian',
      role: UserRole.guardian,
    ),
    'doctor@test.com': AuthUser(
      id: '3',
      name: 'Dr. Smith',
      role: UserRole.doctor,
    ),
  };

  // Dummy authentication
  Future<AuthUser?> login(String email, String password) async {
    // Simulate network delay
    await Future.delayed(const Duration(seconds: 1));

    // For testing: any password works, just check email exists
    return _users[email];
  }

  Future<void> logout() async {
    // Simulate network delay
    await Future.delayed(const Duration(milliseconds: 500));
  }
}
