import '../../core/models/user_model.dart';

/// Utility that maps a user model to a route for role-based navigation.
class RoleGuard {
  /// Returns the route name for the given user.
  ///
  /// If user is null, fall back to role selection.
  static String routeForUser(SmartWheelchairUser? user) {
    if (user == null) return '/role_selection';
    final type = user.userType.toLowerCase();
    if (type == 'patient') return '/patient_home';
    // treat 'guardian' as caregiver
    if (type == 'guardian' || type == 'caregiver') return '/caregiver_home';
    // fallback
    return '/role_selection';
  }
}
