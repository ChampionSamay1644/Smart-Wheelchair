enum UserRole {
  patient,
  guardian;

  String get displayName {
    switch (this) {
      case UserRole.patient:
        return 'Patient';
      case UserRole.guardian:
        return 'Guardian';
    }
  }
}

enum VitalStatus {
  normal,
  warning,
  critical;

  String get displayName {
    switch (this) {
      case VitalStatus.normal:
        return 'Normal';
      case VitalStatus.warning:
        return 'Warning';
      case VitalStatus.critical:
        return 'Critical';
    }
  }
}
