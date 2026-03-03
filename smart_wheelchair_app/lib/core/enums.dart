enum UserRole {
  patient,
  guardian,
  doctor;

  String get displayName {
    switch (this) {
      case UserRole.patient:
        return 'Patient';
      case UserRole.guardian:
        return 'Guardian';
      case UserRole.doctor:
        return 'Doctor';
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
