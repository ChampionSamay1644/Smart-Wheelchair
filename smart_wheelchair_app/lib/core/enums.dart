enum UserRole {
  patient,
  guardian;

  String get displayName {
    switch (this) {
      case UserRole.patient:
        return 'Patient';
      case UserRole.guardian:
        // Use a user-friendly label 'Caregiver' in the UI while keeping
        // the internal value 'guardian' for backwards compatibility.
        return 'Caregiver';
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
