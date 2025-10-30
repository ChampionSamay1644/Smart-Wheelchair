class SmartWheelchairUser {
  final String uid;
  final String email;
  final String name;
  final String userType; // 'patient' or 'guardian'
  final String? guardianId; // Only for patients
  final List<String>? patientIds; // Only for guardians
  final Map<String, dynamic>? healthData;
  final Map<String, dynamic>? preferences;

  SmartWheelchairUser({
    required this.uid,
    required this.email,
    required this.name,
    required this.userType,
    this.guardianId,
    this.patientIds,
    this.healthData,
    this.preferences,
  });

  factory SmartWheelchairUser.fromMap(Map<String, dynamic> data) {
    return SmartWheelchairUser(
      uid: data['uid'] ?? '',
      email: data['email'] ?? '',
      name: data['name'] ?? '',
      userType: data['userType'] ?? '',
      guardianId: data['guardianId'],
      patientIds: List<String>.from(data['patientIds'] ?? []),
      healthData: data['healthData'],
      preferences: data['preferences'],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'email': email,
      'name': name,
      'userType': userType,
      'guardianId': guardianId,
      'patientIds': patientIds,
      'healthData': healthData,
      'preferences': preferences,
    };
  }
}
