import 'package:cloud_firestore/cloud_firestore.dart';

class InviteCode {
  final String code;
  final String guardianId;
  final String guardianName;
  final String? guardianEmail;
  final DateTime expiresAt;
  final bool used;
  final DateTime createdAt;

  const InviteCode({
    required this.code,
    required this.guardianId,
    required this.guardianName,
    this.guardianEmail,
    required this.expiresAt,
    this.used = false,
    required this.createdAt,
  });

  factory InviteCode.fromFirestore(
    DocumentSnapshot<Map<String, dynamic>> snapshot, [
    SnapshotOptions? options,
  ]) {
    final data = snapshot.data()!;
    return InviteCode(
      code: snapshot.id,
      guardianId: data['guardianId'] as String,
      guardianName: data['guardianName'] as String,
      guardianEmail: data['guardianEmail'] as String?,
      expiresAt: (data['expiresAt'] as Timestamp).toDate(),
      used: data['used'] as bool? ?? false,
      createdAt: (data['createdAt'] as Timestamp).toDate(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'guardianId': guardianId,
      'guardianName': guardianName,
      if (guardianEmail != null) 'guardianEmail': guardianEmail,
      'expiresAt': Timestamp.fromDate(expiresAt),
      'used': used,
      'createdAt': Timestamp.fromDate(createdAt),
    };
  }

  bool get isExpired => DateTime.now().isAfter(expiresAt);
}
