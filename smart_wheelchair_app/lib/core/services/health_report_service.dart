import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import '../models/user_model.dart';

class HealthReport {
  final String id;
  final String patientId;
  final String patientName;
  final DateTime startDate;
  final DateTime endDate;
  final String? downloadUrl;
  final DateTime createdAt;
  final Map<String, dynamic> summary;

  HealthReport({
    required this.id,
    required this.patientId,
    required this.patientName,
    required this.startDate,
    required this.endDate,
    this.downloadUrl,
    required this.createdAt,
    required this.summary,
  });

  factory HealthReport.fromFirestore(
    DocumentSnapshot<Map<String, dynamic>> snapshot, [
    SnapshotOptions? options,
  ]) {
    final data = snapshot.data()!;
    return HealthReport(
      id: snapshot.id,
      patientId: data['patientId'] as String,
      patientName: data['patientName'] as String,
      startDate: (data['startDate'] as Timestamp).toDate(),
      endDate: (data['endDate'] as Timestamp).toDate(),
      downloadUrl: data['downloadUrl'] as String?,
      createdAt: (data['createdAt'] as Timestamp).toDate(),
      summary: data['summary'] as Map<String, dynamic>,
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'patientId': patientId,
      'patientName': patientName,
      'startDate': Timestamp.fromDate(startDate),
      'endDate': Timestamp.fromDate(endDate),
      if (downloadUrl != null) 'downloadUrl': downloadUrl,
      'createdAt': Timestamp.fromDate(createdAt),
      'summary': summary,
    };
  }
}

class HealthReportService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseStorage _storage = FirebaseStorage.instance;
  SmartWheelchairUser? _currentUser;

  void setUser(SmartWheelchairUser? user) {
    _currentUser = user;
  }

  Future<List<HealthReport>> getPatientReports(String patientId) async {
    if (_currentUser == null) {
      throw Exception('User not authenticated');
    }

    // Check if current user has access to this patient's reports
    if (_currentUser!.userType == 'patient' && _currentUser!.uid != patientId) {
      throw Exception('Unauthorized access');
    }
    if (_currentUser!.userType == 'guardian' &&
        !(_currentUser!.patientIds?.contains(patientId) ?? false)) {
      throw Exception('Unauthorized access');
    }

    final snapshot = await _firestore
        .collection('health_reports')
        .where('patientId', isEqualTo: patientId)
        .orderBy('createdAt', descending: true)
        .get();

    return snapshot.docs.map((doc) => HealthReport.fromFirestore(doc)).toList();
  }

  Future<HealthReport> generateReport({
    required String patientId,
    required String patientName,
    required DateTime startDate,
    required DateTime endDate,
  }) async {
    if (_currentUser == null) {
      throw Exception('User not authenticated');
    }

    // Validate access
    if (_currentUser!.userType == 'patient' && _currentUser!.uid != patientId) {
      throw Exception('Unauthorized access');
    }
    if (_currentUser!.userType == 'guardian' &&
        !(_currentUser!.patientIds?.contains(patientId) ?? false)) {
      throw Exception('Unauthorized access');
    }

    // Fetch health data for the period
    final healthData = await _fetchHealthData(patientId, startDate, endDate);

    // Generate report summary
    final summary = _generateSummary(healthData);

    // Create report document first
    final reportRef = _firestore.collection('health_reports').doc();
    final report = HealthReport(
      id: reportRef.id,
      patientId: patientId,
      patientName: patientName,
      startDate: startDate,
      endDate: endDate,
      createdAt: DateTime.now(),
      summary: summary,
    );

    await reportRef.set(report.toFirestore());

    // Generate PDF and upload to storage
    final pdfBytes = await _generatePDF(report, healthData);
    final storageRef = _storage
        .ref()
        .child('health_reports')
        .child(patientId)
        .child('${report.id}.pdf');

    await storageRef.putData(pdfBytes);
    final downloadUrl = await storageRef.getDownloadURL();

    // Update report with download URL
    await reportRef.update({'downloadUrl': downloadUrl});

    return report.copyWith(downloadUrl: downloadUrl);
  }

  Future<List<Map<String, dynamic>>> _fetchHealthData(
    String patientId,
    DateTime startDate,
    DateTime endDate,
  ) async {
    final startTimestamp = Timestamp.fromDate(startDate);
    final endTimestamp = Timestamp.fromDate(endDate);

    final snapshot = await _firestore
        .collection('health_readings')
        .doc(patientId)
        .collection('readings')
        .where('timestamp', isGreaterThanOrEqualTo: startTimestamp)
        .where('timestamp', isLessThanOrEqualTo: endTimestamp)
        .orderBy('timestamp')
        .get();

    return snapshot.docs.map((doc) => doc.data()).toList();
  }

  Map<String, dynamic> _generateSummary(List<Map<String, dynamic>> healthData) {
    if (healthData.isEmpty) {
      return {
        'dataPoints': 0,
        'averages': <String, double>{},
        'ranges': <String, Map<String, double>>{},
      };
    }

    // Calculate averages and ranges for each metric
    final metrics = healthData.first.keys
        .where((k) => k != 'timestamp')
        .toList();
    final averages = <String, double>{};
    final ranges = <String, Map<String, double>>{};
    final dataPoints = healthData.length;

    for (final metric in metrics) {
      final values = healthData
          .map((d) => d[metric] as num)
          .map((n) => n.toDouble())
          .toList();

      final avg = values.reduce((a, b) => a + b) / values.length;
      final min = values.reduce((a, b) => a < b ? a : b);
      final max = values.reduce((a, b) => a > b ? a : b);

      averages[metric] = avg;
      ranges[metric] = {'min': min, 'max': max};
    }

    return {'dataPoints': dataPoints, 'averages': averages, 'ranges': ranges};
  }

  Future<Uint8List> _generatePDF(
    HealthReport report,
    List<Map<String, dynamic>> healthData,
  ) async {
    // TODO: Implement PDF generation using pdf package
    // For now, return a placeholder PDF
    return Uint8List.fromList([]);
  }
}

extension HealthReportExt on HealthReport {
  HealthReport copyWith({String? downloadUrl}) {
    return HealthReport(
      id: id,
      patientId: patientId,
      patientName: patientName,
      startDate: startDate,
      endDate: endDate,
      downloadUrl: downloadUrl ?? this.downloadUrl,
      createdAt: createdAt,
      summary: summary,
    );
  }
}
