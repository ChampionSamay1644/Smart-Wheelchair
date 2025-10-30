import 'package:flutter/foundation.dart';
import '../models/user_model.dart';
import '../services/health_report_service.dart';

class HealthReportProvider extends ChangeNotifier {
  final HealthReportService _reportService;
  SmartWheelchairUser? _currentUser;
  List<HealthReport> _reports = [];
  bool _loading = false;

  HealthReportProvider(this._reportService);

  List<HealthReport> get reports => List.unmodifiable(_reports);
  bool get isLoading => _loading;

  void setUser(SmartWheelchairUser? user) {
    _currentUser = user;
    _reportService.setUser(user);
    _reports = [];

    if (user != null) {
      // Load reports for current user
      if (user.userType == 'patient') {
        loadReports(user.uid);
      }
    }
    notifyListeners();
  }

  Future<void> loadReports(String patientId) async {
    if (_currentUser == null) return;

    try {
      _loading = true;
      notifyListeners();

      _reports = await _reportService.getPatientReports(patientId);

      _loading = false;
      notifyListeners();
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
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

    try {
      _loading = true;
      notifyListeners();

      final report = await _reportService.generateReport(
        patientId: patientId,
        patientName: patientName,
        startDate: startDate,
        endDate: endDate,
      );

      // Add to local list if it's for the currently viewed patient
      final viewingPatientId = _currentUser!.userType == 'patient'
          ? _currentUser!.uid
          : patientId;

      if (report.patientId == viewingPatientId) {
        _reports.insert(0, report);
      }

      _loading = false;
      notifyListeners();

      return report;
    } catch (e) {
      _loading = false;
      notifyListeners();
      rethrow;
    }
  }
}
