import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

/// Represents the locally cached voice profile status.
class VoiceCacheStatus {
  const VoiceCacheStatus({
    required this.exists,
    this.speakerName,
    this.availableProfiles = const [],
    this.lastChecked,
  });

  final bool exists;
  final String? speakerName;
  final List<String> availableProfiles;
  final DateTime? lastChecked;
}

/// Handles persisting the voice profile status on the Android device.
class VoiceCacheService {
  VoiceCacheService._internal();

  static final VoiceCacheService _instance = VoiceCacheService._internal();

  factory VoiceCacheService() => _instance;

  static const String _cacheFileName = 'voice_profile_cache.json';

  Future<File> _getCacheFile() async {
    final directory = await getApplicationDocumentsDirectory();
    return File('${directory.path}/$_cacheFileName');
  }

  Future<void> saveStatus({
    required bool exists,
    String? speakerName,
    List<String>? availableProfiles,
  }) async {
    final file = await _getCacheFile();
    final payload = {
      'exists': exists,
      'speaker_name': speakerName,
      'available_profiles': availableProfiles ?? <String>[],
      'last_checked': DateTime.now().toIso8601String(),
    };
    await file.writeAsString(jsonEncode(payload));
  }

  Future<VoiceCacheStatus> loadStatus() async {
    try {
      final file = await _getCacheFile();
      if (!await file.exists()) {
        return const VoiceCacheStatus(exists: false);
      }

      final content = await file.readAsString();
      final data = jsonDecode(content) as Map<String, dynamic>;
      final profiles = (data['available_profiles'] as List<dynamic>? ?? [])
          .map((entry) => entry.toString())
          .toList();

      final lastChecked = data['last_checked'] != null
          ? DateTime.tryParse(data['last_checked'].toString())
          : null;

      return VoiceCacheStatus(
        exists: data['exists'] == true,
        speakerName: data['speaker_name']?.toString(),
        availableProfiles: profiles,
        lastChecked: lastChecked,
      );
    } catch (_) {
      return const VoiceCacheStatus(exists: false);
    }
  }

  Future<void> clear() async {
    final file = await _getCacheFile();
    if (await file.exists()) {
      await file.delete();
    }
  }
}
