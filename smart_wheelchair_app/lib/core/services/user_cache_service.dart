import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../enums.dart';

class CachedProfile {
  final String id;
  final String name;
  final String email;
  final UserRole role;
  final DateTime lastActive;

  CachedProfile({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    required this.lastActive,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'email': email,
    'role': role.toString(),
    'lastActive': lastActive.toIso8601String(),
  };

  factory CachedProfile.fromJson(Map<String, dynamic> json) => CachedProfile(
    id: json['id'],
    name: json['name'],
    email: json['email'],
    role: UserRole.values.firstWhere((e) => e.toString() == json['role']),
    lastActive: DateTime.parse(json['lastActive']),
  );
}

class UserCacheService {
  static const String _key = 'cached_user_profiles';

  static Future<void> saveProfile(CachedProfile profile) async {
    final prefs = await SharedPreferences.getInstance();
    final profiles = await getProfiles();
    
    // Remove duplication by email and role
    profiles.removeWhere((p) => p.email == profile.email && p.role == profile.role);
    profiles.insert(0, profile);
    
    // Keep only last 5 profiles
    if (profiles.length > 5) profiles.removeLast();

    final jsonList = profiles.map((p) => jsonEncode(p.toJson())).toList();
    await prefs.setStringList(_key, jsonList);
  }

  static Future<List<CachedProfile>> getProfiles() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList(_key) ?? [];
    return jsonList.map((j) => CachedProfile.fromJson(jsonDecode(j))).toList();
  }

  static Future<void> clearProfiles() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}
