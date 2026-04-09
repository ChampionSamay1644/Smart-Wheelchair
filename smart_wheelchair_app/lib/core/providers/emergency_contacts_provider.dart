import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class EmergencyContact {
  final String id;
  final String name;
  final String phone;

  EmergencyContact({
    required this.id,
    required this.name,
    required this.phone,
  });

  Map<String, String> toJson() => {
        'id': id,
        'name': name,
        'phone': phone,
      };

  factory EmergencyContact.fromJson(Map<String, dynamic> json) => EmergencyContact(
        id: json['id'] as String,
        name: json['name'] as String,
        phone: json['phone'] as String,
      );
}

class EmergencyContactsProvider extends ChangeNotifier {
  List<EmergencyContact> _contacts = [];
  bool _initialized = false;

  List<EmergencyContact> get contacts => _contacts;
  bool get isInitialized => _initialized;

  EmergencyContactsProvider() {
    _loadContacts();
  }

  Future<void> _loadContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getStringList('emergency_contacts') ?? [];
    _contacts = data
        .map((item) => EmergencyContact.fromJson(jsonDecode(item)))
        .toList();
    _initialized = true;
    notifyListeners();
  }

  Future<void> _saveContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final data = _contacts.map((item) => jsonEncode(item.toJson())).toList();
    await prefs.setStringList('emergency_contacts', data);
  }

  Future<void> addContact(String name, String phone) async {
    final contact = EmergencyContact(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: name,
      phone: phone,
    );
    _contacts.add(contact);
    await _saveContacts();
    notifyListeners();
  }

  Future<void> updateContact(String id, String name, String phone) async {
    final index = _contacts.indexWhere((c) => c.id == id);
    if (index != -1) {
      _contacts[index] = EmergencyContact(id: id, name: name, phone: phone);
      await _saveContacts();
      notifyListeners();
    }
  }

  Future<void> deleteContact(String id) async {
    _contacts.removeWhere((c) => c.id == id);
    await _saveContacts();
    notifyListeners();
  }
}
