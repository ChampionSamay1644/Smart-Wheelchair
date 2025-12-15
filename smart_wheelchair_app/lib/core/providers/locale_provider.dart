import 'package:flutter/foundation.dart';

class LocaleProvider extends ChangeNotifier {
  // supported codes: 'en', 'hi', 'mr'
  String _locale = 'en';

  String get locale => _locale;

  void setLocale(String code) {
    if (code == _locale) return;
    _locale = code;
    notifyListeners();
  }
}
