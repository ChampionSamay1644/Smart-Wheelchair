import 'package:flutter/foundation.dart';

class NotificationEvent {
  final String title;
  final String body;
  final DateTime time;
  bool read;

  NotificationEvent({
    required this.title,
    required this.body,
    required this.time,
    this.read = false,
  });
}

class NotificationsProvider extends ChangeNotifier {
  final List<NotificationEvent> _events = [];
  NotificationEvent? _lastEvent;

  List<NotificationEvent> get events => List.unmodifiable(_events.reversed);

  /// The last event that was added. Useful for UI widgets that want to
  /// react immediately to new notifications (for example an overlay panel).
  NotificationEvent? get lastEvent => _lastEvent;

  int get unreadCount => _events.where((e) => !e.read).length;

  void addEvent(String title, String body) {
    _events.add(
      NotificationEvent(title: title, body: body, time: DateTime.now()),
    );
    _lastEvent = _events.isNotEmpty ? _events.last : null;
    notifyListeners();
  }

  void markAllRead() {
    for (var e in _events) {
      e.read = true;
    }
    notifyListeners();
  }
}
