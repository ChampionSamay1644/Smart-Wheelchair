import 'package:flutter/foundation.dart';

import 'bluetooth_service.dart';
import 'wheelchair_websocket_service.dart';

/// Coordinates emergency-stop signals across all available transports.
class EmergencyStopService {
  EmergencyStopService._();

  static final WheelchairWebSocketService _wsService =
      WheelchairWebSocketService();
  static final WheelchairBluetoothService _btService =
      WheelchairBluetoothService();

  /// Broadcasts a stop request over every active transport.
  static Future<void> trigger() async {
    if (_wsService.isConnected) {
      _wsService.emergencyStop();
    }

    if (_btService.isConnected) {
      try {
        await _btService.sendManualCommand('stop');
      } catch (error, stackTrace) {
        debugPrint('Emergency stop over Bluetooth failed: $error');
        debugPrint('$stackTrace');
      }
    }
  }
}
