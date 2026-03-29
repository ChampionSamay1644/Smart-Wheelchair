import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:provider/provider.dart';
import '../../core/providers/api_provider.dart';
import 'package:intl/intl.dart';

class MovementLogPage extends StatelessWidget {
  const MovementLogPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Movement Log'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => context.read<ApiProvider>().fetchHistory(),
          ),
        ],
      ),
      body: Consumer<ApiProvider>(
        builder: (context, api, _) {
          final history = List.from(api.sensorHistory);
          
          if (history.isEmpty) {
            return const Center(
              child: Text('No movement logs found for this device.'),
            );
          }

          // Ensure sorting (newest first)
          history.sort((a, b) {
            final t1 = a['serverTimestamp'] ?? a['timestamp'] ?? 0;
            final t2 = b['serverTimestamp'] ?? b['timestamp'] ?? 0;
            return (t2 as int).compareTo(t1 as int);
          });

          return ListView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: history.length,
            itemBuilder: (context, index) {
              final entry = history[index];
              final motorStatus = entry['motorStatus'];
              if (motorStatus == null) return const SizedBox.shrink();

              final command = motorStatus['lastCommand'] ?? 'S';
              final mode = motorStatus['mode'] ?? 'REMOTE';
              final timestamp = entry['serverTimestamp'] ?? entry['timestamp'] ?? 0;
              final time = DateTime.fromMillisecondsSinceEpoch(timestamp);

              return Card(
                margin: const EdgeInsets.symmetric(vertical: 4),
                child: ListTile(
                  leading: _buildMovementIcon(command, mode),
                  title: Text(_getMovementDescription(command, mode)),
                  subtitle: Text(DateFormat('HH:mm:ss').format(time)),
                  trailing: Text(mode),
                ),
              );
            },
          );
        },
      ),
    );
  }

  Widget _buildMovementIcon(String command, String mode) {
    IconData icon;
    Color color;

    switch (command.toUpperCase()) {
      case 'F':
      case 'FORWARD':
        icon = FontAwesomeIcons.arrowUp;
        color = Colors.green;
      case 'B':
      case 'BACKWARD':
        icon = FontAwesomeIcons.arrowDown;
        color = Colors.orange;
      case 'L':
      case 'LEFT':
        icon = FontAwesomeIcons.arrowLeft;
        color = Colors.blue;
      case 'R':
      case 'RIGHT':
        icon = FontAwesomeIcons.arrowRight;
        color = Colors.blue;
      case 'S':
      case 'STOP':
      default:
        icon = FontAwesomeIcons.circleStop;
        color = Colors.red;
    }

    if (mode.toUpperCase() == 'VOICE') {
      color = Colors.purple;
    }

    return CircleAvatar(
      backgroundColor: color.withOpacity(0.2),
      child: FaIcon(icon, color: color, size: 16),
    );
  }

  String _getMovementDescription(String command, String mode) {
    String action;
    switch (command.toUpperCase()) {
      case 'F':
      case 'FORWARD':
        action = 'Moving Forward';
      case 'B':
      case 'BACKWARD':
        action = 'Reversing';
      case 'L':
      case 'LEFT':
        action = 'Turning Left';
      case 'R':
      case 'RIGHT':
        action = 'Turning Right';
      case 'S':
      case 'STOP':
        action = 'Stopped';
      default:
        action = 'Unknown Command';
    }
    
    if (mode.toUpperCase() == 'VOICE') {
      return 'Voice: $action';
    }
    return action;
  }
}
