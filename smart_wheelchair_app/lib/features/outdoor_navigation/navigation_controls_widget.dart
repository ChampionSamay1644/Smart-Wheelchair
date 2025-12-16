import 'package:flutter/material.dart';

class NavigationControlsWidget extends StatelessWidget {
  final bool isNavigating;
  final bool isPaused;
  final VoidCallback onStart;
  final VoidCallback onStop;
  final VoidCallback onTogglePause;

  const NavigationControlsWidget({
    super.key,
    required this.isNavigating,
    required this.isPaused,
    required this.onStart,
    required this.onStop,
    required this.onTogglePause,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton.icon(
              onPressed: isNavigating ? null : onStart,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Start'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ElevatedButton.icon(
              onPressed: isNavigating ? onTogglePause : null,
              icon: Icon(isPaused ? Icons.play_circle : Icons.pause),
              label: FittedBox(
                fit: BoxFit.scaleDown,
                child: Text(isPaused ? 'Resume' : 'Pause'),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ElevatedButton.icon(
              onPressed: isNavigating ? onStop : null,
              icon: const Icon(Icons.stop),
              label: const Text('Stop'),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            ),
          ),
        ],
      ),
    );
  }
}
