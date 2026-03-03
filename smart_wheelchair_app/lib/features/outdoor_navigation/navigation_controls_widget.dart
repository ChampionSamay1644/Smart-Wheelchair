import 'package:flutter/material.dart';

class NavigationControlsWidget extends StatelessWidget {
  final bool isNavigating;
  final VoidCallback onStart;
  final VoidCallback onStop;

  const NavigationControlsWidget({
    super.key,
    required this.isNavigating,
    required this.onStart,
    required this.onStop,
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
              label: const Text('Start Navigation'),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: ElevatedButton.icon(
              onPressed: isNavigating ? onStop : null,
              icon: const Icon(Icons.stop),
              label: const Text('Stop Navigation'),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            ),
          ),
        ],
      ),
    );
  }
}
