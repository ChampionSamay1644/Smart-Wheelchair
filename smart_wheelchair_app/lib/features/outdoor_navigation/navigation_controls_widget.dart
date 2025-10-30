import 'package:flutter/material.dart';

class NavigationControlsWidget extends StatelessWidget {
  final bool isNavigating;
  final bool isPaused;
  final VoidCallback onStart;
  final VoidCallback onStop;
  final VoidCallback onPauseResume;

  const NavigationControlsWidget({
    super.key,
    required this.isNavigating,
    required this.isPaused,
    required this.onStart,
    required this.onStop,
    required this.onPauseResume,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildControlButton(
            onPressed: isNavigating ? null : onStart,
            icon: Icons.play_arrow,
            label: 'Start',
            context: context,
          ),
          _buildControlButton(
            onPressed: isNavigating ? onPauseResume : null,
            icon: isPaused ? Icons.play_arrow : Icons.pause,
            label: isPaused ? 'Resume' : 'Pause',
            context: context,
          ),
          _buildControlButton(
            onPressed: isNavigating ? onStop : null,
            icon: Icons.stop,
            label: 'Stop',
            context: context,
            color: Colors.red,
          ),
        ],
      ),
    );
  }

  Widget _buildControlButton({
    required VoidCallback? onPressed,
    required IconData icon,
    required String label,
    required BuildContext context,
    Color? color,
  }) {
    return SizedBox(
      width: 100,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          padding: const EdgeInsets.symmetric(vertical: 12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 24),
            const SizedBox(height: 4),
            Text(label, style: const TextStyle(fontSize: 12)),
          ],
        ),
      ),
    );
  }
}
