import 'dart:async';

import 'package:flutter/material.dart';

class HoldButton extends StatefulWidget {
  final Widget child;
  final Duration holdDuration;
  final VoidCallback onHold;
  final Color? backgroundColor;

  const HoldButton({
    super.key,
    required this.child,
    required this.onHold,
    this.holdDuration = const Duration(seconds: 2),
    this.backgroundColor,
  });

  @override
  State<HoldButton> createState() => _HoldButtonState();
}

class _HoldButtonState extends State<HoldButton> {
  Timer? _timer;

  void _startHold() {
    _timer?.cancel();
    _timer = Timer(widget.holdDuration, () {
      widget.onHold();
    });
    // optional visual state could be added here
  }

  void _cancelHold() {
    _timer?.cancel();
    // cancel without extra UI state for now
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _startHold(),
      onTapUp: (_) => _cancelHold(),
      onTapCancel: () => _cancelHold(),
      child: Container(
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: widget.backgroundColor ?? Colors.transparent,
        ),
        child: widget.child,
      ),
    );
  }
}
