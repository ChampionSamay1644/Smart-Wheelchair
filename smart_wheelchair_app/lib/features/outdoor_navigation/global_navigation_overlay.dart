import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/providers/outdoor_navigation_provider.dart';
import 'map_widget.dart';

class GlobalNavigationOverlay extends StatefulWidget {
  const GlobalNavigationOverlay({super.key});

  @override
  State<GlobalNavigationOverlay> createState() => _GlobalNavigationOverlayState();
}

class _GlobalNavigationOverlayState extends State<GlobalNavigationOverlay> {
  // Mini map state
  double _pipX = 16.0;
  double _pipY = 100.0;
  double _pipScale = 1.0;
  final double _pipSize = 160.0;

  // Notification Banner State
  bool _bannerVisible = false;
  Timer? _hideTimer;
  Timer? _intervalTimer;
  
  // Tracking state to detect transitions
  bool _wasNavigating = false;
  double? _lastDistance;

  @override
  void initState() {
    super.initState();
    debugPrint('🎨 GlobalNavigationOverlay: Mounted');
    
    // Trigger 3: Every 4 minutes (240 seconds)
    _intervalTimer = Timer.periodic(const Duration(minutes: 4), (timer) {
      final nav = context.read<OutdoorNavigationProvider>();
      if (nav.state.isNavigating) {
        debugPrint('🎨 Notification Trigger: 4-minute interval');
        _showBanner();
      }
    });
  }

  @override
  void dispose() {
    debugPrint('🎨 GlobalNavigationOverlay: Disposing');
    _hideTimer?.cancel();
    _intervalTimer?.cancel();
    super.dispose();
  }

  void _showBanner() {
    if (!mounted) return;
    setState(() => _bannerVisible = true);
    
    // Auto-hide after 15 seconds
    _hideTimer?.cancel();
    _hideTimer = Timer(const Duration(seconds: 15), () {
      if (mounted) setState(() => _bannerVisible = false);
    });
  }

  void _dismissBanner() {
    debugPrint('🎨 Notification: Manually dismissed');
    _hideTimer?.cancel();
    setState(() => _bannerVisible = false);
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return Consumer<OutdoorNavigationProvider>(
      builder: (context, nav, _) {
        final st = nav.state;
        
        debugPrint('🎨 Overlay Build: isNavigating=${st.isNavigating}, isMapOpen=${st.isMapPageOpen}, pos=${st.currentPosition != null}, bannerVisible=$_bannerVisible');

        // Show only if navigating and we have a route
        if (!st.isNavigating || st.routePoints.isEmpty) {
          if (_wasNavigating) {
             debugPrint('🎨 Overlay: Navigation ended or route lost');
             _wasNavigating = false;
          }
          return const SizedBox.shrink();
        }

        // Trigger 1: Start of Navigation
        if (!_wasNavigating && st.isNavigating) {
          debugPrint('🎨 Notification Trigger: Navigation session started');
          _wasNavigating = true;
          WidgetsBinding.instance.addPostFrameCallback((_) => _showBanner());
        }

        // Trigger 2: Close to Turn (< 50m) or VERY Close (< 10m)
        if (st.nextTurnDistance != null) {
          final dist = st.nextTurnDistance!;
          final isClose = dist < 50.0;
          final isVeryClose = dist < 10.0;
          final wasClose = (_lastDistance ?? 999.0) < 50.0;
          final wasVeryClose = (_lastDistance ?? 999.0) < 10.0;
          
          if ((isClose && !wasClose && !_bannerVisible) || (isVeryClose && !wasVeryClose)) {
            debugPrint('🎨 Notification Trigger: ${isVeryClose ? "VERY" : ""} Close to turn (${dist.toStringAsFixed(0)}m)');
            WidgetsBinding.instance.addPostFrameCallback((_) => _showBanner());
          }
          _lastDistance = dist;
        }

        return Stack(
          children: [
            // Top Notification Panel (Event-driven popup)
            if (_bannerVisible && st.routeSteps.isNotEmpty)
              _buildNotificationBanner(st),

            // Draggable Mini Map PiP (only if map page is NOT open)
            if (!st.isMapPageOpen && st.currentPosition != null)
              _buildMiniMapPip(st, screenSize),
          ],
        );
      },
    );
  }

  Widget _buildNotificationBanner(NavigationState st) {
    // Show only the "Active" step or the next one
    final activeIndex = st.currentStepIndex;
    if (activeIndex >= st.routeSteps.length) return const SizedBox.shrink();

    final step = st.routeSteps[activeIndex];
    final distance = st.nextTurnDistance ?? (step['distance'] as num?)?.toDouble() ?? 0.0;
    final totalDistance = st.totalRemainingDistance ?? 0.0;
    final instruction = st.currentInstruction ?? 'Continue on journey';

    return Positioned(
      top: 50,
      left: 12,
      right: 12,
      child: GestureDetector(
        onVerticalDragEnd: (details) {
          if (details.primaryVelocity != null && details.primaryVelocity! < -300) {
            _dismissBanner();
          }
        },
        child: TweenAnimationBuilder<double>(
          tween: Tween(begin: -100, end: 0),
          duration: const Duration(milliseconds: 500),
          curve: Curves.easeOutBack,
          builder: (context, value, child) {
            return Transform.translate(
              offset: Offset(0, value),
              child: Opacity(
                opacity: (1 + (value / 100)).clamp(0.0, 1.0),
                child: child,
              ),
            );
          },
          child: _buildInstructionCard(
            context,
            instruction,
            distance,
            totalDistance,
            activeIndex,
            st.routeSteps.length,
          ),
        ),
      ),
    );
  }

  Widget _buildMiniMapPip(NavigationState st, Size screenSize) {
    debugPrint('🎨 Rendering Mini Map PiP at ($_pipX, $_pipY) | Pos=${st.currentPosition}');
    
    return Positioned(
      left: _pipX,
      top: _pipY,
      child: Material(
        color: Colors.transparent,
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        child: GestureDetector(
          onPanUpdate: (details) {
            setState(() {
               _pipX += details.delta.dx;
               _pipY += details.delta.dy;

               // Clamp to screen bounds
               if (_pipX < 0) _pipX = 0;
               if (_pipY < 0) _pipY = 0;
               if (_pipX > screenSize.width - _pipSize * _pipScale) {
                 _pipX = screenSize.width - _pipSize * _pipScale;
               }
               if (_pipY > screenSize.height - _pipSize * _pipScale) {
                 _pipY = screenSize.height - _pipSize * _pipScale;
               }
            });
          },
          onDoubleTap: () {
            setState(() {
              _pipScale = _pipScale == 1.0 ? 1.5 : 1.0;
              // Re-clamp
              if (_pipX > screenSize.width - _pipSize * _pipScale) {
                 _pipX = screenSize.width - _pipSize * _pipScale;
              }
              if (_pipY > screenSize.height - _pipSize * _pipScale) {
                 _pipY = screenSize.height - _pipSize * _pipScale;
              }
            });
          },
          child: Transform.scale(
            scale: _pipScale,
            alignment: Alignment.topLeft,
            child: Container(
              width: _pipSize,
              height: _pipSize,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.blueAccent.withValues(alpha: 0.8), width: 4),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.3),
                    blurRadius: 15,
                    spreadRadius: 2,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: IgnorePointer(
                  ignoring: true,
                  child: MapWidget(
                    currentPosition: st.currentPosition!,
                    routePoints: st.routePoints,
                    destination: st.destination,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildInstructionCard(BuildContext context, String instruction, double distance, double totalDistance, int index, int total) {
    final isVeryClose = distance < 10.0;
    
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isVeryClose 
            ? [Colors.orange[800]!, Colors.red[700]!] // Alert colors for immediate turn
            : [Colors.green[800]!, Colors.green[600]!],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: isVeryClose ? Colors.red.withValues(alpha: 0.5) : Colors.black.withValues(alpha: 0.4),
            blurRadius: isVeryClose ? 20 : 12,
            spreadRadius: isVeryClose ? 2 : 0,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              _buildInstructionIcon(instruction, isVeryClose),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      children: [
                        Text(
                          '${distance.toStringAsFixed(0)}m',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: isVeryClose ? 32 : 26,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        if (isVeryClose)
                          Padding(
                            padding: const EdgeInsets.only(left: 8.0),
                            child: const Text(
                              'TURN NOW!',
                              style: TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 14),
                            ),
                          ),
                      ],
                    ),
                    Text(
                      instruction,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Remaining: ${(totalDistance / 1000).toStringAsFixed(1)} km to destination',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.8),
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.keyboard_arrow_up, color: Colors.white54, size: 20),
            ],
          ),
          const SizedBox(height: 8),
          Container(
            height: 4,
            width: 40,
            decoration: BoxDecoration(
              color: Colors.white24,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInstructionIcon(String instruction, bool isVeryClose) {
    IconData icon = Icons.navigation;
    if (instruction.toLowerCase().contains('left')) icon = Icons.turn_left;
    if (instruction.toLowerCase().contains('right')) icon = Icons.turn_right;
    if (instruction.toLowerCase().contains('straight')) icon = Icons.straight;
    if (instruction.toLowerCase().contains('arrive')) icon = Icons.location_on;

    return Container(
      padding: EdgeInsets.all(isVeryClose ? 10 : 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.2),
        shape: BoxShape.circle,
        border: isVeryClose ? Border.all(color: Colors.white, width: 2) : null,
      ),
      child: Icon(icon, color: Colors.white, size: isVeryClose ? 38 : 32),
    );
  }
}

