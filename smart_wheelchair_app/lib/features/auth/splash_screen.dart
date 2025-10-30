import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';
import '../../core/services/auth_service.dart';

// --- Constants ---
const Color _kSplashBackground = Color(0xFF0A101F); // Base color
const int _kParticleAnimDurationMs = 1800;
const int _kParticleStartDelayMs = 1000;

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _particleController;

  final List<Offset> _particlePositions = [
    Offset(40, 40), // Top-left
    Offset(340, 50), // Top-right
    Offset(20, 120), // Mid-left
    Offset(360, 140), // Mid-right
    Offset(30, 240), // Low-left
    Offset(350, 220), // Low-right
    Offset(90, 310), // Bottom-left (near text)
    Offset(290, 320), // Bottom-right (near text)
  ];

  @override
  void initState() {
    super.initState();

    _particleController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: _kParticleAnimDurationMs),
    );

    Future.delayed(const Duration(milliseconds: _kParticleStartDelayMs), () {
      if (mounted) {
        _particleController.repeat(reverse: true);
      }
    });

    _checkAuthAndNavigate();
  }

  Future<void> _checkAuthAndNavigate() async {
    await Future.delayed(const Duration(seconds: 2)); // Minimum splash time

    if (!mounted) return;

    final authService = Provider.of<AuthService>(context, listen: false);
    final user = await authService.currentUser;

    if (!mounted) return;

    if (user != null) {
      // User is logged in, navigate to appropriate dashboard
      if (user.userType == 'patient') {
        Navigator.pushReplacementNamed(context, '/patient_dashboard');
      } else {
        Navigator.pushReplacementNamed(context, '/guardian_dashboard');
      }
    } else {
      // No user logged in, go to role selection
      Navigator.pushReplacementNamed(context, '/role_selection');
    }
  }

  @override
  void dispose() {
    _particleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kSplashBackground,
      body: Stack(
        children: [
          // --- Coded Noise Background ---
          Positioned.fill(child: CustomPaint(painter: NoisePainter())),

          // --- UI Content (Particles, Logo, Text) ---
          Center(
            child: SizedBox(
              width: 380, // Our coordinate space
              height: 500, // Our coordinate space
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // --- Particles ---
                  ...List.generate(_particlePositions.length, (i) {
                    final phase = (i / _particlePositions.length) * 2 * pi;
                    return AnimatedBuilder(
                      animation: _particleController,
                      builder: (context, child) {
                        final t = _particleController.value;
                        final dy = sin((t * 2 * pi) + phase) * 15.0;

                        return Positioned(
                          left: _particlePositions[i].dx,
                          top: _particlePositions[i].dy,
                          child: Transform.translate(
                            offset: Offset(0, dy),
                            child: child,
                          ),
                        );
                      },
                      child: Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          // --- REPLACED ---
                          color: Colors.cyanAccent.withValues(alpha: 0.9),
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              // --- REPLACED ---
                              color: Colors.cyanAccent.withValues(alpha: 0.8),
                              blurRadius: 12,
                              spreadRadius: 2,
                            ),
                          ],
                        ),
                      ),
                    );
                  }),

                  // --- Logo card ---
                  Positioned(
                    top: 60,
                    child: Container(
                      width: 220,
                      height: 220,
                      decoration: BoxDecoration(
                        boxShadow: [
                          BoxShadow(
                            // --- REPLACED ---
                            color: Colors.black.withValues(alpha: 0.7),
                            blurRadius: 20,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: Image.asset(
                          'assets/logo.jpg',
                          fit: BoxFit.contain,
                          errorBuilder: (context, error, stackTrace) =>
                              const Center(
                                child: Icon(
                                  Icons.accessible,
                                  size: 90,
                                  color: Colors.white70,
                                ),
                              ),
                        ),
                      ),
                    ),
                  ),

                  // --- App name and subtitle ---
                  Positioned(
                    top: 300,
                    child: Column(
                      children: [
                        Text(
                          'SmartNav',
                          style: Theme.of(context).textTheme.headlineSmall
                              ?.copyWith(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 34,
                              ),
                        ).animate().fadeIn(duration: 600.ms).slideY(begin: 0.2),
                        const SizedBox(height: 12),
                        Text(
                          'Freedom through intelligent mobility',
                          style: Theme.of(context).textTheme.bodyLarge
                              ?.copyWith(color: Colors.white70, fontSize: 18),
                          textAlign: TextAlign.center,
                        ).animate().fadeIn(delay: 300.ms, duration: 700.ms),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// --- Custom Painter for the Noise Background ---
class NoisePainter extends CustomPainter {
  final Random _random = Random(0);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;
    int density = 5000;

    for (int i = 0; i < density; i++) {
      final opacity = _random.nextDouble() * 0.06 + 0.02;
      // --- REPLACED ---
      paint.color = Colors.white.withValues(alpha: opacity);
      final x = _random.nextDouble() * size.width;
      final y = _random.nextDouble() * size.height;
      canvas.drawRect(Rect.fromLTWH(x, y, 1, 1), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
