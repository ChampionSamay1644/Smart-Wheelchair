import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:provider/provider.dart';
import '../../core/providers/auth_provider.dart';
import '../../core/providers/api_provider.dart';
import '../../core/enums.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _checkAuthAndNavigate();
  }

  Future<void> _checkAuthAndNavigate() async {
    // Show splash for at least 2 seconds
    final startTime = DateTime.now();
    
    final auth = context.read<AuthProvider>();
    final api = context.read<ApiProvider>();

    // Wait for providers to initialize from storage
    int retries = 0;
    while ((!auth.isInitialized || !api.apiUrl.toString().contains('http')) && retries < 15) {
      debugPrint('⏳ Waiting for Storage providers initialization... (Attempt ${retries + 1})');
      await Future.delayed(const Duration(milliseconds: 300));
      retries++;
    }

    if (api.selectedDeviceId != null) {
       debugPrint('🔌 API: Restored Device preference: ${api.selectedDeviceId}');
    }

    if (!mounted) return;

    if (auth.isAuthenticated) {
      debugPrint('✅ SESSION RESTORED: User=${auth.currentUser?.name}, Role=${auth.userRole}');
      
      // If we have a device ID, ensure polling is started
      if (api.selectedDeviceId != null) {
        api.startPolling();
        api.reportPresence(auth.userRole == UserRole.patient ? 'patient' : 'guardian', name: auth.currentUser?.name);
      }

      final route = auth.userRole == UserRole.patient ? '/patient_dashboard' : '/guardian_dashboard';
      Navigator.pushReplacementNamed(context, route);
    } else {
      debugPrint('🚫 NO SESSION FOUND: Directing to Role Selection.');
      Navigator.pushReplacementNamed(context, '/role_selection');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).primaryColor,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Wheelchair Icon Animation
            Icon(Icons.accessible, size: 120, color: Colors.white)
                .animate()
                .fadeIn(duration: 600.ms)
                .scale(delay: 200.ms)
                .then()
                .shimmer(duration: 1200.ms),
            const SizedBox(height: 24),
            // Motivational Quote
            Text(
                  'Freedom through intelligent mobility.',
                  style: Theme.of(
                    context,
                  ).textTheme.headlineSmall?.copyWith(color: Colors.white),
                  textAlign: TextAlign.center,
                )
                .animate()
                .fadeIn(delay: 400.ms, duration: 800.ms)
                .slideY(begin: 0.2, end: 0),
          ],
        ),
      ),
    );
  }
}
