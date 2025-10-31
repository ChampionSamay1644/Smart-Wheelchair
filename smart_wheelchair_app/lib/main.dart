// ignore_for_file: avoid_print

import 'package:flutter/material.dart';
import 'package:smart_wheelchair_app/features/outdoor_navigation/outdoor_navigation_page.dart';
import 'features/auth/splash_screen.dart';
import 'features/auth/role_selection_page.dart';
import 'package:provider/provider.dart';
import 'core/providers/auth_provider.dart';
import 'core/providers/notifications_provider.dart';
import 'core/providers/movement_log_provider.dart';
// doctor dashboard removed
import 'features/dashboard/guardian_dashboard.dart';
import 'core/widgets/hold_button.dart';
import 'features/dashboard/health_status_page.dart';
import 'features/dashboard/movement_log_page.dart';
import 'joystick_control_page.dart';
import 'manual_control_page.dart';
import 'remote_control_page.dart';
import 'settings_page.dart';
import 'voice_control_page.dart';

import 'package:firebase_core/firebase_core.dart';
import 'core/services/firebase_service.dart';
import 'core/services/auth_service.dart';
import 'core/services/location_service.dart';
import 'core/services/notification_service.dart';
import 'core/services/health_report_service.dart';
import 'core/providers/location_provider.dart';
import 'core/providers/health_report_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Firebase core
  await Firebase.initializeApp();

  // Initialize FirebaseService (helper for analytics/messaging, if any)
  // Run initialize in background to avoid blocking app startup and causing frame skips
  FirebaseService.initialize();

  // Initialize core services
  final authService = AuthService();
  final locationService = LocationService();
  final healthReportService = HealthReportService();

  // Initialize notification system asynchronously after app start
  final notificationsProvider = NotificationsProvider();
  final notificationService = NotificationService(notificationsProvider);
  // Don't block startup - initialize in background
  notificationService.initialize().catchError((e) {
    debugPrint('Failed to initialize notifications: $e');
  });

  // Set up error handling
  FlutterError.onError = (details) {
    FlutterError.presentError(details);
    // Log to Firebase Crashlytics later
  };

  runApp(
    MultiProvider(
      providers: [
        // Core services available via Provider
        Provider<AuthService>.value(value: authService),

        // State management providers
        ChangeNotifierProvider(
          create: (_) => AuthProvider(authService: authService),
        ),
        ChangeNotifierProvider.value(value: notificationsProvider),
        ChangeNotifierProvider(
          create: (_) => LocationProvider(locationService),
        ),
        ChangeNotifierProvider(
          create: (_) => HealthReportProvider(healthReportService),
        ),
        ChangeNotifierProvider(create: (_) => MovementLogProvider()),
      ],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartNav Wheelchair',
      theme: ThemeData(
        primaryColor: Colors.blue,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          secondary: Colors.orange,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.all(20),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
          ),
        ),
      ),
      initialRoute: '/splash',
      routes: {
        '/splash': (context) => const SplashScreen(),
        '/role_selection': (context) => const RoleSelectionPage(),
        '/': (context) =>
            const MyHomePage(title: 'Patient Dashboard'), // Patient Dashboard
        '/patient_dashboard': (context) =>
            const MyHomePage(title: 'Patient Dashboard'),
        '/guardian_dashboard': (context) => const GuardianDashboard(),
        '/health_status': (context) => const HealthStatusPage(),
        '/movement_log': (context) => const MovementLogPage(),
        '/manual_control': (context) => ManualControlPage(),
        '/joystick_control': (context) => JoystickControlPage(),
        '/voice_control': (context) => VoiceControlPage(),
        '/remote_control': (context) => RemoteControlPage(),
        '/settings': (context) => SettingsPage(),
        '/location': (context) => const OutdoorNavigationPage(),
      },
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).primaryColor,
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 35,
                  height: 35,
                  decoration: BoxDecoration(
                    color: const Color(0xFF0A101F),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  padding: const EdgeInsets.all(4),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Image.asset(
                      'assets/logo.jpg',
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) =>
                          const SizedBox.shrink(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                const Text(
                  'SmartNav',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                  ),
                ),
              ],
            ),
          ],
        ),
        elevation: 4,
        actions: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: const [
                Icon(Icons.battery_full, color: Colors.white, size: 20),
                SizedBox(width: 2),
                Text(
                  '75%',
                  style: TextStyle(color: Colors.white, fontSize: 12),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: GestureDetector(
              onTap: () => Navigator.pushNamed(context, '/health_status'),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Icon(Icons.favorite, color: Colors.red, size: 20),
                  SizedBox(width: 2),
                  Text(
                    '72',
                    style: TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ],
              ),
            ),
          ),
          IconButton(
            iconSize: 20,
            padding: const EdgeInsets.all(8),
            icon: const Icon(Icons.help, color: Colors.white),
            onPressed: () {
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(const SnackBar(content: Text('Help requested!')));
            },
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.blue),
              child: Text(
                'Patient Menu',
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
            ),
            ListTile(
              leading: Icon(Icons.settings),
              title: Text('Settings'),
              onTap: () => Navigator.pushNamed(context, '/settings'),
            ),
            ListTile(
              leading: Icon(Icons.logout),
              title: Text('Logout'),
              onTap: () async {
                // If AuthProvider is available, call logout
                try {
                  await context.read<AuthProvider>().logout();
                } catch (_) {}
                if (!context.mounted) return;
                Navigator.pushReplacementNamed(context, '/role_selection');
              },
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            flex: 1,
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.grey[300]!),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withAlpha((0.8 * 255).round()),
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Stack(
                children: [
                  Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.camera_alt, size: 48, color: Colors.white54),
                        SizedBox(height: 8),
                        Text(
                          'Camera Preview',
                          style: TextStyle(color: Colors.white54, fontSize: 16),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          Expanded(
            flex: 1,
            child: Container(
              padding: const EdgeInsets.all(16),
              child: GridView.count(
                crossAxisCount: 2,
                childAspectRatio: 1,
                mainAxisSpacing: 16,
                crossAxisSpacing: 16,
                children: [
                  _buildControlButton(
                    context,
                    'Remote Control',
                    Icons.route,
                    Colors.blue[700]!,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => RemoteControlPage(),
                      ),
                    ),
                  ),
                  // merged manual and joystick into single Drive Control
                  _buildControlButton(
                    context,
                    'Manual Control',
                    Icons.sports_esports,
                    Colors.purple[700]!,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => JoystickControlPage(),
                      ),
                    ),
                  ),
                  _buildControlButton(
                    context,
                    'Voice Control',
                    Icons.mic,
                    Colors.orange[700]!,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => VoiceControlPage(),
                      ),
                    ),
                  ),
                  _buildControlButton(
                    context,
                    'Map Navigation',
                    Icons.map,
                    Colors.green[700]!,
                    () => Navigator.pushNamed(context, '/location'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: SizedBox(
        height: 64,
        width: 64,
        child: FloatingActionButton(
          backgroundColor: Colors.red,
          onPressed: null,
          child: HoldButton(
            holdDuration: const Duration(seconds: 2),
            onHold: () {
              // Notify local notification center (guardians listening)
              try {
                context.read<NotificationsProvider>().addEvent(
                  'Emergency',
                  'Patient triggered emergency stop',
                );
              } catch (e) {
                // provider not available in some test contexts
              }
              // Also log emergency in movement logs (and Firestore)
              try {
                context.read<MovementLogProvider>().addEntry(
                  'emergency',
                  'EMERGENCY STOP',
                );
              } catch (_) {}
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('EMERGENCY STOP ACTIVATED'),
                  backgroundColor: Colors.red,
                  duration: Duration(seconds: 2),
                ),
              );
              debugPrint('EMERGENCY STOP ACTIVATED');
            },
            child: const Icon(Icons.warning_amber_rounded, size: 32),
          ),
        ),
      ),
    );
  }

  Widget _buildControlButton(
    BuildContext context,
    String label,
    IconData icon,
    Color color,
    VoidCallback onPressed,
  ) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [color, color.withAlpha((0.8 * 255).round())],
        ),
        borderRadius: BorderRadius.circular(15),
        boxShadow: [
          BoxShadow(
            color: color.withAlpha((0.8 * 255).round()),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(15),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 40, color: Colors.white),
              const SizedBox(height: 8),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
