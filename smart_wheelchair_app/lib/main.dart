// ignore_for_file: avoid_print

import 'package:flutter/material.dart';
import 'package:smart_wheelchair_app/features/outdoor_navigation/outdoor_navigation_page.dart';
import 'package:provider/provider.dart';
import 'core/providers/auth_provider.dart';
import 'features/auth/splash_screen.dart';
import 'features/auth/role_selection_page.dart';
import 'features/dashboard/doctor_dashboard.dart';
import 'features/dashboard/guardian_dashboard.dart';
import 'features/dashboard/health_status_page.dart';
import 'features/dashboard/movement_log_page.dart';
import 'joystick_control_page.dart';
import 'manual_control_page.dart';
import 'remote_control_page.dart';
import 'settings_page.dart';
import 'voice_control_page.dart';

void main() {
  runApp(
    MultiProvider(
      providers: [ChangeNotifierProvider(create: (_) => AuthProvider())],
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
      initialRoute: '/',
      routes: {
        '/': (context) => const SplashScreen(),
        '/role_selection': (context) => const RoleSelectionPage(),
        '/patient_dashboard': (context) =>
            const MyHomePage(title: 'Patient Dashboard'),
        '/doctor_dashboard': (context) => const DoctorDashboard(),
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

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

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
        title: const Text(
          'SmartNav',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 24,
          ),
        ),
        elevation: 4,
        actions: [
          IconButton(
            icon: const Icon(
              Icons.help,
              color: Colors.white,
            ), // Request Help icon
            onPressed: () {
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(const SnackBar(content: Text('Help requested!')));
            },
          ),
          IconButton(
            icon: const Icon(Icons.map, color: Colors.white),
            onPressed: () {
              Navigator.pushNamed(context, '/location'); // Map route
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
              leading: Icon(Icons.health_and_safety),
              title: Text('Health Status'),
              onTap: () => Navigator.pushNamed(context, '/health_status'),
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
                await context.read<AuthProvider>().logout();
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
              child: const Stack(
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
                  _buildControlButton(
                    context,
                    'Manual Control',
                    Icons.gamepad,
                    Colors.green[700]!,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ManualControlPage(),
                      ),
                    ),
                  ),
                  _buildControlButton(
                    context,
                    'Joystick Control',
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
          onPressed: () {
            print('EMERGENCY STOP ACTIVATED');
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('EMERGENCY STOP ACTIVATED'),
                backgroundColor: Colors.red,
                duration: Duration(seconds: 2),
              ),
            );
          },
          child: const Icon(Icons.warning_amber_rounded, size: 32),
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
