// ignore_for_file: avoid_print

import 'package:flutter/material.dart';
import 'package:smart_wheelchair_app/features/outdoor_navigation/outdoor_navigation_page.dart';
import 'package:provider/provider.dart';
import 'core/providers/auth_provider.dart';
import 'core/providers/connection_provider.dart';
import 'core/providers/camera_feed_provider.dart';
import 'bluetooth_connection_page.dart';
import 'core/providers/bluetooth_provider.dart';
import 'widgets/connection_dialog.dart';
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
import 'services/emergency_stop_service.dart';

void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => ConnectionProvider()),
        ChangeNotifierProvider(create: (_) => CameraFeedProvider()),
        ChangeNotifierProvider(create: (_) => BluetoothProvider()),
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
        '/bluetooth_connection': (context) => const BluetoothConnectionPage(),
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
  bool _hasPromptedConnectionDialog = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final connection = Provider.of<ConnectionProvider>(context);
    if (_hasPromptedConnectionDialog) {
      return;
    }
    if (!connection.isInitialized) {
      return;
    }
    if (connection.hasValidConfig) {
      _hasPromptedConnectionDialog = true;
      return;
    }

    _hasPromptedConnectionDialog = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      ConnectionDialog.show(context, barrierDismissible: false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).primaryColor,
        title: const FittedBox(
          fit: BoxFit.scaleDown,
          alignment: Alignment.centerLeft,
          child: Text(
            'Smart Wheelchair',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 24,
            ),
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
          const _ConnectionStatusAction(),
          const _BluetoothStatusAction(),
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
              child: Consumer<CameraFeedProvider>(
                builder: (context, camera, _) {
                  if (camera.hasFrame && camera.latestFrame != null) {
                    return ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Image.memory(
                            camera.latestFrame!,
                            fit: BoxFit.cover,
                            gaplessPlayback: true,
                          ),
                          Positioned(
                            left: 16,
                            bottom: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.black.withAlpha(150),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(
                                    Icons.videocam,
                                    color: Colors.white,
                                    size: 16,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    camera.streamName ?? 'Camera',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w600,
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

                  final statusText =
                      camera.error ?? 'Waiting for camera stream';
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.camera_alt,
                          size: 48,
                          color: Colors.white.withAlpha(150),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          statusText,
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 16,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  );
                },
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
          onPressed: () async {
            final messenger = ScaffoldMessenger.of(context);
            await EmergencyStopService.trigger();
            if (!mounted) return;
            messenger.showSnackBar(
              const SnackBar(
                content: Text('Emergency stop sent'),
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

class _ConnectionStatusAction extends StatelessWidget {
  const _ConnectionStatusAction();

  @override
  Widget build(BuildContext context) {
    return Consumer<ConnectionProvider>(
      builder: (context, provider, _) {
        final isConnecting = provider.isConnecting;
        final isConnected = provider.isConnected;
        final icon = isConnected
            ? Icons.wifi
            : isConnecting
            ? Icons.wifi_tethering
            : Icons.wifi_off;
        final tooltip = isConnected
            ? 'Connected to ${provider.ipAddress}:${provider.port}'
            : isConnecting
            ? 'Connecting to wheelchair...'
            : 'Tap to connect to wheelchair';

        return IconButton(
          icon: Icon(
            icon,
            color: isConnected
                ? Colors.lightGreenAccent
                : isConnecting
                ? Colors.orangeAccent
                : Colors.white,
          ),
          tooltip: tooltip,
          onPressed: () => ConnectionDialog.show(context),
        );
      },
    );
  }
}

class _BluetoothStatusAction extends StatelessWidget {
  const _BluetoothStatusAction();

  @override
  Widget build(BuildContext context) {
    return Consumer<BluetoothProvider>(
      builder: (context, provider, _) {
        final isAdapterOn = provider.isAdapterOn;
        final isConnected = provider.isConnected;
        final isConnecting = provider.isConnecting;

        IconData icon;
        Color color;
        String tooltip;

        if (!isAdapterOn) {
          icon = Icons.bluetooth_disabled;
          color = Colors.redAccent;
          tooltip = 'Bluetooth adapter disabled';
        } else if (isConnected) {
          icon = Icons.bluetooth_connected;
          color = Colors.lightBlueAccent;
          tooltip = 'Connected to ${provider.connectedAddress ?? 'wheelchair'}';
        } else if (isConnecting) {
          icon = Icons.bluetooth_searching;
          color = Colors.orangeAccent;
          tooltip = 'Connecting to wheelchair...';
        } else {
          icon = Icons.bluetooth;
          color = Colors.white;
          tooltip = 'Tap to connect via Bluetooth';
        }

        return IconButton(
          icon: Icon(icon, color: color),
          tooltip: tooltip,
          onPressed: () {
            Navigator.pushNamed(context, '/bluetooth_connection');
          },
        );
      },
    );
  }
}
