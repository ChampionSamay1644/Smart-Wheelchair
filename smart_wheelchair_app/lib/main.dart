// ignore_for_file: avoid_print

import 'dart:async';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:smart_wheelchair_app/features/outdoor_navigation/outdoor_navigation_page.dart';
import 'package:provider/provider.dart';
import 'core/providers/auth_provider.dart';
import 'core/providers/connection_provider.dart';
import 'core/providers/camera_feed_provider.dart';
import 'core/providers/bluetooth_provider.dart';
import 'core/providers/outdoor_navigation_provider.dart';
import 'core/providers/emergency_contacts_provider.dart';
import 'core/providers/alert_provider.dart';
import 'features/dashboard/notification_panel.dart';
import 'bluetooth_connection_page.dart';
import 'widgets/connection_dialog.dart';
import 'features/auth/splash_screen.dart';
import 'features/auth/role_selection_page.dart';
import 'features/dashboard/guardian_dashboard.dart';
import 'features/dashboard/health_status_page.dart';
import 'features/dashboard/medical_info_page.dart';
import 'features/dashboard/movement_log_page.dart';
import 'joystick_control_page.dart';
import 'manual_control_page.dart';
import 'settings_page.dart';
import 'voice_control_page.dart';
import 'services/emergency_stop_service.dart';
import 'core/providers/locale_provider.dart';
import 'core/localization.dart';
import 'features/settings/emergency_contacts_page.dart';
import 'services/api_service.dart';
import 'core/providers/api_provider.dart';
import 'core/enums.dart';
import 'features/auth/login_page.dart';
import 'core/services/sync_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  await ApiService().init();
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => ApiProvider()),
        ChangeNotifierProvider(create: (_) => ConnectionProvider()),
        ChangeNotifierProvider(create: (_) => CameraFeedProvider()),
        ChangeNotifierProxyProvider<ApiProvider, BluetoothProvider>(
          create: (context) => BluetoothProvider(context.read<ApiProvider>()),
          update: (context, api, bluetooth) => bluetooth!,
        ),
        ChangeNotifierProvider(create: (_) => LocaleProvider()),
        ChangeNotifierProvider(create: (_) => EmergencyContactsProvider()),
        ChangeNotifierProxyProvider2<ConnectionProvider, ApiProvider, OutdoorNavigationProvider>(
          create: (context) => OutdoorNavigationProvider(
            context.read<ConnectionProvider>(),
            context.read<ApiProvider>(),
          ),
          update: (context, connection, api, navigation) => navigation!,
        ),
        ChangeNotifierProxyProvider<ApiProvider, AlertProvider>(
          create: (_) => AlertProvider(),
          update: (_, api, alert) => alert!..updateDeviceId(api.selectedDeviceId),
        ),
        Provider(create: (_) => SyncService()),
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
      title: 'SmartNav',
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
        '/login': (context) => LoginPage(
              role: ModalRoute.of(context)!.settings.arguments as UserRole,
            ),
        '/patient_dashboard': (context) =>
            const MyHomePage(title: 'Patient Dashboard'),
        '/guardian_dashboard': (context) => const GuardianDashboard(),
        '/medical_info': (context) => const MedicalInfoPage(),
        '/health_status': (context) => const HealthStatusPage(),
        '/movement_log': (context) => const MovementLogPage(),
        '/manual_control': (context) => ManualControlPage(),
        '/joystick_control': (context) => JoystickControlPage(),
        '/voice_control': (context) => VoiceControlPage(),
        '/settings': (context) => SettingsPage(),
        '/location': (context) => const OutdoorNavigationPage(),
        '/bluetooth_connection': (context) => const BluetoothConnectionPage(),
        '/emergency_contacts': (context) => const EmergencyContactsPage(),
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

  bool _isHoldingEmergency = false;
  double _emergencyProgress = 0.0;
  Timer? _emergencyTimer;

  void _startEmergencyHold() {
    setState(() {
      _isHoldingEmergency = true;
      _emergencyProgress = 0.0;
    });
    _emergencyTimer = Timer.periodic(const Duration(milliseconds: 50), (timer) async {
      setState(() {
        _emergencyProgress += 0.01; // 50ms * 100 = 5s
      });
      if (_emergencyProgress >= 1.0) {
        _emergencyTimer?.cancel();
        _triggerEmergency();
      }
    });
  }

  void _cancelEmergencyHold() {
    _emergencyTimer?.cancel();
    setState(() {
      _isHoldingEmergency = false;
      _emergencyProgress = 0.0;
    });
  }

  Future<void> _triggerEmergency() async {
    final messenger = ScaffoldMessenger.of(context);
    await EmergencyStopService.trigger();
    // Also create a backend alert if connected
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    if (apiProvider.selectedDeviceId != null) {
      await ApiService().createAlert(
        apiProvider.selectedDeviceId!,
        'Emergency Button',
        'Patient triggered emergency stop (5s hold verified)',
      );
    }
    
    if (!mounted) return;
    messenger.showSnackBar(
      const SnackBar(
        content: Text('EMERGENCY STOP ACTIVATED!'),
        backgroundColor: Colors.red,
        duration: Duration(seconds: 4),
      ),
    );
    setState(() {
      _isHoldingEmergency = false;
      _emergencyProgress = 0.0;
    });
  }

  Timer? _syncTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ApiProvider>().reportPresence('patient');
      _startBackgroundSync();
    });
  }

  void _startBackgroundSync() {
    final syncService = context.read<SyncService>();
    final authProvider = context.read<AuthProvider>();
    
    // Periodically push mock sensor data if in mock mode (always for now)
    _syncTimer = Timer.periodic(const Duration(seconds: 3), (timer) async {
      final user = authProvider.currentUser;
      if (user == null) return;
      
      // Use 'test_patient_123' for ASAP interconnectivity demo, or user.id
      final uid = 'test_patient_123'; 
      
      // Mock data generation
      final random = DateTime.now().second;
      final heartRate = 72.0 + (random % 10);
      final spo2 = 98.0 + (random % 2);
      
      await syncService.updatePatientStatus(
        uid: uid,
        heartRate: heartRate,
        spo2: spo2,
        lat: 19.0760, // Mumbai coordinates
        lng: 72.8777,
        isEmergency: _isHoldingEmergency,
      );
    });
  }

  @override
  void dispose() {
    _emergencyTimer?.cancel();
    _syncTimer?.cancel();
    super.dispose();
  }

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
        elevation: 4,
        // Title with app logo and localized title
        title: Row(
          children: [
            // asset logo with fallback
            ClipOval(
              child: Image.asset(
                'assets/logo.jpg',
                height: 36,
                width: 36,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) =>
                    const FlutterLogo(size: 36),
              ),
            ),
            const SizedBox(width: 12),
            Text(
              tr(context, 'app_title'),
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
          ],
        ),
        actions: [
          const _ConnectionStatusAction(),
          const _BluetoothStatusAction(),
          IconButton(
            icon: const Icon(Icons.map, color: Colors.white),
            onPressed: () {
              Navigator.pushNamed(context, '/location'); // Map route
            },
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(48),
          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 16.0,
              vertical: 8.0,
            ),
            child: Row(
              children: [
                DropdownButton<String>(
                  value: context.watch<LocaleProvider>().locale,
                  underline: const SizedBox(),
                  items: const [
                    DropdownMenuItem(value: 'en', child: Text('English')),
                    DropdownMenuItem(value: 'hi', child: Text('हिन्दी')),
                    DropdownMenuItem(value: 'mr', child: Text('मराठी')),
                  ],
                  onChanged: (code) {
                    if (code != null) {
                      context.read<LocaleProvider>().setLocale(code);
                    }
                  },
                ),
                const Spacer(),
              ],
            ),
          ),
        ),
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.blue),
              child: Text(
                tr(context, 'patient_menu'),
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
            ),
            ListTile(
              leading: Icon(Icons.health_and_safety),
              title: Text(tr(context, 'health_status')),
              onTap: () => Navigator.pushNamed(context, '/health_status'),
            ),
            ListTile(
              leading: Icon(Icons.medical_services),
              title: Text('Medical Info'),
              onTap: () => Navigator.pushNamed(context, '/medical_info'),
            ),
            ListTile(
              leading: Icon(Icons.settings),
              title: Text(tr(context, 'settings')),
              onTap: () => Navigator.pushNamed(context, '/settings'),
            ),
            ListTile(
              leading: Icon(Icons.logout),
              title: Text(tr(context, 'logout')),
              onTap: () async {
                await context.read<AuthProvider>().logout();
                if (!context.mounted) return;
                Navigator.pushReplacementNamed(
                  context,
                  '/role_selection',
                );
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
              // move camera down slightly by increasing top margin
              margin: const EdgeInsets.fromLTRB(16, 24, 16, 16),
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
              padding: const EdgeInsets.all(14),
              child: GridView.count(
                crossAxisCount: 2,
                childAspectRatio: 1,
                mainAxisSpacing: 14,
                crossAxisSpacing: 14,
                children: [
                  _buildControlButton(
                    context,
                    tr(context, 'manual_control'),
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
                    tr(context, 'joystick_control'),
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
                    tr(context, 'voice_control'),
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
        height: 80,
        width: 80,
        child: GestureDetector(
          onLongPressStart: (_) => _startEmergencyHold(),
          onLongPressEnd: (_) => _cancelEmergencyHold(),
          child: Stack(
            alignment: Alignment.center,
            children: [
              if (_isHoldingEmergency)
                SizedBox(
                  height: 80,
                  width: 80,
                  child: CircularProgressIndicator(
                    value: _emergencyProgress,
                    strokeWidth: 8,
                    color: Colors.red[900],
                    backgroundColor: Colors.red[100],
                  ),
                ),
              FloatingActionButton(
                backgroundColor: _isHoldingEmergency ? Colors.red[900] : Colors.red,
                elevation: _isHoldingEmergency ? 0 : 6,
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Hold for 5 seconds to trigger emergency'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                },
                child: const Icon(Icons.warning_amber_rounded, size: 36),
              ),
            ],
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
        borderRadius: BorderRadius.circular(13),
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
          borderRadius: BorderRadius.circular(13),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 30, color: Colors.white),
              const SizedBox(height: 4),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
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
