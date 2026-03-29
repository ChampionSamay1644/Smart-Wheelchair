import 'dart:async';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/providers/auth_provider.dart';
import '../../core/providers/outdoor_navigation_provider.dart';
import '../../core/providers/api_provider.dart';
import 'package:provider/provider.dart';
import '../../core/providers/alert_provider.dart';
import 'notification_panel.dart';
import '../outdoor_navigation/map_widget.dart';
import 'package:latlong2/latlong.dart';
import '../../core/services/sync_service.dart';
import 'guardian_info_page.dart';

class GuardianDashboard extends StatefulWidget {
  const GuardianDashboard({super.key});

  @override
  State<GuardianDashboard> createState() => _GuardianDashboardState();
}

class _GuardianDashboardState extends State<GuardianDashboard> {
  Map<String, dynamic>? _realtimeStatus;
  StreamSubscription? _statusSubscription;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ApiProvider>().reportPresence('guardian');
      _setupRealtimeSync();
    });
  }

  void _setupRealtimeSync() {
    final syncService = context.read<SyncService>();
    // For ASAP implementation, we listen to a 'test_patient' or based on auth
    // In a real app, this would be the UID of the patient the guardian is watching.
    _statusSubscription = syncService.listenToPatientStatus('test_patient_123').listen((status) {
      if (mounted) {
        setState(() {
          _realtimeStatus = status;
        });
      }
    });
  }

  @override
  void dispose() {
    _statusSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Header layer
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
            decoration: BoxDecoration(
              color: Theme.of(context).primaryColor,
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(20),
                bottomRight: Radius.circular(20),
              ),
            ),
            child: SafeArea(
              top: false,
              bottom: false,
              child: Row(
                children: [
                  // App logo + title
                  Image.asset(
                    'assets/logo.jpg',
                    height: 36,
                    errorBuilder: (c, e, s) => const FlutterLogo(size: 36),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'SmartNav',
                          style: Theme.of(context).textTheme.headlineSmall
                              ?.copyWith(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Guardian Dashboard',
                          style: Theme.of(
                            context,
                          ).textTheme.bodyMedium?.copyWith(color: Colors.white70),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.refresh, color: Colors.white),
                    tooltip: 'Refresh',
                    onPressed: () {
                      context.read<ApiProvider>().fetchLatestData();
                    },
                  ),
                  IconButton(
                    icon: const Icon(Icons.settings, color: Colors.white),
                    tooltip: 'Settings',
                    onPressed: () => Navigator.pushNamed(context, '/settings'),
                  ),
                  Stack(
                    children: [
                      IconButton(
                        icon: const Icon(Icons.notifications, color: Colors.white),
                        tooltip: 'Notifications',
                        onPressed: () {
                          showDialog(
                            context: context,
                            builder: (context) => const NotificationPanel(),
                          );
                        },
                      ),
                      Consumer<AlertProvider>(
                        builder: (context, alertProvider, child) {
                          if (alertProvider.unreadCount == 0) return const SizedBox.shrink();
                          return Positioned(
                            right: 8,
                            top: 8,
                            child: Container(
                              padding: const EdgeInsets.all(2),
                              decoration: BoxDecoration(
                                color: Colors.red,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              constraints: const BoxConstraints(
                                minWidth: 16,
                                minHeight: 16,
                              ),
                              child: Text(
                                '${alertProvider.unreadCount}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ),
                          );
                        },
                      ),
                    ],
                  ),
                  IconButton(
                    icon: const Icon(Icons.logout, color: Colors.white),
                    tooltip: 'Logout',
                    onPressed: () async {
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
          ),

          // Main screen layer with map
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  // Map area
                  Expanded(
                    child: Card(
                      clipBehavior: Clip.hardEdge,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Consumer<OutdoorNavigationProvider>(
                        builder: (context, nav, _) {
                          final state = nav.state;
                          return Stack(
                            children: [
                              if (state.currentPosition != null)
                                MapWidget(
                                  currentPosition: state.currentPosition!,
                                  routePoints: state.routePoints,
                                  destination: state.destination,
                                )
                              else
                                Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      const CircularProgressIndicator(),
                                      const SizedBox(height: 16),
                                      Consumer<ApiProvider>(
                                        builder: (context, api, _) {
                                          String msg = 'Waiting for Patient GPS...';
                                          if (api.apiUrl == null || api.apiUrl!.isEmpty) {
                                            msg = '⚠️ API URL not configured in Settings';
                                          } else if (api.selectedDeviceId == null) {
                                            msg = '⚠️ Wheelchair Device ID not set';
                                          }
                                          return Text(
                                            msg,
                                            textAlign: TextAlign.center,
                                            style: const TextStyle(fontWeight: FontWeight.bold),
                                          );
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              Positioned(
                                top: 12,
                                left: 12,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.9),
                                    borderRadius: BorderRadius.circular(20),
                                    boxShadow: [
                                      BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 4),
                                    ],
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: const BoxDecoration(
                                          color: Colors.green,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                      const SizedBox(width: 8),
                                      const Text(
                                        'Patient Live',
                                        style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                                      ),
                                      const SizedBox(width: 12),
                                      const VerticalDivider(width: 1, thickness: 1, indent: 4, endIndent: 4),
                                      const SizedBox(width: 12),
                                      Consumer<ApiProvider>(
                                        builder: (context, api, _) {
                                          final battery = (api.latestSensorData['motorStatus']?['battery'] ?? 100.0) as double;
                                          Color batteryColor = Colors.green;
                                          IconData batteryIcon = Icons.battery_full;
                                          
                                          if (battery < 15) {
                                            batteryColor = Colors.red;
                                            batteryIcon = Icons.battery_alert;
                                          } else if (battery < 40) {
                                            batteryColor = Colors.orange;
                                            batteryIcon = Icons.battery_3_bar;
                                          }
                                          
                                          return Row(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              Icon(batteryIcon, size: 14, color: batteryColor),
                                              const SizedBox(width: 4),
                                              Text(
                                                '${battery.toStringAsFixed(0)}%',
                                                style: TextStyle(
                                                  fontSize: 12, 
                                                  fontWeight: FontWeight.bold,
                                                  color: batteryColor,
                                                ),
                                              ),
                                            ],
                                          );
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),

                  // Live Health Ticker
                  _buildLiveHealthTicker(context),

                  const SizedBox(height: 12),

                  // Edit Profile Button (Requested by USER)
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (c) => const GuardianInfoPage()),
                        );
                      },
                      icon: const Icon(Icons.edit_note, size: 20),
                      label: const Text('Edit Guardian Profile'),
                    ),
                  ),

                  const SizedBox(height: 12),

                  // Action buttons row
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                          ),
                          onPressed: () {
                            Navigator.pushNamed(context, '/movement_log');
                          },
                          icon: const FaIcon(FontAwesomeIcons.list, size: 18),
                          label: const Text('Movement Logs'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                          ),
                          onPressed: () {
                            Navigator.pushNamed(context, '/health_status');
                          },
                          icon: const FaIcon(
                            FontAwesomeIcons.heartPulse,
                            size: 18,
                          ),
                          label: const Text('Live Health'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLiveHealthTicker(BuildContext context) {
    return Consumer<ApiProvider>(
      builder: (context, api, _) {
        final data = api.latestSensorData;
        final dht11 = data['dht11'] ?? {};
        final max30100 = data['max30100'] ?? {};
        
        if (api.error != null && api.error!.contains('401')) {
          return Card(
            color: Colors.red.shade50,
            child: ListTile(
              leading: const Icon(Icons.lock, color: Colors.red),
              title: const Text('Authentication Failed', style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
              subtitle: const Text('Wheelchair password is incorrect. Check Settings.'),
              trailing: TextButton(
                onPressed: () => Navigator.pushNamed(context, '/settings'),
                child: const Text('Fix'),
              ),
            ),
          );
        }

        // Precedence: Live Firebase Status > API Polling Data
        final pulse = _realtimeStatus?['heartRate']?.toStringAsFixed(0) ?? 
                    (max30100['ir'] != null ? (max30100['ir'] / 100).toStringAsFixed(0) : '--');
        
        final spo2 = _realtimeStatus?['spo2']?.toStringAsFixed(0) != null ? '${_realtimeStatus?['spo2']?.toStringAsFixed(0)}%' : 
                    (max30100['red'] != null ? '${(max30100['red'] / 100).toStringAsFixed(0)}%' : '--%');
        
        final temp = _realtimeStatus?['temperature']?.toStringAsFixed(1) != null ? '${_realtimeStatus?['temperature']?.toStringAsFixed(1)}°C' : 
                    '${dht11['temperature'] ?? '--'}°C';

        return Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(child: _buildStat('Pulse', pulse, Colors.red)),
                const VerticalDivider(width: 1),
                Expanded(child: _buildStat('SpO2', spo2, Colors.blue)),
                const VerticalDivider(width: 1),
                Expanded(child: _buildStat('Temp', temp, Colors.orange)),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildStat(String label, String value, Color color) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label, 
          style: const TextStyle(fontSize: 12, color: Colors.grey),
          overflow: TextOverflow.ellipsis,
        ),
        const SizedBox(height: 4),
        Text(
          value, 
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }
}
