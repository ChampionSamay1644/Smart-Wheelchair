import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/providers/auth_provider.dart';
import 'package:provider/provider.dart';
import '../outdoor_navigation/map_widget.dart';
import 'package:latlong2/latlong.dart';

class GuardianDashboard extends StatelessWidget {
  const GuardianDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    // Guardian name intentionally not shown here; using app brand title instead

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
                    'assets/logo.png',
                    height: 36,
                    errorBuilder: (c, e, s) => const FlutterLogo(size: 36),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'SmartNav',
                        style: Theme.of(context)
                            .textTheme
                            .headlineSmall
                            ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Guardian Dashboard',
                        style: Theme.of(context)
                            .textTheme
                            .bodyMedium
                            ?.copyWith(color: Colors.white70),
                      ),
                    ],
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.notifications, color: Colors.white),
                    tooltip: 'Notifications',
                    onPressed: () {
                      // TO: show notifications / emergency timestamps
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Notifications tapped')),
                      );
                    },
                  ),
                  IconButton(
                    icon: const Icon(Icons.logout, color: Colors.white),
                    tooltip: 'Logout',
                    onPressed: () async {
                      await context.read<AuthProvider>().logout();
                      if (!context.mounted) return;
                      Navigator.pushReplacementNamed(context, '/role_selection');
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
                      child: MapWidget(
                        currentPosition: const LatLng(
                          28.6139,
                          77.2090,
                        ), // placeholder
                        routePoints: const [],
                        destination: null,
                      ),
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
                            // TO: show movement logs for patient (mocked)
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
                            // Open health status page which uses the same mock charts
                            Navigator.pushNamed(context, '/health_status');
                          },
                          icon: const FaIcon(
                            FontAwesomeIcons.heartPulse,
                            size: 18,
                          ),
                          label: const Text('Health Data'),
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

  // Helper cards removed — not used in the redesigned guardian dashboard.
}
