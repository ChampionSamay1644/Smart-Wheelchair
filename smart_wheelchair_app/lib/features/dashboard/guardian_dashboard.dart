import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/providers/auth_provider.dart';
import 'package:provider/provider.dart';

class GuardianDashboard extends StatelessWidget {
  const GuardianDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().currentUser;

    return Scaffold(
      appBar: AppBar(
        title: Text('Guardian: ${user?.name ?? ""}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await context.read<AuthProvider>().logout();
              if (!context.mounted) return;
              Navigator.pushReplacementNamed(context, '/role_selection');
            },
          ),
        ],
      ),
      body: GridView.count(
        padding: const EdgeInsets.all(16),
        crossAxisCount: 2,
        mainAxisSpacing: 16,
        crossAxisSpacing: 16,
        children: [
          _buildCard(
            context,
            'Patient Location',
            FontAwesomeIcons.locationDot,
            Colors.blue,
            () {
              Navigator.pushNamed(context, '/patient_location');
            },
          ),
          _buildCard(
            context,
            'Health Status',
            FontAwesomeIcons.heartPulse,
            Colors.red,
            () {
              Navigator.pushNamed(context, '/health_status');
            },
          ),
          _buildCard(
            context,
            'Emergency Alert',
            FontAwesomeIcons.bell,
            Colors.orange,
            () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Emergency alert sent to patient'),
                  backgroundColor: Colors.orange,
                ),
              );
            },
          ),
          _buildCard(
            context,
            'Stop Wheelchair',
            FontAwesomeIcons.stop,
            Colors.red,
            () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Stop Wheelchair?'),
                  content: const Text(
                    'Are you sure you want to stop the wheelchair?',
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () {
                        Navigator.pop(context);
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Wheelchair stopped'),
                            backgroundColor: Colors.red,
                          ),
                        );
                      },
                      child: const Text('Stop'),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildCard(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Card(
      elevation: 4,
      child: InkWell(
        onTap: onTap,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FaIcon(icon, size: 48, color: color),
            const SizedBox(height: 16),
            Text(
              title,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ],
        ),
      ),
    );
  }
}
