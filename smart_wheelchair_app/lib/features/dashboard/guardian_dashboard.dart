import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/providers/auth_provider.dart';
import '../../features/outdoor_navigation/outdoor_navigation_page.dart';
import '../../core/providers/notifications_provider.dart';
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
          // notification bell with unread count
          Consumer<NotificationsProvider>(
            builder: (context, np, _) {
              return IconButton(
                icon: Stack(
                  children: [
                    const Icon(Icons.notifications),
                    if (np.unreadCount > 0)
                      Positioned(
                        right: 0,
                        top: 0,
                        child: Container(
                          padding: const EdgeInsets.all(2),
                          decoration: const BoxDecoration(
                            color: Colors.red,
                            shape: BoxShape.circle,
                          ),
                          constraints: const BoxConstraints(
                            minWidth: 16,
                            minHeight: 16,
                          ),
                          child: Center(
                            child: Text(
                              np.unreadCount.toString(),
                              style: const TextStyle(
                                fontSize: 10,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
                onPressed: () {
                  // Open a simple dialog listing notifications
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('Notifications'),
                      content: SizedBox(
                        width: double.maxFinite,
                        child: ListView(
                          children: np.events
                              .map(
                                (e) => ListTile(
                                  title: Text(e.title),
                                  subtitle: Text(e.body),
                                  trailing: Text(
                                    '${e.time.hour}:${e.time.minute.toString().padLeft(2, '0')}',
                                    style: const TextStyle(fontSize: 12),
                                  ),
                                ),
                              )
                              .toList(),
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () {
                            np.markAllRead();
                            Navigator.pop(context);
                          },
                          child: const Text('Mark all read'),
                        ),
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Close'),
                        ),
                      ],
                    ),
                  );
                },
              );
            },
          ),
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
      body: Column(
        children: [
          // Map taking up most of the screen
          Expanded(
            flex: 3, // Takes up 75% of the screen
            child: Container(
              margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withAlpha(25),
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: const OutdoorNavigationPage(remoteView: true),
              ),
            ),
          ),
          // Bottom control strip
          Container(
            margin: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Expanded(
                  child: _buildControlButton(
                    context,
                    'Movement Log',
                    FontAwesomeIcons.list,
                    Colors.purple,
                    () => Navigator.pushNamed(context, '/movement_log'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _buildControlButton(
                    context,
                    'Health Status',
                    FontAwesomeIcons.heartPulse,
                    Colors.red,
                    () => Navigator.pushNamed(context, '/health_status'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControlButton(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Container(
      height: 60,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [color, color.withAlpha((0.8 * 255).round())],
        ),
        borderRadius: BorderRadius.circular(15),
        boxShadow: [
          BoxShadow(
            color: color.withAlpha((0.3 * 255).round()),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(15),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              FaIcon(icon, size: 24, color: Colors.white),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
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
