import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/providers/alert_provider.dart';
import 'package:intl/intl.dart';

class NotificationPanel extends StatelessWidget {
  const NotificationPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        width: MediaQuery.of(context).size.width * 0.85,
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.75,
        ),
        padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
        child: Consumer<AlertProvider>(
          builder: (context, alertProvider, child) {
            final hasAlerts = alertProvider.alerts.isNotEmpty;
            return Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header row
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Notifications',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    if (hasAlerts)
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.red.shade50,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          '${alertProvider.alerts.length}',
                          style: const TextStyle(
                            color: Colors.red,
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 16),

                // Alert list
                if (alertProvider.isLoading && alertProvider.alerts.isEmpty)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 40),
                    child: Center(child: CircularProgressIndicator()),
                  )
                else if (!hasAlerts)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 40),
                    child: Center(
                      child: Column(
                        children: [
                          Icon(Icons.notifications_none, size: 48, color: Colors.grey[400]),
                          const SizedBox(height: 8),
                          Text('All clear!', style: TextStyle(color: Colors.grey[500], fontSize: 15)),
                        ],
                      ),
                    ),
                  )
                else
                  Flexible(
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: alertProvider.alerts.length,
                      separatorBuilder: (_, __) => const Divider(height: 24),
                      itemBuilder: (context, index) {
                        final alert = alertProvider.alerts[index];
                        final ts = alert['timestamp'] as int;
                        final time = DateTime.fromMillisecondsSinceEpoch(ts);
                        final timeStr = DateFormat('dd MMM, HH:mm').format(time);
                        final isRead = alert['read'] == true;
                        final isCritical = (alert['title'] ?? '').toString().contains('Emergency') ||
                            (alert['severity'] ?? '') == 'critical';

                        return Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              margin: const EdgeInsets.only(top: 6, right: 10),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: isRead
                                    ? Colors.transparent
                                    : (isCritical ? Colors.red : Colors.blue),
                              ),
                            ),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    alert['title'] ?? 'Alert',
                                    style: TextStyle(
                                      fontWeight: isRead ? FontWeight.normal : FontWeight.bold,
                                      fontSize: 15,
                                      color: isCritical ? Colors.red[700] : null,
                                    ),
                                  ),
                                  const SizedBox(height: 3),
                                  Text(
                                    alert['message'] ?? '',
                                    style: TextStyle(color: Colors.grey[600], fontSize: 13),
                                  ),
                                  const SizedBox(height: 3),
                                  Text(
                                    timeStr,
                                    style: TextStyle(color: Colors.grey[400], fontSize: 11),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ),

                const SizedBox(height: 20),
                const Divider(),
                const SizedBox(height: 8),

                // Action buttons
                OverflowBar(
                  alignment: MainAxisAlignment.spaceBetween,
                  overflowAlignment: OverflowBarAlignment.end,
                  spacing: 4,
                  children: [
                    if (hasAlerts) ...[
                      TextButton.icon(
                        onPressed: () => alertProvider.markAllAsRead(),
                        icon: const Icon(Icons.done_all, size: 16),
                        label: const Text('Mark read', style: TextStyle(fontSize: 12)),
                        style: TextButton.styleFrom(
                          foregroundColor: Colors.blue,
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                        ),
                      ),
                      TextButton.icon(
                        onPressed: () async {
                          final confirmed = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                              title: const Text('Clear all notifications?'),
                              content: const Text('This will permanently delete all notifications.'),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, false),
                                  child: const Text('Cancel'),
                                ),
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, true),
                                  style: TextButton.styleFrom(foregroundColor: Colors.red),
                                  child: const Text('Clear All'),
                                ),
                              ],
                            ),
                          );
                          if (confirmed == true) {
                            await alertProvider.clearAll();
                          }
                        },
                        icon: const Icon(Icons.delete_sweep, size: 16),
                        label: const Text('Clear All', style: TextStyle(fontSize: 12)),
                        style: TextButton.styleFrom(
                          foregroundColor: Colors.red,
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                        ),
                      ),
                    ],
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Close'),
                    ),
                  ],
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
