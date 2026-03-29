import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../../core/enums.dart';
import '../../../core/providers/outdoor_navigation_provider.dart';
import 'package:provider/provider.dart';

class RoleSelectionPage extends StatelessWidget {
  const RoleSelectionPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 48),
              Text(
                'Welcome to\nSmartNav',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),
              Text(
                'Select your role',
                style: Theme.of(context).textTheme.titleLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              Expanded(
                child: GridView.count(
                  crossAxisCount: 2,
                  mainAxisSpacing: 16,
                  crossAxisSpacing: 16,
                  children: [
                    _buildRoleCard(
                      context,
                      UserRole.patient,
                      FontAwesomeIcons.wheelchair,
                      Colors.blue,
                    ),
                    _buildRoleCard(
                      context,
                      UserRole.guardian,
                      FontAwesomeIcons.userShield,
                      Colors.green,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRoleCard(
    BuildContext context,
    UserRole role,
    IconData icon,
    Color color,
  ) {
    return Card(
      elevation: 4,
      child: InkWell(
        onTap: () => _onRoleSelected(context, role),
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              FaIcon(icon, size: 48, color: color),
              const SizedBox(height: 16),
              Text(
                role.displayName,
                style: Theme.of(context).textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _onRoleSelected(BuildContext context, UserRole role) {
    context.read<OutdoorNavigationProvider>().setUserRole(role);
    Navigator.pushReplacementNamed(
      context,
      '/login',
      arguments: role,
    );
  }
}
