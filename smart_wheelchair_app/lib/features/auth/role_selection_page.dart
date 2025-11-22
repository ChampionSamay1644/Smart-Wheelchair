import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../../core/enums.dart';

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
                'Welcome to\nSmart Wheelchair',
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
                    _buildRoleCard(
                      context,
                      UserRole.doctor,
                      FontAwesomeIcons.userDoctor,
                      Colors.red,
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
    String route;
    switch (role) {
      case UserRole.patient:
        route = '/patient_dashboard';
      case UserRole.doctor:
        route = '/doctor_dashboard';
      case UserRole.guardian:
        route = '/guardian_dashboard';
    }
    Navigator.pushReplacementNamed(context, route);
  }
}
