import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/enums.dart';
import '../../core/providers/auth_provider.dart';

class LoginPage extends StatefulWidget {
  final UserRole role;

  const LoginPage({super.key, required this.role});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _onLogin() async {
    if (!_formKey.currentState!.validate()) return;

    final success = await context.read<AuthProvider>().login(
      _emailController.text,
      _passwordController.text,
    );

    if (!mounted) return;

    if (success) {
      // Navigate to appropriate dashboard based on role
      String route = switch (widget.role) {
        UserRole.patient => '/patient_dashboard',
        UserRole.guardian => '/guardian_dashboard',
        UserRole.doctor => '/doctor_dashboard',
      };

      Navigator.pushReplacementNamed(context, route);
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Invalid credentials')));
    }
  }

  @override
  Widget build(BuildContext context) {
    // Pre-fill email for testing based on role
    _emailController.text = switch (widget.role) {
      UserRole.patient => 'patient@test.com',
      UserRole.guardian => 'guardian@test.com',
      UserRole.doctor => 'doctor@test.com',
    };

    return Scaffold(
      appBar: AppBar(title: Text('Login as ${widget.role.displayName}')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Icon(
              switch (widget.role) {
                UserRole.patient => FontAwesomeIcons.wheelchair,
                UserRole.guardian => FontAwesomeIcons.userShield,
                UserRole.doctor => FontAwesomeIcons.userDoctor,
              },
              size: 64,
              color: Theme.of(context).primaryColor,
            ),
            const SizedBox(height: 32),
            TextFormField(
              controller: _emailController,
              decoration: const InputDecoration(
                labelText: 'Email',
                prefixIcon: Icon(Icons.email),
              ),
              keyboardType: TextInputType.emailAddress,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Please enter your email';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),
            TextFormField(
              controller: _passwordController,
              decoration: const InputDecoration(
                labelText: 'Password',
                prefixIcon: Icon(Icons.lock),
              ),
              obscureText: true,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Please enter your password';
                }
                return null;
              },
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: context.watch<AuthProvider>().isLoading
                  ? null
                  : _onLogin,
              child: context.watch<AuthProvider>().isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Login'),
            ),
            const SizedBox(height: 16),
            TextButton(
              onPressed: () {
                // Forgot password not implemented yet
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Forgot password not implemented'),
                  ),
                );
              },
              child: const Text('Forgot Password?'),
            ),
          ],
        ),
      ),
    );
  }
}
