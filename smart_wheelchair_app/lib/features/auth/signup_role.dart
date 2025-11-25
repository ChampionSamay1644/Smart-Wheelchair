import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/enums.dart';
import '../../core/providers/auth_provider.dart';
import 'role_guard.dart';

class SignupRolePage extends StatefulWidget {
  final UserRole? role;
  const SignupRolePage({super.key, this.role});

  @override
  State<SignupRolePage> createState() => _SignupRolePageState();
}

class _SignupRolePageState extends State<SignupRolePage> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> _onSignup() async {
    if (!_formKey.currentState!.validate()) return;

    final role = widget.role ?? UserRole.patient;
    final authProvider = context.read<AuthProvider>();
    try {
      final success = await authProvider.signUp(
        email: _emailController.text.trim(),
        password: _passwordController.text.trim(),
        name: _nameController.text.trim(),
        role: role,
      );

      if (!mounted) return;
      if (success) {
        // Redirect based on role stored in user data
        final route = RoleGuard.routeForUser(authProvider.currentUser);
        Navigator.pushReplacementNamed(context, route);
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(authProvider.error ?? 'Signup failed')),
        );
      }
    } catch (e, st) {
      // Show full error so developer can see stacktrace from platform channel / Pigeon
      final text = '$e\n${st.toString()}';
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Error'),
          content: SingleChildScrollView(child: Text(text)),
          actions: [TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Close'))],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final role = widget.role ?? UserRole.patient;
    return Scaffold(
      appBar: AppBar(title: Text('Sign up as ${role.displayName}')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(labelText: 'Full name'),
                validator: (v) => (v == null || v.isEmpty) ? 'Enter name' : null,
              ),
              TextFormField(
                controller: _emailController,
                decoration: const InputDecoration(labelText: 'Email'),
                keyboardType: TextInputType.emailAddress,
                validator: (v) => (v == null || v.isEmpty) ? 'Enter email' : null,
              ),
              TextFormField(
                controller: _passwordController,
                decoration: const InputDecoration(labelText: 'Password'),
                obscureText: true,
                validator: (v) => (v == null || v.length < 6) ? '6+ chars' : null,
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _onSignup,
                child: const Text('Sign up'),
              ),
              const SizedBox(height: 8),
              TextButton(
                onPressed: () => Navigator.pushReplacementNamed(context, '/login', arguments: role),
                child: const Text('Already have an account? Login'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
