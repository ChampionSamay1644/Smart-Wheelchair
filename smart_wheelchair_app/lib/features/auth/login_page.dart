import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../../core/enums.dart';
import '../../core/providers/auth_provider.dart';
import '../../core/providers/api_provider.dart';
import '../../core/services/user_cache_service.dart';
import '../dashboard/guardian_info_page.dart';

class LoginPage extends StatefulWidget {
  final UserRole role;

  const LoginPage({super.key, required this.role});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _deviceIdController = TextEditingController();
  final _devicePasswordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  CachedProfile? _cachedProfile;
  bool _showManualLogin = false;

  @override
  void initState() {
    super.initState();
    _loadCache();
  }

  Future<void> _loadCache() async {
    final profiles = await UserCacheService.getProfiles();
    final profile = profiles.where((p) => p.role == widget.role).firstOrNull;
    
    if (profile != null) {
      setState(() {
        _cachedProfile = profile;
        _deviceIdController.text = profile.id; // Cached device ID
      });
    }
  }

  @override
  void dispose() {
    _deviceIdController.dispose();
    _devicePasswordController.dispose();
    super.dispose();
  }

  void _onLogin() async {
    if (!_formKey.currentState!.validate()) return;

    final deviceId = _deviceIdController.text.trim();
    final devicePassword = _devicePasswordController.text.trim();

    debugPrint('🚀 UI LOGIN ATTEMPT: role=${widget.role}, deviceId=$deviceId');

    final success = await context.read<AuthProvider>().login(
      deviceId,
      devicePassword,
      role: widget.role,
    );

    if (!mounted) return;

    if (success) {
      try {
        final apiProvider = context.read<ApiProvider>();
        
        await apiProvider.selectDevice(
          deviceId,
          password: devicePassword
        );

        if (!mounted) return;

        // Navigate to appropriate dashboard based on role
        String route = switch (widget.role) {
          UserRole.patient => '/patient_dashboard',
          UserRole.guardian => '/guardian_dashboard',
        };
        Navigator.pushReplacementNamed(context, route);
      } catch (e) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Device Connection Error: ${e.toString()}'),
            backgroundColor: Colors.orange,
          ),
        );
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Invalid Device ID or Password'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
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
              },
              size: 56,
              color: Theme.of(context).primaryColor,
            ),
            const SizedBox(height: 24),
            
            // PROFILE SWITCHER UI
            if (_cachedProfile != null && !_showManualLogin) ...[
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.blue.withOpacity(0.2)),
                ),
                child: Column(
                  children: [
                    CircleAvatar(
                      radius: 35,
                      backgroundColor: Colors.blue[100],
                      child: Text(
                        _cachedProfile!.role.displayName[0].toUpperCase(),
                        style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Welcome Back, ${_cachedProfile!.name}',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    Text(
                      'Device: ${_cachedProfile!.id}',
                      style: const TextStyle(color: Colors.grey),
                    ),
                    const SizedBox(height: 24),
                    
                    TextFormField(
                      controller: _devicePasswordController,
                      decoration: InputDecoration(
                        labelText: 'Enter ${widget.role.displayName} Password',
                        prefixIcon: const Icon(Icons.lock),
                        border: const OutlineInputBorder(),
                        filled: true,
                        fillColor: Colors.white,
                      ),
                      obscureText: true,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Password is required to log in';
                        }
                        return null;
                      },
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: _onLogin,
                        child: const Text('Login securely'),
                      ),
                    ),
                    TextButton(
                      onPressed: () => setState(() => _showManualLogin = true),
                      child: const Text('Wait, connect to another device'),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ] else ...[
              Text(
                'Secure Device Authentication',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.grey),
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _deviceIdController,
                decoration: const InputDecoration(
                  labelText: 'Wheelchair Device ID',
                  hintText: 'e.g. ALPHA_1',
                  prefixIcon: Icon(Icons.qr_code_scanner),
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter Device ID';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _devicePasswordController,
                decoration: InputDecoration(
                  labelText: '${widget.role.displayName} Password',
                  hintText: 'The secret key for this wheelchair',
                  prefixIcon: const Icon(Icons.security),
                  border: const OutlineInputBorder(),
                ),
                obscureText: true,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter Password';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 32),
              ElevatedButton(
                onPressed: context.watch<AuthProvider>().isLoading
                    ? null
                    : _onLogin,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: context.watch<AuthProvider>().isLoading
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Connect & Login'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
