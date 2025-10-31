import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class GuardianSignupPage extends StatefulWidget {
  const GuardianSignupPage({super.key});

  @override
  State<GuardianSignupPage> createState() => _GuardianSignupPageState();
}

class _GuardianSignupPageState extends State<GuardianSignupPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _nameController = TextEditingController();
  final _patientCodeController = TextEditingController();
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _nameController.dispose();
    _patientCodeController.dispose();
    super.dispose();
  }

  Future<String?> _findPatientIdByCode(String code) async {
    final snapshot = await FirebaseFirestore.instance
        .collection('users')
        .where('userType', isEqualTo: 'patient')
        .where('connectionCode', isEqualTo: code)
        .limit(1)
        .get();

    if (snapshot.docs.isEmpty) return null;
    return snapshot.docs.first.id;
  }

  Future<void> _signUp() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final messenger = ScaffoldMessenger.of(context);

    try {
      final email = _emailController.text.trim();
      final password = _passwordController.text.trim();
      final name = _nameController.text.trim();
      final patientCode = _patientCodeController.text.trim();

      // 1. Verify patient code exists first
      final patientId = await _findPatientIdByCode(patientCode);
      if (patientId == null) {
        throw Exception('Invalid patient connection code');
      }

      // 2. Create user with Firebase Auth
      final credential = await FirebaseAuth.instance
          .createUserWithEmailAndPassword(email: email, password: password);

      // 3. Save user data to Firestore
      await FirebaseFirestore.instance
          .collection('users')
          .doc(credential.user!.uid)
          .set({
            'email': email,
            'name': name,
            'userType': 'guardian',
            'patientIds': [patientId],
            'createdAt': FieldValue.serverTimestamp(),
          });

      // Link patient to guardian
      await FirebaseFirestore.instance
          .collection('users')
          .doc(patientId)
          .update({'guardianId': credential.user!.uid});

      if (!mounted) return;

      // Success - show message and navigate
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Account created and linked to patient!'),
          backgroundColor: Colors.green,
        ),
      );

      // Navigate to home
      Navigator.pushReplacementNamed(context, '/guardian_dashboard');
    } catch (e) {
      debugPrint('SIGNUP FAILED: $e');
      if (!mounted) return;

      // Show error in UI
      messenger.showSnackBar(
        SnackBar(
          content: Text(
            e is FirebaseAuthException
                ? switch (e.code) {
                    'email-already-in-use' =>
                      'This email is already registered',
                    'weak-password' => 'Password is too weak',
                    'invalid-email' => 'Invalid email address',
                    _ => 'Sign up failed: ${e.message}',
                  }
                : 'Sign up failed: $e',
          ),
          backgroundColor: Colors.red,
        ),
      );

      setState(() {
        _errorMessage = e is FirebaseAuthException
            ? switch (e.code) {
                'email-already-in-use' => 'This email is already registered',
                'weak-password' => 'Password is too weak',
                'invalid-email' => 'Invalid email address',
                _ => 'Sign up failed: ${e.message}',
              }
            : 'Sign up failed: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Guardian Sign Up'), elevation: 0),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 20),
              Text(
                'Create Your Guardian Account',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: 'Full Name',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.person),
                ),
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Please enter your name';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _emailController,
                decoration: const InputDecoration(
                  labelText: 'Email',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.email),
                ),
                keyboardType: TextInputType.emailAddress,
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Please enter an email';
                  }
                  if (!value.contains('@')) {
                    return 'Please enter a valid email';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _passwordController,
                decoration: const InputDecoration(
                  labelText: 'Password',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.lock),
                ),
                obscureText: true,
                validator: (value) {
                  if (value == null || value.length < 6) {
                    return 'Password must be at least 6 characters';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _patientCodeController,
                decoration: const InputDecoration(
                  labelText: 'Patient Connection Code',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.link),
                  helperText: 'Enter the 6-digit code from your patient',
                ),
                validator: (value) {
                  if (value == null || value.trim().length != 6) {
                    return 'Please enter a valid 6-digit code';
                  }
                  return null;
                },
              ),
              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Text(
                  _errorMessage!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                  textAlign: TextAlign.center,
                ),
              ],
              const SizedBox(height: 32),
              ElevatedButton(
                onPressed: _isLoading ? null : _signUp,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? const SizedBox(
                        height: 20,
                        width: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Sign Up'),
              ),
              const SizedBox(height: 16),
              TextButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text('Already have an account? Sign in'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
