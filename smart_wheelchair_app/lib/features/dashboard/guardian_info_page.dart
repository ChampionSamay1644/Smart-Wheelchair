import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/providers/api_provider.dart';
import '../../services/api_service.dart';

class GuardianInfoPage extends StatefulWidget {
  final bool isInitialSetup;
  const GuardianInfoPage({super.key, this.isInitialSetup = false});

  @override
  State<GuardianInfoPage> createState() => _GuardianInfoPageState();
}

class _GuardianInfoPageState extends State<GuardianInfoPage> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _phoneController = TextEditingController();
  final _emailController = TextEditingController();
  final _relationshipController = TextEditingController();
  final _notesController = TextEditingController();
  
  bool _isLoading = true;
  bool _isSaving = false;
  late bool _isEditing;

  @override
  void initState() {
    super.initState();
    _isEditing = widget.isInitialSetup;
    _loadGuardianInfo();
  }

  Future<void> _loadGuardianInfo() async {
    final apiProvider = context.read<ApiProvider>();
    if (apiProvider.selectedDeviceId == null) {
      if (mounted) setState(() => _isLoading = false);
      return;
    }

    try {
      // Reusing medical info logic for now, or we can use a dedicated node
      final info = await ApiService().fetchMedicalInfo(apiProvider.selectedDeviceId!);
      final guardian = info['guardian'] ?? {};
      
      if (guardian.isNotEmpty) {
        _nameController.text = guardian['name']?.toString() ?? '';
        _phoneController.text = guardian['phone']?.toString() ?? '';
        _emailController.text = guardian['email']?.toString() ?? '';
        _relationshipController.text = guardian['relationship']?.toString() ?? '';
        _notesController.text = guardian['notes']?.toString() ?? '';
      }
    } catch (e) {
      debugPrint('Error loading guardian info: $e');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _saveGuardianInfo() async {
    if (!_formKey.currentState!.validate()) return;

    final apiProvider = context.read<ApiProvider>();
    if (apiProvider.selectedDeviceId == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No device selected')),
      );
      return;
    }

    setState(() => _isSaving = true);

    try {
      // Get current info first to merge
      final currentInfo = await ApiService().fetchMedicalInfo(apiProvider.selectedDeviceId!);
      
      final guardianData = {
        'name': _nameController.text,
        'phone': _phoneController.text,
        'email': _emailController.text,
        'relationship': _relationshipController.text,
        'notes': _notesController.text,
        'updatedAt': DateTime.now().toIso8601String(),
      };

      currentInfo['guardian'] = guardianData;

      await ApiService().updateMedicalInfo(apiProvider.selectedDeviceId!, currentInfo);
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Guardian profile updated successfully')),
        );
        
        if (widget.isInitialSetup) {
          Navigator.pushReplacementNamed(context, '/guardian_dashboard');
        } else {
          Navigator.pop(context);
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving guardian info: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    _phoneController.dispose();
    _emailController.dispose();
    _relationshipController.dispose();
    _notesController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    String title = widget.isInitialSetup 
        ? 'Setup Guardian Profile' 
        : (_isEditing ? 'Edit Guardian Profile' : 'View Guardian Profile');

    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        automaticallyImplyLeading: !widget.isInitialSetup,
        actions: [
          if (!widget.isInitialSetup && !_isLoading)
            IconButton(
              icon: Icon(_isEditing ? Icons.visibility : Icons.edit),
              onPressed: () => setState(() => _isEditing = !_isEditing),
              tooltip: _isEditing ? 'Switch to View' : 'Switch to Edit',
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Center(
                      child: CircleAvatar(
                        radius: 50,
                        child: Icon(Icons.shield_outlined, size: 50),
                      ),
                    ),
                    const SizedBox(height: 24),
                    const Text(
                      'Guardian Information',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'This information helps emergency services contact the right person.',
                      style: TextStyle(color: Colors.grey),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 32),
                    TextFormField(
                      controller: _nameController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Full Name',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.person),
                      ),
                      validator: (v) => v?.isEmpty ?? true ? 'Name is required' : null,
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _relationshipController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Relationship (e.g. Son, Nurse)',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.family_restroom),
                      ),
                      validator: (v) => v?.isEmpty ?? true ? 'Required' : null,
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _phoneController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Phone Number',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.phone),
                      ),
                      keyboardType: TextInputType.phone,
                      validator: (v) => v?.isEmpty ?? true ? 'Phone is required' : null,
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _emailController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Email Address',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.email),
                      ),
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _notesController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Additional Notes',
                        hintText: 'Any specific instructions for emergencies...',
                        border: OutlineInputBorder(),
                      ),
                      maxLines: 3,
                    ),
                    const SizedBox(height: 40),
                    if (_isEditing)
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: _isSaving ? null : _saveGuardianInfo,
                        child: _isSaving 
                          ? const CircularProgressIndicator(color: Colors.white)
                          : Text(widget.isInitialSetup ? 'COMPLETE SETUP' : 'SAVE CHANGES'),
                      ),
                    if (widget.isInitialSetup) ...[
                      const SizedBox(height: 16),
                      TextButton(
                        onPressed: () => Navigator.pushReplacementNamed(context, '/guardian_dashboard'),
                        child: const Text('ID Prefer to do this later'),
                      ),
                    ],
                  ],
                ),
              ),
            ),
    );
  }
}
