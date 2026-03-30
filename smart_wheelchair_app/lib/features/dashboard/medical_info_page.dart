import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/providers/api_provider.dart';
import '../../services/api_service.dart';

class MedicalInfoPage extends StatefulWidget {
  final bool isInitialSetup;
  const MedicalInfoPage({super.key, this.isInitialSetup = false});

  @override
  State<MedicalInfoPage> createState() => _MedicalInfoPageState();
}

class _MedicalInfoPageState extends State<MedicalInfoPage> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _ageController = TextEditingController();
  final _bloodGroupController = TextEditingController();
  final _allergiesController = TextEditingController();
  final _conditionsController = TextEditingController();
  final _emergencyContactController = TextEditingController();
  
  bool _isLoading = true;
  bool _isSaving = false;
  late bool _isEditing;

  @override
  void initState() {
    super.initState();
    _isEditing = widget.isInitialSetup;
    _loadMedicalInfo();
  }

  Future<void> _loadMedicalInfo() async {
    final apiProvider = context.read<ApiProvider>();
    if (apiProvider.selectedDeviceId == null) {
      setState(() => _isLoading = false);
      return;
    }

    try {
      final info = await ApiService().fetchMedicalInfo(apiProvider.selectedDeviceId!);
      if (info.isNotEmpty) {
        _nameController.text = info['name']?.toString() ?? '';
        _ageController.text = info['age']?.toString() ?? '';
        _bloodGroupController.text = info['bloodGroup']?.toString() ?? '';
        _allergiesController.text = info['allergies']?.toString() ?? '';
        _conditionsController.text = info['conditions']?.toString() ?? '';
        _emergencyContactController.text = info['emergencyContact']?.toString() ?? '';
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading medical info: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _saveMedicalInfo() async {
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
      final info = {
        'name': _nameController.text,
        'age': _ageController.text,
        'bloodGroup': _bloodGroupController.text,
        'allergies': _allergiesController.text,
        'conditions': _conditionsController.text,
        'emergencyContact': _emergencyContactController.text,
      };

      await ApiService().updateMedicalInfo(apiProvider.selectedDeviceId!, info);
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Medical information saved successfully')),
        );
        
        if (widget.isInitialSetup) {
          Navigator.pushReplacementNamed(context, '/patient_dashboard');
        } else {
          Navigator.pop(context);
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving medical info: $e')),
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
    _ageController.dispose();
    _bloodGroupController.dispose();
    _allergiesController.dispose();
    _conditionsController.dispose();
    _emergencyContactController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    String title = widget.isInitialSetup 
        ? 'Setup Patient Profile' 
        : (_isEditing ? 'Edit Patient Profile' : 'View Patient Profile');

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
          if (_isEditing && !_isLoading)
            IconButton(
              icon: _isSaving 
                ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.save),
              onPressed: _isSaving ? null : _saveMedicalInfo,
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text(
                      'Patient Details',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _nameController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Full Name',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.person),
                      ),
                      validator: (v) => v?.isEmpty ?? true ? 'Required' : null,
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            controller: _ageController,
                            enabled: _isEditing,
                            decoration: const InputDecoration(
                              labelText: 'Age',
                              border: OutlineInputBorder(),
                            ),
                            keyboardType: TextInputType.number,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: TextFormField(
                            controller: _bloodGroupController,
                            enabled: _isEditing,
                            decoration: const InputDecoration(
                              labelText: 'Blood Group',
                              border: OutlineInputBorder(),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 24),
                    const Text(
                      'Medical Background',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _allergiesController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Allergies',
                        hintText: 'e.g. Peanuts, Penicillin',
                        border: OutlineInputBorder(),
                      ),
                      maxLines: 2,
                    ),
                    const SizedBox(height: 12),
                    TextFormField(
                      controller: _conditionsController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Chronic Conditions',
                        hintText: 'e.g. Diabetes, Hypertension',
                        border: OutlineInputBorder(),
                      ),
                      maxLines: 2,
                    ),
                    const SizedBox(height: 24),
                    const Text(
                      'Emergency',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _emergencyContactController,
                      enabled: _isEditing,
                      decoration: const InputDecoration(
                        labelText: 'Emergency Contact Number',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.phone),
                      ),
                      keyboardType: TextInputType.phone,
                    ),
                    const SizedBox(height: 32),
                    if (_isEditing)
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                        onPressed: _isSaving ? null : _saveMedicalInfo,
                        child: _isSaving 
                          ? const CircularProgressIndicator(color: Colors.white)
                          : Text(widget.isInitialSetup ? 'COMPLETE SETUP' : 'UPDATE INFORMATION'),
                      ),
                    if (widget.isInitialSetup) ...[
                      const SizedBox(height: 16),
                      TextButton(
                        onPressed: () => Navigator.pushReplacementNamed(context, '/patient_dashboard'),
                        child: const Text('I\'ll do this later'),
                      ),
                    ],
                  ],
                ),
              ),
            ),
    );
  }
}
