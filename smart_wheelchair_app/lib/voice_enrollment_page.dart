// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/wheelchair_websocket_service.dart';
import 'services/voice_cache_service.dart';
import 'package:shared_preferences/shared_preferences.dart';

class VoiceEnrollmentPage extends StatefulWidget {
  final WheelchairWebSocketService wsService;

  const VoiceEnrollmentPage({required this.wsService});

  @override
  State<VoiceEnrollmentPage> createState() => _VoiceEnrollmentPageState();
}

class _VoiceEnrollmentPageState extends State<VoiceEnrollmentPage> {
  static const List<Map<String, String>> _promptPhrases = [
    {
      'label': 'English',
      'phrase': 'My voice is my password, verify my identity',
    },
    {
      'label': 'Hindi',
      'phrase': 'मेरी आवाज मेरा पासवर्ड है, मेरी पहचान सत्यापित करें',
    },
    {
      'label': 'Marathi',
      'phrase': 'माझा आवाज माझा पासवर्ड आहे, माझी ओळख सत्यापित करा',
    },
  ];

  FlutterSoundRecorder? _audioRecorder;
  final TextEditingController _nameController = TextEditingController();
  final VoiceCacheService _voiceCacheService = VoiceCacheService();

  bool _isRecording = false;
  bool _isProcessing = false;
  String _statusMessage = 'Enter your name and record your voice';
  String _selectedGender = 'male';
  int _recommendedSamples = 3;
  int _samplesRecorded = 0;
  int? _activeSampleIndex;
  String _currentPrompt = _promptPhrases.first['phrase']!;

  StreamSubscription? _messageSubscription;
  StreamSubscription? _audioStreamSubscription;

  @override
  void initState() {
    super.initState();
    _audioRecorder = FlutterSoundRecorder();
    _audioRecorder!.openRecorder();
    _setupMessageListener();
    _prefillStoredName();
  }

  @override
  void dispose() {
    _messageSubscription?.cancel();
    _audioStreamSubscription?.cancel();
    _audioRecorder?.closeRecorder();
    _nameController.dispose();
    super.dispose();
  }

  void _setupMessageListener() {
    _messageSubscription = widget.wsService.messageStream.listen((message) {
      final type = message['type'] as String?;

      switch (type) {
        case 'enrollment_started':
          final total =
              (message['total_samples'] as num?)?.toInt() ??
              _recommendedSamples;
          final idx =
              (message['sample_index'] as num?)?.toInt() ??
              (_samplesRecorded + 1);
          final prompt = message['recommended_prompt']?.toString();
          setState(() {
            _isProcessing = false;
            _recommendedSamples = total;
            _activeSampleIndex = idx;
            _currentPrompt =
                prompt ??
                _promptPhrases[(idx - 1) % _promptPhrases.length]['phrase']!;
            _statusMessage = 'Recording sample $idx of $total... Speak clearly';
          });
          break;

        case 'enrollment_processing':
          setState(() {
            _isProcessing = true;
            _statusMessage = 'Processing your voice...';
          });
          break;

        case 'enrollment_sample_received':
          final recorded =
              (message['samples_recorded'] as num?)?.toInt() ??
              (_samplesRecorded + 1);
          final total =
              (message['total_samples'] as num?)?.toInt() ??
              _recommendedSamples;
          final prompt = message['recommended_prompt']?.toString();
          final infoMessage = message['message']?.toString();
          setState(() {
            _isProcessing = false;
            _isRecording = false;
            _activeSampleIndex = null;
            _samplesRecorded = recorded;
            _recommendedSamples = total;
            _currentPrompt =
                prompt ??
                _promptPhrases[recorded % _promptPhrases.length]['phrase']!;
            _statusMessage =
                infoMessage ??
                'Sample $recorded saved. ${total - recorded} more recommended.';
          });
          _showSnackBar('Sample $recorded saved.');
          break;

        case 'enrollment_success':
          final name = message['speaker_name']?.toString() ?? 'user';
          final recorded =
              (message['samples_recorded'] as num?)?.toInt() ??
              _samplesRecorded;
          final total =
              (message['total_samples'] as num?)?.toInt() ??
              _recommendedSamples;
          setState(() {
            _isProcessing = false;
            _isRecording = false;
            _activeSampleIndex = null;
            _samplesRecorded = recorded;
            _recommendedSamples = total < 3 ? 3 : total;
            _currentPrompt =
                _promptPhrases[_samplesRecorded %
                    _promptPhrases.length]['phrase']!;
            _statusMessage = 'Success! Voice profile created for $name';
          });
          _nameController.text = name;
          Future.microtask(() => _persistEnrollment(name));
          _showSuccessDialog(name);
          break;

        case 'enrollment_error':
          setState(() {
            _isProcessing = false;
            _isRecording = false;
            _activeSampleIndex = null;
            _statusMessage = 'Error: ${message['message']}';
          });
          _showSnackBar('Error: ${message['message']}');
          break;
      }
    });
  }

  void _prefillStoredName() {
    Future.microtask(() async {
      final prefs = await SharedPreferences.getInstance();
      final cachedName = prefs.getString('voice_profile_name');
      if (cachedName != null && cachedName.isNotEmpty) {
        _nameController.text = cachedName;
      }
    });
  }

  Future<void> _persistEnrollment(String name) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('voice_profile_name', name);
    await prefs.setBool('voice_profile_exists', true);
    await prefs.setBool('initial_voice_setup_completed', true);
    await _voiceCacheService.saveStatus(
      exists: true,
      speakerName: name,
      availableProfiles: [name],
    );
  }

  Future<void> _startEnrollment() async {
    final name = _nameController.text.trim().toLowerCase();

    if (name.isEmpty) {
      _showSnackBar('Please enter your name');
      return;
    }

    if (!widget.wsService.isConnected) {
      _showSnackBar('Please connect to wheelchair first');
      return;
    }

    if (!await Permission.microphone.isGranted) {
      final status = await Permission.microphone.request();
      if (!status.isGranted) {
        _showSnackBar('Microphone permission required');
        return;
      }
    }

    try {
      final sampleIndex = _samplesRecorded + 1;
      final promptData =
          _promptPhrases[(sampleIndex - 1) % _promptPhrases.length];
      final promptPhrase = promptData['phrase']!;
      final promptLabel = promptData['label']!;

      widget.wsService.startVoiceEnrollment(
        name,
        _selectedGender,
        sampleIndex: sampleIndex,
        totalSamples: _recommendedSamples,
        prompt: promptPhrase,
      );

      // Create a StreamController to receive audio data
      final controller = StreamController<Uint8List>();

      // Listen to the stream and send audio chunks
      _audioStreamSubscription = controller.stream.listen((Uint8List data) {
        widget.wsService.sendAudioChunk(data);
      });

      // Start recording to stream
      await _audioRecorder!.startRecorder(
        toStream: controller.sink,
        codec: Codec.pcm16,
        sampleRate: 16000,
        numChannels: 1,
      );

      setState(() {
        _isRecording = true;
        if (sampleIndex > _recommendedSamples) {
          _recommendedSamples = sampleIndex;
        }
        _activeSampleIndex = sampleIndex;
        _currentPrompt = promptPhrase;
        _statusMessage =
            'Recording sample $sampleIndex of $_recommendedSamples...\nSpeak ($promptLabel): "$promptPhrase"';
      });

      // Auto-stop after 5 seconds
      Future.delayed(const Duration(seconds: 5), () {
        if (_isRecording) {
          _stopEnrollment();
        }
      });
    } catch (e) {
      print('Error starting enrollment: $e');
      _showSnackBar('Failed to start recording: $e');
    }
  }

  Future<void> _stopEnrollment() async {
    if (!_isRecording) return;

    try {
      await _audioStreamSubscription?.cancel();
      _audioStreamSubscription = null;

      if (_audioRecorder != null) {
        await _audioRecorder!.stopRecorder();
      }

      final name = _nameController.text.trim().toLowerCase();
      final sampleIndex = _activeSampleIndex ?? (_samplesRecorded + 1);
      widget.wsService.stopVoiceEnrollment(
        name,
        _selectedGender,
        sampleIndex: sampleIndex,
        totalSamples: _recommendedSamples,
        finalize: false,
      );

      setState(() {
        _isRecording = false;
        _activeSampleIndex = null;
        _statusMessage = 'Processing...';
      });
    } catch (e) {
      print('Error stopping enrollment: $e');
      _showSnackBar('Error stopping recording: $e');
    }
  }

  void _showSuccessDialog(String name) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('Success!'),
        content: Text(
          'Voice profile created for $name.\n\nYou can now use voice commands with authentication.',
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context, true);
            },
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  void _finishEnrollmentEarly() {
    final name = _nameController.text.trim().toLowerCase();

    if (name.isEmpty) {
      _showSnackBar('Please enter your name first');
      return;
    }

    if (!widget.wsService.isConnected) {
      _showSnackBar('Connect to the wheelchair before finishing enrollment');
      return;
    }

    if (_isRecording) {
      _showSnackBar('Stop the current recording before finishing');
      return;
    }

    if (_samplesRecorded == 0) {
      _showSnackBar('Record at least one sample before finishing enrollment');
      return;
    }

    widget.wsService.finishVoiceEnrollment(name);
    setState(() {
      _isProcessing = true;
      _statusMessage =
          'Finalizing enrollment with $_samplesRecorded sample${_samplesRecorded == 1 ? '' : 's'}...';
    });
  }

  void _addOptionalSample() {
    if (_isRecording || _isProcessing) {
      _showSnackBar(
        'Wait for the current step to finish before adding another sample',
      );
      return;
    }

    setState(() {
      _recommendedSamples += 1;
      _statusMessage =
          'Optional sample added. Ready for sample ${_samplesRecorded + 1}.';
    });
  }

  Widget _buildSampleProgressCard() {
    final recommended = _recommendedSamples < 1 ? 1 : _recommendedSamples;
    final num recordedNum = _samplesRecorded.clamp(0, recommended);
    final int recorded = recordedNum.toInt();
    final double progressValue = recommended == 0
        ? 0.0
        : (recordedNum.toDouble() / recommended.toDouble()).clamp(0.0, 1.0);
    final nextPromptData = _promptPhrases[recorded % _promptPhrases.length];
    final nextPromptLabel = nextPromptData['label']!;
    final nextPromptPhrase = _currentPrompt;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue.withAlpha((0.08 * 255).round()),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.blue.withAlpha((0.25 * 255).round())),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Enrollment Progress',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              Text(
                '$recorded / $recommended samples',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: progressValue,
              minHeight: 10,
              backgroundColor: Colors.white.withAlpha((0.4 * 255).round()),
              valueColor: const AlwaysStoppedAnimation<Color>(Colors.blue),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Next recommended phrase ($nextPromptLabel):',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            '"$nextPromptPhrase"',
            style: const TextStyle(fontSize: 14, color: Colors.black87),
          ),
          const SizedBox(height: 12),
          Text(
            'Tip: record in different languages for stronger authentication.',
            style: TextStyle(fontSize: 13, color: Colors.blueGrey[700]),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Add Your Voice'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Icon(Icons.record_voice_over, size: 80, color: Colors.blue),
            const SizedBox(height: 24),
            const Text(
              'Voice Enrollment',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            Text(
              _statusMessage,
              style: TextStyle(fontSize: 16, color: Colors.grey[700]),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            _buildSampleProgressCard(),
            const SizedBox(height: 24),

            // Name input
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Your Name',
                hintText: 'e.g., samay',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.person),
              ),
              enabled: !_isRecording && !_isProcessing,
            ),

            const SizedBox(height: 24),

            // Gender selection
            const Text(
              'Gender:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            DropdownMenu<String>(
              initialSelection: _selectedGender,
              enabled: !(_isRecording || _isProcessing),
              label: const Text('Gender'),
              leadingIcon: const Icon(Icons.person_outline),
              dropdownMenuEntries: const [
                DropdownMenuEntry(value: 'male', label: 'Male'),
                DropdownMenuEntry(value: 'female', label: 'Female'),
              ],
              onSelected: (value) {
                if (value == null) return;
                setState(() {
                  _selectedGender = value;
                });
              },
            ),

            const SizedBox(height: 32),

            // Instructions
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue.withAlpha((0.1 * 255).round()),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Instructions:',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text('1. Enter your name (lowercase, no spaces)'),
                  Text('2. Select your gender'),
                  Text('3. Tap "Record Sample" (repeat at least 3 times)'),
                  Text(
                    '4. Speak the suggested phrase (rotate languages each time)',
                  ),
                  Text('5. Tap "Finish Enrollment" once you are satisfied'),
                ],
              ),
            ),

            const SizedBox(height: 32),

            // Record button
            if (!_isProcessing)
              ElevatedButton.icon(
                onPressed: _isRecording ? _stopEnrollment : _startEnrollment,
                icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                label: Text(
                  _isRecording
                      ? 'Stop Sample ${_activeSampleIndex ?? _samplesRecorded + 1}'
                      : 'Record Sample ${_samplesRecorded + 1}',
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isRecording ? Colors.red : Colors.blue,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  textStyle: const TextStyle(fontSize: 18),
                ),
              )
            else
              const Center(child: CircularProgressIndicator()),
            if (!_isProcessing) ...[
              const SizedBox(height: 16),
              Wrap(
                alignment: WrapAlignment.center,
                spacing: 12,
                runSpacing: 8,
                children: [
                  OutlinedButton.icon(
                    onPressed: _isRecording ? null : _addOptionalSample,
                    icon: const Icon(Icons.add_circle_outline),
                    label: const Text('Add Optional Sample'),
                  ),
                  OutlinedButton.icon(
                    onPressed: (_samplesRecorded == 0 || _isRecording)
                        ? null
                        : _finishEnrollmentEarly,
                    icon: const Icon(Icons.done_all),
                    label: const Text('Finish Enrollment'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
