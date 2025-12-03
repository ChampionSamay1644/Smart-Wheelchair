// ignore_for_file: use_key_in_widget_constructors, avoid_print

import 'dart:async';
import 'dart:typed_data';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'core/providers/connection_provider.dart';
import 'services/voice_cache_service.dart';
import 'services/wheelchair_websocket_service.dart';
import 'services/emergency_stop_service.dart';
import 'voice_enrollment_page.dart';
import 'widgets/connection_dialog.dart';

class VoiceControlPage extends StatefulWidget {
  @override
  State<VoiceControlPage> createState() => _VoiceControlPageState();
}

class _VoiceControlPageState extends State<VoiceControlPage> {
  final WheelchairWebSocketService _wsService = WheelchairWebSocketService();
  final VoiceCacheService _voiceCacheService = VoiceCacheService();

  FlutterSoundRecorder? _audioRecorder;
  StreamSubscription<Map<String, dynamic>>? _messageSubscription;
  StreamSubscription<Uint8List>? _recordingDataSubscription;
  StreamSubscription<ConnectivityResult>? _connectivitySubscription;
  SharedPreferences? _prefs;

  bool _isListening = false;
  bool _isProcessing = false;
  String _lastCommand = 'No command yet';
  String _lastTranscription = '';
  String _statusMessage = 'Not connected';
  double _confidence = 0.0;
  String? _lastSpeakerName;
  double? _lastSpeakerScore;
  bool? _lastSpeakerVerified;
  double? _requiredSpeakerThreshold;
  bool? _lastCommandExecuted;
  String _commandFeedbackMessage = '';
  bool _isWifiAvailable = true;
  bool _voiceProfileExists = false;
  String? _voiceProfileName;
  List<String> _availableVoiceProfiles = [];
  bool _initialVoiceSetupCompleted = false;
  bool _hasPromptedInitialEnrollment = false;
  bool _hasPromptedConnectionDialog = false;
  bool _lastConnectionState = false;

  @override
  void initState() {
    super.initState();
    _audioRecorder = FlutterSoundRecorder();
    _audioRecorder!.openRecorder();
    _initConnectivity();
    _initialize();
  }

  void _initialize() {
    Future.microtask(() async {
      _subscribeToMessages();
      await _loadPreferences();
      await _loadVoiceCache();
      await _requestPermissions();
      if (!mounted) {
        return;
      }
      final connection = context.read<ConnectionProvider>();
      if (connection.isConnected) {
        setState(() {
          _statusMessage = 'Connected to wheelchair';
        });
        _requestVoiceStatus();
      }
    });
  }

  Future<void> _initConnectivity() async {
    try {
      final result = await Connectivity().checkConnectivity();
      _updateConnectivityState(result);
    } catch (_) {
      // Ignore connectivity errors; assume Wi-Fi unavailable until the next event.
      _updateConnectivityState(ConnectivityResult.none);
    }

    _connectivitySubscription?.cancel();
    _connectivitySubscription = Connectivity().onConnectivityChanged.listen(
      _updateConnectivityState,
    );
  }

  void _updateConnectivityState(ConnectivityResult result) {
    final wifiActive =
        result == ConnectivityResult.wifi ||
        result == ConnectivityResult.ethernet;
    if (!mounted) {
      _isWifiAvailable = wifiActive;
      return;
    }

    if (_isWifiAvailable != wifiActive) {
      setState(() {
        _isWifiAvailable = wifiActive;
      });

      if (!wifiActive) {
        _showSnackBar(
          'Wi-Fi is off. Connect to the wheelchair network to use voice control.',
        );
      }
    }
  }

  Future<bool> _ensureWifiReady() async {
    final result = await Connectivity().checkConnectivity();
    _updateConnectivityState(result);
    if (!_isWifiAvailable) {
      if (mounted) {
        _showSnackBar('Cannot start voice capture without Wi-Fi.');
      }
      return false;
    }
    return true;
  }

  String _speakerConfidenceLabel() {
    if (_lastSpeakerScore == null) {
      return '';
    }
    final normalized = _lastSpeakerScore!.clamp(0.0, 1.0);
    final threshold = _requiredSpeakerThreshold?.clamp(0.0, 1.0);
    final thresholdLabel = threshold != null
        ? ' (needs ≥ ${(threshold * 100).toStringAsFixed(0)}%)'
        : '';
    return ' | ${(normalized * 100).toStringAsFixed(0)}%$thresholdLabel';
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final connection = Provider.of<ConnectionProvider>(context);
    if (!_hasPromptedConnectionDialog &&
        connection.isInitialized &&
        !connection.hasValidConfig) {
      _hasPromptedConnectionDialog = true;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        ConnectionDialog.show(context, barrierDismissible: false);
      });
    } else if (connection.hasValidConfig && !_hasPromptedConnectionDialog) {
      _hasPromptedConnectionDialog = true;
    }
  }

  @override
  void dispose() {
    _messageSubscription?.cancel();
    _recordingDataSubscription?.cancel();
    _connectivitySubscription?.cancel();
    _audioRecorder?.closeRecorder();
    super.dispose();
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    _prefs = prefs;
    if (!mounted) {
      return;
    }
    setState(() {
      _voiceProfileName = prefs.getString('voice_profile_name');
      _voiceProfileExists = prefs.getBool('voice_profile_exists') ?? false;
      _initialVoiceSetupCompleted =
          prefs.getBool('initial_voice_setup_completed') ?? false;
    });
  }

  Future<void> _loadVoiceCache() async {
    final status = await _voiceCacheService.loadStatus();
    if (!mounted) {
      return;
    }
    setState(() {
      _voiceProfileExists = _voiceProfileExists || status.exists;
      if (status.speakerName != null && status.speakerName!.isNotEmpty) {
        _voiceProfileName ??= status.speakerName;
      }
      _availableVoiceProfiles = status.availableProfiles;
    });
  }

  Future<void> _requestPermissions() async {
    final micStatus = await Permission.microphone.request();
    if (!micStatus.isGranted) {
      _showSnackBar('Microphone permission is required for voice control');
    }
  }

  void _subscribeToMessages() {
    _messageSubscription?.cancel();
    _messageSubscription = _wsService.messageStream.listen(
      _handleWebSocketMessage,
      onError: (error) {
        if (!mounted) return;
        setState(() {
          _statusMessage = 'Connection error';
          _isListening = false;
          _isProcessing = false;
        });
      },
    );
  }

  void _handleWebSocketMessage(Map<String, dynamic> message) {
    if (!mounted) {
      return;
    }

    final type = message['type'] as String?;
    switch (type) {
      case 'connection':
        setState(() {
          _statusMessage = message['message']?.toString() ?? 'Connected';
          _isProcessing = false;
        });
        _requestVoiceStatus();
        break;
      case 'connection_failed':
        setState(() {
          final reason = message['message']?.toString();
          _statusMessage = reason ?? 'Connection failed';
          _isListening = false;
          _isProcessing = false;
        });
        break;
      case 'disconnected':
        setState(() {
          final reason = message['message']?.toString();
          _statusMessage = reason ?? 'Disconnected from server';
          _isListening = false;
          _isProcessing = false;
        });
        break;
      case 'reconnecting':
        setState(() {
          final reason = message['message']?.toString();
          _statusMessage = reason ?? 'Reconnecting...';
          _isListening = false;
          _isProcessing = false;
        });
        break;
      case 'reconnect_exhausted':
        setState(() {
          final reason = message['message']?.toString();
          _statusMessage = reason ?? 'Unable to reach wheelchair server';
          _isListening = false;
          _isProcessing = false;
        });
        _showSnackBar('Connection attempts stopped. Update the IP and retry.');
        break;
      case 'recording_started':
        setState(() {
          _statusMessage = 'Recording...';
        });
        break;
      case 'processing':
        setState(() {
          _isProcessing = true;
          _statusMessage = 'Processing command...';
        });
        break;
      case 'command_recognized':
        final command = message['command']?.toString() ?? 'Unknown';
        final confidence = (message['confidence'] as num?)?.toDouble() ?? 0.0;
        final transcription = message['transcription']?.toString() ?? command;
        final speakerData = message['speaker'] as Map<String, dynamic>?;
        final speakerName = speakerData?['name']?.toString();
        final speakerScore = (speakerData?['score'] as num?)?.toDouble();
        final speakerVerified = speakerData?['verified'] == true;
        final requiredThreshold = (speakerData?['required_threshold'] as num?)
            ?.toDouble();
        final executed = message['executed'] == true;
        final feedbackMessage =
            message['message']?.toString() ??
            (executed
                ? "Command '$command' executed"
                : 'Authentication failed. Command blocked.');
        setState(() {
          _isProcessing = false;
          _isListening = false;
          _lastCommand = command.toUpperCase();
          _confidence = confidence;
          _statusMessage = executed ? 'Command executed' : feedbackMessage;
          _lastTranscription = transcription;
          _lastSpeakerName = (speakerName != null && speakerName.isNotEmpty)
              ? speakerName
              : null;
          _lastSpeakerScore = speakerScore;
          _lastSpeakerVerified = (speakerName != null && speakerName.isNotEmpty)
              ? speakerVerified
              : null;
          _requiredSpeakerThreshold = requiredThreshold;
          _lastCommandExecuted = executed;
          _commandFeedbackMessage = feedbackMessage;
        });
        _showSnackBar(
          executed ? '✓ ${feedbackMessage.trim()}' : feedbackMessage.trim(),
        );
        break;
      case 'command_not_recognized':
        final transcription = message['transcription']?.toString() ?? '';
        final speakerData = message['speaker'] as Map<String, dynamic>?;
        final speakerName = speakerData?['name']?.toString();
        final speakerScore = (speakerData?['score'] as num?)?.toDouble();
        final speakerVerified = speakerData?['verified'] == true;
        setState(() {
          _isProcessing = false;
          _isListening = false;
          _statusMessage = 'Command not recognized';
          _lastTranscription = transcription;
          _confidence = (message['confidence'] as num?)?.toDouble() ?? 0.0;
          _lastSpeakerName = (speakerName != null && speakerName.isNotEmpty)
              ? speakerName
              : null;
          _lastSpeakerScore = speakerScore;
          _lastSpeakerVerified = (speakerName != null && speakerName.isNotEmpty)
              ? speakerVerified
              : null;
          _requiredSpeakerThreshold =
              (speakerData?['required_threshold'] as num?)?.toDouble();
          _lastCommandExecuted = false;
          _commandFeedbackMessage = 'Command not recognized';
        });
        _showSnackBar('✗ Could not recognize command. Please try again.');
        break;
      case 'auto_stop':
        _showSnackBar('Wheelchair stopped automatically');
        break;
      case 'error':
        final error = message['message']?.toString() ?? 'Unknown error';
        setState(() {
          _isProcessing = false;
          _isListening = false;
          _statusMessage = 'Error: $error';
        });
        _showSnackBar('Error: $error');
        break;
      case 'voice_profile_status':
        final exists = message['exists'] == true;
        final speakerName = (message['speaker_name'] as String?)
            ?.trim()
            .toLowerCase();
        final profiles =
            (message['available_profiles'] as List<dynamic>?)
                ?.map((value) => value.toString())
                .toList() ??
            <String>[];
        _updateVoiceProfileStatus(
          exists: exists,
          speakerName: speakerName,
          availableProfiles: profiles,
        );
        break;
    }
  }

  Future<void> _requestVoiceStatus() async {
    if (!_wsService.isConnected) {
      return;
    }
    if (_voiceProfileName != null && _voiceProfileName!.isNotEmpty) {
      _wsService.checkVoiceProfileStatus(speakerName: _voiceProfileName);
    } else {
      _wsService.checkVoiceProfileStatus();
    }
  }

  Future<void> _updateVoiceProfileStatus({
    required bool exists,
    String? speakerName,
    List<String>? availableProfiles,
  }) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;

    await prefs.setBool('voice_profile_exists', exists);

    final normalizedName = (speakerName != null && speakerName.isNotEmpty)
        ? speakerName
        : _voiceProfileName;

    if (normalizedName != null && normalizedName.isNotEmpty) {
      await prefs.setString('voice_profile_name', normalizedName);
    }

    final profiles = availableProfiles ?? <String>[];
    await _voiceCacheService.saveStatus(
      exists: exists,
      speakerName: normalizedName,
      availableProfiles: profiles.isNotEmpty
          ? profiles
          : (normalizedName != null && normalizedName.isNotEmpty)
          ? <String>[normalizedName]
          : <String>[],
    );

    if (!mounted) {
      return;
    }

    setState(() {
      _voiceProfileExists = exists;
      if (normalizedName != null && normalizedName.isNotEmpty) {
        _voiceProfileName = normalizedName;
      }
      _availableVoiceProfiles = profiles.isNotEmpty
          ? profiles
          : (_voiceProfileName != null && _voiceProfileName!.isNotEmpty)
          ? <String>[_voiceProfileName!]
          : <String>[];
    });

    if (exists) {
      await _setInitialSetupCompleted(true);
      _hasPromptedInitialEnrollment = false;
    } else {
      await _setInitialSetupCompleted(false);
      _promptVoiceEnrollmentIfNeeded();
    }
  }

  Future<void> _setInitialSetupCompleted(bool value) async {
    final prefs = _prefs ?? await SharedPreferences.getInstance();
    _prefs = prefs;
    await prefs.setBool('initial_voice_setup_completed', value);

    if (!mounted) {
      return;
    }

    if (_initialVoiceSetupCompleted != value) {
      setState(() {
        _initialVoiceSetupCompleted = value;
      });
    }
  }

  Future<void> _promptVoiceEnrollmentIfNeeded() async {
    if (!mounted) {
      return;
    }
    if (_voiceProfileExists || _hasPromptedInitialEnrollment) {
      return;
    }

    _hasPromptedInitialEnrollment = true;

    await showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (context) => PopScope(
        canPop: false,
        onPopInvokedWithResult: (didPop, result) {},
        child: AlertDialog(
          title: const Text('Voice Enrollment Required'),
          content: const Text(
            'You need to enroll your voice profile on the wheelchair system '
            'before using voice commands. Tap below to start the enrollment process.',
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: const Text('Start Enrollment'),
            ),
          ],
        ),
      ),
    );

    if (!mounted) {
      return;
    }

    _hasPromptedInitialEnrollment = false;
    await _navigateToEnrollment(force: true);
  }

  Future<void> _showDisconnectConfirmation() async {
    if (!mounted) {
      return;
    }

    final provider = context.read<ConnectionProvider>();
    final shouldDisconnect = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Disconnect WebSocket'),
        content: const Text(
          'Disconnecting will stop commands until you reconnect. Do you want to continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Disconnect'),
          ),
        ],
      ),
    );

    if (shouldDisconnect == true) {
      await provider.disconnect(userInitiated: true);
      if (!mounted) return;
      setState(() {
        _statusMessage = 'Disconnected';
        _isProcessing = false;
        _isListening = false;
      });
      _showSnackBar('Disconnected from wheelchair control system');
    }
  }

  Future<void> _startRecording() async {
    if (!_wsService.isConnected) {
      _showSnackBar('Please connect to the wheelchair first');
      return;
    }

    if (!await _ensureWifiReady()) {
      return;
    }

    if (!await Permission.microphone.isGranted) {
      await _requestPermissions();
      if (!await Permission.microphone.isGranted) {
        return;
      }
    }

    try {
      _wsService.startRecording();
      final controller = StreamController<Uint8List>();
      _recordingDataSubscription = controller.stream.listen((data) {
        _wsService.sendAudioChunk(data);
      });

      await _audioRecorder!.startRecorder(
        toStream: controller.sink,
        codec: Codec.pcm16,
        sampleRate: 16000,
        numChannels: 1,
      );

      setState(() {
        _isListening = true;
        _statusMessage = 'Listening... Speak your command';
      });

      Future.delayed(const Duration(seconds: 5), () {
        if (_isListening) {
          _stopRecording();
        }
      });
    } catch (e) {
      print('Error starting recording: $e');
      _showSnackBar('Failed to start recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (!_isListening) {
      return;
    }

    try {
      if (_audioRecorder != null) {
        await _audioRecorder!.stopRecorder();
      }
      await _recordingDataSubscription?.cancel();
      _recordingDataSubscription = null;

      _wsService.stopRecording();

      setState(() {
        _isListening = false;
        _statusMessage = 'Processing...';
      });
    } catch (e) {
      print('Error stopping recording: $e');
      _showSnackBar('Error stopping recording: $e');
    }
  }

  void _emergencyStop() {
    EmergencyStopService.trigger();
    _showSnackBar('🚨 Emergency stop sent');
  }

  Future<void> _navigateToEnrollment({bool force = false}) async {
    if (!_wsService.isConnected) {
      _showSnackBar('Please connect to the wheelchair first');
      return;
    }

    final result = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (context) => VoiceEnrollmentPage(wsService: _wsService),
      ),
    );

    _hasPromptedInitialEnrollment = false;

    await _loadPreferences();
    _requestVoiceStatus();

    if (force && result != true && !_voiceProfileExists) {
      _promptVoiceEnrollmentIfNeeded();
    }
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final connection = context.watch<ConnectionProvider>();
    final isConnected = connection.isConnected;
    final isConnecting = connection.isConnecting;

    if (isConnected && !_lastConnectionState) {
      _lastConnectionState = true;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        setState(() {
          _statusMessage = 'Connected to wheelchair';
        });
        _requestVoiceStatus();
      });
    } else if (!isConnected && _lastConnectionState) {
      _lastConnectionState = false;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (!mounted) return;
        setState(() {
          _statusMessage = 'Not connected';
          _isListening = false;
          _isProcessing = false;
        });
      });
    }

    final connectionIcon = !_isWifiAvailable
        ? Icons.wifi_off
        : isConnected
        ? Icons.wifi
        : isConnecting
        ? Icons.wifi_tethering
        : Icons.wifi_off;

    final statusColor = !_isWifiAvailable
        ? Colors.red
        : isConnected
        ? Colors.green
        : isConnecting
        ? Colors.orange
        : Colors.red;

    final statusMessage = !_isWifiAvailable
        ? 'Wi-Fi disconnected'
        : _statusMessage.isNotEmpty
        ? _statusMessage
        : isConnected
        ? 'Connected to wheelchair'
        : isConnecting
        ? 'Connecting...'
        : 'Not connected';

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Voice Control'),
            const SizedBox(width: 8),
            Tooltip(
              message: !_isWifiAvailable
                  ? 'Wi-Fi disconnected'
                  : isConnected
                  ? 'Connected to wheelchair'
                  : isConnecting
                  ? 'Connecting to wheelchair'
                  : 'Not connected',
              child: Icon(
                Icons.circle,
                size: 12,
                color: !_isWifiAvailable
                    ? Colors.red
                    : isConnected
                    ? Colors.green
                    : isConnecting
                    ? Colors.orange
                    : Colors.red,
              ),
            ),
          ],
        ),
        backgroundColor: Theme.of(context).primaryColor,
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add),
            onPressed: _navigateToEnrollment,
            tooltip: 'Add Your Voice',
          ),
          IconButton(
            icon: const Icon(Icons.wifi),
            onPressed: () => ConnectionDialog.show(context),
            tooltip: 'Connection Settings',
          ),
          if (isConnected)
            PopupMenuButton<String>(
              icon: const Icon(Icons.more_vert),
              onSelected: (value) {
                if (value == 'disconnect') {
                  _showDisconnectConfirmation();
                }
              },
              itemBuilder: (context) => const [
                PopupMenuItem(value: 'disconnect', child: Text('Disconnect')),
              ],
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: statusColor.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Icon(connectionIcon, color: statusColor),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      statusMessage,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: statusColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _voiceProfileExists
                    ? Colors.blue.withValues(alpha: 0.08)
                    : Colors.red.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        _voiceProfileExists
                            ? Icons.verified_user
                            : Icons.person_off,
                        color: _voiceProfileExists ? Colors.blue : Colors.red,
                      ),
                      const SizedBox(width: 12),
                      Text(
                        'Voice Profile Status',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _voiceProfileExists
                        ? 'Active profile: '
                              '${_voiceProfileName ?? (_availableVoiceProfiles.isNotEmpty ? _availableVoiceProfiles.first : 'not set')}'
                        : 'No voice profile detected on the wheelchair. Please enroll your voice before using commands.',
                    style: const TextStyle(fontSize: 16),
                  ),
                  if (_availableVoiceProfiles.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Profiles on Pi: ${_availableVoiceProfiles.join(', ')}',
                      style: const TextStyle(color: Colors.black54),
                    ),
                  ],
                  const SizedBox(height: 12),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: OutlinedButton.icon(
                      onPressed: _navigateToEnrollment,
                      icon: const Icon(Icons.person_add_alt_1),
                      label: Text(
                        _voiceProfileExists
                            ? 'Update Voice Profile'
                            : 'Add Voice Profile',
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Center(
              child: Container(
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isListening
                      ? Colors.red.withValues(alpha: 0.2)
                      : _isProcessing
                      ? Colors.orange.withValues(alpha: 0.2)
                      : Colors.blue.withValues(alpha: 0.1),
                ),
                child: IconButton(
                  iconSize: 64,
                  icon: Icon(
                    _isListening
                        ? Icons.mic
                        : _isProcessing
                        ? Icons.hourglass_empty
                        : Icons.mic_none,
                    color: _isListening
                        ? Colors.red
                        : _isProcessing
                        ? Colors.orange
                        : Colors.blue,
                  ),
                  onPressed: (!isConnected || isConnecting || _isProcessing)
                      ? null
                      : _isListening
                      ? _stopRecording
                      : _startRecording,
                ),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              !_isWifiAvailable
                  ? 'Enable Wi-Fi to use voice control'
                  : !isConnected
                  ? (isConnecting
                        ? 'Attempting to connect to the wheelchair...'
                        : 'Tap the WebSocket button to reconnect')
                  : _isListening
                  ? 'Listening... Speak your command'
                  : _isProcessing
                  ? 'Processing your voice command...'
                  : 'Tap the microphone to start a command',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  const Text(
                    'Last Command:',
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _lastCommand,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  if (_lastCommandExecuted != null) ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color:
                            (_lastCommandExecuted == true
                                    ? Colors.green
                                    : Colors.red)
                                .withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Wrap(
                        crossAxisAlignment: WrapCrossAlignment.center,
                        spacing: 6,
                        children: [
                          Icon(
                            _lastCommandExecuted == true
                                ? Icons.check_circle
                                : Icons.block,
                            size: 18,
                            color: _lastCommandExecuted == true
                                ? Colors.green
                                : Colors.red,
                          ),
                          Text(
                            _commandFeedbackMessage.isNotEmpty
                                ? _commandFeedbackMessage
                                : (_lastCommandExecuted == true
                                      ? 'Command executed'
                                      : 'Command blocked'),
                            style: TextStyle(
                              color: _lastCommandExecuted == true
                                  ? Colors.green
                                  : Colors.red,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                  if (_lastTranscription.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Heard: $_lastTranscription',
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.black54,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                  if (_confidence > 0) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Confidence: ${(_confidence * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(fontSize: 14, color: Colors.grey),
                    ),
                  ],
                  if ((_lastSpeakerName != null &&
                          _lastSpeakerName!.isNotEmpty) ||
                      ((_lastSpeakerScore ?? 0) > 0)) ...[
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          _lastSpeakerVerified == true
                              ? Icons.verified
                              : Icons.person,
                          color: _lastSpeakerVerified == true
                              ? Colors.green
                              : Colors.grey,
                          size: 18,
                        ),
                        const SizedBox(width: 6),
                        Flexible(
                          child: Text(
                            'Speaker: '
                            '${(_lastSpeakerName != null && _lastSpeakerName!.isNotEmpty) ? _lastSpeakerName : 'Unknown'}'
                            '${_speakerConfidenceLabel()}',
                            style: const TextStyle(
                              fontSize: 14,
                              color: Colors.black87,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 24),
            Align(
              alignment: Alignment.center,
              child: TextButton.icon(
                onPressed: isConnected ? _requestVoiceStatus : null,
                icon: const Icon(Icons.refresh),
                label: const Text('Refresh voice status'),
              ),
            ),
            const SizedBox(height: 48),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.red,
        onPressed: _emergencyStop,
        child: const Icon(Icons.warning, color: Colors.white),
      ),
    );
  }
}
