# ♿ SmartNav - An IoT-Enabled AI Smart Wheelchair

<p align="center">
  <img src="Poster/poster.png" alt="Project Poster Banner" width="600px">
</p>

> ⚠️ **Note:** This project is currently under active development.  
> Hardware schematics and remaining sensor integrations are in progress.

---

## 🎯 About The Project

The **AI-Powered Smart Wheelchair (SmartNav)** is an intelligent assistive mobility system designed to enhance **safety, independence, and healthcare accessibility** for individuals with physical impairments.

Unlike traditional wheelchairs, SmartNav integrates:

- Voice-controlled navigation  
- AI-powered authentication  
- Health monitoring  
- Emergency communication  

Our vision is to transform a wheelchair into an **intelligent mobility companion** that empowers users while ensuring safety.

---

## 🚀 Vision & Objectives

We aim to build a **cost-effective, intelligent wheelchair system** that minimizes caregiver dependency and maximizes user independence.

### ✅ Key Objectives

- **Enhance Safety** → Intelligent automated collision avoidance  
- **Promote Independence** → Hands-free AI voice navigation  
- **Enable Health Monitoring** → Real-time vitals tracking  
- **Emergency Communication** → GSM-based SMS alerts  
- **Reliable Multimodal Control** → Voice + App + Joystick support  

---

## 📌 Overview

SmartNav operates in two primary modes:

1. **Online LLM Mode**  
   Authenticated users can ask general queries to a lightweight LLM.

2. **Command Mode**  
   Authenticated users control wheelchair movement using voice commands.

---

## ✨ Core Features

| Feature | Description |
|----------|------------|
| 🎤 **Multilingual Speech-to-Text** | Supports Hindi, English, and Marathi |
| 🔐 **Voice Authentication** | Secure speaker verification before control access |
| 🛑 **Collision Avoidance** | Ultrasonic-based obstacle detection |
| ❤️ **Health Monitoring** | Pulse + Temperature monitoring (integration in progress) |
| 📍 **Location Tracking** | GPS-based position detection (integration pending) |
| 📧 **Emergency SMS Alerts** | GSM-based alert system (integration pending) |
| 🕹️ **Multimodal Controls** | Voice, mobile app, and joystick |

---

## 🛠️ Technology Stack

### 🖥 Controllers
- **Primary Controller** → Raspberry Pi 4B  
- **Real-Time Control** → Raspberry Pi (same unit handles control tasks)  
- **Peripheral Communication** → ESP module (GSM communication only)

> Note: ESP is used strictly as a peripheral device to interface with the GSM module for sending alerts.

---

### 💻 Software
- Python (High-level AI & system control)
- C++ (Low-level hardware interfacing where required)

---

### 🔍 Sensors (Integration Pending)

- MAX30100 – Pulse sensor  
- DHT11 – Temperature sensor  
- Neo-6M – GPS module  
- Ultrasonic Sensors – Obstacle detection  

---

### 📡 Communication
- SIM800L GSM module (SMS alerts)

---

## 🌍 Multilingual Command Support

SmartNav currently supports:

- English  
- Hindi  
- Marathi  

### Example Commands

**Forward**
- English: "forward", "go ahead"
- Hindi: "आगे बढ़ो"
- Marathi: "पुढे जा"

**Backward**
- English: "back"
- Hindi: "पीछे जाओ"
- Marathi: "मागे जा"

**Left / Right / Stop**
- Fully supported in all three languages.

The system uses fuzzy matching and phonetic correction for improved recognition accuracy.

---

## 🧠 AI Models Used

### 1️⃣ Speech Recognition
- Whisper Tiny (optimized for Raspberry Pi)
- On-device processing

### 2️⃣ Voice Authentication
- Resemblyzer-based speaker embeddings
- Similarity threshold validation

### 3️⃣ Text-to-Speech
- Piper TTS (offline)

### 4️⃣ Conversational Mode
- Llama 3.2-3B-Instruct (Optional LLM mode)

---

## 🔄 System Architecture

The wheelchair control system is built on a modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│                   Voice Input Pipeline                  │
└───────────────┬───────────────────────────┬─────────────┘
                │                           │
┌───────────────▼───────┐       ┌───────────▼───────────┐
│  Voice Authentication │       │ Command Recognition   │
│  - Speaker verification│       │ - Speech-to-text      │
│  - Profile matching    │       │ - Language detection  │
└───────────────┬───────┘       └───────────┬───────────┘
                │                           │
                │           ┌───────────────▼───────────┐
                │           │ Command Processing        │
                │           │ - Intent classification   │
                │           │ - Fuzzy matching         │
                │           └───────────────┬───────────┘
                │                           │
┌───────────────▼───────┐       ┌───────────▼───────────┐
│  LLM Query Mode        │       │ Motion Control        │
│  - Question answering  │       │ - Direction commands  │
│  - Information access  │       │ - Speed control       │
└───────────────┬───────┘       └───────────┬───────────┘
                │                           │
┌───────────────▼───────────────────────────▼───────────┐
│                     Response System                   │
│              (Text-to-Speech Feedback)                │
└─────────────────────────────────────────────────────────┘
```

### 🔊 Voice Authentication System

The voice authentication system uses a combination of:
- **Voice embedding generation** - Creates a unique 256-dimensional voice signature
- **Similarity scoring** - Compares live voice to stored profiles
- **Adaptive thresholding** - Adjusts recognition strictness based on environment

---

## 🔗 Bluetooth Control Pipeline

The Raspberry Pi can now accept wheelchair commands from both the WebSocket
voice channel and a dedicated Bluetooth RFCOMM transport. The mobile app keeps a
continuous Bluetooth link for manual buttons and joystick streaming, while
WebSocket remains available for voice authentication and LLM features.

### Raspberry Pi Setup

1. Install system dependencies (run on the Pi):
   ```bash
   sudo apt update
   sudo apt install -y bluetooth bluez python3-bluez
   ```
2. Install Python requirements (inside the project venv if used):
   ```bash
   pip install -r requirements_rpi.txt
   ```
3. Enable the RFCOMM profile and agent:
   ```bash
   sudo systemctl enable bluetooth
   sudo systemctl start bluetooth
   sudo bluetoothctl agent NoInputNoOutput
   sudo bluetoothctl default-agent
   ```
4. Pair the Android device once (from either side):
   ```bash
   sudo bluetoothctl pair AA:BB:CC:DD:EE:FF
   sudo bluetoothctl trust AA:BB:CC:DD:EE:FF
   ```
5. Launch the wheelchair server:
   ```bash
   python3 rpi_websocket_server.py
   ```
   The server starts both the WebSocket listener and the Bluetooth dispatcher
   (`bluetooth_controller.py`). Manual and joystick commands are funneled through
   a single-threaded command queue, so voice and Bluetooth never fight over GPIO.

### Android App Setup

1. From `smart_wheelchair_app/`, fetch dependencies:
   ```bash
   flutter pub get
   ```
2. Build or run the app:
   ```bash
   flutter run
   ```
3. Pair the phone with the Raspberry Pi in Android system settings. The app
   remembers the last paired device and reconnects automatically on launch.
4. Open **Settings ▸ Bluetooth Control** in the app to confirm connection status
   or manually switch devices. Manual buttons and joystick now send commands over
   Bluetooth at ~20 Hz, while voice enrollment and command recognition continue
   to use the WebSocket pipeline when Wi-Fi is available.

### Safety Guarantees

- Every control source ultimately flows through `CommandDispatcher`, enforcing
  sticky manual commands, 200 ms joystick timeouts, and a shared emergency stop.
- Bluetooth disconnects trigger an automatic stop.
- All acknowledgements include the originating channel (`source`), making log
  analysis straightforward.

### 📡 Command Recognition Pipeline

1. **Audio Capture**: High-quality audio recording with noise reduction
2. **Preprocessing**: Normalization and filtering for better recognition
3. **Transcription**: Speech-to-text via Whisper tiny model
4. **Language Detection**: Automatic identification of spoken language
5. **Command Matching**: Fuzzy matching to predefined command set
6. **Feedback**: Language-specific audio response

---

## 📈 Project Status

### ✅ Software
- 100% Implemented  
- Voice authentication  
- Multilingual command recognition  
- Bluetooth control pipeline  
- LLM query mode  

### 🚧 Hardware Integration Pending
- MAX30100 sensor data fetching  
- DHT11 sensor data fetching  
- Neo-6M GPS data integration  
- GSM alert triggering system  

---

## 🎯 Current Development Focus

1. Sensor data acquisition (MAX30100, DHT11, Neo-6M)
2. GSM-based emergency alert integration
3. Hardware testing and validation
4. Real-world safety testing

---

## 📽️ Demo

Demo video coming soon.

---

## 📞 Contact

For queries, collaboration, or technical discussion:

- samaypandey2022@kccemsr.edu.in  
- nishalpoojary2022@kccemsr.edu.in  
- aaryawalve2022@kccemsr.edu.in  

---

## 📃 License

This project is licensed under the MIT License.
