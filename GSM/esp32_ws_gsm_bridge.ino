/*
 * ESP32 WebSocket SERVER -> GSM Emergency Bridge
 * -----------------------------------------------
 * The ESP32 runs a tiny WebSocket server on port 81.
 * The Raspberry Pi (or test script) connects directly to the ESP32
 * and sends an emergency_alert JSON. The ESP32 sends the SMS via SIM800L.
 *
 * Architecture:
 *   RPi / test_script  ---WS--->  ESP32 (server :81)  ---UART--->  SIM800L  ---SMS--->  Contact
 *
 * Hardware:
 *   GPIO17 (ESP32 TX)  ->  SIM800L RX
 *   GPIO16 (ESP32 RX)  <-  SIM800L TX
 *   Common GND
 *   SIM800L: dedicated 3.7-4.2V / 2A supply (NOT from ESP32 3.3V!)
 *
 * Libraries (Arduino Library Manager):
 *   - ArduinoJson  (Benoit Blanchon)
 *   - WebSockets   (Markus Sattler)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// Wi-Fi credentials
// ---------------------------------------------------------------------------
static const char *WIFI_SSID     = "Pandey";
static const char *WIFI_PASSWORD = "RaMoSa171116*";

// ---------------------------------------------------------------------------
// WebSocket server port (ESP32 listens here)
// ---------------------------------------------------------------------------
static const uint16_t WS_PORT = 81;
static WebSocketsServer wsServer(WS_PORT);

// ---------------------------------------------------------------------------
// Emergency contact (can be overridden per-message via JSON)
// ---------------------------------------------------------------------------
static const char *DEFAULT_RECEIVER = "+919326632001";
static const char *DEFAULT_SENDER   = "+910000000000";  // your SIM (informational)

// ---------------------------------------------------------------------------
// SIM800L UART  (GPIO17=TX -> SIM RX, GPIO16=RX <- SIM TX)
// ---------------------------------------------------------------------------
static const int           GSM_TX_PIN = 17;
static const int           GSM_RX_PIN = 16;
static const unsigned long GSM_BAUD   = 9600;
static HardwareSerial      gsmSerial(2);  // UART2

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
static void   onWsEvent(uint8_t clientNum, WStype_t type, uint8_t *payload, size_t length);
static void   handleEmergencyJson(uint8_t clientNum, const String &raw);
static String buildSmsMessage(const JsonVariantConst &doc);
static bool   sendATCommand(const String &cmd, const char *expect, unsigned long timeoutMs = 5000);
static bool   ensureGsmReady();
static bool   sendSms(const String &receiver, const String &message);

// ===========================================================================
// Arduino setup
// ===========================================================================
void setup() {
    Serial.begin(115200);
    delay(800);
    Serial.println("\n=== ESP32 WS-SERVER -> GSM Bridge ===");

    // ---- Wi-Fi ----
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("Connecting to Wi-Fi '%s'", WIFI_SSID);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t0 > 20000) {
            Serial.println("\n[FATAL] Wi-Fi timed out. Restarting...");
            delay(500);
            ESP.restart();
        }
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nWi-Fi OK  IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("WebSocket server listening on ws://%s:%u\n",
                  WiFi.localIP().toString().c_str(), WS_PORT);

    // ---- GSM ----
    gsmSerial.begin(GSM_BAUD, SERIAL_8N1, GSM_RX_PIN, GSM_TX_PIN);
    delay(500);
    Serial.println("Initialising SIM800L...");
    ensureGsmReady();

    // ---- WebSocket server ----
    wsServer.begin();
    wsServer.onEvent(onWsEvent);

    Serial.println("\nReady! Waiting for emergency_alert from RPi or test script...");
}

// ===========================================================================
// Arduino loop
// ===========================================================================
void loop() {
    wsServer.loop();

    // Reconnect Wi-Fi if dropped
    static unsigned long lastWifiCheck = 0;
    if (millis() - lastWifiCheck > 10000) {
        lastWifiCheck = millis();
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WiFi] Lost – reconnecting...");
            WiFi.reconnect();
        }
    }
}

// ===========================================================================
// WebSocket event handler
// ===========================================================================
static void onWsEvent(uint8_t clientNum, WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED: {
            IPAddress ip = wsServer.remoteIP(clientNum);
            Serial.printf("[WS] Client #%u connected from %s\n", clientNum, ip.toString().c_str());
            // Send welcome/ready status
            String welcome = "{\"type\":\"esp32_ready\",\"client\":\"esp32_gsm_bridge\",\"status\":\"online\"}";
            wsServer.sendTXT(clientNum, welcome);
            break;
        }

        case WStype_DISCONNECTED:
            Serial.printf("[WS] Client #%u disconnected\n", clientNum);
            break;

        case WStype_TEXT: {
            String msg = String((char *)payload);
            Serial.printf("[WS] Message from client #%u (%u bytes)\n", clientNum, length);

            // Peek at message type
            StaticJsonDocument<64> peek;
            if (deserializeJson(peek, msg) == DeserializationError::Ok) {
                const char *msgType = peek["type"] | "";
                if (strcmp(msgType, "emergency_alert") == 0) {
                    Serial.println("[WS] Emergency alert received!");
                    handleEmergencyJson(clientNum, msg);
                } else if (strcmp(msgType, "ping") == 0) {
                    wsServer.sendTXT(clientNum, "{\"type\":\"pong\"}");
                } else {
                    Serial.printf("[WS] Ignored type: %s\n", msgType);
                }
            } else {
                Serial.println("[WS] Invalid JSON received");
            }
            break;
        }

        case WStype_ERROR:
            Serial.printf("[WS] Error on client #%u\n", clientNum);
            break;

        default:
            break;
    }
}

// ===========================================================================
// Handle emergency_alert JSON
// ===========================================================================
static void handleEmergencyJson(uint8_t clientNum, const String &raw) {
    StaticJsonDocument<1024> doc;
    DeserializationError err = deserializeJson(doc, raw);
    if (err) {
        Serial.printf("[ERR] JSON parse failed: %s\n", err.f_str());
        wsServer.sendTXT(clientNum,
            "{\"type\":\"emergency_ack\",\"status\":\"error\",\"reason\":\"bad_json\"}");
        return;
    }

    const char *receiver = doc["receiver_number"] | DEFAULT_RECEIVER;
    String sms = buildSmsMessage(doc);

    Serial.println("--- SMS to be sent ---");
    Serial.println(sms);
    Serial.println("----------------------");

    // Ack immediately so the sender knows we got it
    wsServer.sendTXT(clientNum,
        "{\"type\":\"emergency_ack\",\"status\":\"sending\",\"client\":\"esp32_gsm_bridge\"}");

    if (!ensureGsmReady()) {
        Serial.println("[ERR] GSM not ready");
        wsServer.sendTXT(clientNum,
            "{\"type\":\"emergency_ack\",\"status\":\"gsm_error\",\"reason\":\"modem_not_ready\"}");
        return;
    }

    bool ok = sendSms(receiver, sms);
    if (ok) {
        Serial.println("[GSM] SMS sent!");
        wsServer.sendTXT(clientNum,
            "{\"type\":\"emergency_ack\",\"status\":\"sms_sent\",\"client\":\"esp32_gsm_bridge\"}");
    } else {
        Serial.println("[ERR] SMS failed");
        wsServer.sendTXT(clientNum,
            "{\"type\":\"emergency_ack\",\"status\":\"sms_failed\",\"client\":\"esp32_gsm_bridge\"}");
    }
}

// ===========================================================================
// Build SMS text from JSON payload
// ===========================================================================
static String buildSmsMessage(const JsonVariantConst &doc) {
    const char *source = doc["source"] | "system";
    const char *reason = doc["reason"] | "emergency";
    float temp     = doc["temperature_c"]  | -1.0f;
    float humidity = doc["humidity_pct"]   | -1.0f;
    float spo2     = doc["spo2_percent"]   | -1.0f;
    float hr       = doc["heart_rate_bpm"] | -1.0f;
    float lat      = doc["latitude"]       |  0.0f;
    float lng      = doc["longitude"]      |  0.0f;
    float alt      = doc["altitude_m"]     |  0.0f;

    char msg[320];
    snprintf(msg, sizeof(msg),
        "EMERGENCY! Wheelchair alert.\n"
        "Reason: %s (%s)\n"
        "Vitals: Temp %.1fC Hum %.0f%% HR %.0fbpm SpO2 %.0f%%\n"
        "GPS: %.5f,%.5f Alt %.0fm\n"
        "maps.google.com/?q=%.5f,%.5f",
        reason, source,
        temp, humidity, hr, spo2,
        lat, lng, alt,
        lat, lng
    );
    return String(msg);
}

// ===========================================================================
// AT command helpers
// ===========================================================================
static bool sendATCommand(const String &command, const char *expect, unsigned long timeoutMs) {
    while (gsmSerial.available()) gsmSerial.read();  // flush

    if (command.length() > 0) {
        gsmSerial.println(command);
        Serial.printf("[AT] >>> %s\n", command.c_str());
    }

    unsigned long start = millis();
    String line;
    while (millis() - start < timeoutMs) {
        while (gsmSerial.available()) {
            char c = gsmSerial.read();
            if (c == '\r') continue;
            if (c == '\n') {
                if (line.length() > 0) {
                    Serial.printf("[AT] <<< %s\n", line.c_str());
                    if (line.indexOf(expect) >= 0) return true;
                    line = "";
                }
            } else {
                line += c;
            }
        }
        delay(10);
    }
    Serial.println("[AT] Timeout");
    return false;
}

static bool ensureGsmReady() {
    for (int i = 0; i < 5; i++) {
        if (sendATCommand("AT", "OK", 2000)) {
            sendATCommand("AT+CMGF=1", "OK");
            sendATCommand("AT+CSCS=\"GSM\"", "OK");
            Serial.println("[GSM] Modem ready");
            return true;
        }
        delay(600);
    }
    Serial.println("[GSM] Modem not responding");
    return false;
}

static bool sendSms(const String &receiver, const String &message) {
    Serial.printf("[GSM] Sending to %s...\n", receiver.c_str());
    String cmd = String("AT+CMGS=\"") + receiver + "\"";
    if (!sendATCommand(cmd, ">", 8000)) {
        Serial.println("[GSM] Failed to enter message mode");
        return false;
    }
    gsmSerial.print(message);
    gsmSerial.write(26);  // Ctrl+Z
    if (!sendATCommand("", "OK", 60000)) {
        Serial.println("[GSM] Send timed out");
        return false;
    }
    return true;
}
