/*
 * ESP HTTP -> GSM Bridge
 * ----------------------
 * Listens for HTTP POST requests at /emergency, expects JSON payloads like:
 * {
 *   "message": "Emergency stop triggered...",
 *   "sender_number": "+910000000000",
 *   "receiver_number": "+910000000001"
 * }
 *
 * When received, the ESP sends the message to the attached GSM modem via
 * AT commands so that the modem can deliver the SMS to the configured number.
 *
 * Tested with ESP32 + SIM800L. Adjust the pin mappings and serial baud rate
 * to match your hardware.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// Wi-Fi configuration (must match the network used by the Raspberry Pi)
// ---------------------------------------------------------------------------
static const char *WIFI_SSID = "Pandey";
static const char *WIFI_PASSWORD = "RaMoSa171116*";

// ---------------------------------------------------------------------------
// GSM UART configuration
// ---------------------------------------------------------------------------
static const int GSM_TX_PIN = 17;      // ESP TX -> GSM RX
static const int GSM_RX_PIN = 16;      // ESP RX <- GSM TX
static const unsigned long GSM_BAUD = 9600;
static HardwareSerial gsmSerial(1);

// ---------------------------------------------------------------------------
// HTTP server configuration
// ---------------------------------------------------------------------------
static const uint16_t HTTP_PORT = 80;
static WebServer server(HTTP_PORT);

// ---------------------------------------------------------------------------
// Message template (fallback when payload omits message field)
// ---------------------------------------------------------------------------
static const char *DEFAULT_TEMPLATE =
    "Emergency stop triggered by {source} due to {reason}. "
    "Location: {latitude}, {longitude}. Heart rate: {heart_rate_bpm} bpm. "
    "SpO2: {spo2_percent}%.";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static bool sendATCommand(const String &command, const char *expect, unsigned long timeoutMs = 5000);
static bool ensureGsmReady();
static bool sendSms(const String &sender, const String &receiver, const String &message);
static String substituteTemplate(const String &tmpl, const JsonVariantConst &data);

// ---------------------------------------------------------------------------
// HTTP Handlers
// ---------------------------------------------------------------------------
static void handleRoot() {
  server.send(200, "text/plain", "ESP emergency GSM bridge online");
}

static void handleEmergency() {
  if (server.method() != HTTP_POST) {
    server.send(405, "text/plain", "Only POST supported");
    return;
  }

  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "Missing request body");
    return;
  }

  const String &body = server.arg("plain");
  StaticJsonDocument<2048> doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    server.send(400, "text/plain", String("Invalid JSON: ") + err.f_str());
    return;
  }

  const String sender = doc["sender_number"].as<String>();
  const String receiver = doc["receiver_number"].as<String>();
  String message = doc["message"].as<String>();
  if (message.isEmpty()) {
    message = substituteTemplate(DEFAULT_TEMPLATE, doc);
  }

  if (sender.isEmpty() || receiver.isEmpty() || message.isEmpty()) {
    server.send(400, "text/plain", "Payload missing sender_number, receiver_number, or message");
    return;
  }

  if (!ensureGsmReady()) {
    server.send(500, "text/plain", "GSM module not ready");
    return;
  }

  bool ok = sendSms(sender, receiver, message);
  if (!ok) {
    server.send(500, "text/plain", "GSM send failed");
    return;
  }

  StaticJsonDocument<128> resp;
  resp["status"] = "ok";
  resp["message_length"] = message.length();
  String response;
  serializeJson(resp, response);
  server.send(200, "application/json", response);
}

static void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// ---------------------------------------------------------------------------
// Arduino lifecycle
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("ESP HTTP->GSM bridge starting...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to %s", WIFI_SSID);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++attempts % 20 == 0) {
      Serial.println();
      Serial.printf("Still connecting (%d)\n", attempts);
    }
  }
  Serial.println();
  Serial.print("Wi-Fi connected. IP address: ");
  Serial.println(WiFi.localIP());

  gsmSerial.begin(GSM_BAUD, SERIAL_8N1, GSM_RX_PIN, GSM_TX_PIN);
  delay(300);
  ensureGsmReady();

  server.on("/", handleRoot);
  server.on("/emergency", handleEmergency);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.printf("HTTP server ready on port %u\n", HTTP_PORT);
}

void loop() {
  server.handleClient();
}

// ---------------------------------------------------------------------------
// Helper implementations
// ---------------------------------------------------------------------------
static bool sendATCommand(const String &command, const char *expect, unsigned long timeoutMs) {
  while (gsmSerial.available()) {
    gsmSerial.read();
  }
  gsmSerial.println(command);
  unsigned long start = millis();
  String line;
  while (millis() - start < timeoutMs) {
    while (gsmSerial.available()) {
      char c = gsmSerial.read();
      if (c == '\r') {
        continue;
      }
      if (c == '\n') {
        if (line.length() > 0) {
          Serial.println(String("[GSM] ") + line);
          if (line.indexOf(expect) >= 0) {
            return true;
          }
          line = "";
        }
      } else {
        line += c;
      }
    }
    delay(10);
  }
  Serial.println("[GSM] Timeout waiting for response");
  return false;
}

static bool ensureGsmReady() {
  Serial.println("Ensuring GSM modem is ready...");
  for (int i = 0; i < 5; ++i) {
    if (sendATCommand("AT", "OK")) {
      // Enable SMS text mode
      sendATCommand("AT+CMGF=1", "OK");
      // Optional: set caller ID text encoding to UTF-8 (may not be supported)
      sendATCommand("AT+CSCS=\"GSM\"", "OK");
      return true;
    }
    delay(500);
  }
  Serial.println("GSM modem did not respond to AT");
  return false;
}

static bool sendSms(const String &, const String &receiver, const String &message) {
  Serial.printf("Sending SMS to %s\n", receiver.c_str());

  if (!sendATCommand(String("AT+CMGS=\"") + receiver + "\"", ">")) {
    Serial.println("Failed to enter message mode");
    return false;
  }

  gsmSerial.print(message);
  gsmSerial.write(26);  // Ctrl+Z to send

  if (!sendATCommand("", "OK", 15000)) {
    Serial.println("SMS send failed or timed out");
    return false;
  }

  Serial.println("SMS sent successfully");
  return true;
}

static String substituteTemplate(const String &tmpl, const JsonVariantConst &data) {
  String result = tmpl;
  struct Replacement {
    const char *key;
  } replacements[] = {
      {"source"},          {"reason"},           {"latitude"},
      {"longitude"},       {"altitude_m"},       {"heart_rate_bpm"},
      {"spo2_percent"},    {"gps_timestamp"},    {"health_timestamp"},
  };

  for (const auto &item : replacements) {
    String placeholder = String("{") + item.key + "}";
    if (result.indexOf(placeholder) >= 0) {
      String value = data[item.key].as<String>();
      if (value.isEmpty()) {
        value = "unknown";
      }
      result.replace(placeholder, value);
    }
  }

  return result;
}
