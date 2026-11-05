/*
 * Real-Time AIoT AQI Telemetry Node Firmware (ESP32-WROOM-32)
 * ===========================================================
 * Hardware allocation (per `Tìm hiểu thêm.pdf` Section I.2.1 - Conflict-free buses):
 *   1. Bosch BME280 (Temperature, Relative Humidity, Pressure)
 *      - Bus : Dedicated I2C (Wire)
 *      - Pins: GPIO21 (SDA), GPIO22 (SCL) — 4.7 kOhm pull-ups, I2C addr 0x76 or 0x77
 *   2. Plantower PMS7003 (PM1.0, PM2.5, PM10 laser scattering)
 *      - Bus : Dedicated Hardware UART2 (Serial2, 9600 8N1, active mode 32-byte frame)
 *      - Pins: GPIO16 (RX2 <- PMS7003 TX), GPIO17 (TX2 -> PMS7003 RX)
 *      - Isolated from UART0 (USB debug/programming) to prevent frame corruption.
 *
 * Libraries required in Arduino IDE / PlatformIO:
 *   - Adafruit BME280 Library + Adafruit Unified Sensor
 *   - ArduinoJson (or built-in snprintf JSON formatter below)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <time.h>

// ── 1. Network & Server Configuration ─────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Point to the PC/VPS running `python src/serving/server.py` (Port 5000)
const char* SERVER_URL    = "http://192.168.1.100:5000/api/telemetry";

// NTP Timezone: Hanoi (UTC+7)
const char* NTP_SERVER    = "pool.ntp.org";
const long  GMT_OFFSET_S  = 7 * 3600;
const int   DST_OFFSET_S  = 0;

// ── 2. Hardware Pin Definitions (Section I.2.1) ───────────────────────────────
#define BME_SDA_PIN  21
#define BME_SCL_PIN  22
#define PMS_RX2_PIN  16
#define PMS_TX2_PIN  17

// ── 3. Sampling & EMA Filtering Constants ─────────────────────────────────────
const unsigned long SAMPLE_INTERVAL_MS  = 5000UL;    // Read sensors every 5 s
const unsigned long PUBLISH_INTERVAL_MS = 60000UL;   // POST telemetry every 60 s
const float         EMA_ALPHA           = 0.25f;     // Exponential moving average smoothing

Adafruit_BME280 bme;
bool bmeAvailable = false;

struct SensorState {
  bool  initialized = false;
  float pm25        = 0.0f;
  float pm10        = 0.0f;
  float temperature = 25.0f;
  float humidity    = 75.0f;
  float pressure    = 1010.0f;
} state;

unsigned long lastSampleMs  = 0;
unsigned long lastPublishMs = 0;

// ── 4. PMS7003 Active-Mode 32-Byte Frame Parser ───────────────────────────────
bool readPMS7003(float &pm25_out, float &pm10_out) {
  while (Serial2.available() >= 32) {
    if (Serial2.peek() != 0x42) {
      Serial2.read();
      continue;
    }
    uint8_t buf[32];
    Serial2.readBytes(buf, 32);

    if (buf[1] != 0x4D) {
      continue;
    }

    uint16_t frameLen = (uint16_t(buf[2]) << 8) | buf[3];
    if (frameLen != 28) {
      continue;
    }

    uint16_t checksum = 0;
    for (int i = 0; i < 30; i++) {
      checksum += buf[i];
    }
    uint16_t expected = (uint16_t(buf[30]) << 8) | buf[31];
    if (checksum != expected) {
      continue;
    }

    // Atmospheric environment concentration (ug/m3): bytes 10..15
    uint16_t pm25_atm = (uint16_t(buf[12]) << 8) | buf[13];
    uint16_t pm10_atm = (uint16_t(buf[14]) << 8) | buf[15];

    if (pm25_atm <= 1000 && pm10_atm <= 1500) {
      pm25_out = float(pm25_atm);
      pm10_out = float(pm10_atm);
      return true;
    }
  }
  return false;
}

// ── 5. ISO-8601 Local Timestamp Helper ────────────────────────────────────────
String getIsoTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 200)) {
    return "";
  }
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+07:00", &timeinfo);
  return String(buf);
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Connection timed out; will retry.");
  }
}

void setup() {
  // UART0 for debug logging only
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== ESP32 AIoT AQI Telemetry Node (PMS7003 + BME280) ===");

  // Dedicated UART2 for PMS7003 (9600 8N1 on GPIO16/17)
  Serial2.begin(9600, SERIAL_8N1, PMS_RX2_PIN, PMS_TX2_PIN);

  // Dedicated I2C bus for BME280 on GPIO21/22
  Wire.begin(BME_SDA_PIN, BME_SCL_PIN);
  bmeAvailable = bme.begin(0x76, &Wire) || bme.begin(0x77, &Wire);
  if (bmeAvailable) {
    // Configure indoor/outdoor weather monitoring oversampling & IIR filter
    bme.setSampling(
      Adafruit_BME280::MODE_NORMAL,
      Adafruit_BME280::SAMPLING_X2,   // Temperature
      Adafruit_BME280::SAMPLING_X16,  // Pressure
      Adafruit_BME280::SAMPLING_X1,   // Humidity
      Adafruit_BME280::FILTER_X16,
      Adafruit_BME280::STANDBY_MS_500
    );
    Serial.println("[BME280] Initialized on I2C (GPIO21/22).");
  } else {
    Serial.println("[BME280] WARNING: Sensor not detected on 0x76/0x77!");
  }

  connectWiFi();
  configTime(GMT_OFFSET_S, DST_OFFSET_S, NTP_SERVER);
}

void loop() {
  unsigned long now = millis();

  // 1. Sample PMS7003 + BME280 every SAMPLE_INTERVAL_MS
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;

    float rawPm25 = state.pm25;
    float rawPm10 = state.pm10;
    bool gotPms = readPMS7003(rawPm25, rawPm10);

    float rawTemp = state.temperature;
    float rawRh   = state.humidity;
    float rawPres = state.pressure;
    if (bmeAvailable) {
      float t = bme.readTemperature();
      float h = bme.readHumidity();
      float p = bme.readPressure() / 100.0f; // Pa -> hPa
      if (!isnan(t) && t > -20.0f && t < 70.0f) rawTemp = t;
      if (!isnan(h) && h >= 0.0f && h <= 100.0f) rawRh = h;
      if (!isnan(p) && p >= 850.0f && p <= 1100.0f) rawPres = p;
    }

    if (!state.initialized && (gotPms || bmeAvailable)) {
      state.pm25        = rawPm25;
      state.pm10        = rawPm10;
      state.temperature = rawTemp;
      state.humidity    = rawRh;
      state.pressure    = rawPres;
      state.initialized = true;
    } else if (state.initialized) {
      if (gotPms) {
        state.pm25 = EMA_ALPHA * rawPm25 + (1.0f - EMA_ALPHA) * state.pm25;
        state.pm10 = EMA_ALPHA * rawPm10 + (1.0f - EMA_ALPHA) * state.pm10;
      }
      state.temperature = EMA_ALPHA * rawTemp + (1.0f - EMA_ALPHA) * state.temperature;
      state.humidity    = EMA_ALPHA * rawRh   + (1.0f - EMA_ALPHA) * state.humidity;
      state.pressure    = EMA_ALPHA * rawPres + (1.0f - EMA_ALPHA) * state.pressure;
    }

    Serial.printf(
      "[Sample] PM2.5=%.1f ug/m3 | PM10=%.1f ug/m3 | T=%.1f C | RH=%.1f %% | P=%.1f hPa\n",
      state.pm25, state.pm10, state.temperature, state.humidity, state.pressure
    );
  }

  // 2. Publish filtered telemetry to Backend every PUBLISH_INTERVAL_MS
  if (state.initialized && (now - lastPublishMs >= PUBLISH_INTERVAL_MS)) {
    lastPublishMs = now;
    connectWiFi();

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(SERVER_URL);
      http.addHeader("Content-Type", "application/json");

      String ts = getIsoTimestamp();
      char payload[320];
      if (ts.length() > 0) {
        snprintf(
          payload, sizeof(payload),
          "{\"timestamp\":\"%s\",\"pm25\":%.1f,\"pm10\":%.1f,\"temperature\":%.1f,\"relativeHumidity\":%.1f,\"pressure\":%.1f}",
          ts.c_str(), state.pm25, state.pm10, state.temperature, state.humidity, state.pressure
        );
      } else {
        snprintf(
          payload, sizeof(payload),
          "{\"pm25\":%.1f,\"pm10\":%.1f,\"temperature\":%.1f,\"relativeHumidity\":%.1f,\"pressure\":%.1f}",
          state.pm25, state.pm10, state.temperature, state.humidity, state.pressure
        );
      }

      int code = http.POST((uint8_t*)payload, strlen(payload));
      if (code > 0) {
        String resp = http.getString();
        Serial.printf("[HTTP %d] %s\n", code, resp.c_str());
      } else {
        Serial.printf("[HTTP Error] %s\n", http.errorToString(code).c_str());
      }
      http.end();
    }
  }
}

