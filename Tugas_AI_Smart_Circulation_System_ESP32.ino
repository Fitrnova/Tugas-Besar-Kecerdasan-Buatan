#include <DHT.h>
#include <DHT_U.h>
#include <ESP32Servo.h>

// Definisi pin
#define DHTPIN 4        // Pin digital untuk DHT11
#define DHTTYPE DHT11   // Tipe sensor DHT (DHT11 atau DHT22)
#define MQ7_PIN 34      // Pin analog untuk sensor MQ7 (GPIO34 adalah ADC1_CH6)
#define LDR_PIN 35      // Pin analog untuk sensor LDR (GPIO35 adalah ADC1_CH7)
#define PIR_PIN 13      // Pin digital untuk sensor PIR
#define SERVO_PIN 14    // Pin digital untuk servo
#define RELAY_PIN 2     // Pin digital untuk relay

// Inisialisasi sensor
DHT dht(DHTPIN, DHTTYPE);
Servo myServo;

void setup() {
  Serial.begin(115200); // Kecepatan baud komunikasi serial
  dht.begin();

  pinMode(MQ7_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);

  myServo.attach(SERVO_PIN); // Melampirkan servo ke pin
  myServo.write(90); // Set posisi awal servo ke 90 derajat

  Serial.println("ESP32 Ready. Mengirim data sensor...");
}

void loop() {
  // Pembacaan sensor DHT11
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // Pastikan pembacaan berhasil
  if (isnan(h) || isnan(t)) {
    Serial.println("Gagal membaca dari sensor DHT!");
    return;
  }

  // Pembacaan sensor MQ7 (nilai analog)
  int mq7_analog_value = analogRead(MQ7_PIN);
  // Konversi nilai analog ke PPM CO2 memerlukan kalibrasi dan kurva sensitivitas sensor MQ7.
  // Untuk contoh ini, kita akan mengirim nilai analog mentah atau perkiraan sederhana.
  // Anda perlu mencari kurva sensitivitas MQ7 untuk konversi yang akurat.
  // Contoh perkiraan (bukan nilai akurat):
  float co2_ppm = map(mq7_analog_value, 0, 4095, 400, 5000); // Asumsi 0-4095 ke 400-5000 ppm

  // Pembacaan sensor LDR (nilai analog)
  int ldr_analog_value = analogRead(LDR_PIN);
  // Konversi nilai analog LDR ke Lux juga memerlukan kalibrasi.
  // Contoh perkiraan (bukan nilai akurat):
  float light_lux = map(ldr_analog_value, 0, 4095, 0, 1000); // Asumsi 0-4095 ke 0-1000 lux

  // Pembacaan sensor PIR
  int motion_detected = digitalRead(PIR_PIN); // 1 jika terdeteksi, 0 jika tidak

  // Mengirim data melalui Serial Port ke komputer
  // Format: Temperature,Humidity,CO2,LightIntensity,MotionDetected
  Serial.print(t);
  Serial.print(",");
  Serial.print(h);
  Serial.print(",");
  Serial.print(co2_ppm);
  Serial.print(",");
  Serial.print(light_lux);
  Serial.print(",");
  Serial.println(motion_detected);

  // Memproses perintah dari komputer
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Hapus spasi atau karakter newline

    if (command.startsWith("SERVO:")) {
      int angle = command.substring(6).toInt(); // Ambil angka setelah "SERVO:"
      if (angle >= 0 && angle <= 180) {
        myServo.write(angle);
        Serial.print("Servo diatur ke ");
        Serial.print(angle);
        Serial.println(" derajat.");
      } else {
        Serial.println("Sudut servo tidak valid (0-180).");
      }
    } else if (command == "RELAY_ON") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("Relay ON.");
    } else if (command == "RELAY_OFF") {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("Relay OFF.");
    }
  }

  delay(2000); // Jeda setiap 2 detik sebelum pengiriman data berikutnya
}