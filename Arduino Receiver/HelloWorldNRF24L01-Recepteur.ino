// ========== RECEPTEUR AVEC ACK ET PILOTAGE SERVO+MOTEUR L298 ==========
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>

// ==== Pins ====
#define CE_PIN     9
#define CSN_PIN    10
#define SERVO_PIN  6
#define IN1_PIN    4
#define IN2_PIN    5
#define ENA_PIN    3  // PWM

const byte ADDRESS[6] = "PIPE1";

struct DataPacket {
  int8_t  angleServo;   // -90…+90
  uint8_t motorPower;   // 0…100 %
  uint8_t motorDir;     // 0 = arrière, 1 = avant
};
struct AckPacket {
  uint8_t status;       // 1 = reçu
};

RF24 radio(CE_PIN, CSN_PIN);
Servo monServo;
DataPacket packet;
AckPacket ack = {1};

// État courant pour maintien
int8_t       lastAngle       = 0;
uint8_t      lastPower       = 0;
uint8_t      lastDir         = 1;
unsigned long lastReceiveTime = 0;
const unsigned long holdDuration = 1000;  // ms

// Constrain angle and convert -90..+90 -> 0..180
void piloterServo(int8_t angle) {
  angle = constrain(angle, -90, 90);
  monServo.write(angle + 90);
}

// Apply motor power and direction
void piloterMoteur(uint8_t power, uint8_t dir) {
  if (power == 0) {
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, LOW);
    analogWrite(ENA_PIN, 0);
  } else if (dir == 1) {
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, HIGH);
    analogWrite(ENA_PIN, map(power, 0, 100, 0, 255));
  } else {
    digitalWrite(IN1_PIN, HIGH);
    digitalWrite(IN2_PIN, LOW);
    analogWrite(ENA_PIN, map(power, 0, 100, 0, 255));
  }
}

void setup() {
  Serial.begin(9600);

  // Servo init
  monServo.attach(SERVO_PIN);
  monServo.write(90);

  // L298 init
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  analogWrite(ENA_PIN, 0);

  // NRF24L01 init
  radio.begin();
  radio.openReadingPipe(1, ADDRESS);
  radio.setPALevel(RF24_PA_LOW);
  radio.enableDynamicPayloads();
  radio.enableAckPayload();
  radio.startListening();

  Serial.println(F("Récepteur servo+moteur prêt"));
}

void loop() {
  if (radio.available()) {
    // Lecture du paquet
    radio.read(&packet, sizeof(packet));
    lastAngle       = packet.angleServo;
    lastPower       = packet.motorPower;
    lastDir         = packet.motorDir;
    lastReceiveTime = millis();

    Serial.print(F("Reçu → angle: "));
    Serial.print(lastAngle);
    Serial.print(F("°, "));
    Serial.print(lastPower);
    Serial.print(F("%, dir: "));
    Serial.println(lastDir ? F("avant") : F("arrière"));

    piloterServo(lastAngle);
    piloterMoteur(lastPower, lastDir);

    // Envoi de l'ACK payload
    radio.writeAckPayload(1, &ack, sizeof(ack));

  } else if (millis() - lastReceiveTime < holdDuration) {
    // Maintien de la dernière commande
    piloterServo(lastAngle);
    piloterMoteur(lastPower, lastDir);

  } else {
    // >1s sans commande -> arrêt moteur
    if (lastPower != 0) {
      Serial.println(F("1s sans commande, arrêt moteur"));
      lastPower = 0;
      piloterMoteur(lastPower, lastDir);
    }
    // servo reste en place
  }
}
