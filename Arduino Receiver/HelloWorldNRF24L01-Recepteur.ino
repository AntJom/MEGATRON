#include <Servo.h>
#include <SPI.h>
#include <RF24.h>

#define pinCE   7             // Broche "CE" du NRF24L01
#define pinCSN  8             // Broche "CSN" du NRF24L01
#define tunnel  "PIPE1"       // Nom du tunnel (5 caractères)

const int servoPin = 5;       // Broche de commande du servo
Servo myServo;
int currentAngle = 90;        // Angle initial du servo

RF24 radio(pinCE, pinCSN);    // Instanciation du module NRF24L01
const byte adresse[6] = tunnel;
char message[32];             // Buffer temporaire pour le message

void setup() {
  Serial.begin(9600);
  
  // Initialisation du servo
  myServo.attach(servoPin);
  myServo.write(currentAngle);
  Serial.print("Initialisation du servo, angle = ");
  Serial.println(currentAngle);

  // Initialisation du module NRF24L01 en mode réception avec ack payloads
  radio.begin();
  radio.setPALevel(RF24_PA_MIN);
  radio.openReadingPipe(0, adresse);
  radio.enableAckPayload();  // Activation de l'ack payload
  radio.startListening();
}

void loop() {
  if (radio.available()) {
    // Utiliser un buffer pour ne traiter que le dernier message disponible
    char lastMessage[32];
    memset(lastMessage, 0, sizeof(lastMessage));
    
    while (radio.available()) {
      memset(message, 0, sizeof(message));
      radio.read(&message, sizeof(message));
      strcpy(lastMessage, message);  // Conserver le dernier message lu
      delay(5);  // Petit délai pour permettre l'accumulation de messages
    }
    
    Serial.print("Message reçu : ");
    Serial.println(lastMessage);
    
    // Conversion du message en entier pour obtenir la nouvelle position du servo
    int newAngle = atoi(lastMessage);
    if (newAngle < 0) newAngle = 0;
    if (newAngle > 180) newAngle = 180;
    
    setServoAngle(newAngle);
    
    // Envoi de l'accusé de réception via l'ack payload
    const char ackPayload[] = "OK";
    radio.writeAckPayload(0, ackPayload, sizeof(ackPayload));
  }
}

// Fonction pour modifier l'angle du servomoteur
void setServoAngle(int angle) {
  currentAngle = angle;
  myServo.write(currentAngle);
  Serial.print("Servo déplacé à l'angle : ");
  Serial.println(currentAngle);
}
