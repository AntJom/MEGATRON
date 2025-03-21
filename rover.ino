#include <Servo.h>

// Définir les broches pour les capteurs SRF05
const int trigPin[4] = {4, 6, 8, 10};
const int echoPin[4] = {5, 7, 9, 11};
long distances[4] = {0, 0, 0, 0};

// Définir les broches pour la LED
const int GreenledPin = 2;  // Broche pour la LED verte
const int RedledPin = 3;

unsigned long lastSensorUpdate = 0;
unsigned long lastBlinkUpdate = 0;
const unsigned long sensorInterval = 200;  // Intervalle de lecture des capteurs (ms)
const unsigned long blinkInterval = 500;
bool obstacleDetected = false;


// Variables pour le servomoteur
const int servoPin = 5;
Servo myServo;
int currentAngle = 90;

enum State { MANUAL, AUTOMATIC, MANUAL_ERROR, AUTOMATIC_ERROR };
State currentState = MANUAL;

unsigned long sensorReadTime = 0;
unsigned long obstacleCheckTime = 0;
unsigned long serialCommandTime = 0;
unsigned long ledUpdateTime = 0;

void setup() {
  // Initialiser la communication série
  Serial.begin(9600);

  // Configurer les broches des capteurs
  for (int i = 0; i < 4; i++) {
    pinMode(trigPin[i], OUTPUT);
    pinMode(echoPin[i], INPUT);
  }
    // Configurer les broches des LEDs
  pinMode(GreenledPin, OUTPUT);
  pinMode(RedledPin, OUTPUT);

  // Attacher le servomoteur
  myServo.attach(servoPin);
  myServo.write(currentAngle);
}

void loop() {
  // Gérer la lecture des capteurs de manière non bloquante
  if (millis() - lastSensorUpdate >= sensorInterval) {
    unsigned long startTime = micros();
    readAllSensors();
    lastSensorUpdate = millis();
  }
    // Gérer les LEDs en fonction de l'état actuel
    unsigned long startTime = micros();
  updateLEDs();
  ledUpdateTime = micros() - startTime;

  // Détecter les obstacles
  startTime = micros();
  checkObstacles();
  obstacleCheckTime = micros() - startTime;

  // Gérer les commandes série
  startTime = micros();
  handleSerialCommands();
  serialCommandTime = micros() - startTime;
  // Détecter les obstacles
  checkObstacles();


  switch (currentState) {
    case MANUAL:
      // Mode manuel : rien à faire ici, les commandes sont gérées par handleSerialCommands()
      break;
    case AUTOMATIC:
      // Mode automatique : déplacer le rover (à implémenter)
      break;
    case MANUAL_ERROR:
    case AUTOMATIC_ERROR:
      // En cas d'erreur, arrêter le rover
      stopRover();
      break;
  }


  printDistances();
}

// Fonction pour lire tous les capteurs
void readAllSensors() {
  for (int i = 0; i < 4; i++) {
    distances[i] = getDistance(i);
  }
}

// Fonction pour mesurer la distance d'un capteur
long getDistance(int sensorIndex) {
  digitalWrite(trigPin[sensorIndex], LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin[sensorIndex], HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin[sensorIndex], LOW);

  long duration = pulseIn(echoPin[sensorIndex], HIGH);
  return duration * 0.034 / 2;  // Convertir en cm
}

// Fonction pour détecter les obstacles
void checkObstacles() {
  obstacleDetected = false;
  for (int i = 0; i < 4; i++) {
    if (distances[i] < 20) {
      obstacleDetected = true;
      break;
    }
  }
    // Mettre à jour l'état en fonction de la détection d'obstacles
  if (obstacleDetected) {
    if (currentState == MANUAL) currentState = MANUAL_ERROR;
    if (currentState == AUTOMATIC) currentState = AUTOMATIC_ERROR;
  } else {
    if (currentState == MANUAL_ERROR) currentState = MANUAL;
    if (currentState == AUTOMATIC_ERROR) currentState = AUTOMATIC;
  }
}
void updateLEDs() {
  static bool ledState = false;
  unsigned long currentMillis = millis();

  switch (currentState) {
    case MANUAL:
      digitalWrite(GreenledPin, HIGH);
      digitalWrite(RedledPin, LOW);
      break;
    case AUTOMATIC:
      digitalWrite(GreenledPin, LOW);
      digitalWrite(RedledPin, HIGH);
      break;
    case MANUAL_ERROR:
    case AUTOMATIC_ERROR:
      if (currentMillis - lastBlinkUpdate >= blinkInterval) {
        ledState = !ledState;
        digitalWrite(currentState == MANUAL_ERROR ? GreenledPin : RedledPin, ledState);
        lastBlinkUpdate = currentMillis;
      }
      break;
  }
}

// Fonction pour afficher les distances mesurées
void printDistances() {
  for (int i = 0; i < 4; i++) {
    Serial.print("Capteur ");
    Serial.print(i);
    Serial.print(": ");
    Serial.print(distances[i]);
    Serial.println(" cm");
  }
}
void stopRover() {
  Serial.println("Rover stopped");
  // Implémenter l'arrêt du rover ici
}

void handleSerialCommands() {
  if (Serial.available()) {
    String input = Serial.readString();
    input.trim();  // Supprimer les espaces inutiles

    if  (input.toInt() >= 0 && input.toInt() <= 180) {
      setServoAngle(input.toInt());
    }

    // AJOUTER ICI activer la fonction TOGGLEMODE
  }
}


void toggleMode() {
  if (currentState == MANUAL || currentState == MANUAL_ERROR) {
    currentState = AUTOMATIC;
  } else {
    currentState = MANUAL;
  }
  Serial.println(currentState == MANUAL ? "Mode: MANUAL" : "Mode: AUTOMATIC");
}

void setServoAngle(int angle) {
  currentAngle = angle;
  myServo.write(currentAngle);
  Serial.print("Servo déplacé à l'angle : ");
  Serial.println(currentAngle);
}
