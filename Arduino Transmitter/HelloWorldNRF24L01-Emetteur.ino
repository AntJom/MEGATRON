#include <SPI.h>
#include <RF24.h>

#define pinCE   7             // Broche "CE" du NRF24L01
#define pinCSN  8             // Broche "CSN" du NRF24L01
#define tunnel  "PIPE1"       // Nom du tunnel (5 caractères)

RF24 radio(pinCE, pinCSN);    // Instanciation du NRF24L01
const byte adresse[6] = tunnel;
const int BUFFER_SIZE = 32;   // Taille maximale du message

// Buffer pour stocker la dernière commande d'angle
String lastCommand = "";

void setup() {
  Serial.begin(9600);
  
  // Initialisation du module NRF24L01 en mode émission avec auto-ack
  radio.begin();
  radio.openWritingPipe(adresse);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();  // On commence en mode émission
  
  Serial.println("====================================");
  Serial.println("       Mode Emission NRF24L01       ");
  Serial.println("====================================");
  Serial.println("Saisissez votre commande et appuyez sur Entrée:");
}

void loop() {
  // Lecture de la commande depuis le moniteur série
  if (Serial.available()) {
    char buffer[BUFFER_SIZE];
    int len = Serial.readBytesUntil('\n', buffer, BUFFER_SIZE - 1);
    buffer[len] = '\0';  // Terminaison de la chaîne
    // Met à jour le buffer avec la dernière commande reçue
    lastCommand = String(buffer);
  }
  
  // Si une commande est présente dans le buffer, on l'envoie
  if (lastCommand.length() > 0) {
    bool success = radio.write(lastCommand.c_str(), BUFFER_SIZE);
    
    Serial.println("====================================");
    if (success) {
      Serial.print("Commande envoyée : ");
      Serial.println(lastCommand);
      
      // Passage en mode écoute pour attendre l'accusé de réception (ACK)
      radio.startListening();
      unsigned long startTime = millis();
      bool ackReceived = false;
      while (millis() - startTime < 500) { // Timeout de 500 ms
        if (radio.available()) {
          char ack[BUFFER_SIZE];
          memset(ack, 0, sizeof(ack));
          radio.read(&ack, sizeof(ack));
          String ackStr = String(ack);
          if (ackStr.indexOf("OK") != -1) {
            ackReceived = true;
            break;
          }
        }
      }
      radio.stopListening();
      
      if (ackReceived) {
        Serial.println("Accusé de réception reçu : OK");
        // On vide le buffer pour éviter l'envoi répété
        lastCommand = "";
      } else {
        Serial.println("Aucun accusé reçu, tentative de renvoi...");
        // On conserve la dernière commande pour réessayer
      }
    } else {
      Serial.println("Erreur lors de l'envoi de la commande !");
    }
    Serial.println("====================================");
    delay(100); // Petit délai pour éviter un envoi trop fréquent
  }
}
