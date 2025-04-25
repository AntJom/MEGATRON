// ========== TRANSMETTEUR AVEC ACK ET SAISIE UTILISATEUR ==========
#include <SPI.h>
#include <RF24.h>

#define CE_PIN   7
#define CSN_PIN  8
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
DataPacket packet = {0, 0, 1};
AckPacket ack;

// Lit la saisie série de l'utilisateur sous forme "angle,power,dir"
void lireSaisieUtilisateur() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    int idx1 = line.indexOf(',');
    int idx2 = line.indexOf(',', idx1 + 1);
    if (idx1 > 0 && idx2 > idx1) {
      int8_t a = line.substring(0, idx1).toInt();
      uint8_t p = line.substring(idx1 + 1, idx2).toInt();
      uint8_t d = line.substring(idx2 + 1).toInt();
      packet.angleServo = constrain(a, -90, 90);
      packet.motorPower = constrain(p, 0, 100);
      packet.motorDir   = (d == 1 ? 1 : 0);
      Serial.print("Commande définie → angle: ");
      Serial.print(packet.angleServo);
      Serial.print("°, puissance: ");
      Serial.print(packet.motorPower);
      Serial.print("%, dir: ");
      Serial.println(packet.motorDir ? "avant" : "arrière");
    } else {
      Serial.println("Format invalide. Utilisez: angle,power,dir (ex: -45,75,1)");
    }
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial) {}
  Serial.println("Entrez: angle(-90..+90),puissance(0..100),dir(0=arrière,1=avant)");

  radio.begin();
  radio.openWritingPipe(ADDRESS);
  radio.setPALevel(RF24_PA_LOW);
  radio.enableDynamicPayloads();
  radio.enableAckPayload();
  radio.stopListening();
}

void loop() {
  // Mettre à jour le paquet si l'utilisateur a saisi de nouvelles valeurs
  lireSaisieUtilisateur();

  // Envoi du paquet
  bool ok = radio.write(&packet, sizeof(packet));
  Serial.print("Envoi → ");
  Serial.println(ok ? "OK" : "KO");

  // Lecture de l'ACK si disponible
  if (ok && radio.isAckPayloadAvailable()) {
    radio.read(&ack, sizeof(ack));
    Serial.print("ACK statut: ");
    Serial.println(ack.status);
  }

  delay(10);
}