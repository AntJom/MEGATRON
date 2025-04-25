import serial
import time

# Constantes de configuration
DEFAULT_PORT = "COM5"      # Port série par défaut (ex: COM3 ou /dev/ttyUSB0)
DEFAULT_BAUD = 9600        # Vitesse en bauds par défaut
INIT_DELAY   = 2.0         # Délai pour laisser l'Arduino redémarrer (secondes)
TIMEOUT      = 1.0         # Timeout pour lecture/écriture (secondes)

class DataSender:
    """
    Classe pour envoyer des triplets (angle, puissance, dir) sur un port série Arduino.
    Usage importé :
        from serial_comms import DataSender
        sender = DataSender()
        sender.open()
        sender.send(angle=0, power=75, direction=1)
        sender.close()
    """
    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 timeout=TIMEOUT, init_delay=INIT_DELAY):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.init_delay = init_delay
        self._ser = None

    def open(self):
        """Ouvre le port série et attend un court instant."""
        if self._ser and self._ser.is_open:
            return
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except serial.SerialException as e:
            raise RuntimeError(f"Impossible d'ouvrir le port {self.port}: {e}")
        # Laisser l'Arduino se réinitialiser
        time.sleep(self.init_delay)

    def close(self):
        """Ferme le port série si ouvert."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            self._ser = None

    def send(self, angle: int, power: int, direction: int):
        """
        Envoie un triplet (angle,power,dir) formaté 'angle,power,dir\n'.
        :param angle: entier entre -90 et +90
        :param power: entier entre 0 et 100
        :param direction: 0 (arrière) ou 1 (avant)
        """
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Port série non ouvert. Appelle d'abord open().")
        # Clamp des valeurs
        a = max(-90, min(90, int(angle)))
        p = max(0, min(100, int(power)))
        d = 1 if int(direction) == 1 else 0
        msg = f"{a},{p},{d}\n"
        self._ser.write(msg.encode('utf-8'))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _interactive_loop():
    """
    Fonction utilisée si ce fichier est exécuté directement. 
    Demande des valeurs à l'utilisateur et les envoie en boucle jusqu'à 'q'.
    """
    sender = DataSender()
    try:
        sender.open()
        print(f"Connecté sur {sender.port} @ {sender.baud} bauds. (q pour quitter)")
        while True:
            s = input("Entrez angle(-90..+90),puissance(0..100),dir(0=arrière,1=avant) > ").strip()
            if s.lower() == 'q':
                break
            parts = s.split(',')
            if len(parts) != 3:
                print("Format invalide, ex : -45,75,1")
                continue
            try:
                a = int(parts[0])
                p = int(parts[1])
                d = int(parts[2])
            except ValueError:
                print("Valeurs non entières !")
                continue
            sender.send(angle=a, power=p, direction=d)
            print(f"Envoyé → angle: {a}°, puissance: {p}%, dir: {'avant' if d==1 else 'arrière'}")
    finally:
        sender.close()
        print("Déconnexion.")


if __name__ == "__main__":
    _interactive_loop()
