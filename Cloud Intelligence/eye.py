import cv2
import socket
import struct
import numpy as np
import time
import json

HOST = '192.168.1.66'  # Mettez a jour avec l'IP du serveur
PORT = 5000

# Taille (en octets) de l'entete indiquant la longueur (8 octets)
header_size = struct.calcsize("Q")

while True:
    try:
        print(f"[INFO] Tentative de connexion a {HOST}:{PORT}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("[INFO] Connecte au serveur.")
        data = b""
        while True:
            start_recv = time.time()
            # Reception de la longueur du bloc metadata (8 octets)
            while len(data) < header_size:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    raise Exception("Fermeture de la connexion lors de la reception de l'entete metadata.")
                data += packet
            meta_len = struct.unpack("Q", data[:header_size])[0]
            data = data[header_size:]
            
            # Reception du bloc metadata
            while len(data) < meta_len:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    raise Exception("Fermeture de la connexion lors de la reception du bloc metadata.")
                data += packet
            meta_bytes = data[:meta_len]
            data = data[meta_len:]
            metadata = json.loads(meta_bytes.decode("utf-8"))
            
            # Reception de la longueur de l'image (8 octets)
            while len(data) < header_size:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    raise Exception("Fermeture de la connexion lors de la reception de l'entete image.")
                data += packet
            img_len = struct.unpack("Q", data[:header_size])[0]
            data = data[header_size:]
            
            # Reception de l'image encodee
            while len(data) < img_len:
                packet = client_socket.recv(4 * 1024)
                if not packet:
                    raise Exception("Fermeture de la connexion lors de la reception de l'image.")
                data += packet
            img_data = data[:img_len]
            data = data[img_len:]
            
            recv_time = time.time() - start_recv
            
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("[WARNING] Echec du decodage de la frame, passe a la suivante.")
                continue

            # Calcul de la latence et du poids de l'image
            latency_ms = recv_time * 1000
            weight = len(img_data)
            preset = metadata.get("preset", "N/A")

            # Superposition des informations sur l'image
            cv2.putText(frame, f"Preset: {preset}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Weight: {weight} octets", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("[INFO] Fermeture du client...")
        break
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass
        # Reconnecter immediatement (pas de delai d'attente)
        print("[INFO] Deconnecte du serveur. Reconnexion immediatement...")

cv2.destroyAllWindows()
