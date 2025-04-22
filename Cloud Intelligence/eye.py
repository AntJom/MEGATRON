import cv2
import socket
import struct
import time
import numpy as np
import json

# ------------------------
# CONSTANTES
# ------------------------
HOST = "megatron.local"   # mettre l'IP du serveur
PORT = 5000
HEADER_FMT = "Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
WINDOW = "Stream Client"

def main():
    cv2.namedWindow(WINDOW)
    while True:
        print(f"[INFO] Tentative de connexion a {HOST}:{PORT}")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            print("[INFO] Connecte au serveur")
            data = b""

            while True:
                # lire metadata length
                while len(data) < HEADER_SIZE:
                    packet = s.recv(4096)
                    if not packet:
                        raise Exception("Deconnexion serveur")
                    data += packet
                mlen = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])[0]
                data = data[HEADER_SIZE:]

                # lire metadata
                while len(data) < mlen:
                    packet = s.recv(4096)
                    if not packet:
                        raise Exception("Deconnexion serveur")
                    data += packet
                meta = json.loads(data[:mlen].decode())
                data = data[mlen:]

                # lire image length
                while len(data) < HEADER_SIZE:
                    packet = s.recv(4096)
                    if not packet:
                        raise Exception("Deconnexion serveur")
                    data += packet
                ilen = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])[0]
                data = data[HEADER_SIZE:]

                # lire image
                while len(data) < ilen:
                    packet = s.recv(4096)
                    if not packet:
                        raise Exception("Deconnexion serveur")
                    data += packet
                img_bytes = data[:ilen]
                data = data[ilen:]

                # decode
                frame = cv2.imdecode(
                    np.frombuffer(img_bytes, np.uint8),
                    cv2.IMREAD_UNCHANGED
                )
                if frame is None:
                    print("[WARN] Echec decode frame")
                    continue

                # calcul latence
                latency = (time.time() - meta["timestamp"]) * 1000
                w, h = meta.get("resolution", ["?", "?"])
                size = len(img_bytes)

                # affichage info
                cv2.putText(frame, f"Res:{w}x{h}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Latence:{latency:.1f}ms", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Taille:{size}octets", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Fermeture client")
                    raise KeyboardInterrupt
                if key == ord('+'):
                    s.send(b'+')
                    print("[INFO] Demande preset +")
                if key == ord('-'):
                    s.send(b'-')
                    print("[INFO] Demande preset -")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            print("[INFO] Reconnexion dans 1s")
            time.sleep(1)
        finally:
            try:
                s.close()
            except:
                pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
