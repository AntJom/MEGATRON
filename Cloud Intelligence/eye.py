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

# Socket et buffer internes
_stream_sock = None
_stream_buf = b""

# ------------------------
# Fonctions de streaming internes
# ------------------------
def _read_raw_stream():
    """
    Lit le flux réseau et renvoie (ret, frame, meta, img_size).
    meta: dict JSON, img_size: nombre d'octets de l'image brute
    """
    global _stream_sock, _stream_buf
    if _stream_sock is None:
        raise RuntimeError("Stream non initialisé !")

    # 1) Lire la longueur des metadata
    while len(_stream_buf) < HEADER_SIZE:
        packet = _stream_sock.recv(4096)
        if not packet:
            return False, None, None, None
        _stream_buf += packet
    mlen = struct.unpack(HEADER_FMT, _stream_buf[:HEADER_SIZE])[0]
    _stream_buf = _stream_buf[HEADER_SIZE:]

    # 2) Lire la metadata (JSON)
    while len(_stream_buf) < mlen:
        packet = _stream_sock.recv(4096)
        if not packet:
            return False, None, None, None
        _stream_buf += packet
    raw_meta = _stream_buf[:mlen]
    meta = json.loads(raw_meta.decode())
    _stream_buf = _stream_buf[mlen:]

    # 3) Lire la longueur de l'image
    while len(_stream_buf) < HEADER_SIZE:
        packet = _stream_sock.recv(4096)
        if not packet:
            return False, None, None, None
        _stream_buf += packet
    ilen = struct.unpack(HEADER_FMT, _stream_buf[:HEADER_SIZE])[0]
    _stream_buf = _stream_buf[HEADER_SIZE:]

    # 4) Lire l'image
    while len(_stream_buf) < ilen:
        packet = _stream_sock.recv(4096)
        if not packet:
            return False, None, None, None
        _stream_buf += packet
    img_bytes = _stream_buf[:ilen]
    _stream_buf = _stream_buf[ilen:]

    # Décoder avec OpenCV
    frame = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_UNCHANGED
    )
    if frame is None:
        print("[WARN] Échec du décodage de la frame")
        return False, None, None, None

    return True, frame, meta, len(img_bytes)


def init_stream(host: str = HOST, port: int = PORT):
    """
    Initialise la connexion au serveur de streaming.
    """
    global _stream_sock, _stream_buf
    _stream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _stream_sock.connect((host, port))
    _stream_buf = b""
    print(f"[STREAM] Connecté à {host}:{port}")


def read_stream_frame():
    """
    Lit une trame du flux réseau.
    Renvoie (ret, frame) sans metadata.
    """
    ret, frame, _, _ = _read_raw_stream()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return ret, frame


def send_control(cmd: bytes):
    """
    Envoie une commande unique ('+' ou '-') au serveur.
    """
    if _stream_sock:
        try:
            _stream_sock.send(cmd)
            print(f"[STREAM] Commande envoyée: {cmd!r}")
        except Exception as e:
            print(f"[ERROR] Impossible d'envoyer la commande: {e}")


def close_stream():
    """
    Ferme la connexion au serveur.
    """
    global _stream_sock
    if _stream_sock:
        _stream_sock.close()
        _stream_sock = None
        print("[STREAM] Déconnecté")


# ------------------------
# Mode autonome
# ------------------------
def main():
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        init_stream()
        while True:
            ret, frame, meta, size = _read_raw_stream()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ret:
                print("[STREAM] Flux interrompu")
                break

            # Extraire et afficher les infos de metadata
            ts = meta.get("timestamp", None)
            if ts is not None:
                latency = (time.time() - ts) * 1000
                res = meta.get("resolution", [None, None])
                w, h = res if len(res) == 2 else (None, None)
                cv2.putText(frame, f"Res:{w}x{h}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Latence:{latency:.1f}ms", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f"Taille:{size}octets", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Fermeture du client de streaming")
                break
            elif key == ord('+'):
                send_control(b'+')
            elif key == ord('-'):
                send_control(b'-')

    except KeyboardInterrupt:
        print("[INFO] Interruption clavier")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        close_stream()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()