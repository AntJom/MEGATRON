import cv2
import socket
import struct
import time
import numpy as np
import json

# ------------------------
# Definition des presets disponibles
# ------------------------
PRESETS = {
    "HIGH": {
        "quality": 90,        # Qualite elevee, couleur, resolution complete
        "use_gray": False,
        "resize_factor": 1.0
    },
    "MEDIUM": {
        "quality": 60,        # Qualite intermediaire, couleur, resolution reduite
        "use_gray": False,
        "resize_factor": 0.75
    },
    "LOW": {
        "quality": 30,        # Faible qualite, conversion en niveaux de gris, resolution reduite
        "use_gray": True,
        "resize_factor": 0.5
    }
}

# Choix du preset par defaut
DEFAULT_PRESET = "HIGH"

# Mode demo : en mode demo, le serveur fait cycler automatiquement le preset toutes les 5 secondes.
DEMO_MODE = False

# Mode verbose (True pour afficher tous les messages de debug, False pour n'afficher que l'essentiel)
VERBOSE = False

# ------------------------
# Fonctions d'initialisation
# ------------------------
def get_local_ip():
    """Retourne l'IP locale du serveur en creant une connexion UDP vers une adresse externe."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def init_camera(width, height):
    """Initialise la camera USB avec la resolution specifiee."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def init_server_socket(host, port):
    """Initialise et retourne un socket serveur ecoute sur l'IP et le port donnes."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    return server_socket

# ------------------------
# Gestion du preset en mode demo
# ------------------------
def update_preset_demo(state):
    """
    En mode demo, fait cycler le preset parmi les cles de PRESETS toutes les 5 secondes.
    Le dictionnaire d'etat (state) doit contenir les cles "preset" et "last_preset_update".
    """
    current_time = time.time()
    if current_time - state["last_preset_update"] >= 5:
        preset_list = list(PRESETS.keys())
        current_index = preset_list.index(state["preset"])
        new_index = (current_index + 1) % len(preset_list)
        state["preset"] = preset_list[new_index]
        state["last_preset_update"] = current_time
        if VERBOSE:
            print(f"[DEMO] Nouveau preset: {state['preset']}")

# ------------------------
# Traitement de l'image selon le preset
# ------------------------
def apply_preset(frame, preset):
    """
    Applique les modifications definies dans le preset sur la frame.
      - Conversion en niveaux de gris si necessaire,
      - Redimensionnement.
    Retourne la frame traitee et la valeur de quality associee.
    """
    settings = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    quality = settings["quality"]
    use_gray = settings["use_gray"]
    resize_factor = settings["resize_factor"]

    processed = frame.copy()
    if use_gray:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if resize_factor != 1.0:
        processed = cv2.resize(processed, (0, 0), fx=resize_factor, fy=resize_factor)
    return processed, quality

def encode_frame(frame, quality):
    """
    Encode la frame en JPEG avec la qualite donnee.
    Retourne les donnees encodees ou None en cas d'echec.
    """
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return None
    return buffer.tobytes()

# ------------------------
# Fonction de streaming
# ------------------------
def stream_video(conn, cap, state):
    """
    Pour chaque frame lue depuis la camera, le serveur (en mode demo s'il est actif)
    met a jour le preset automatiquement toutes les 5 secondes, applique le preset a l'image,
    encode l'image et envoie au client un paquet contenant :
      - La longueur (8 octets) du bloc metadata,
      - Le bloc metadata (JSON indiquant le preset),
      - La longueur (8 octets) des donnees image,
      - Les donnees image encodees.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Echec de lecture de la frame depuis la camera")
            break

        if state["demo_mode"]:
            update_preset_demo(state)

        # Appliquer le preset courant
        processed_frame, quality = apply_preset(frame, state["preset"])
        encoded_data = encode_frame(processed_frame, quality)
        if encoded_data is None:
            print("[ERROR] Echec d'encodage de la frame")
            continue

        # Creer le bloc metadata
        metadata = {"preset": state["preset"]}
        meta_bytes = json.dumps(metadata).encode("utf-8")
        meta_length = len(meta_bytes)
        img_length = len(encoded_data)

        try:
            # Envoyer la taille du bloc metadata (8 octets)
            conn.sendall(struct.pack("Q", meta_length))
            # Envoyer le bloc metadata
            conn.sendall(meta_bytes)
            # Envoyer la taille de l'image (8 octets)
            conn.sendall(struct.pack("Q", img_length))
            # Envoyer l'image encodee
            conn.sendall(encoded_data)
        except Exception as e:
            print(f"[WARNING] Connection perdue: {e}")
            break

        if VERBOSE:
            print(f"[DEBUG] Envoye : preset={state['preset']}, image={img_length} octets, quality={quality}")

# ------------------------
# Fonction principale du serveur
# ------------------------
def main():
    host = get_local_ip()
    port = 5000
    print(f"[INFO] Server IP: {host}")

    cap = init_camera(1280, 960)
    server_socket = init_server_socket(host, port)
    print(f"[INFO] Server ecoute sur {host}:{port}")

    # Etat initial, incluant le preset courant et le temps du dernier changement (pour mode demo)
    state = {
        "preset": DEFAULT_PRESET,
        "demo_mode": DEMO_MODE,
        "last_preset_update": time.time()
    }

    while True:
        print("[INFO] En attente d'un client...")
        try:
            conn, addr = server_socket.accept()
            print(f"[INFO] Connection etablie avec {addr}")
            stream_video(conn, cap, state)
        except Exception as ex:
            print(f"[ERROR] Exception: {ex}")
        finally:
            try:
                conn.close()
            except:
                pass
            print("[INFO] Client deconnecte, attente de reconnexion...")

    cap.release()
    server_socket.close()

if __name__ == '__main__':
    main()
