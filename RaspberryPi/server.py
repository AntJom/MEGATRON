import socket
import struct
import time
import json
import errno
from picamera2 import Picamera2
import cv2

# ------------------------
# CONSTANTES
# ------------------------
PORT = 5000
RESOLUTION = (3280, 2464)
CAM_FORMAT = "BGR888"
HEADER_FMT = "Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

PRESETS = {
    "HIGH":   {"quality": 100, "use_gray": False, "resize_factor": 0.5},
    "MEDIUM": {"quality": 60, "use_gray": False, "resize_factor": 0.5},
    "LOW":    {"quality": 30, "use_gray": False,  "resize_factor": 0.5},
}
PRESET_LIST = list(PRESETS.keys())
DEFAULT_PRESET = "HIGH"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return "127.0.0.1"
    finally:
        s.close()

def init_camera(res):
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": res, "format": CAM_FORMAT})
    cam.configure(cfg)
    cam.start()
    print(f"[INFO] Camera demarree en resolution {res}")
    return cam

def init_server(host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[INFO] Serveur ecoute sur {host}:{port}")
    return srv

def apply_preset(frame, preset):
    s = PRESETS[preset]
    f = frame.copy()
    if s["use_gray"]:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if s["resize_factor"] != 1.0:
        f = cv2.resize(f, (0, 0),
                       fx=s["resize_factor"], fy=s["resize_factor"],
                       interpolation=cv2.INTER_AREA)
    return f, s["quality"]

def encode_frame(frame, quality):
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ret else None

def send_all_non_blocking(sock, data):
    """
    Envoie integralement 'data' sur la socket non-bloquante 'sock'.
    En cas de EAGAIN/EWOULDBLOCK, attend tres breve et continue.
    """
    total_sent = 0
    length = len(data)
    while total_sent < length:
        try:
            sent = sock.send(data[total_sent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            total_sent += sent
        except BlockingIOError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                time.sleep(0.001)
                continue
            else:
                raise

def stream_video(conn, cam, res):
    conn.setblocking(False)
    preset = DEFAULT_PRESET
    orig_w, orig_h = res
    print(f"[INFO] Demarrage du streaming avec preset {preset}")
    while True:
        frame = cam.capture_array()
        proc, quality = apply_preset(frame, preset)
        data = encode_frame(proc, quality)
        if not data:
            continue

        factor = PRESETS[preset]["resize_factor"]
        w = int(orig_w * factor)
        h = int(orig_h * factor)

        ts = time.time()
        meta = json.dumps({
            "timestamp": ts,
            "resolution": [w, h]
        }).encode()

        try:
            send_all_non_blocking(conn, struct.pack(HEADER_FMT, len(meta)))
            send_all_non_blocking(conn, meta)
            send_all_non_blocking(conn, struct.pack(HEADER_FMT, len(data)))
            send_all_non_blocking(conn, data)
        except Exception:
            print("[WARN] Client deconnecte ou erreur d'envoi")
            break

        try:
            cmd = conn.recv(1)
            idx = PRESET_LIST.index(preset)
            if cmd == b"-" and idx < len(PRESET_LIST) - 1:
                preset = PRESET_LIST[idx + 1]
                print(f"[INFO] Changement preset -> {preset}")
            elif cmd == b"+" and idx > 0:
                preset = PRESET_LIST[idx - 1]
                print(f"[INFO] Changement preset -> {preset}")
        except BlockingIOError:
            pass

def main():
    host = get_local_ip()
    cam = init_camera(RESOLUTION)
    srv = init_server(host, PORT)
    while True:
        print("[INFO] En attente d'un client...")
        conn, addr = srv.accept()
        print(f"[INFO] Client connecte: {addr}")
        stream_video(conn, cam, RESOLUTION)
        conn.close()
        print("[INFO] Client deconnecte")

if __name__ == "__main__":
    main()
