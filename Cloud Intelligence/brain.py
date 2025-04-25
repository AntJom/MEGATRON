import os
import sys
import cv2
import math
import numpy as np
import time
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import re
import glob
from collections import deque

from serial_comms import DataSender

# Import du client de streaming
from eye import init_stream, read_stream_frame, send_control, close_stream

# ——————————————————————————————————
# CONFIGURATION (globals)
# ——————————————————————————————————
MODEL_PATH = 'yolo11x-pose.pt'
CALIB_DIR = 'calib_data'

DRAW_SKELETON = True
DRAW_BOXES = True
DISPLAY_PROCESS_TIME = False

HORIZONTAL_THRESHOLD_DEG = 15.0
SCORE_OK = 80.0

DEEPSORT_MAX_AGE = 50
DEEPSORT_N_INIT = 10
DEEPSORT_NMS_MAX_OVERLAP = 0.3
DEEPSORT_MAX_IOU_DISTANCE = 0.9
DEEPSORT_MAX_COSINE_DISTANCE = 0.6
IOU_MAPPING_THRESHOLD = 0.5

HOLD_TIME = 5.0
FRAME_GAP = 0.5
MARGIN_PX = 50

# Anthropometric constants (pour calibration future)
REAL_HEIGHT_CM   = 175.0       # taille du sujet pour la calibration
DIST_CORRECTION  = 1.0         # facteur de correction global si besoin
REAL_DIST_CM     = 200.0       # distance caméra–sujet lors de la calibration
REAL_SPAN_CM     = REAL_HEIGHT_CM * 7.0/9.0  # envergure poignet→poignet de référence

PROPORTIONS = {
    'shoulder_shoulder':       0.26,
    'left_shoulder_elbow':     0.186,
    'left_elbow_wrist':        0.146,
    'right_shoulder_elbow':    0.186,
    'right_elbow_wrist':       0.146
}
REAL_SEGMENTS = {
    **{k: PROPORTIONS[k] * REAL_HEIGHT_CM for k in PROPORTIONS},
    'span_wrist_wrist': REAL_SPAN_CM
}

# COCO keypoint indices
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH = 5, 6
LEL, REL = 7, 8
LWR, RWR = 9, 10
LHIP, RHIP = 11, 12
LKNE, RKNE = 13, 14
LANK, RANK = 15, 16
CONF_THRESH = 0.2

# Joint indices for RANSAC shoulders + wrists
JOINT_INDICES = (7, 9, 8, 10)

# Global state
times = {'start': None, 'last': None, 'prev_frame': None}
state = {'mode': 'manual', 'auto_phase': 'search', 'selected_id': None}

# variables a envoyer sur le port com
COM_ANGLE = 0
COM_DIRECTION = 0 # 1 = arrière
COM_MOTOR_POWER = 0

cap = None
model = None
tracker = None

# Basculer entre webcam (False) et streaming réseau (True)
USE_STREAM = True

# Historique des distances initialisé à 10 zéros
DIST_HISTORY = deque([0.0] * 10, maxlen=10)
# Moyenne courante initiale = 0
COM_DISTANCE = 0.0

# ——————————————————————————————————
# INITIALIZATION
# ——————————————————————————————————
def init_camera(source=0, buffer_size=1):
    global cap
    if USE_STREAM:
        init_stream()    # initialise le socket de streaming
        cap = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"Impossible d'ouvrir le flux vidéo: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

def init_model():
    global model
    model = YOLO(MODEL_PATH)

def init_tracker():
    global tracker
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nms_max_overlap=DEEPSORT_NMS_MAX_OVERLAP,
        max_iou_distance=DEEPSORT_MAX_IOU_DISTANCE,
        max_cosine_distance=DEEPSORT_MAX_COSINE_DISTANCE
    )

# ——————————————————————————————————
# PROCESSING FUNCTIONS (non-distance)
# ——————————————————————————————————
def read_frame():
    if USE_STREAM:
        frame = None
        ret   = False
        # on lit MAX_FLUSH fois et on ne garde que la dernière image valide
        for _ in range(1):
            ok, img = read_stream_frame()
            if not ok:
                break
            frame = img   # on ne stocke que l’image
            ret   = True
        return ret, frame

    else:
        # pour VideoCapture OpenCV, on jette tout le buffer
        while cap.grab():
            pass
        return cap.retrieve()

def infer_pose(frame):
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or not res.boxes:
        return res, np.zeros((0,17,2)), np.zeros((0,17)), np.zeros((0,4),int), np.array([])
    xy    = res.keypoints.xy.cpu().numpy()
    conf  = res.keypoints.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes,'conf') else np.ones(len(boxes))
    return res, xy, conf, boxes, confs

def score_ransac(xy_person, conf_person, thresh_deg):
    pts = [tuple(xy_person[i]) for i in JOINT_INDICES if conf_person[i] > CONF_THRESH]
    if len(pts) < 2:
        return 0.0
    best = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            p1, p2 = pts[i], pts[j]
            angle = abs(math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])))
            angle = min(angle, 180 - angle)
            if angle > thresh_deg:
                continue
            inliers = sum(
                min(abs(math.degrees(math.atan2(p[1]-p1[1], p[0]-p1[0]))),
                    180 - abs(math.degrees(math.atan2(p[1]-p1[1], p[0]-p1[0]))))
                <= thresh_deg for p in pts
            )
            best = max(best, inliers)
    return 100.0 * best / len(pts)

def update_duration(is_tpose: bool) -> bool:
    now = time.time()
    if not is_tpose:
        times['start'] = times['last'] = None
        return False
    if times['last'] and now - times['last'] > FRAME_GAP:
        times['start'] = now
    if times['start'] is None:
        times['start'] = now
    times['last'] = now
    return (now - times['start']) >= HOLD_TIME

def update_tracker(boxes, confs, frame):
    dets = [([int(x1),int(y1),int(x2-x1),int(y2-y1)], float(c), 'person')
            for (x1,y1,x2,y2), c in zip(boxes, confs)]
    return tracker.update_tracks(dets, frame=frame)

def compute_angle(p1, p2):
    global COM_ANGLE
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    temp_angle = abs(np.degrees(np.arctan2(dy, dx)))
    if (temp_angle - 90) > 80:
        COM_ANGLE = 80
    elif (temp_angle - 90) < -80:
        COM_ANGLE = -80
    else:
        COM_ANGLE = temp_angle
    return temp_angle

def annotate_angle(frame, center):
    h, w = frame.shape[:2]
    ref = (w//2, h)
    cv2.line(frame, ref, center, (255,0,255), 2)
    ang = compute_angle(center, ref)
    cv2.putText(frame, f"{ang:.1f}°", (ref[0]+10,ref[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    return ang

# ——————————————————————————————————
# NOUVELLES FONCTIONS POUR L’ESTIMATION DE DISTANCE
# ——————————————————————————————————
def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return abs(np.degrees(np.arctan2(dy, dx)))

def pixel_segments_with_weights(xy, conf):
    pairs = {
        'shoulder_shoulder':    (LSH, RSH),
        'left_shoulder_elbow':  (LSH, LEL),
        'left_elbow_wrist':     (LEL, LWR),
        'right_shoulder_elbow': (RSH, REL),
        'right_elbow_wrist':    (REL, RWR),
        'span_wrist_wrist':     (LWR, RWR)
    }
    px, w = {}, {}
    for name, (i, j) in pairs.items():
        if conf[i] > CONF_THRESH and conf[j] > CONF_THRESH:
            dist_px = np.hypot(*(xy[i] - xy[j]))
            ang     = angle_between(xy[i], xy[j])
            weight  = max(0.0, np.cos(np.radians(ang)))
            px[name] = dist_px
            w[name]  = weight
    return px, w

def detect(model, frame):
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or not res.boxes:
        return None, None, None
    box = res.boxes.xyxy.cpu().numpy()[0].astype(int)
    xy  = res.keypoints.xy.cpu().numpy()[0]
    conf= res.keypoints.conf.cpu().numpy()[0]
    return xy, conf, box

def calibrate_focal_sum(model: YOLO, calib_dir: str = CALIB_DIR) -> float:
    """
    Calibre la focale en médiane sur toutes les images de calibration de `calib_dir`.
    Chaque fichier doit être nommé comme: <ts>_<span_cm>cm_<dist_cm>cm.<ext>
    """
    # Regex pour extraire span et distance depuis le nom de fichier
    pattern = re.compile(r'.*_(\d+(?:\.\d+)?)cm_(\d+(?:\.\d+)?)cm\.\w+$')
    focals = []

    # Parcours de toutes les images du dossier
    for filepath in glob.glob(os.path.join(calib_dir, '*')):
        fname = os.path.basename(filepath)
        m = pattern.match(fname)
        if not m:
            # nom non conforme, on ignore
            continue

        span_cm = float(m.group(1))
        dist_cm = float(m.group(2))

        img = cv2.imread(filepath)
        if img is None:
            # image illisible, on ignore
            continue

        # détection des points clés
        xy, conf, _ = detect(model, img)
        if xy is None:
            # pas de détection, on ignore
            continue

        # segments en pixels et poids
        px, w = pixel_segments_with_weights(xy, conf)
        sum_wpx = sum(w[s] * px[s] for s in px)

        # recalcul des longueurs réelles, en injectant span_cm
        real_segments = {
            **{k: PROPORTIONS[k] * REAL_HEIGHT_CM for k in PROPORTIONS},
            'span_wrist_wrist': span_cm
        }
        sum_wreal = sum(w[s] * real_segments[s] for s in px)

        # si insuffisant, on ignore
        if sum_wpx <= 1e-6 or sum_wreal <= 1e-6:
            continue

        # focale pour cette image
        focals.append(sum_wpx * dist_cm / sum_wreal)

    if not focals:
        raise RuntimeError("Calibration impossible : aucune image valide dans le dossier.")

    # on retourne la médiane des focales calculées
    return float(np.median(focals))

def estimate_distance_sum(xy, conf, focal_sum: float) -> float:
    global COM_DISTANCE, DIST_HISTORY
    px, w = pixel_segments_with_weights(xy, conf)
    sum_wpx   = sum(w[s] * px[s] for s in px)
    sum_wreal = sum(w[s] * REAL_SEGMENTS[s] for s in px)
    if sum_wpx <= 1e-6 or sum_wreal <= 1e-6:
        return None
    
    # 3) distance instantanée
    dist = focal_sum * (sum_wreal / sum_wpx) * DIST_CORRECTION

    # 4) mise à jour de l’historique
    DIST_HISTORY.append(dist)
    # 5) calcul de la moyenne glissante
    COM_DISTANCE = sum(DIST_HISTORY) / len(DIST_HISTORY)
    return focal_sum * (sum_wreal / sum_wpx) * DIST_CORRECTION

def draw_overlay(frame, box, dist=None):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if dist is not None:
        cv2.putText(frame,
                    f"{dist:.1f} cm",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

# ——————————————————————————————————
# Fonctions de tracking et mapping améliorées
# ——————————————————————————————————
def select_tpose_target(tracks, boxes, xy_all, conf_all, ref_index):
    MAX_KEYPOINT_DIST = 100.0
    ref_pts = xy_all[ref_index][[LSH, RSH, LWR, RWR]]
    ref_center = np.mean(ref_pts, axis=0)
    for t in tracks:
        if not t.is_confirmed():
            continue
        tb = tuple(map(int, t.to_ltrb()))
        ious = np.array([iou(tb, b) for b in boxes])
        if ious.size == 0:
            continue
        idx = int(np.argmax(ious))
        if ious[idx] < IOU_MAPPING_THRESHOLD:
            continue
        cand_pts = xy_all[idx][[LSH, RSH, LWR, RWR]]
        cand_center = np.mean(cand_pts, axis=0)
        if np.linalg.norm(ref_center - cand_center) < MAX_KEYPOINT_DIST:
            return t.track_id
    return None

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter) if (aA + aB - inter) > 0 else 0.0

def calibrate_step(frame, xy, conf, boxes):
    ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    span_cm = float(input('Envergure (cm): '))
    dist_cm = float(input('Distance (cm): '))
    ann = frame.copy()
    fname = f"{ts}_{int(span_cm)}cm_{int(dist_cm)}cm.jpg"
    path = os.path.join(CALIB_DIR, fname)
    cv2.imwrite(path, ann)
    print(f"Saved calibration: {path}")

# ——————————————————————————————————
# MAIN LOOP
# ——————————————————————————————————
def run():
    global cap, tracker, state, times
    global COM_ANGLE, COM_MOTOR_POWER, COM_DIRECTION
    win = 'T-pose Detector'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    init_camera()
    init_model()

    # calibration unique à partir d'une image de référence
    focal_sum = calibrate_focal_sum(model, 'calib_data')
    print(f"Focale calibrée = {focal_sum:.1f}px")

    init_tracker()
    times['prev_frame'] = time.perf_counter()

    sender = DataSender(port="COM5", baud=9600)
    sender.open()

    manual_timer_start = 0

    while True:
        now = time.perf_counter()
        ret, frame = read_frame()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            state['mode'] = 'manual'
        elif key == ord('a'):
            state['mode'] = 'auto'
            state['auto_phase'] = 'search'
            times['start'] = times['last'] = None
        elif key == ord('c'):
            state['mode'] = 'config'
        elif key == ord('+') and USE_STREAM:
            send_control(b'+')
        elif key == ord('-') and USE_STREAM:
            send_control(b'-')
        elif key in (ord('q'), 27):
            break
        elif key == ord('s'):   # ←
            COM_ANGLE = -90
            COM_MOTOR_POWER = 100
            COM_DIRECTION = 0
            manual_timer_start = time.time()
        elif key == ord('d'): # →
            COM_ANGLE = 90
            COM_MOTOR_POWER = 100
            COM_DIRECTION = 0
            manual_timer_start = time.time()
        elif key == ord('z'): # ↑
            COM_ANGLE = 0
            COM_MOTOR_POWER = 100
            COM_DIRECTION = 0
            manual_timer_start = time.time()
        elif key == ord('w'): # ↓
            COM_ANGLE = 0
            COM_MOTOR_POWER = 100
            COM_DIRECTION = 1
            manual_timer_start = time.time()

        if time.time() - manual_timer_start > 2:
            COM_ANGLE = 0
            COM_MOTOR_POWER = 0
            COM_DIRECTION = 1

        if state['mode'] in ('config', 'auto'):
            res, xy_all, conf_all, boxes, confs = infer_pose(frame)

        if state['mode'] == 'config':
            COM_MOTOR_POWER = 0
            sender.send(angle=0, power=0, direction=0)
            # 1) Affichage du squelette ou de l'image brute
            img = res.plot() if DRAW_SKELETON else frame.copy()

            # 2) On calcule un score pour chaque boîte et on choisit la couleur
            tpose_scores = []
            for i, box in enumerate(boxes):
                sc = score_ransac(xy_all[i], conf_all[i], HORIZONTAL_THRESHOLD_DEG)
                tpose_scores.append(sc)
                color = (0, 255, 0) if sc >= SCORE_OK else (0, 255, 255)
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 3) Affichage
            cv2.imshow(win, img)

            # 4) Si la T-pose (au moins une boîte verte) est tenue assez longtemps, on calibre
            if update_duration(any(sc >= SCORE_OK for sc in tpose_scores)):
                calibrate_step(frame, xy_all, conf_all, boxes)

        elif state['mode'] == 'manual':
            disp = frame.copy()
            cv2.putText(disp, 'MODE MANUEL', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            com_texts = [
                f"Angle       : {COM_ANGLE}",
                f"Motor power : {COM_MOTOR_POWER}",
                f"Direction   : {COM_DIRECTION}"
            ]
            for i, txt in enumerate(com_texts):
                # on commence à y=80 puis +30 pixels par ligne
                y = 80 + i * 30
                cv2.putText(disp, txt, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            cv2.imshow(win, disp)
            
            sender.send(angle=COM_ANGLE, power=COM_MOTOR_POWER, direction=COM_DIRECTION)



        else:  # auto mode
            COM_MOTOR_POWER = 0
            if state['auto_phase'] == 'search':
                img = res.plot() if DRAW_SKELETON else frame.copy()
                for i, box in enumerate(boxes):
                    sc = score_ransac(xy_all[i], conf_all[i], HORIZONTAL_THRESHOLD_DEG)
                    color = (255, 0, 0) if sc >= SCORE_OK else (0, 255, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                
                cv2.imshow(win, img)

                tpose_scores = [
                    score_ransac(xy_all[j], conf_all[j], HORIZONTAL_THRESHOLD_DEG)
                    for j in range(len(boxes))
                ]
                if update_duration(any(sc >= SCORE_OK for sc in tpose_scores)):
                    tpose_index = int(np.argmax(tpose_scores))
                    tracks = update_tracker(boxes, confs, frame)
                    state['selected_id'] = select_tpose_target(
                        tracks, boxes, xy_all, conf_all, tpose_index
                    )
                    if state['selected_id']:
                        state['auto_phase'] = 'track'

            else:  # tracking phase
                tracks = update_tracker(boxes, confs, frame)
                ann = frame.copy()
                for t in tracks:
                    if not (t.is_confirmed() and t.track_id == state['selected_id']):
                        continue

                    # Si pas de box ni de keypoints, on skippe
                    if len(boxes) == 0 or len(xy_all) == 0:
                        continue

                    x1, y1, x2, y2 = map(int, t.to_ltrb())
                    # calcul des IoU avec les boîtes détectées
                    ious = [iou((x1, y1, x2, y2), b) for b in boxes]
                    if not ious:
                        continue

                    idx = int(np.argmax(ious))
                    # là on est sûrs que idx < len(xy_all)
                    xy, conf = xy_all[idx], conf_all[idx]
                    dist = estimate_distance_sum(xy, conf, focal_sum)

                    draw_overlay(ann, (x1, y1, x2, y2), dist)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    annotate_angle(ann, (cx, cy))

                    if COM_DISTANCE > 130:
                        COM_MOTOR_POWER = 100
                    else:
                        COM_MOTOR_POWER = 0

                    sender.send(angle=-(COM_ANGLE-90), power=COM_MOTOR_POWER, direction=COM_DIRECTION)

                    # Affichage des valeurs COM_ en haut à droite
                    com_texts = [
                        f"Angle       : {-(COM_ANGLE-90)}",
                        f"Motor power : {COM_MOTOR_POWER}",
                        f"Direction   : {COM_DIRECTION}"
                    ]
                    for i, txt in enumerate(com_texts):
                        y = 40 + (i+1)*30   # on décale chaque ligne de 30px
                        cv2.putText(ann, txt,
                                    (20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,  # taille un peu plus petite qu’en titre
                                    (255, 255, 255), 2)

                    break
                cv2.imshow(win, ann)

                if not any(t.track_id == state['selected_id'] and t.is_confirmed() for t in tracks):
                    state['auto_phase'] = 'search'
                    times['start'] = times['last'] = None
                    state['selected_id'] = None
                    

        if DISPLAY_PROCESS_TIME:
            print(f"[{state['mode'].upper()}:{state.get('auto_phase','')}] "
                  f"{(time.perf_counter()-now)*1000:.1f} ms")
            
        

    # fermeture de la source vidéo
    if USE_STREAM:
        close_stream()
    else:
        cap.release()
    cv2.destroyAllWindows()
    sender.close()

if __name__ == '__main__':
    run()
