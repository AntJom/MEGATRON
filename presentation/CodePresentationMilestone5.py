import cv2
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# =============================================================================
# Paramètres d'affichage (taille minimale pour nos fenêtres redimensionnables)
# =============================================================================
window_sizes = {
    "DeepSORT Tracking": (768, 432),
    "Center & Ratio": (768, 432),
    "YOLO Detections": (768, 432),
    "Candidate Tracking": (768, 432)
}

# =============================================================================
# Fonctions utilitaires pour le redimensionnement et l'affichage
# =============================================================================
def resize_with_aspect_ratio(image, target_width, target_height):
    """
    Redimensionne l'image pour qu'elle tienne entièrement dans (target_width, target_height)
    tout en conservant son ratio d'aspect (mode "contain" pour voir tout le FOV).
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def show_in_window(window_name, image):
    """
    Affiche 'image' dans la fenêtre 'window_name' en adaptant l'image aux dimensions
    actuelles de la zone d'affichage de la fenêtre tout en conservant son ratio.
    """
    try:
        # Récupère la taille actuelle de la zone d'affichage de la fenêtre
        x, y, win_width, win_height = cv2.getWindowImageRect(window_name)
    except Exception:
        win_width, win_height = window_sizes.get(window_name, (768, 432))
    
    # Si la taille retournée semble invalide, on utilise la taille par défaut
    def_width, def_height = window_sizes.get(window_name, (768, 432))
    if win_width < def_width or win_height < def_height:
        win_width, win_height = def_width, def_height

    # Redimensionnement de l'image en mode "contain" pour afficher tout le FOV
    resized_image = resize_with_aspect_ratio(image, win_width, win_height)
    
    # Création d'un canvas noir de la taille de la fenêtre
    canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    h_new, w_new = resized_image.shape[:2]
    
    # Centrage de l'image redimensionnée sur le canvas
    x_offset = (win_width - w_new) // 2
    y_offset = (win_height - h_new) // 2
    canvas[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = resized_image
    
    cv2.imshow(window_name, canvas)

# =============================================================================
# Configuration et initialisation
# =============================================================================
CLASSES_TO_DETECT = ["person"]
CONFIDENCE_THRESHOLD = 0.6  
TRACKING_CONFIRMATION_TIME = 4.0  # Durée pendant laquelle le ratio doit rester < 0.6 pour confirmer

# Initialisation du modèle YOLO
yoloModel = YOLO("yolov10x.pt")

# Initialisation de DeepSORT
tracker = DeepSort(max_age=50, n_init=10, nms_max_overlap=10, 
                   max_iou_distance=0.9, max_cosine_distance=0.6)

# Capture vidéo depuis la caméra
cap = cv2.VideoCapture(0)
if cap is None or not cap.isOpened():
    print("Warning: unable to open video source")
    exit(1)

# --- MODIFICATION : Définir une résolution plus large pour obtenir le FOV complet ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# =============================================================================
# Création et configuration des fenêtres (redimensionnables)
# =============================================================================
for name, (w, h) in window_sizes.items():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)

# Positionnement des fenêtres en grille (2 colonnes x 2 lignes)
cv2.moveWindow("YOLO Detections", 0, 0)                  # Haut gauche
cv2.moveWindow("DeepSORT Tracking", window_sizes["DeepSORT Tracking"][0], 0)            # Haut droite
cv2.moveWindow("Center & Ratio", 0, window_sizes["DeepSORT Tracking"][1])          # Bas gauche
cv2.moveWindow("Candidate Tracking", 
               window_sizes["DeepSORT Tracking"][0], 
               window_sizes["DeepSORT Tracking"][1])  # Bas droite

# =============================================================================
# Variables globales pour le suivi de la cible
# =============================================================================
candidate_id = None         # ID de la cible candidate (None si aucune)
candidate_start_time = 0    # Temps d'apparition de la candidate
tracking_confirmed = False  # Indique si la candidate est confirmée pour le tracking

# =============================================================================
# Variables globales pour le lissage du point de suivi
# =============================================================================
smoothed_state = {
    "x": 0,
    "y": 0,
    "vx": 0,
    "vy": 0,
    "initialized": False
}

# =============================================================================
# Fonctions d'affichage pour chaque fenêtre
# =============================================================================
def update_tracking_window(frame, tracks, candidate_id, tracking_confirmed):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        if candidate_id is not None and track_id == candidate_id and tracking_confirmed:
            # Pour la cible candidate confirmée : rectangle et texte en rouge, traits plus épais
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, f"Tracking Candidate (ID {track_id})", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Pour les autres personnes détectées : rectangle et texte en vert, traits plus épais
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"person ID {track_id}", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    show_in_window("DeepSORT Tracking", frame)

def update_center_ratio_window(frame, tracks):
    height, width, _ = frame.shape
    blank = 255 * np.ones((height, width, 3), dtype=np.uint8)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        box_width = x2 - x1
        box_height = y2 - y1
        ratio = box_height / box_width if box_width != 0 else 0
        # Augmentation de la taille du point et de la police
        cv2.circle(blank, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.putText(blank, f"{ratio:.2f}", (center_x + 10, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    show_in_window("Center & Ratio", blank)

def update_yolo_window(frame, yolo_detections_boxes):
    frame_yolo = frame.copy()
    for (x1, y1, x2, y2) in yolo_detections_boxes:
        # Rectangle et texte pour les détections YOLO avec des traits plus épais
        cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(frame_yolo, "YOLO", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    show_in_window("YOLO Detections", frame_yolo)

def update_candidate_tracking_window_with_smoothing(frame, candidate_track):
    blank = 255 * np.ones_like(frame, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, candidate_track.to_ltrb())
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Dessiner le point immédiat (en rouge) avec un cercle plus grand
    cv2.circle(blank, (center_x, center_y), 8, (0, 0, 255), -1)
    
    # Mettre à jour et dessiner le point lissé (en bleu)
    update_smoothed_candidate_point(blank, (center_x, center_y))
    
    show_in_window("Candidate Tracking", blank)

def update_smoothed_candidate_point(frame, target_center):
    global smoothed_state
    attraction = 0.02
    friction = 0.85
    threshold = 2
    
    target_x, target_y = target_center
    
    if not smoothed_state["initialized"]:
        smoothed_state["x"] = target_x
        smoothed_state["y"] = target_y
        smoothed_state["vx"] = 0
        smoothed_state["vy"] = 0
        smoothed_state["initialized"] = True

    old_x, old_y = smoothed_state["x"], smoothed_state["y"]

    dx = target_x - smoothed_state["x"]
    dy = target_y - smoothed_state["y"]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist > threshold:
        direction_x = dx / dist
        direction_y = dy / dist
        force = dist * attraction
        smoothed_state["vx"] += direction_x * force
        smoothed_state["vy"] += direction_y * force
    else:
        smoothed_state["x"] = target_x
        smoothed_state["y"] = target_y
        smoothed_state["vx"] = 0
        smoothed_state["vy"] = 0

    smoothed_state["vx"] *= friction
    smoothed_state["vy"] *= friction

    smoothed_state["x"] += smoothed_state["vx"]
    smoothed_state["y"] += smoothed_state["vy"]

    vec_old = np.array([target_x - old_x, target_y - old_y])
    vec_new = np.array([target_x - smoothed_state["x"], target_y - smoothed_state["y"]])
    if np.dot(vec_old, vec_new) <= 0:
        smoothed_state["x"] = target_x
        smoothed_state["y"] = target_y
        smoothed_state["vx"] = 0
        smoothed_state["vy"] = 0

    x_lisse = int(smoothed_state["x"])
    y_lisse = int(smoothed_state["y"])
    # Point lissé en bleu avec un cercle agrandi
    cv2.circle(frame, (x_lisse, y_lisse), 8, (255, 0, 0), -1)

    height, width, _ = frame.shape
    ref_point = (width // 2, height)
    # Point de référence en vert
    cv2.circle(frame, ref_point, 8, (0, 255, 0), -1)

    dir_x = x_lisse - ref_point[0]
    dir_y = y_lisse - ref_point[1]
    magnitude = np.sqrt(dir_x**2 + dir_y**2)
    
    if magnitude > 1e-3:
        unit_dir_x = dir_x / magnitude
        unit_dir_y = dir_y / magnitude
        small_line_length = 40  # Allongement de la ligne pour plus de visibilité
        endpoint = (int(ref_point[0] + unit_dir_x * small_line_length),
                    int(ref_point[1] + unit_dir_y * small_line_length))
        cv2.line(frame, ref_point, endpoint, (255, 0, 255), 3)
        angle_deg = np.degrees(np.arctan2(dir_x, -dir_y))
        cv2.putText(frame, f"{angle_deg:.1f} deg", (endpoint[0] + 10, endpoint[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

# =============================================================================
# Boucle principale
# =============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo depuis la caméra.")
        break

    # Détection avec YOLOv8
    yoloResult = yoloModel(frame)
    detections = []             # Format attendu par DeepSORT : [ [x, y, largeur, hauteur], score, class_id ]
    yolo_detections_boxes = []  # Pour affichage : (x1, y1, x2, y2)
    
    for result in yoloResult[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        class_id = int(result.cls[0])
        class_name = yoloModel.names[class_id] if hasattr(yoloModel, 'names') else str(class_id)
        
        if class_name in CLASSES_TO_DETECT and conf >= CONFIDENCE_THRESHOLD:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
            yolo_detections_boxes.append((x1, y1, x2, y2))
    
    # Mise à jour du tracker DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # Actualisation des différentes fenêtres d'affichage
    update_tracking_window(frame.copy(), tracks, candidate_id, tracking_confirmed)
    update_center_ratio_window(frame, tracks)
    update_yolo_window(frame, yolo_detections_boxes)

    # --- LOGIQUE DE SÉLECTION ET DE TRACKING D'UNE SEULE CIBLE ---
    candidate_track_obj = None
    candidate_found = False

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        box_width = x2 - x1
        box_height = y2 - y1
        ratio = box_height / box_width if box_width != 0 else 0

        hlRatio = 0.83

        if candidate_id is None:
            if ratio < hlRatio:
                candidate_id = track.track_id
                candidate_start_time = time.time()
                candidate_track_obj = track
                candidate_found = True
                break
        else:
            if track.track_id == candidate_id:
                candidate_found = True
                candidate_track_obj = track
                if not tracking_confirmed:
                    if ratio < hlRatio:
                        if time.time() - candidate_start_time >= TRACKING_CONFIRMATION_TIME:
                            tracking_confirmed = True
                    else:
                        candidate_id = None
                        candidate_start_time = 0
                        tracking_confirmed = False
                break

    if not candidate_found:
        candidate_id = None
        candidate_start_time = 0
        tracking_confirmed = False
        smoothed_state["initialized"] = False

    if tracking_confirmed and candidate_track_obj is not None:
        update_candidate_tracking_window_with_smoothing(frame.copy(), candidate_track_obj)
    else:
        blank_candidate = 255 * np.ones_like(frame, dtype=np.uint8)
        cv2.putText(blank_candidate, "No Candidate Tracking", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        show_in_window("Candidate Tracking", blank_candidate)

    key = cv2.waitKey(10)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
