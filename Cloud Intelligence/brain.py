import cv2
import numpy as np
import time
import serial
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import mediapipe as mp

# =============================================================================
# Paramètres d'affichage
# =============================================================================
window_sizes = {
    "DeepSORT Tracking": (768, 432),
    "Center & Ratio": (768, 432),
    "YOLO Detections": (768, 432),
    "Candidate Tracking": (768, 432)
}

# =============================================================================
# Variables globales
# =============================================================================
last_com_send_time = 0
candidate_id = None         
candidate_start_time = 0    
tracking_confirmed = False  

smoothed_state = {
    "x": 0,
    "y": 0,
    "vx": 0,
    "vy": 0,
    "initialized": False
}

# =============================================================================
# Fonctions utilitaires d'affichage et de redimensionnement
# =============================================================================

def resize_with_aspect_ratio(image, target_width, target_height):
    """
    Redimensionne une image sans déformer son ratio.

    Entrées:
        image (np.ndarray): Image source.
        target_width (int): Largeur maximale souhaitée.
        target_height (int): Hauteur maximale souhaitée.

    Sortie:
        resized_image (np.ndarray): Image redimensionnée pour tenir dans la zone cible tout en conservant le ratio.

    Cette fonction calcule les nouvelles dimensions en fonction du ratio largeur/hauteur d'origine
    et applique un redimensionnement bilinéaire.
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
    Affiche une image dans une fenêtre avec centrage et redimensionnement automatique.

    Entrées:
        window_name (str): Nom de la fenêtre OpenCV.
        image (np.ndarray): Image à afficher.

    Sortie:
        None (affiche la fenêtre directement).

    Cette fonction récupère la taille de la fenêtre, ajuste l'image via resize_with_aspect_ratio,
    crée un canevas noir aux dimensions de la fenêtre, centre l'image redimensionnée,
    puis utilise cv2.imshow pour l'affichage.
    """
    try:
        x, y, win_width, win_height = cv2.getWindowImageRect(window_name)
    except Exception:
        win_width, win_height = window_sizes.get(window_name, (768, 432))
    def_width, def_height = window_sizes.get(window_name, (768, 432))
    if win_width < def_width or win_height < def_height:
        win_width, win_height = def_width, def_height
    resized_image = resize_with_aspect_ratio(image, win_width, win_height)
    canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    h_new, w_new = resized_image.shape[:2]
    x_offset = (win_width - w_new) // 2
    y_offset = (win_height - h_new) // 2
    canvas[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = resized_image
    cv2.imshow(window_name, canvas)

# =============================================================================
# Fonction de traitement d'angle pour le servo
# =============================================================================

def angle(input_angle):
    """
    Transforme un angle de visée en commande pour servo-moteur.

    Entrée:
        input_angle (float): Angle calculé en degrés autour de l'axe vertical.

    Sortie:
        servo_angle (float): Valeur centrée autour de 90°, bornée et mise à l'échelle pour le servo.

    La fonction borne l'angle à [-33, 33] puis le convertit en plage [90 - 33*(8/6), 90 + 33*(8/6)].
    """
    clamped = max(-33, min(33, input_angle))
    return (90 + clamped * (8/6))

# =============================================================================
# Fonctions d'affichage pour le suivi (YOLO / DeepSORT)
# =============================================================================

def update_tracking_window(frame, tracks, candidate_id, tracking_confirmed):
    """
    Dessine les boîtes de suivi DeepSORT et indique le candidat si confirmé.

    Entrées:
        frame (np.ndarray): Image sur laquelle dessiner les boîtes.
        tracks (list): Liste d'objets track renvoyés par DeepSort.
        candidate_id (int | None): ID du track sélectionné comme candidat.
        tracking_confirmed (bool): Indique si le suivi du candidat est confirmé après délai.

    Sortie:
        None (affiche la fenêtre de suivi).

    Parcourt les tracks confirmés, dessine en rouge la boîte du candidat confirmé,
    et en vert les autres personnes, puis affiche avec show_in_window.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        if candidate_id is not None and track_id == candidate_id and tracking_confirmed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, f"Tracking Candidate (ID {track_id})", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"person ID {track_id}", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    show_in_window("DeepSORT Tracking", frame)


def update_center_ratio_window(frame, tracks):
    """
    Affiche le centre de chaque bbox et leur ratio hauteur/largeur.

    Entrées:
        frame (np.ndarray): Image source pour calculs de dimensions.
        tracks (list): Liste d'objets track DeepSort.

    Sortie:
        None (affiche une fenêtre avec points et ratios).

    Crée un canevas blanc de la taille de l'image, calcule centre et ratio pour chaque track,
    dessine un cercle rouge au centre et affiche le ratio à côté.
    """
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
        cv2.circle(blank, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.putText(blank, f"{ratio:.2f}", (center_x + 10, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    show_in_window("Center & Ratio", blank)


def update_yolo_window(frame, yolo_detections_boxes):
    """
    Affiche les détections YOLO sur l'image.

    Entrées:
        frame (np.ndarray): Image sur laquelle dessiner.
        yolo_detections_boxes (list of tuples): Liste de boîtes (x1, y1, x2, y2).

    Sortie:
        None (affiche les boîtes YOLO dans une fenêtre dédiée).

    Clone l'image, dessine chaque bbox en jaune avec le label "YOLO", puis affiche.
    """
    frame_yolo = frame.copy()
    for (x1, y1, x2, y2) in yolo_detections_boxes:
        cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(frame_yolo, "YOLO", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    show_in_window("YOLO Detections", frame_yolo)


def update_candidate_tracking_window_with_smoothing(frame, candidate_track):
    """
    Affiche la position lissée du candidat dans une fenêtre dédiée.

    Entrées:
        frame (np.ndarray): Image de référence pour les dimensions.
        candidate_track (Track object): Objet track du candidat.

    Sortie:
        None (affiche la trajectoire lissée).

    Extrait le centre du bbox du candidat, met à jour l'état lissé via update_smoothed_candidate_point,
    puis affiche le résultat sur un canevas blanc.
    """
    blank = 255 * np.ones_like(frame, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, candidate_track.to_ltrb())
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    cv2.circle(blank, (center_x, center_y), 8, (0, 0, 255), -1)
    update_smoothed_candidate_point(blank, (center_x, center_y))
    show_in_window("Candidate Tracking", blank)


def update_smoothed_candidate_point(frame, target_center):
    """
    Calcule et trace la position lissée du point cible avec envoi au servo.

    Entrées:
        frame (np.ndarray): Canevas sur lequel dessiner.
        target_center (tuple): Coordonnées (x, y) du point à lisser.

    Sortie:
        None (dessine sur frame et envoie angle si nécessaire).

    Implémente un modèle de masse-ressort avec friction pour lisser le mouvement,
    trace l'ancien et nouveau vecteur, calcule l'angle pour le servo,
    et envoie la commande via port série si >1s depuis dernier envoi.
    """
    global smoothed_state, ser, last_com_send_time
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
    cv2.circle(frame, (x_lisse, y_lisse), 8, (255, 0, 0), -1)
    height, width, _ = frame.shape
    ref_point = (width // 2, height)
    cv2.circle(frame, ref_point, 8, (0, 255, 0), -1)
    dir_x = x_lisse - ref_point[0]
    dir_y = y_lisse - ref_point[1]
    magnitude = np.sqrt(dir_x**2 + dir_y**2)
    if magnitude > 1e-3:
        unit_dir_x = dir_x / magnitude
        unit_dir_y = dir_y / magnitude
        small_line_length = 40
        endpoint = (int(ref_point[0] + unit_dir_x * small_line_length),
                    int(ref_point[1] + unit_dir_y * small_line_length))
        cv2.line(frame, ref_point, endpoint, (255, 0, 255), 3)
        angle_deg = np.degrees(np.arctan2(dir_x, -dir_y))
        cv2.putText(frame, f"{angle_deg:.1f} deg", (endpoint[0] + 10, endpoint[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        # Traitement de l'angle pour le servo
        processed_angle = angle(angle_deg)
        current_time = time.time()
        if current_time - last_com_send_time >= 1:
            if ser is not None and ser.is_open:
                try:
                    message = f"{processed_angle:.1f}\n"
                    ser.write(message.encode('utf-8'))
                    last_com_send_time = current_time
                except Exception as e:
                    print("Erreur lors de l'envoi via le port COM:", e)
        cv2.putText(frame, f"Angle: {processed_angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# =============================================================================
# Calibration via une image de référence (Mediapipe)
# =============================================================================

def calibrate_from_reference(ref_image_path, known_height_cm, known_distance_cm):
    """
    Calibre la caméra à partir d'une image de référence en calculant la focale.

    Entrées:
        ref_image_path (str): Chemin vers l'image de référence contenant une pose humaine.
        known_height_cm (float): Hauteur réelle de la personne en cm (envergure bras étendus).
        known_distance_cm (float): Distance de la personne à la caméra en cm.

    Sortie:
        focal_length (float | None): Focale effective calculée en pixels, ou None si échec.

    Utilise Mediapipe pour détecter la pose, extrait la distance pixel entre poignets,
    puis applique le modèle du trou de serrure (pinhole) pour estimer la focale.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        print("Erreur : Impossible de charger l'image de référence.")
        return None
    height_img, width_img, _ = ref_image.shape

    image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        print("Aucune pose détectée dans l'image de référence.")
        return None

    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_x = left_wrist.x * width_img
    left_y = left_wrist.y * height_img
    right_x = right_wrist.x * width_img
    right_y = right_wrist.y * height_img

    arm_span_pixels = math.dist((left_x, left_y), (right_x, right_y))
    print(f"Envergure dans l'image de référence : {arm_span_pixels:.2f} pixels")

    focal_length = (arm_span_pixels * known_distance_cm) / known_height_cm
    print(f"Focale calculée : {focal_length:.2f} pixels")

    pose.close()
    return focal_length

# =============================================================================
# Partie principale : fusion des deux traitements
# =============================================================================
if __name__ == "__main__":
    # --- Calibration par image de référence ---
    reference_image_path = "reference.jpg"
    known_height_cm = 175
    known_distance_cm = 222

    focal_length = calibrate_from_reference(reference_image_path, known_height_cm, known_distance_cm)
    if focal_length is None:
        print("Calibration échouée.")
        exit(1)

    # --- Initialisation YOLO et DeepSORT ---
    CLASSES_TO_DETECT = ["person"]
    CONFIDENCE_THRESHOLD = 0.6  
    TRACKING_CONFIRMATION_TIME = 4.0

    yoloModel = YOLO("yolo11n.pt")
    tracker = DeepSort(max_age=50, n_init=10, nms_max_overlap=10, 
                       max_iou_distance=0.9, max_cosine_distance=0.6)

    try:
        ser = serial.Serial('COM5', 9600)
        time.sleep(2)  # Attente de la réinitialisation de l'Arduino
    except Exception as e:
        print("Erreur lors de l'ouverture du port COM:", e)
        ser = None

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("Warning: unable to open video source")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # --- Création des fenêtres d'affichage ---
    for name, (w, h) in window_sizes.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h)

    cv2.moveWindow("YOLO Detections", 0, 0)
    cv2.moveWindow("DeepSORT Tracking", window_sizes["DeepSORT Tracking"][0], 0)
    cv2.moveWindow("Center & Ratio", 0, window_sizes["DeepSORT Tracking"][1])
    cv2.moveWindow("Candidate Tracking", window_sizes["DeepSORT Tracking"][0], window_sizes["DeepSORT Tracking"][1])
    cv2.namedWindow("Distance Estimation", cv2.WINDOW_NORMAL)

    # --- Initialisation de Mediapipe pour la pose ---
    mp_pose_instance = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Impossible de lire la vidéo depuis la caméra.")
                break

            # ---------------------------
            # Traitement par YOLO / DeepSORT
            # ---------------------------
            frame_tracking = frame.copy()
            yoloResult = yoloModel(frame_tracking)
            detections = []              # Format: ([x, y, w, h], conf, class_id)
            yolo_detections_boxes = []   # Format: (x1, y1, x2, y2)

            for result in yoloResult[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                conf = result.conf[0].item()
                class_id = int(result.cls[0])
                class_name = yoloModel.names[class_id] if hasattr(yoloModel, 'names') else str(class_id)
                if class_name in CLASSES_TO_DETECT and conf >= CONFIDENCE_THRESHOLD:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                    yolo_detections_boxes.append((x1, y1, x2, y2))

            tracks = tracker.update_tracks(detections, frame=frame_tracking)
            update_tracking_window(frame_tracking.copy(), tracks, candidate_id, tracking_confirmed)
            update_center_ratio_window(frame_tracking, tracks)
            update_yolo_window(frame_tracking, yolo_detections_boxes)

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
                update_candidate_tracking_window_with_smoothing(frame_tracking.copy(), candidate_track_obj)
            else:
                blank_candidate = 255 * np.ones_like(frame_tracking, dtype=np.uint8)
                cv2.putText(blank_candidate, "No Candidate Tracking", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                show_in_window("Candidate Tracking", blank_candidate)

            # ---------------------------
            # Traitement de la pose (Mediapipe) et estimation de distance
            # ---------------------------
            frame_pose = frame.copy()
            image_rgb = cv2.cvtColor(frame_pose, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_pose = mp_pose_instance.process(image_rgb)
            image_rgb.flags.writeable = True
            image_pose = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            height_img, width_img, _ = image_pose.shape

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                def to_pixel(pt):
                    return (int(pt.x * width_img), int(pt.y * height_img))

                lsh = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER])
                rsh = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
                leb = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW])
                reb = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW])
                lwr = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST])
                rwr = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST])
                lhip = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP])
                rhip = to_pixel(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])

                def eucl_dist(a, b):
                    return math.dist(a, b)

                left_upper_arm = eucl_dist(lsh, leb)
                left_forearm = eucl_dist(leb, lwr)
                right_upper_arm = eucl_dist(rsh, reb)
                right_forearm = eucl_dist(reb, rwr)
                left_trunk = eucl_dist(lsh, lhip)
                right_trunk = eucl_dist(rsh, rhip)
                shoulder_width = eucl_dist(lsh, rsh)

                distances = [
                    left_upper_arm,
                    left_forearm,
                    right_upper_arm,
                    right_forearm,
                    left_trunk,
                    right_trunk,
                    shoulder_width
                ]
                global_distance = np.median(distances)
                if global_distance != 0:
                    ratio_distance = 1.6
                    estimated_distance = ((known_height_cm * focal_length) / global_distance ) * ratio_distance
                else:
                    estimated_distance = 0

                cv2.putText(image_pose, f"Distance: {estimated_distance:.2f} cm", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                for ptA, ptB, color in [
                    (lsh, leb, (255, 0, 0)),
                    (leb, lwr, (0, 255, 0)),
                    (rsh, reb, (255, 0, 0)),
                    (reb, rwr, (0, 255, 0)),
                    (lsh, lhip, (0, 165, 255)),
                    (rsh, rhip, (0, 165, 255)),
                    (lsh, rsh, (0, 255, 255))
                ]:
                    cv2.line(image_pose, ptA, ptB, color, 3)
                    cv2.circle(image_pose, ptA, 5, (0, 0, 255), -1)
                    cv2.circle(image_pose, ptB, 5, (0, 0, 255), -1)

                mp_drawing.draw_landmarks(image_pose, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            else:
                cv2.putText(image_pose, "Aucune pose detectee", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Distance Estimation", image_pose)

            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None and ser.is_open:
            ser.close()
        mp_pose_instance.close()
