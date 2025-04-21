import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def init_video_capture(source: int = 0) -> cv2.VideoCapture:
    """
    Initialise et retourne un VideoCapture pour la webcam.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def load_pose_model(model_path: str = 'yolo11x-pose.pt') -> YOLO:
    """
    Charge et retourne le modèle YOLO-pose.
    """
    return YOLO(model_path)

def compute_segment_lengths(xy: np.ndarray, conf: np.ndarray, conf_thresh: float = 0.2) -> dict:
    """
    Calcule les longueurs pixel des segments : 
    poignet-coude, coude-épaule, épaule-épaule pour chaque côté.
    """
    segments = {}
    # Indices COCO
    LSH, RSH = 5, 6
    LEL, REL = 7, 8
    LWR, RWR = 9, 10
    
    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    # Gauche
    if conf[LWR] > conf_thresh and conf[LEL] > conf_thresh:
        segments['Left Wrist-Elbow'] = dist(xy[LWR], xy[LEL])
    if conf[LEL] > conf_thresh and conf[LSH] > conf_thresh:
        segments['Left Elbow-Shoulder'] = dist(xy[LEL], xy[LSH])
    if conf[LSH] > conf_thresh and conf[RSH] > conf_thresh:
        segments['Shoulder-Shoulder'] = dist(xy[LSH], xy[RSH])
    
    # Droite
    if conf[RWR] > conf_thresh and conf[REL] > conf_thresh:
        segments['Right Wrist-Elbow'] = dist(xy[RWR], xy[REL])
    if conf[REL] > conf_thresh and conf[RSH] > conf_thresh:
        segments['Right Elbow-Shoulder'] = dist(xy[REL], xy[RSH])
    # épaule-épaule déjà compté
    
    return segments

def setup_realtime_plot():
    """
    Initialise le plot matplotlib en mode interactif.
    """
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Frame')
    ax.set_ylabel('Longueur (px)')
    ax.set_title('Taille des segments du bras en temps réel')
    lines = {}
    for label in ['Left Wrist-Elbow', 'Left Elbow-Shoulder', 
                  'Shoulder-Shoulder', 'Right Wrist-Elbow', 'Right Elbow-Shoulder']:
        line, = ax.plot([], [], label=label)
        lines[label] = line
    ax.legend()
    return fig, ax, lines

def update_plot(ax, lines, data, frame_idx):
    """
    Met à jour le plot avec les nouvelles longueurs.
    """
    for label, values in data.items():
        line = lines[label]
        xdata = list(range(len(values)))
        line.set_data(xdata, values)
    ax.relim()
    ax.autoscale_view()

def run_pose_and_plot(source: int = 0, model_path: str = 'yolo11x-pose.pt'):
    cap = init_video_capture(source)
    model = load_pose_model(model_path)
    fig, ax, lines = setup_realtime_plot()

    # Structure pour stocker les longueurs
    data = {label: [] for label in lines.keys()}
    frame_idx = 0
    window_name = "Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            res = model(frame, verbose=False)[0]
            annotated = res.plot() if hasattr(res, 'plot') else frame

            # Extraction keypoints
            if res.keypoints is not None:
                xy = res.keypoints.xy.cpu().numpy()[0]
                conf = res.keypoints.conf.cpu().numpy()[0]
                segs = compute_segment_lengths(xy, conf)
                # Append aux listes
                for label, values in data.items():
                    values.append(segs.get(label, np.nan))
                # Mise à jour du plot
                update_plot(ax, lines, data, frame_idx)
                plt.pause(0.001)

            # Affichage vidéo
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Laisser le graphique ouvert
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    run_pose_and_plot(0, 'yolo11x-pose.pt')

