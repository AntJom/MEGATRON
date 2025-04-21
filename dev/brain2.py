import os
import sys
import cv2
import math
import numpy as np
import time
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

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

# Global state
times = {'start': None, 'last': None, 'prev_frame': None}
state = {'mode': 'manual', 'auto_phase': 'search', 'selected_id': None}

cap = None
model = None
tracker = None
calibration_images = []
focal_length = 0.0
ref_span_cm = 0.0

JOINT_INDICES = (7, 9, 8, 10)

# ——————————————————————————————————
# INITIALIZATION
# ——————————————————————————————————
def init_camera(source=0, buffer_size=1):
    global cap
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

def load_calibration():
    global calibration_images
    os.makedirs(CALIB_DIR, exist_ok=True)
    calibration_images = []
    for fn in os.listdir(CALIB_DIR):
        if fn.lower().endswith('.jpg'):
            parts = fn[:-4].split('_')
            if len(parts) == 3:
                _, sc, dc = parts
                try:
                    span = float(sc.replace('cm',''))
                    dist = float(dc.replace('cm',''))
                except:
                    continue
                calibration_images.append((os.path.join(CALIB_DIR, fn), span, dist))

def compute_pixel_span(path):
    img = cv2.imread(path)
    if img is None:
        return None
    _, xy, conf, _, _ = infer_pose(img)
    spans = [np.hypot(pts[10][0]-pts[9][0], pts[10][1]-pts[9][1])
             for pts, cf in zip(xy, conf) if cf[9]>0.2 and cf[10]>0.2]
    return max(spans) if spans else None

def init_calibration():
    global focal_length, ref_span_cm
    load_calibration()
    f_vals, spans = [], []
    for path, span_cm, dist_cm in calibration_images:
        px = compute_pixel_span(path)
        if px and px>1e-3:
            f_vals.append(px * dist_cm / span_cm)
            spans.append(span_cm)
    if not f_vals:
        focal_length, ref_span_cm = 0.0, 0.0
    else:
        median_f = np.median(f_vals)
        filt = [(f_vals[i], spans[i]) for i in range(len(f_vals)) if abs(f_vals[i]-median_f)/median_f<0.3]
        if filt:
            fs, ss = zip(*filt)
            focal_length, ref_span_cm = float(np.mean(fs)), float(np.mean(ss))
        else:
            focal_length, ref_span_cm = float(median_f), float(np.median(spans))

# ——————————————————————————————————
# PROCESSING FUNCTIONS
# ——————————————————————————————————
def read_frame():
    return cap.read()

def infer_pose(frame):
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or not res.boxes:
        return res, np.zeros((0,17,2)), np.zeros((0,17)), np.zeros((0,4),int), np.array([])
    xy = res.keypoints.xy.cpu().numpy()
    conf = res.keypoints.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes,'conf') else np.ones(len(boxes))
    return res, xy, conf, boxes, confs

def score_ransac(xy, conf, thresh_deg):
    pts = [tuple(xy[i]) for i in JOINT_INDICES if conf[i]>0.2]
    if len(pts)<2: return 0.0
    best=0
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            p1,p2=pts[i],pts[j]
            ang=abs(math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0])))
            ang=min(ang,180-ang)
            if ang>thresh_deg: continue
            inl=sum(min(abs(math.degrees(math.atan2(p[1]-p1[1],p[0]-p1[0]))),
                       180-abs(math.degrees(math.atan2(p[1]-p1[1],p[0]-p1[0]))))<=thresh_deg
                    for p in pts)
            best=max(best,inl)
    return 100.0*best/len(pts)

def update_duration(is_tpose):
    now=time.time()
    if not is_tpose:
        times['start']=times['last']=None
        return False
    if times['last'] and now-times['last']>FRAME_GAP:
        times['start']=now
    if not times['start']:
        times['start']=now
    times['last']=now
    return (now-times['start'])>=HOLD_TIME

def update_tracker(boxes, confs, frame):
    dets=[([int(x1),int(y1),int(x2-x1),int(y2-y1)], float(c), 'person')
          for (x1,y1,x2,y2),c in zip(boxes,confs)]
    return tracker.update_tracks(dets, frame=frame)

def compute_angle(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return abs(np.degrees(np.arctan2(dy, dx)))

def annotate_angle(frame, center):
    h,w=frame.shape[:2]
    ref=(w//2,h)
    cv2.line(frame, ref, center, (255,0,255),2)
    ang=compute_angle(center, ref)
    cv2.putText(frame, f"{ang:.1f}°", (ref[0]+10,ref[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
    return ang

def estimate_distance(xy, conf):
    spans=[np.hypot(pts[10][0]-pts[9][0],pts[10][1]-pts[9][1])
           for pts,cf in zip(xy,conf) if cf[9]>0.2 and cf[10]>0.2]
    if not spans or focal_length<=0:
        return None
    return float(focal_length*ref_span_cm/max(spans))

def select_tpose_target(tracks, boxes, xy, conf):
    for t in tracks:
        if not t.is_confirmed(): continue
        tb=tuple(map(int,t.to_ltrb()))
        ious=np.array([iou(tb,b) for b in boxes])
        if ious.size==0 or ious.max()<IOU_MAPPING_THRESHOLD: continue
        return t.track_id
    return None

def iou(boxA, boxB):
    xA,yA=max(boxA[0],boxB[0]),max(boxA[1],boxB[1])
    xB,yB=min(boxA[2],boxB[2]),min(boxA[3],boxB[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    aA=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    aB=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter/(aA+aB-inter) if (aA+aB-inter)>0 else 0.0

# Calibration helper (replaces missing calibrate_step)


# Ajouter avant run()
def calibrate_step(frame, xy, conf, boxes):
    """Effectue une étape de calibration en mode CONFIG"""
    ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    xl, yl = int(xy[0][9][0]), int(xy[0][9][1])
    xr, yr = int(xy[0][10][0]), int(xy[0][10][1])
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
    win = 'T-pose Detector'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    init_camera()
    init_model()
    init_tracker()
    init_calibration()
    times['prev_frame'] = time.perf_counter()

    while True:
        now = time.perf_counter()
        ret, frame = read_frame()
        if not ret: break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            state['mode'] = 'manual'
        elif key == ord('a'):
            state['mode'] = 'auto'
            state['auto_phase'] = 'search'
            times['start'] = times['last'] = None
        elif key == ord('c'):
            state['mode'] = 'config'
        elif key in (ord('q'), 27):
            break

        if state['mode'] in ('config', 'auto'):
            res, xy, conf, boxes, confs = infer_pose(frame)
        if state['mode'] == 'config':
            if update_duration(score_ransac(xy[0], conf[0], HORIZONTAL_THRESHOLD_DEG) >= SCORE_OK if len(xy)>0 else False):
                calibrate_step(frame, xy, conf, boxes)
            img = res.plot() if DRAW_SKELETON else frame
            cv2.imshow(win, img)

        elif state['mode'] == 'manual':
            disp = frame.copy()
            cv2.putText(disp, 'MODE MANUEL', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
            cv2.imshow(win, disp)

        else:  # auto
            if state['auto_phase'] == 'search':
                img = res.plot() if DRAW_SKELETON else frame.copy()
                for i, box in enumerate(boxes):
                    sc = score_ransac(xy[i], conf[i], HORIZONTAL_THRESHOLD_DEG)
                    color = (0,255,0) if sc>=SCORE_OK else (0,255,255)
                    x1,y1,x2,y2 = box
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,2)
                cv2.imshow(win, img)
                if update_duration(any(score_ransac(xy[j], conf[j], HORIZONTAL_THRESHOLD_DEG)>=SCORE_OK for j in range(len(boxes)))):
                    tracks = update_tracker(boxes, confs, frame)
                    state['selected_id'] = select_tpose_target(tracks, boxes, xy, conf)
                    if state['selected_id']:
                        state['auto_phase'] = 'track'
            else:
                tracks = update_tracker(boxes, confs, frame)
                ann = frame.copy()
                for t in tracks:
                    if t.is_confirmed() and t.track_id==state['selected_id']:
                        x1,y1,x2,y2 = map(int, t.to_ltrb())
                        cv2.rectangle(ann,(x1,y1),(x2,y2),(255,0,0),2)
                        dist = estimate_distance(xy, conf)
                        if dist:
                            cv2.putText(ann, f"{dist:.1f} cm", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                        cx,cy=(x1+x2)//2,(y1+y2)//2
                        annotate_angle(ann,(cx,cy))
                        break
                cv2.imshow(win, ann)
                if not any(t.track_id==state['selected_id'] and t.is_confirmed() for t in tracks):
                    state['auto_phase']='search'
                    times['start']=times['last']=None
                    state['selected_id']=None

        if DISPLAY_PROCESS_TIME:
            print(f"[{state['mode'].upper()}:{state.get('auto_phase','')}] {(time.perf_counter()-now)*1000:.1f} ms")

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    run()
