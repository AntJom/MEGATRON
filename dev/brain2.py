import os
import sys
import cv2
import math
import numpy as np
import time
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ——————————————————————————————————
# INDICES & RANSAC
# ——————————————————————————————————
JOINT_INDICES = (7, 9, 8, 10)  # épaules + poignets

def score_ransac(xy: np.ndarray, conf: np.ndarray, thresh_deg: float) -> float:
    pts = [tuple(xy[i]) for i in JOINT_INDICES if conf[i] > 0.2]
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
                min(
                    abs(math.degrees(math.atan2(p[1]-p1[1], p[0]-p1[0]))),
                    180 - abs(math.degrees(math.atan2(p[1]-p1[1], p[0]-p1[0])))
                ) <= thresh_deg
                for p in pts
            )
            best = max(best, inliers)
    return 100.0 * best / len(pts)

@dataclass
class Config:
    model_path: str = 'yolo11x-pose.pt'
    calib_dir: str = 'calib_data'
    draw_skeleton: bool = True
    draw_boxes: bool = True
    display_process_time: bool = True
    horizontal_threshold_deg: float = 15.0
    score_ok: float = 80.0
    deepsort_max_age: int = 50
    deepsort_n_init: int = 10
    deepsort_nms_max_overlap: float = 0.3
    deepsort_max_iou_distance: float = 0.9
    deepsort_max_cosine_distance: float = 0.6
    iou_mapping_threshold: float = 0.5
    smooth_attraction: float = 0.02
    smooth_friction: float = 0.85
    smooth_threshold: float = 2.0
    config_hold_sec: float = 5.0
    config_frame_gap: float = 0.5
    config_margin_px: int = 50

class Mode(Enum):
    MANUAL = 'manual'
    AUTO   = 'auto'
    CONFIG = 'config'

class CameraHandler:
    def __init__(self, source: int = 0, buffer_size: int = 1):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): sys.exit(f"Impossible d'ouvrir le flux vidéo: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    def release(self) -> None:
        self.cap.release()

class YOLOHandler:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    def infer(self, frame: np.ndarray):
        res = self.model(frame, verbose=False)[0]
        if res.keypoints is None or not res.boxes:
            return res, np.zeros((0,17,2)), np.zeros((0,17)), np.zeros((0,4),int), np.array([])
        xy    = res.keypoints.xy.cpu().numpy()
        conf  = res.keypoints.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes,'conf') else np.ones(len(boxes))
        return res, xy, conf, boxes, confs

class TPoseDurationChecker:
    def __init__(self, hold_time: float, max_gap: float):
        self.hold_time = hold_time
        self.max_gap = max_gap
        self.start_time: Optional[float] = None
        self.last_time: Optional[float] = None
    def update(self, is_tpose: bool) -> bool:
        now = time.time()
        if not is_tpose:
            self.start_time = None
            self.last_time = None
            return False
        if self.last_time and now - self.last_time > self.max_gap:
            self.start_time = now
        if self.start_time is None:
            self.start_time = now
        self.last_time = now
        return (now - self.start_time) >= self.hold_time

class DeepSortHandler:
    def __init__(self, cfg: Config):
        self.tracker = DeepSort(
            max_age=cfg.deepsort_max_age,
            n_init=cfg.deepsort_n_init,
            nms_max_overlap=cfg.deepsort_nms_max_overlap,
            max_iou_distance=cfg.deepsort_max_iou_distance,
            max_cosine_distance=cfg.deepsort_max_cosine_distance
        )
    def update(self, boxes: np.ndarray, confs: np.ndarray, frame: np.ndarray):
        dets = [([int(x1),int(y1),int(x2-x1),int(y2-y1)], float(c), 'person')
                for (x1,y1,x2,y2), c in zip(boxes, confs)]
        return self.tracker.update_tracks(dets, frame=frame)

class AngleCalculator:
    @staticmethod
    def compute(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        return abs(np.degrees(np.arctan2(dy, dx)))
    @staticmethod
    def annotate(frame: np.ndarray, center: Tuple[int,int]):
        h, w = frame.shape[:2]
        ref = (w//2, h)
        cv2.line(frame, ref, center, (255,0,255), 2)
        ang = AngleCalculator.compute(center, ref)
        cv2.putText(frame, f"{ang:.1f}°", (ref[0]+10, ref[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2)
        return ang

class CalibrationHandler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.calib_dir, exist_ok=True)
        self.calibs = self._load_calibrations()
        self.focal_length, self.ref_span_cm = self._compute_focal_and_refspan()
    def _load_calibrations(self) -> List[Dict]:
        out = []
        for fn in os.listdir(self.cfg.calib_dir):
            if not fn.lower().endswith('.jpg'): continue
            parts = fn[:-4].split('_')
            if len(parts) != 3: continue
            _, sc, dc = parts
            try:
                span_cm = float(sc.replace('cm',''))
                dist_cm = float(dc.replace('cm',''))
            except ValueError:
                continue
            out.append({'path': os.path.join(self.cfg.calib_dir,fn),'span_cm':span_cm,'dist_cm':dist_cm})
        return out
    def _get_pixel_span(self,path:str)->Optional[float]:
        img=cv2.imread(path)
        if img is None: return None
        yolo=YOLOHandler(self.cfg.model_path)
        _, xy, conf, _, _ = yolo.infer(img)
        spans=[]
        for pts, cf in zip(xy, conf):
            if cf[9]>0.2 and cf[10]>0.2:
                spans.append(np.hypot(pts[10][0]-pts[9][0],pts[10][1]-pts[9][1]))
        return max(spans) if spans else None
    def _compute_focal_and_refspan(self)->Tuple[float,float]:
        fs, ss = [], []
        for c in self.calibs:
            px=self._get_pixel_span(c['path'])
            if px and px>1e-3:
                fs.append(px * c['dist_cm']/c['span_cm']); ss.append(c['span_cm'])
        if not fs: return 0.0,0.0
        med = np.median(fs)
        filtered=[(fs[i],ss[i]) for i,v in enumerate(fs) if abs(v-med)/med<0.3]
        if filtered:
            fs_f, ss_f = zip(*filtered)
            return float(np.mean(fs_f)), float(np.mean(ss_f))
        return med, float(np.median(ss))

class DistanceEstimator:
    def __init__(self,calib:CalibrationHandler):
        self.focal=calib.focal_length; self.ref_span=calib.ref_span_cm
    def estimate(self,xy:np.ndarray,conf:np.ndarray)->Optional[float]:
        spans=[np.hypot(pts[10][0]-pts[9][0],pts[10][1]-pts[9][1]) for pts,cf in zip(xy,conf) if cf[9]>0.2 and cf[10]>0.2]
        if not spans or self.focal<=0: return None
        return float(self.focal*self.ref_span/max(spans))

class TPoseApp:
    def __init__(self,cfg:Config):
        self.cfg=cfg
        self.camera=CameraHandler(0)
        self.inferer=YOLOHandler(cfg.model_path)
        self.ds=DeepSortHandler(cfg)
        self.pose_checker=TPoseDurationChecker(cfg.config_hold_sec,cfg.config_frame_gap)
        self.calib=CalibrationHandler(cfg)
        self.dist_est=DistanceEstimator(self.calib)
        self.mode=Mode.MANUAL
        self.auto_phase='search'
        self.selected_id=None

    def run(self):
        win='T-pose Detector'
        cv2.namedWindow(win,cv2.WINDOW_NORMAL)
        while True:
            start=time.perf_counter()
            ret,frame=self.camera.read()
            if not ret: break
            key=cv2.waitKey(1)&0xFF
            if key==ord('m'): self.mode=Mode.MANUAL
            elif key==ord('a'): self.mode=Mode.AUTO; self.auto_phase='search'; self.pose_checker=TPoseDurationChecker(self.cfg.config_hold_sec,self.cfg.config_frame_gap)
            elif key==ord('c'): self.mode=Mode.CONFIG
            elif key in (ord('q'),27): break

            if self.mode==Mode.CONFIG:
                res,xy,conf,boxes,confs=self.inferer.infer(frame)
                if self.pose_checker.update(score_ransac(xy[0],conf[0],self.cfg.horizontal_threshold_deg)>=self.cfg.score_ok if len(xy)>0 else False):
                    self._do_calibration(frame,xy,conf,boxes)
                img=res.plot() if self.cfg.draw_skeleton else frame
                cv2.imshow(win,img)

            elif self.mode==Mode.MANUAL:
                disp=frame.copy(); cv2.putText(disp,'MODE MANUEL',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2); cv2.imshow(win,disp)

            else:
                res,xy,conf,boxes,confs=self.inferer.infer(frame)
                if self.auto_phase=='search':
                    img=res.plot() if self.cfg.draw_skeleton else frame.copy()
                    for i,box in enumerate(boxes):
                        score=score_ransac(xy[i],conf[i],self.cfg.horizontal_threshold_deg)
                        color=(0,255,0) if score>=self.cfg.score_ok else (0,255,255)
                        x1,y1,x2,y2=box; cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                    cv2.imshow(win,img)
                    if self.pose_checker.update(any(score_ransac(xy[j],conf[j],self.cfg.horizontal_threshold_deg)>=self.cfg.score_ok for j in range(len(boxes)))):
                        tracks=self.ds.update(boxes,confs,frame)
                        self.selected_id=self._select_tpose_target(tracks,boxes,xy,conf)
                        if self.selected_id: self.auto_phase='track'
                else:
                    tracks=self.ds.update(boxes,confs,frame)
                    ann=frame.copy()
                    for t in tracks:
                        if t.is_confirmed() and t.track_id==self.selected_id:
                            x1,y1,x2,y2=map(int,t.to_ltrb())
                            cv2.rectangle(ann,(x1,y1),(x2,y2),(255,0,0),2)
                            # affiche distance
                            dist=self.dist_est.estimate(xy,conf)
                            if dist: cv2.putText(ann,f"{dist:.1f} cm",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                            # affiche angle de centre lissé
                            cx,cy=(x1+x2)//2,(y1+y2)//2
                            AngleCalculator.annotate(ann,(cx,cy))
                            break
                    cv2.imshow(win,ann)
                    if not any(t.track_id==self.selected_id and t.is_confirmed() for t in tracks):
                        self.auto_phase='search'; self.pose_checker=TPoseDurationChecker(self.cfg.config_hold_sec,self.cfg.config_frame_gap); self.selected_id=None

            if self.cfg.display_process_time:
                print(f"[{self.mode.value.upper()}:{self.auto_phase if self.mode==Mode.AUTO else ''}] {(time.perf_counter()-start)*1000:.1f} ms")

        self.camera.release(); cv2.destroyAllWindows()

    def _do_calibration(self, frame, xy, conf, boxes):
        ts=datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        xl,yl=xy[0][9]; xr,yr=xy[0][10]
        span_cm=float(input('Envergure (cm): '))
        dist_cm=float(input('Distance (cm): '))
        ann=frame.copy(); cv2.circle(ann,(int(xl),int(yl)),6,(0,0,255),-1); cv2.circle(ann,(int(xr),int(yr)),6,(0,0,255),-1)
        cv2.putText(ann,f"{ts} | {span_cm:.0f}cm | {dist_cm:.0f}cm",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        fname=f"{ts}_{int(span_cm)}cm_{int(dist_cm)}cm.jpg"; path=os.path.join(self.cfg.calib_dir,fname); cv2.imwrite(path,ann); print(f"Saved calibration: {path}")

    def _select_tpose_target(self, tracks, boxes, xy, conf)->Optional[int]:
        for t in tracks:
            if not t.is_confirmed(): continue
            tb=tuple(map(int,t.to_ltrb())); ious=np.array([self._iou(tb,b) for b in boxes])
            if ious.size==0 or ious.max()<self.cfg.iou_mapping_threshold: continue
            return t.track_id
        return None

    @staticmethod
    def _iou(boxA,boxB): xA,yA=max(boxA[0],boxB[0]),max(boxA[1],boxB[1]); xB,yB=min(boxA[2],boxB[2]),min(boxA[3],boxB[3]); inter=max(0,xB-xA)*max(0,yB-yA); areaA=(boxA[2]-boxA[0])*(boxA[3]-boxA[1]); areaB=(boxB[2]-boxB[0])*(boxB[3]-boxB[1]); return inter/(areaA+areaB-inter) if (areaA+areaB-inter)>0 else 0.0

if __name__=='__main__':
    cfg=Config(); app=TPoseApp(cfg); app.run()
