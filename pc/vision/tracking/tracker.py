"""Hedef Takip Modülü — ByteTrack + servo Kalman filtresi.

Tüm tespit yöntemlerinden (YOLO, HSV, YOLO-ROI) gelen hedefler ortak
sınır kutusu formatına dönüştürülüp birleşik listede toplanır; böylece
yöntemler arası geçişte kimlikler korunur. ByteTrack, dahili Kalman
filtresiyle sonraki konumu tahmin eder; IoU + Macar algoritması ile
eşleştirme yapar; yüksek güvenli tespitler önce, düşük güvenliler
telafi turunda eşleştirilir.

Servo kararlılığı için, takip edilen hedefin merkez koordinatına
ByteTrack'ten bağımsız ikinci bir Kalman filtresi uygulanır.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import supervision as sv

import config
from detection.yolo_detector import Detection


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki xyxy kutusu arasındaki IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _box_center_dist(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def boxes_same_object(a: np.ndarray, b: np.ndarray,
                      iou_threshold: float | None = None,
                      center_ratio: float | None = None) -> bool:
    """IoU veya merkez yakınlığına göre iki kutunun aynı nesneye ait olup olmadığı."""
    iou_threshold = config.DEDUPE_IOU if iou_threshold is None else iou_threshold
    center_ratio = config.DEDUPE_CENTER_RATIO if center_ratio is None else center_ratio
    if _box_iou(a, b) > iou_threshold:
        return True
    max_side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return _box_center_dist(a, b) < max_side * center_ratio


def detections_same_object(a: Detection, b: Detection,
                           iou_threshold: float | None = None,
                           center_ratio: float | None = None) -> bool:
    """Detection çifti için aynı-nesne testi."""
    if a.class_id != b.class_id:
        return False
    return boxes_same_object(a.as_xyxy(), b.as_xyxy(), iou_threshold, center_ratio)


@dataclass
class TrackedTarget:
    track_id: int
    det: Detection
    age: int = 0                 # kaç karedir takipte
    misses: int = 0              # ardışık kayıp kare sayısı
    center_history: list = field(default_factory=list)
    servo_corrections: list = field(default_factory=list)


class TargetTracker:
    def __init__(self, fps: int = 30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACK_HIGH_CONF,
            minimum_matching_threshold=config.TRACK_MATCH_IOU,
            lost_track_buffer=config.TRACK_BUFFER,
            frame_rate=fps,
            minimum_consecutive_frames=2,
        )
        self.targets: dict[int, TrackedTarget] = {}

    @staticmethod
    def _suppress_overlapping(tracked: sv.Detections) -> sv.Detections:
        """Aynı sınıfta yüksek örtüşen takiplerden yalnızca en güvenilir olanı bırak.

        YOLO bazen aynı nesneye kaydırmalı iki kutu üretir; dedupe kaçırsa
        ByteTrack aynı karede iki kimlik açar. Bu filtre çıktıyı tekilleştirir.
        """
        n = len(tracked)
        if n <= 1:
            return tracked

        order = sorted(range(n), key=lambda i: tracked.confidence[i], reverse=True)
        keep: list[int] = []
        for i in order:
            cls_i = int(tracked.class_id[i])
            box_i = tracked.xyxy[i]
            if any(int(tracked.class_id[j]) == cls_i
                   and boxes_same_object(box_i, tracked.xyxy[j],
                                         iou_threshold=config.TRACK_DEDUPE_IOU)
                   for j in keep):
                continue
            keep.append(i)
        return tracked[keep]

    @staticmethod
    def _to_sv(detections: list[Detection]) -> sv.Detections:
        """Birleşik tespit listesi -> supervision formatı.

        Gürültü tespiti önlemek için conf < TRACK_LOW_CONF (0.1) olanlar filtrelenir;
        TRACK_LOW_CONF <= conf < TRACK_HIGH_CONF (0.1..0.5) arasındaki düşük güvenli
        tespitler ByteTrack'in 2. tur telafi eşleştirmesinde kullanılır.
        """
        valid_dets = [d for d in detections if d.conf >= config.TRACK_LOW_CONF]
        if not valid_dets:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([d.as_xyxy() for d in valid_dets]),
            confidence=np.array([d.conf for d in valid_dets]),
            class_id=np.array([d.class_id for d in valid_dets]),
        )

    def update(self, detections: list[Detection]) -> dict[int, TrackedTarget]:
        """ByteTrack güncellemesi; kimlikleri korunmuş hedef sözlüğü döner."""
        tracked = self.tracker.update_with_detections(self._to_sv(detections))
        tracked = self._suppress_overlapping(tracked)

        seen: set[int] = set()
        for xyxy, conf, cls_id, tid in zip(tracked.xyxy,
                                           tracked.confidence,
                                           tracked.class_id,
                                           tracked.tracker_id):
            tid = int(tid)
            seen.add(tid)
            det = Detection(*xyxy.tolist(), conf=float(conf),
                            class_id=int(cls_id), source="track")
            if tid in self.targets:
                t = self.targets[tid]
                t.det = det
                t.age += 1
                t.misses = 0
            else:
                t = TrackedTarget(track_id=tid, det=det, age=1)
                self.targets[tid] = t
            t.center_history.append((det.cx, det.cy))
            if len(t.center_history) > 60:
                t.center_history.pop(0)

        # ölçüm alınamayan takipler: kayıp sayacı artar, buffer aşılınca düşer
        for tid in list(self.targets):
            if tid not in seen:
                # Çift iz bastırması: aynı sınıfta örtüşen aktif takip varsa hemen düş
                t = self.targets[tid]
                if any(other.det.class_id == t.det.class_id
                       and boxes_same_object(t.det.as_xyxy(), other.det.as_xyxy(),
                                             iou_threshold=config.TRACK_DEDUPE_IOU)
                       for other_id, other in self.targets.items()
                       if other_id in seen):
                    del self.targets[tid]
                    continue
                self.targets[tid].misses += 1
                if self.targets[tid].misses > config.TRACK_BUFFER:
                    del self.targets[tid]

        return self.targets

    @staticmethod
    def stability(t: TrackedTarget) -> float:
        """Merkez geçmişindeki oynaklıktan 0-1 arası kararlılık metriği."""
        if len(t.center_history) < 5:
            return 0.0
        pts = np.array(t.center_history[-20:])
        jitter = float(np.mean(np.std(pts, axis=0)))
        return 1.0 / (1.0 + jitter / 10.0)


class ServoKalman:
    """Servoya giden hedef merkez koordinatı için sabit-hız Kalman filtresi.
    Ölçüm gürültüsünü bastırarak kararlı konum kestirimi sağlar."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # durum: [x, y, vx, vy], ölçüm: [x, y]
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-2
        self.initialized = False

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.initialized = True
        self.kf.predict()
        est = self.kf.correct(meas)
        return float(est[0]), float(est[1])

    def predict_only(self) -> tuple[float, float]:
        """Ölçüm yokken yalnızca tahmin adımı — takip sürekliliği."""
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])
