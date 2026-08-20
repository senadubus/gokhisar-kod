"""Hedef Takip Modülü — BotSORT motion + kararlı kendi ID + KTR Kalman.

KTR 4.2.2.4:
  • BotSORT: hareket / düşük-conf telafi (ham track_id kullanılmaz)
  • Kararlı #id: IoU+merkez eşleme; yeni id için conf + ardışık onay
  • Ölçüm yokken TrackKalman predict (coast)
  • ServoKalman ayrı (nişan)

Akış:
  tespitler (+ kare) → BotSORT (kutu) → uzaysal ID eşleme → TrackKalman
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT

import config
from detection.yolo_detector import Detection


def _as_float(x) -> float:
    """NumPy 2.x: float(array([v])) TypeError verir — tek skaler çıkar."""
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    return float(x)


def _as_int(x) -> int:
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    return int(x)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki xyxy kutusu arasındaki IoU."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _box_center_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def boxes_same_object(a: np.ndarray, b: np.ndarray,
                      iou_threshold: float | None = None,
                      center_ratio: float | None = None) -> bool:
    """IoU veya merkez yakınlığına göre iki kutunun aynı nesneye ait olup olmadığı."""
    iou_threshold = config.DEDUPE_IOU if iou_threshold is None else iou_threshold
    center_ratio = config.DEDUPE_CENTER_RATIO if center_ratio is None else center_ratio
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
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


def _detection_at_center(
    det: Detection, cx: float, cy: float, *, conf: float | None = None
) -> Detection:
    """Aynı boyutta, yeni merkezli kutu (Kalman tahmini için)."""
    w = max(1.0, det.x2 - det.x1)
    h = max(1.0, det.y2 - det.y1)
    return Detection(
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
        conf=det.conf if conf is None else conf,
        class_id=det.class_id,
        source=det.source,
    )


class _DetBatch:
    """Ultralytics BotSORT'un beklediği Results-benzeri tespit paketi."""

    __slots__ = ("_xyxy", "conf", "cls")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self._xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self._xyxy.T
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx) -> _DetBatch:
        return _DetBatch(self._xyxy[idx], self.conf[idx], self.cls[idx])


class TrackKalman:
    """Sabit-hız Kalman — merkez (cx, cy) + hız (vx, vy).

    KTR: ölçüm varken doğru/yumuşat; ölçüm yokken gelecek tahmini.
    Hızlı balon için süreç gürültüsü yüksek, ölçüme güven yüksek tutulur
    (kutu geride kalmasın).
    """

    def __init__(self) -> None:
        q = float(getattr(config, "TRACK_KALMAN_PROCESS", 5e-2))
        r = float(getattr(config, "TRACK_KALMAN_MEASURE", 1e-2))
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePost = np.array(
                [[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32
            )
            self.initialized = True
        self.kf.predict()
        est = self.kf.correct(meas)
        return _as_float(est[0, 0]), _as_float(est[1, 0])

    def predict_only(self) -> tuple[float, float]:
        """Ölçüm yok — KTR 'yalnız tahmin adımı'."""
        if not self.initialized:
            return 0.0, 0.0
        pred = self.kf.predict()
        return _as_float(pred[0, 0]), _as_float(pred[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        s = self.kf.statePost
        return _as_float(s[2, 0]), _as_float(s[3, 0])


@dataclass
class TrackedTarget:
    track_id: int
    det: Detection
    age: int = 0                 # kaç karedir takipte
    misses: int = 0              # ardışık kayıp kare sayısı
    center_history: list = field(default_factory=list)
    servo_corrections: list = field(default_factory=list)
    predicted: bool = False      # bu karede kutu Kalman tahmini mi?


def _botsort_args(fps: int = 30) -> SimpleNamespace:
    """Config → Ultralytics BotSORT argümanları.

    match_thresh maliyet eşiğidir (1 − IoU); TRACK_MATCH_IOU IoU alt sınırına
    çevrilir. new_track_thresh ayrı ve yüksek tutulur: düşük conf gürültü
    yalnızca mevcut izi besler, yeni ID açmaz. ReID kapalı.
    """
    iou_min = float(getattr(config, "TRACK_MATCH_IOU", 0.25))
    new_track = float(
        getattr(config, "TRACK_NEW_TRACK_CONF", config.TRACK_HIGH_CONF)
    )
    return SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=float(config.TRACK_HIGH_CONF),
        track_low_thresh=float(config.TRACK_LOW_CONF),
        new_track_thresh=new_track,
        track_buffer=int(config.TRACK_BUFFER),
        match_thresh=max(0.1, min(0.99, 1.0 - iou_min)),
        fuse_score=True,
        gmc_method=str(getattr(config, "TRACK_GMC_METHOD", "sparseOptFlow")),
        proximity_thresh=float(getattr(config, "TRACK_PROXIMITY_THRESH", 0.5)),
        appearance_thresh=float(getattr(config, "TRACK_APPEARANCE_THRESH", 0.8)),
        with_reid=bool(getattr(config, "TRACK_WITH_REID", False)),
        model="auto",
        frame_rate=max(1, int(fps)),
    )


@dataclass
class _LostSlot:
    """Düşen iz — aynı konumda yeni tespit gelince eski #id'yi geri ver."""
    track_id: int
    box: np.ndarray
    class_id: int
    age_frames: int = 0


@dataclass
class _PendingNew:
    """Yeni iz adayı — TRACK_CONFIRM_FRAMES dolmadan #id verilmez."""
    box: np.ndarray
    conf: float
    class_id: int
    hits: int = 1


class TargetTracker:
    def __init__(self, fps: int = 30):
        self._fps = max(1, int(fps))
        self.tracker = BOTSORT(_botsort_args(self._fps))
        self.targets: dict[int, TrackedTarget] = {}
        self._kalmans: dict[int, TrackKalman] = {}
        self._next_id = 1
        self._recent_lost: list[_LostSlot] = []
        self._pending: list[_PendingNew] = []

    def set_fps(self, fps: int) -> None:
        """Ölçülen pipeline FPS — tampon süresi sabit kare; yeniden kurma yok."""
        self._fps = max(1, int(fps))

    def _kf(self, track_id: int) -> TrackKalman:
        if track_id not in self._kalmans:
            self._kalmans[track_id] = TrackKalman()
        return self._kalmans[track_id]

    def _remember_lost(self, tid: int, t: TrackedTarget) -> None:
        self._recent_lost.append(
            _LostSlot(
                track_id=tid,
                box=np.asarray(t.det.as_xyxy(), dtype=np.float64),
                class_id=t.det.class_id,
                age_frames=0,
            )
        )

    def _drop(self, track_id: int, *, remember: bool = True) -> None:
        t = self.targets.get(track_id)
        if remember and t is not None:
            self._remember_lost(track_id, t)
        self.targets.pop(track_id, None)
        self._kalmans.pop(track_id, None)

    def _prune_recent_lost(self) -> None:
        limit = int(getattr(config, "TRACK_ID_REUSE_FRAMES", 120))
        kept: list[_LostSlot] = []
        for slot in self._recent_lost:
            slot.age_frames += 1
            if slot.age_frames <= limit and slot.track_id not in self.targets:
                kept.append(slot)
        self._recent_lost = kept

    def _assoc_score(self, box: np.ndarray, other: np.ndarray) -> float:
        """IoU + merkez yakınlığı; BotSORT id'sinden bağımsız yapıştırma skoru."""
        iou_min = float(getattr(config, "TRACK_ASSOCIATE_IOU", 0.10))
        center_ratio = float(
            getattr(config, "TRACK_ASSOCIATE_CENTER", config.DEDUPE_CENTER_RATIO)
        )
        iou = _box_iou(box, other)
        if iou >= iou_min:
            return 1.0 + iou  # IoU eşleşmeleri öncelikli
        if boxes_same_object(box, other, iou_threshold=iou_min, center_ratio=center_ratio):
            dist = _box_center_dist(box, other)
            max_side = max(
                other[2] - other[0], other[3] - other[1],
                box[2] - box[0], box[3] - box[1], 1.0,
            )
            return 0.5 + max(0.0, 1.0 - dist / (max_side * center_ratio))
        return -1.0

    def _match_existing(
        self, box: np.ndarray, class_id: int, claimed: set[int]
    ) -> int | None:
        best_id, best = None, 0.0
        for tid, t in self.targets.items():
            if tid in claimed or t.det.class_id != class_id:
                continue
            score = self._assoc_score(box, t.det.as_xyxy())
            if score > best:
                best_id, best = tid, score
        return best_id

    def _match_lost(
        self, box: np.ndarray, class_id: int, claimed: set[int]
    ) -> int | None:
        best_id, best = None, 0.0
        for slot in self._recent_lost:
            if slot.track_id in claimed or slot.class_id != class_id:
                continue
            if slot.track_id in self.targets:
                continue
            score = self._assoc_score(box, slot.box)
            if score > best:
                best_id, best = slot.track_id, score
        if best_id is not None:
            self._recent_lost = [
                s for s in self._recent_lost if s.track_id != best_id
            ]
        return best_id

    def _mint_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _update_pending(
        self, unmatched: list[tuple[np.ndarray, float, int]]
    ) -> list[tuple[int, np.ndarray, float, int]]:
        """Onaysız adayları güncelle; eşiği geçenleri yeni kararlı id olarak döndür."""
        confirm = int(getattr(config, "TRACK_CONFIRM_FRAMES", 4))
        new_conf = float(getattr(config, "TRACK_NEW_TRACK_CONF", 0.55))
        promoted: list[tuple[int, np.ndarray, float, int]] = []
        next_pending: list[_PendingNew] = []

        for box, conf, cls_id in unmatched:
            if conf < new_conf:
                continue
            best_i, best = -1, 0.0
            for i, pend in enumerate(self._pending):
                if pend.class_id != cls_id:
                    continue
                score = self._assoc_score(box, pend.box)
                if score > best:
                    best_i, best = i, score
            if best_i >= 0:
                pend = self._pending.pop(best_i)
                pend.box = box
                pend.conf = conf
                pend.hits += 1
                if pend.hits >= confirm:
                    tid = self._match_lost(box, cls_id, set()) or self._mint_id()
                    promoted.append((tid, box, conf, cls_id))
                else:
                    next_pending.append(pend)
            else:
                next_pending.append(
                    _PendingNew(box=box, conf=conf, class_id=cls_id, hits=1)
                )

        self._pending = next_pending
        return promoted

    def _coast_missing(self, tid: int) -> None:
        """Ölçüm yok: Kalman predict + kutuyu ilerlet (hayalet donmasın)."""
        t = self.targets[tid]
        t.misses += 1
        if t.misses > config.TRACK_BUFFER:
            self._drop(tid, remember=True)
            return
        kf = self._kf(tid)
        if not kf.initialized:
            return
        cx, cy = kf.predict_only()
        conf = max(0.05, t.det.conf * 0.92)
        t.det = _detection_at_center(t.det, cx, cy, conf=conf)
        t.predicted = True
        t.center_history.append((cx, cy))
        if len(t.center_history) > 60:
            t.center_history.pop(0)

    @staticmethod
    def _suppress_overlapping(tracks: np.ndarray) -> np.ndarray:
        """Aynı sınıfta yüksek örtüşen takiplerden yalnızca en güvenilir olanı bırak."""
        if tracks.size == 0:
            return tracks
        n = tracks.shape[0]
        if n <= 1:
            return tracks

        order = sorted(range(n), key=lambda i: float(tracks[i, 5]), reverse=True)
        keep: list[int] = []
        for i in order:
            cls_i = int(tracks[i, 6])
            box_i = tracks[i, :4]
            if any(
                int(tracks[j, 6]) == cls_i
                and boxes_same_object(
                    box_i, tracks[j, :4], iou_threshold=config.TRACK_DEDUPE_IOU
                )
                for j in keep
            ):
                continue
            keep.append(i)
        return tracks[keep]

    @staticmethod
    def _to_batch(detections: list[Detection]) -> _DetBatch:
        """Birleşik tespit listesi → BotSORT giriş formatı."""
        valid = [d for d in detections if d.conf >= config.TRACK_LOW_CONF]
        if not valid:
            return _DetBatch(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return _DetBatch(
            np.array([d.as_xyxy() for d in valid], dtype=np.float32),
            np.array([d.conf for d in valid], dtype=np.float32),
            np.array([d.class_id for d in valid], dtype=np.float32),
        )

    def _apply_measurement(
        self, tid: int, box: np.ndarray, conf: float, cls_id: int
    ) -> None:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        raw = Detection(
            x1, y1, x2, y2, conf=conf, class_id=cls_id, source="track"
        )
        self._kf(tid).update(raw.cx, raw.cy)
        if tid in self.targets:
            t = self.targets[tid]
            t.det = raw
            t.age += 1
            t.misses = 0
            t.predicted = False
            t.track_id = tid
        else:
            t = TrackedTarget(track_id=tid, det=raw, age=1, predicted=False)
            self.targets[tid] = t
        t.center_history.append((raw.cx, raw.cy))
        if len(t.center_history) > 60:
            t.center_history.pop(0)

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> dict[int, TrackedTarget]:
        """BotSORT motion + kararlı kendi ID.

        BotSORT'un ürettiği ham track_id kullanılmaz; kutular mevcut / kayıp
        izlere IoU+merkez ile yapıştırılır. Yeni #id yalnız yüksek conf ve
        ``TRACK_CONFIRM_FRAMES`` peş peşe adaylıktan sonra verilir.
        """
        self._prune_recent_lost()
        batch = self._to_batch(detections)
        tracked = self.tracker.update(batch, frame)

        rows: list[tuple[np.ndarray, float, int]] = []
        if tracked is not None and len(tracked) > 0:
            arr = np.asarray(tracked, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = self._suppress_overlapping(arr)
            for row in arr:
                rows.append(
                    (
                        np.asarray(row[:4], dtype=np.float64),
                        _as_float(row[5]),
                        _as_int(row[6]),
                    )
                )

        # BotSORT boş kaldıysa ham tespitlerle ID sürekliliğini koru
        if not rows:
            for det in detections:
                if det.conf < config.TRACK_LOW_CONF:
                    continue
                rows.append(
                    (
                        np.asarray(det.as_xyxy(), dtype=np.float64),
                        float(det.conf),
                        int(det.class_id),
                    )
                )
            # aynı sınıf örtüşen ham kutuları tekilleştir
            rows.sort(key=lambda r: r[1], reverse=True)
            kept: list[tuple[np.ndarray, float, int]] = []
            for box, conf, cls_id in rows:
                if any(
                    c == cls_id
                    and boxes_same_object(box, b, iou_threshold=config.TRACK_DEDUPE_IOU)
                    for b, _, c in kept
                ):
                    continue
                kept.append((box, conf, cls_id))
            rows = kept

        if not rows:
            self._pending.clear()
            for tid in list(self.targets):
                self._coast_missing(tid)
            return self.targets

        # Önce güçlü conf, sonra mevcut izlere yapıştır
        rows.sort(key=lambda r: r[1], reverse=True)
        seen: set[int] = set()
        claimed: set[int] = set()
        unmatched: list[tuple[np.ndarray, float, int]] = []

        for box, conf, cls_id in rows:
            tid = self._match_existing(box, cls_id, claimed)
            if tid is None:
                tid = self._match_lost(box, cls_id, claimed)
            if tid is not None:
                claimed.add(tid)
                seen.add(tid)
                self._apply_measurement(tid, box, conf, cls_id)
            else:
                unmatched.append((box, conf, cls_id))

        for tid, box, conf, cls_id in self._update_pending(unmatched):
            if tid in claimed:
                continue
            claimed.add(tid)
            seen.add(tid)
            self._apply_measurement(tid, box, conf, cls_id)

        for tid in list(self.targets):
            if tid in seen:
                continue
            t = self.targets[tid]
            if any(
                other.det.class_id == t.det.class_id
                and boxes_same_object(
                    t.det.as_xyxy(),
                    other.det.as_xyxy(),
                    iou_threshold=config.TRACK_DEDUPE_IOU,
                )
                for other_id, other in self.targets.items()
                if other_id in seen
            ):
                self._drop(tid, remember=False)
                continue
            self._coast_missing(tid)

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
    """Servoya giden merkez — yumuşak CV Kalman (sıçramasız).

    lead_time varsayılan 0: hız gürültüsü tavan/fırlama yapıyordu.
    Çıkış kareler arası SERVO_TARGET_MAX_JUMP_PX ile sınırlanır.
    """

    def __init__(self, lead_time: float | None = None):
        self.lead_time = float(
            lead_time
            if lead_time is not None
            else getattr(config, "SERVO_KALMAN_LEAD_S", 0.0)
        )
        self._max_jump = float(getattr(config, "SERVO_TARGET_MAX_JUMP_PX", 40.0))
        self._max_speed = 600.0  # px/s — absürt vy/vx kes
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        # Ölçüme daha çok güven; hızı az serbest bırak (Sena 0.8 çok agresifti)
        self.kf.processNoiseCov = np.diag([0.08, 0.08, 0.15, 0.15]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag([0.08, 0.08]).astype(np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._set_dt(0.033)
        self.initialized = False
        self._last_out: tuple[float, float] | None = None

    def _set_dt(self, dt: float) -> None:
        dt = max(0.001, min(float(dt), 0.2))
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kf.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self._last_out = None

    def _clamp_speed(self) -> None:
        sp = self.kf.statePost
        vx = float(np.clip(_as_float(sp[2, 0]), -self._max_speed, self._max_speed))
        vy = float(np.clip(_as_float(sp[3, 0]), -self._max_speed, self._max_speed))
        sp[2, 0] = np.float32(vx)
        sp[3, 0] = np.float32(vy)

    def _limit_jump(self, x: float, y: float) -> tuple[float, float]:
        if self._last_out is None:
            self._last_out = (x, y)
            return x, y
        lx, ly = self._last_out
        dx = max(-self._max_jump, min(self._max_jump, x - lx))
        dy = max(-self._max_jump, min(self._max_jump, y - ly))
        out = (lx + dx, ly + dy)
        self._last_out = out
        return out

    def update(
        self, cx: float, cy: float, dt: float = 0.033
    ) -> tuple[float, float]:
        self._set_dt(dt)
        if not self.initialized:
            self.kf.statePost = np.array(
                [[np.float32(cx)], [np.float32(cy)], [0], [0]], dtype=np.float32
            )
            self.kf.statePre = self.kf.statePost.copy()
            self.initialized = True
            return self._limit_jump(float(cx), float(cy))

        self.kf.predict()
        measurement = np.array(
            [[np.float32(cx)], [np.float32(cy)]], dtype=np.float32
        )
        estimate = self.kf.correct(measurement)
        self._clamp_speed()
        x = _as_float(estimate[0, 0])
        y = _as_float(estimate[1, 0])
        vx = _as_float(self.kf.statePost[2, 0])
        vy = _as_float(self.kf.statePost[3, 0])
        if self.lead_time > 0.0:
            x += vx * self.lead_time
            y += vy * self.lead_time
        return self._limit_jump(x, y)

    def predict_only(self, dt: float = 0.033) -> tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        # Miss'te son çıkışı tut — hayalet hızla tavana gitmesin
        if self._last_out is not None:
            return self._last_out
        self._set_dt(dt)
        pred = self.kf.predict()
        self._clamp_speed()
        return self._limit_jump(_as_float(pred[0, 0]), _as_float(pred[1, 0]))
