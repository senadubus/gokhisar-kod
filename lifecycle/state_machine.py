"""Hedef Yaşam Döngüsü + Kilitlenme + İmha Değerlendirme.

Her hedef için DETECT -> VALIDATE -> TRACK -> EVALUATE -> TARGET_LOCK ->
DESTROYED durumları tanımlıdır; geçişler belirlenen koşullarla yapılır.

İmha doğrulaması üç koşulun EŞ ZAMANLI sağlanmasını gerektirir:
  1) belirli süre yeniden tespit edilememe,
  2) takip zincirinin sonlanması,
  3) güven skorlarının eşik altında kalması.
Koşullar tam sağlanmazsa hedef yeniden takip/doğrulamaya alınır
(geçici algılama kayıplarından kaynaklı yanlış imha kararlarını önler).
"""
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import config
from iff.friend_foe import IFFLabel
from tracking.tracker import TrackedTarget


class TargetState(Enum):
    DETECT = auto()
    VALIDATE = auto()
    TRACK = auto()
    EVALUATE = auto()
    TARGET_LOCK = auto()
    DESTROYED = auto()


@dataclass
class TargetRecord:
    track_id: int
    state: TargetState = TargetState.DETECT
    iff: IFFLabel = IFFLabel.UNKNOWN
    lock_stable_frames: int = 0
    fired: bool = False
    fire_time: float = 0.0


class TargetLifecycleManager:
    def __init__(self):
        self.records: dict[int, TargetRecord] = {}

    def get(self, track_id: int) -> TargetRecord:
        if track_id not in self.records:
            self.records[track_id] = TargetRecord(track_id)
        return self.records[track_id]

    # ---------- durum geçişleri ----------
    def on_validated(self, track_id: int):
        rec = self.get(track_id)
        if rec.state in (TargetState.DETECT, TargetState.VALIDATE):
            rec.state = TargetState.TRACK

    def on_iff(self, track_id: int, label: IFFLabel):
        rec = self.get(track_id)
        rec.iff = label
        if rec.state is TargetState.TRACK and label is IFFLabel.FOE:
            rec.state = TargetState.EVALUATE

    def on_selected_for_lock(self, track_id: int):
        rec = self.get(track_id)
        if rec.state is TargetState.EVALUATE:
            rec.state = TargetState.TARGET_LOCK
            rec.lock_stable_frames = 0

    def update_lock(self, rec: TargetRecord, target: TrackedTarget) -> bool:
        """Kilit sürdürme: merkez tolerans bölgesinde kararlıysa
        angajmana hazır (True). Tolerans dışına çıkarsa TRACK'e döner."""
        cx, cy = target.det.cx, target.det.cy
        fx, fy = config.FRAME_CENTER
        err = math.hypot(cx - fx, cy - fy)

        if err <= config.LOCK_TOLERANCE_PX:
            rec.lock_stable_frames += 1
        else:
            rec.lock_stable_frames = 0
            if err > config.LOCK_TOLERANCE_PX * 3:
                rec.state = TargetState.TRACK      # yeniden merkezleme
                return False
        return rec.lock_stable_frames >= config.LOCK_STABLE_FRAMES

    def on_fired(self, track_id: int):
        rec = self.get(track_id)
        rec.fired = True
        rec.fire_time = time.time()

    # ---------- imha değerlendirme ----------
    def evaluate_destroyed(self, rec: TargetRecord,
                           target: TrackedTarget | None) -> bool:
        """Üç koşulun eş zamanlı kontrolü."""
        if not rec.fired:
            return False

        cond_miss = (target is None or
                     target.misses >= config.DESTROY_MISS_FRAMES)
        cond_track_ended = target is None
        cond_low_conf = (target is None or
                         target.det.conf < config.DESTROY_CONF_THRESHOLD)

        if cond_miss and cond_track_ended and cond_low_conf:
            rec.state = TargetState.DESTROYED
            return True

        # Koşullar tam sağlanmadı: kesin imha yok, yeniden takip/doğrulama
        rec.state = TargetState.TRACK
        rec.fired = False
        return False

    def drop(self, track_id: int):
        self.records.pop(track_id, None)
