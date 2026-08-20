"""Dost-Düşman Ayrım Modülü (IFF).

Aşama-2: aktif balona sahip tüm maketler / balon hedefleri doğrudan DÜŞMAN.
Aşama-3: renk tabanlı ayrım — kırmızı işaret = DÜŞMAN, camgöbeği = DOST.

Maket IFF: hedef kutusunun Hue medyanı.
Balon IFF (aşama-3): balon gövdesi kırmızı olduğu için gövdeye bakılmaz;
yalnızca balonun üstündeki işaret bölgesinde kırmızı/camgöbeği piksel
oranına bakılır. Üstünde kırmızı nesne yoksa DÜŞMAN sayılmaz.
"""
from collections import defaultdict, deque
from enum import Enum

import cv2
import numpy as np

import config
from detection.yolo_detector import Detection


class IFFLabel(Enum):
    UNKNOWN = "UNKNOWN"
    FRIEND = "DOST"
    FOE = "DUSMAN"


# Üst işaret ROI'sinde en az bu kadar doygun piksel gerekir
_MARKER_MIN_PIXELS = 25


class FriendFoeClassifier:
    def __init__(self, stage: int = 3):
        self.stage = stage
        # track_id -> son N karedeki etiket geçmişi
        self.history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.IFF_HISTORY_LEN))

    # ---------- tek kare renk sınıflandırması (maket kutusu) ----------
    @staticmethod
    def _classify_frame(frame: np.ndarray, det: Detection) -> IFFLabel:
        x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
        x2, y2 = int(det.x2), int(det.y2)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return IFFLabel.UNKNOWN

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # doygunluğu düşük (gri/siyah arka plan) pikselleri dışla
        sat_mask = hsv[:, :, 1] > 80
        hues = hsv[:, :, 0][sat_mask]
        if hues.size < 10:
            return IFFLabel.UNKNOWN

        hue_median = float(np.median(hues))
        for lo, hi in config.HUE_RED_RANGES:
            if lo <= hue_median <= hi:
                return IFFLabel.FOE
        lo, hi = config.HUE_CYAN_RANGE
        if lo <= hue_median <= hi:
            return IFFLabel.FRIEND
        return IFFLabel.UNKNOWN

    @staticmethod
    def _balloon_marker_roi(
        det: Detection, frame_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        """Balon gövdesinin hemen üstü — maket/işaret arama bandı.

        Gövde kırmızısı IFF'ye girmesin diye alt sınır balonun üst kenarıdır.
        """
        h, w = frame_shape[:2]
        span_w = max(det.w * 1.6, 24.0)
        span_h = max(det.h * 1.3, 24.0)
        x1 = int(max(0, det.cx - span_w / 2))
        x2 = int(min(w, det.cx + span_w / 2))
        y2 = int(max(0, min(h, det.y1)))
        y1 = int(max(0, y2 - span_h))
        return x1, y1, x2, y2

    @staticmethod
    def _classify_marker_region(frame: np.ndarray, roi: tuple[int, int, int, int]) -> IFFLabel:
        """Üst bantta kırmızı vs camgöbeği piksel sayısı (medyan değil)."""
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return IFFLabel.UNKNOWN
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return IFFLabel.UNKNOWN

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1] > 80
        val = hsv[:, :, 2] > 50
        usable = sat & val
        if int(np.count_nonzero(usable)) < _MARKER_MIN_PIXELS:
            return IFFLabel.UNKNOWN

        hue = hsv[:, :, 0]
        red = np.zeros(hue.shape, dtype=bool)
        for lo, hi in config.HUE_RED_RANGES:
            red |= (hue >= lo) & (hue <= hi)
        lo, hi = config.HUE_CYAN_RANGE
        cyan = (hue >= lo) & (hue <= hi)

        red_n = int(np.count_nonzero(red & usable))
        cyan_n = int(np.count_nonzero(cyan & usable))
        if red_n >= _MARKER_MIN_PIXELS and red_n > cyan_n * 1.2:
            return IFFLabel.FOE
        if cyan_n >= _MARKER_MIN_PIXELS and cyan_n > red_n * 1.2:
            return IFFLabel.FRIEND
        return IFFLabel.UNKNOWN

    def _vote(self, track_id: int, label: IFFLabel) -> IFFLabel:
        hist = self.history[track_id]
        hist.append(label)
        foe_votes = sum(1 for l in hist if l is IFFLabel.FOE)
        friend_votes = sum(1 for l in hist if l is IFFLabel.FRIEND)
        if foe_votes >= config.IFF_VOTE_MIN_FRAMES and foe_votes > friend_votes:
            return IFFLabel.FOE
        if friend_votes >= config.IFF_VOTE_MIN_FRAMES and friend_votes > foe_votes:
            return IFFLabel.FRIEND
        return IFFLabel.UNKNOWN

    # ---------- zamansal oylama ----------
    def classify(self, frame: np.ndarray, det: Detection,
                 track_id: int) -> IFFLabel:
        if self.stage == 2:
            # Aşama-2: balonlu tüm maketler düşman kabul edilir
            return IFFLabel.FOE

        label = self._classify_frame(frame, det)
        return self._vote(track_id, label)

    def classify_balloon(
        self,
        frame: np.ndarray,
        balloon: Detection,
        track_id: int,
        marker_det: Detection | None = None,
    ) -> IFFLabel:
        """Aşama-3 balon IFF: yalnız üstündeki kırmızı/camgöbeği nesne.

        ``marker_det`` varsa (eşleşmiş maket kutusu) o bölgeye bakılır;
        yoksa balonun üst bandı taranır. Üstte kırmızı yoksa UNKNOWN.
        """
        if self.stage == 2:
            return IFFLabel.FOE

        if marker_det is not None:
            raw = self._classify_frame(frame, marker_det)
        else:
            roi = self._balloon_marker_roi(balloon, frame.shape)
            raw = self._classify_marker_region(frame, roi)
        return self._vote(track_id, raw)

    def drop(self, track_id: int):
        self.history.pop(track_id, None)
