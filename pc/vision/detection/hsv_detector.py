"""Küçük Hedef Tespiti + Yedek Balon Algılama (HSV boru hattı).

Arka planın siyah, balonların kırmızı olması bilgisinden yararlanır:
HSV dönüşümü -> kırmızı için çift eşikli maskeleme -> morfolojik
temizleme -> kontur analizi -> dairesellik filtresi. Balon adayının
etrafında piksel boyutuna bağlı dinamik ROI üretir; bu ROI, YOLO'da
yeniden değerlendirilmek üzere tespit modülüne verilir.
"""
import cv2
import numpy as np

import config
from detection.yolo_detector import Detection


class HsvBalloonDetector:
    def __init__(self, trigger_frame_threshold: int = 30):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.trigger_frame_threshold = trigger_frame_threshold
        self.mismatch_frame_count = 0

    def update_condition(self, num_objects: int, num_balloons: int) -> bool:
        """30 frame (veya trigger_frame_threshold) boyunca nesne var ama
        nesne sayısı ile balon sayısı eşit olmadığında tetiklenme şartını günceller.

        - num_objects > 0 ve num_objects != num_balloons ise sayaç 1 artırılır.
        - Nesne yoksa veya nesne sayısı balon sayısına eşitse sayaç 0'lanır.
        - Sayaç trigger_frame_threshold eşiğine ulaştığında True döndürür.
        """
        if num_objects > 0 and num_objects != num_balloons:
            self.mismatch_frame_count += 1
        else:
            self.mismatch_frame_count = 0
        return self.mismatch_frame_count >= self.trigger_frame_threshold

    def reset_condition(self) -> None:
        """Sayaç durumunu sıfırlar."""
        self.mismatch_frame_count = 0

    def should_trigger(self, num_objects: int, num_balloons: int) -> bool:
        """Tetiklenme şartının karşılanıp karşılanmadığını günceller ve döndürür."""
        return self.update_condition(num_objects, num_balloons)

    # ---------- ortak HSV boru hattı ----------
    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, config.HSV_RED_LOWER_1, config.HSV_RED_UPPER_1)
        m2 = cv2.inRange(hsv, config.HSV_RED_LOWER_2, config.HSV_RED_UPPER_2)
        mask = cv2.bitwise_or(m1, m2)
        # morfolojik temizleme
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _candidates(self, mask: np.ndarray) -> list[tuple[float, float, float]]:
        """Kontur analizi + dairesellik filtresi.
        Dönüş: (cx, cy, yarıçap) listesi."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < config.MIN_CIRCULARITY:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            out.append((cx, cy, r))
        return out

    # ---------- küçük hedef tespiti ----------
    def detect(self, frame: np.ndarray,
               num_objects: int | None = None,
               num_balloons: int | None = None,
               force: bool = False) -> list[Detection]:
        """HSV tabanlı balon adayları; merkez+yarıçaptan ortak sınır
        kutusu formatına dönüştürülür.

        num_objects ve num_balloons belirtilirse (ve force=False ise),
        yalnızca 30 frame boyunca nesne varlığı ve nesne-balon dengesizliği
        (num_objects > 0 ve num_objects != num_balloons) korunduğunda tespit yapılır.
        Parametreler belirtilmezse veya force=True ise koşulsuz tespit yapılır.
        """
        if not force and num_objects is not None and num_balloons is not None:
            if not self.update_condition(num_objects, num_balloons):
                return []

        mask = self._red_mask(frame)
        detections = []
        for cx, cy, r in self._candidates(mask):
            detections.append(
                Detection(cx - r, cy - r, cx + r, cy + r,
                          conf=0.4,                     # sabit ön güven
                          class_id=config.BALLOON_CLASS_ID,
                          source="hsv")
            )
        return detections

    # ---------- dinamik ROI üretimi ----------
    @staticmethod
    def dynamic_roi(det: Detection, frame_shape) -> tuple[int, int, int, int]:
        """Balon piksel boyutuna bağlı, kare sınırlarına kırpılmış ROI.
        Mesafeye bağlı ölçek değişimlerine doğal uyum sağlar."""
        h, w = frame_shape[:2]
        size = max(det.w, det.h) * config.ROI_SCALE_FACTOR
        x1 = int(max(0, det.cx - size / 2))
        y1 = int(max(0, det.cy - size / 2))
        x2 = int(min(w, det.cx + size / 2))
        y2 = int(min(h, det.cy + size / 2))
        return x1, y1, x2, y2

    @staticmethod
    def upper_roi(det: Detection, frame_shape) -> tuple[int, int, int, int]:
        """Balonun üstündeki bölge (maket doğrulaması için)."""
        h, w = frame_shape[:2]
        span = max(det.w, det.h) * config.ROI_SCALE_FACTOR
        x1 = int(max(0, det.cx - span / 2))
        x2 = int(min(w, det.cx + span / 2))
        y2 = int(max(0, det.y1))
        y1 = int(max(0, det.y1 - span))
        return x1, y1, x2, y2

    # ---------- yedek algılama ----------
    def detect_backup(self, frame: np.ndarray) -> list[Detection]:
        """Sistem performans düşüşünde devreye giren yedek mekanizma:
        aynı HSV boru hattı + boyut/şekil/oran geometrik filtreleri."""
        mask = self._red_mask(frame)
        detections = []
        for cx, cy, r in self._candidates(mask):
            d = 2 * r
            # geometrik filtreler: makul boyut ve en-boy oranı (daire ~1)
            if d < 4 or d > min(frame.shape[:2]) * 0.5:
                continue
            detections.append(
                Detection(cx - r, cy - r, cx + r, cy + r,
                          conf=0.3,
                          class_id=config.BALLOON_CLASS_ID,
                          source="hsv")
            )
        return detections
