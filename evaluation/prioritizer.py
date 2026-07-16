"""Hedef Değerlendirme ve Önceliklendirme Modülü (EVALUATE).

Öncelik puanı = boyut, takip kararlılığı ve servo düzeltme miktarının
ağırlıklı birleşimi. Büyük görünen hedef daha yakın/öncelikli; az servo
düzeltmesi gerektiren hedef daha kararlı sayılır. Değerlendirme yalnızca
DÜŞMAN doğrulanan hedefler üzerinde yapılır; en yüksek puanlı hedef
angajman adayı seçilir.
"""
import numpy as np

import config
from tracking.tracker import TrackedTarget, TargetTracker


class TargetPrioritizer:
    def __init__(self):
        self.frame_area = config.FRAME_WIDTH * config.FRAME_HEIGHT

    def _servo_stability(self, t: TrackedTarget) -> float:
        """Servonun hedefi merkezde tutmak için yaptığı düzeltme
        miktarından kararlılık metriği (az düzeltme = yüksek puan)."""
        if not t.servo_corrections:
            return 0.5
        mean_corr = float(np.mean(np.abs(t.servo_corrections[-20:])))
        return 1.0 / (1.0 + mean_corr / 5.0)

    def score(self, t: TrackedTarget) -> float:
        size_score = min(1.0, t.det.area / (self.frame_area * 0.05))
        track_score = TargetTracker.stability(t)
        servo_score = self._servo_stability(t)
        return (config.W_SIZE * size_score
                + config.W_STABILITY * track_score
                + config.W_SERVO * servo_score)

    def select(self, foes: list[TrackedTarget]) -> TrackedTarget | None:
        """Düşman doğrulanmış hedefler arasından angajman adayını seç."""
        if not foes:
            return None
        ranked = sorted(foes, key=self.score, reverse=True)
        return ranked[0]
