"""4.2.2.7 Hedefe Yönelme — PID kontrol algoritması (Raspberry Pi 5).

Hedefin görüntü merkezine olan hatasını minimize eder:
  P: mevcut hataya bağlı hızlı yönelim
  I: birikmiş hatayı gidererek kalıcı konum hatasını sıfırlar
  D: hata değişim hızını sönümleyerek aşım/titreşimi azaltır
"""
import time


class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = -90.0, out_max: float = 90.0,
                 integral_limit: float = 50.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.monotonic()
        dt = 0.02 if self._prev_time is None else max(1e-4, now - self._prev_time)
        self._prev_time = now

        # P
        p = self.kp * error
        # I (anti-windup sınırlı)
        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i = self.ki * self._integral
        # D
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        out = p + i + d
        return max(self.out_min, min(self.out_max, out))


class PanTiltController:
    """X-Y piksel hatasını pan-tilt açı komutlarına dönüştürür."""

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.cx, self.cy = frame_w / 2, frame_h / 2
        self.pid_pan = PID(kp=0.03, ki=0.002, kd=0.008)
        self.pid_tilt = PID(kp=0.03, ki=0.002, kd=0.008)
        self.pan_angle = 90.0     # servo orta konum
        self.tilt_angle = 90.0

    def step(self, target_x: float, target_y: float) -> tuple[float, float]:
        err_x = target_x - self.cx
        err_y = target_y - self.cy
        self.pan_angle += self.pid_pan.update(err_x) * 0.1
        self.tilt_angle += self.pid_tilt.update(err_y) * 0.1
        self.pan_angle = max(0.0, min(180.0, self.pan_angle))
        self.tilt_angle = max(0.0, min(180.0, self.tilt_angle))
        return self.pan_angle, self.tilt_angle

    def manual(self, dx: float, dy: float) -> tuple[float, float]:
        self.pan_angle = max(0.0, min(180.0, self.pan_angle + dx))
        self.tilt_angle = max(0.0, min(180.0, self.tilt_angle + dy))
        self.pid_pan.reset()
        self.pid_tilt.reset()
        return self.pan_angle, self.tilt_angle
