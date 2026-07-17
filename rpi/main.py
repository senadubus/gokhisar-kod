"""Raspberry Pi 5 ana programı — Atış Kontrol Yazılımı (üst katman).

PC'den gelen X-Y koordinatlarıyla PID yönelimi çalıştırır, yasaklı açı
bölgelerini yazılımsal denetler (bu bölgelere komut STM32'ye iletilmez),
LiDAR ile sınıf-bazlı güvenli angajman mesafesini ve zamansal
kararlılığı doğrular; tüm koşullar sağlanınca STM32'ye "ATES_ET" gönderir.
"""
import json
import socket
import threading
import time

from pid_controller import PanTiltController
from hardware_links import LidarReader, Stm32Link

HOST, PORT = "0.0.0.0", 5005

# Yasaklı açı bölgeleri: (pan_min, pan_max, tilt_min, tilt_max)
FORBIDDEN_ZONES = [
    (0.0, 20.0, 0.0, 180.0),      # sol güvenlik bölgesi
    (160.0, 180.0, 0.0, 180.0),   # sağ güvenlik bölgesi
    (0.0, 180.0, 150.0, 180.0),   # aşağı (operatör) bölgesi
]

# Sınıf bazlı güvenli angajman mesafeleri (cm): class_id -> (min, max)
SAFE_ENGAGE_DISTANCES = {
    0: (300, 1500), 1: (300, 1500), 2: (300, 1500), 3: (300, 1500),
    4: (200, 1500),
}
ENGAGE_STABLE_SECONDS = 1.0       # mesafe içinde kararlı kalma şartı


def in_forbidden_zone(pan: float, tilt: float) -> bool:
    return any(p1 <= pan <= p2 and t1 <= tilt <= t2
               for p1, p2, t1, t2 in FORBIDDEN_ZONES)


class FireControl:
    def __init__(self):
        self.ctrl = PanTiltController()
        self.lidar = LidarReader()
        self.stm32 = Stm32Link()
        self.autonomous = False
        self.engage_request: dict | None = None
        self.in_range_since: float | None = None
        threading.Thread(target=self._status_loop, daemon=True).start()

    # ---------- STM32 geri bildirimi (kapalı çevrim) ----------
    def _status_loop(self):
        while True:
            status = self.stm32.read_status()
            if status:
                print(f"[STM32] {status}")   # üst katmana iletilir/loglanır
            time.sleep(0.01)

    # ---------- açı komutu (yasaklı bölge denetimli) ----------
    def _command_angles(self, pan: float, tilt: float):
        if in_forbidden_zone(pan, tilt):
            print(f"[GUVENLIK] Yasakli bolge: pan={pan:.1f} tilt={tilt:.1f} — iletilmedi")
            return
        self.stm32.send_angles(pan, tilt)

    # ---------- mesaj işleme ----------
    def handle(self, msg: dict):
        mtype = msg.get("type")

        if mtype == "mode":
            self.autonomous = bool(msg["autonomous"])
            self.engage_request = None

        elif mtype == "manual" and not self.autonomous:
            pan, tilt = self.ctrl.manual(msg["dx"], msg["dy"])
            self._command_angles(pan, tilt)

        elif mtype == "target" and self.autonomous:
            pan, tilt = self.ctrl.step(msg["cx"], msg["cy"])
            self._command_angles(pan, tilt)
            self._check_engagement(msg.get("class_id"))

        elif mtype == "engage":
            self.engage_request = msg
            self.in_range_since = None

    # ---------- LiDAR mesafe doğrulaması + zamansal kararlılık ----------
    def _check_engagement(self, class_id):
        if self.engage_request is None or class_id is None:
            return
        dist = self.lidar.distance_cm
        limits = SAFE_ENGAGE_DISTANCES.get(class_id)
        if dist is None or limits is None:
            self.in_range_since = None
            return

        lo, hi = limits
        if lo <= dist <= hi:
            if self.in_range_since is None:
                self.in_range_since = time.monotonic()
            elif time.monotonic() - self.in_range_since >= ENGAGE_STABLE_SECONDS:
                print(f"[ANGAJMAN] class={class_id} dist={dist:.0f}cm — ATES_ET")
                self.stm32.send_fire()
                self.engage_request = None
                self.in_range_since = None
        else:
            self.in_range_since = None   # kararlılık sayacı sıfırlanır


def main():
    fc = FireControl()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[RPi] PC baglantisi bekleniyor {HOST}:{PORT}")

    while True:
        conn, addr = srv.accept()
        print(f"[RPi] PC baglandi: {addr}")
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        fc.handle(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        finally:
            conn.close()
            print("[RPi] PC baglantisi koptu, yeniden bekleniyor")


if __name__ == "__main__":
    main()
