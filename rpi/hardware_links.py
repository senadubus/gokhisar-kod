"""Raspberry Pi 5 donanım bağlantıları: LiDAR okuma + STM32 seri hattı."""
import threading
import time

import serial


class LidarReader:
    """UART LiDAR (ör. TF-Luna, 9 baytlık çerçeve: 0x59 0x59 dist_L dist_H ...).
    Arka plan iş parçacığında sürekli okur; son mesafeyi (cm) tutar."""

    def __init__(self, port: str = "/dev/ttyAMA0", baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.distance_cm: float | None = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            if self.ser.read(1) != b"\x59":
                continue
            if self.ser.read(1) != b"\x59":
                continue
            frame = self.ser.read(7)
            if len(frame) == 7:
                dist = frame[0] | (frame[1] << 8)
                strength = frame[2] | (frame[3] << 8)
                if strength > 100:          # zayıf sinyalleri ele
                    self.distance_cm = float(dist)

    def stop(self):
        self._run = False
        self.ser.close()


class Stm32Link:
    """RPi <-> STM32F411 UART protokolü.
    Komutlar (satır tabanlı ASCII, checksum'lu):
      ANG,<pan>,<tilt>*CS   -> servo açı komutu
      ATES_ET*CS            -> ateşleme komutu
    STM32 geri bildirimi: DURUM,<kod> satırları."""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.lock = threading.Lock()

    @staticmethod
    def _checksum(payload: str) -> str:
        cs = 0
        for ch in payload:
            cs ^= ord(ch)
        return f"{cs:02X}"

    def _send(self, payload: str):
        line = f"{payload}*{self._checksum(payload)}\n"
        with self.lock:
            self.ser.write(line.encode())

    def send_angles(self, pan: float, tilt: float):
        self._send(f"ANG,{pan:.1f},{tilt:.1f}")

    def send_fire(self):
        self._send("ATES_ET")

    def read_status(self) -> str | None:
        """STM32'den gelen durum geri bildirimi (kapalı çevrim)."""
        line = self.ser.readline().decode(errors="ignore").strip()
        return line or None

    def close(self):
        self.ser.close()
