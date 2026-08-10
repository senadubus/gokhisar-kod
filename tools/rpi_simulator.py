#!/usr/bin/env python3
"""Donanımsız uçtan uca test için Raspberry Pi taklidi.

`rpi/main.py`'nin ağ davranışını birebir taklit eder: 5005 portunu dinler,
satır tabanlı JSON okur, `mode` / `manual` / `target` / `engage` mesajlarını
aynı mantıkla işler. Farkı, gerçek `rpi/main.py`'nin yapmadığı bir şeyi de
yapmasıdır: PC'ye telemetri geri yollar. Böylece arayüzün mesafe göstergesi,
yasak bölge uyarısı ve ateşleme bildirimi yolları donanım olmadan test
edilebilir.

`rpi/` altındaki gerçek kod değiştirilmedi; bu dosya yalnızca bir test
aracıdır ve `tools/` altında durur.

Çalıştırma:
    python tools/rpi_simulator.py
    python tools/rpi_simulator.py --distance 450 --verbose

Sonra ayrı bir terminalde:
    GOKHISAR_RPI_HOST=127.0.0.1 python main.py
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Güvenlik kuralları ve protokol sözleşmeden geliyor. Daha önce bunlar
# `rpi/main.py`den içe aktarılıyordu; o yol `hardware_links` üzerinden pyserial'ı
# da sürüklüyor ve içe aktarma başarısız olunca sessizce kopyalanmış
# varsayılanlara düşülüyordu — yani simülatörün gerçeği temsil etmeyi bıraktığı
# görünmez bir durum vardı. `shared` hem hafif hem tek doğru kaynak; RPi ile
# aynı değerleri taşıdığını `tests/test_contract.py` doğruluyor.
from shared import engagement, geometry, protocol
from shared.engagement import in_forbidden_zone


class SimulatedPanTilt:
    """`PanTiltController`'ın basitleştirilmiş, PID'siz karşılığı.

    Simülatörün amacı PID'yi doğrulamak değil, protokolü doğrulamak. Gerçek
    PID'yi taklit etmeye çalışmak yanıltıcı bir "çalışıyor" hissi verirdi.
    """

    def __init__(self, frame_w: int = geometry.FRAME_WIDTH,
                 frame_h: int = geometry.FRAME_HEIGHT):
        self.cx, self.cy = frame_w / 2, frame_h / 2
        self.pan = engagement.SERVO_CENTER_ANGLE
        self.tilt = engagement.SERVO_CENTER_ANGLE

    def step(self, target_x: float, target_y: float) -> tuple[float, float]:
        clamp = engagement.clamp_angle
        self.pan = clamp(self.pan + (target_x - self.cx) * 0.01)
        self.tilt = clamp(self.tilt + (target_y - self.cy) * 0.01)
        return self.pan, self.tilt

    def manual(self, dx: float, dy: float) -> tuple[float, float]:
        clamp = engagement.clamp_angle
        self.pan = clamp(self.pan + dx)
        self.tilt = clamp(self.tilt + dy)
        return self.pan, self.tilt


class SimulatedRpi:
    def __init__(self, conn: socket.socket, distance_cm: float, verbose: bool):
        self.conn = conn
        self.ctrl = SimulatedPanTilt()
        self.distance_cm = distance_cm
        self.verbose = verbose
        self.autonomous = False
        self.engage_request: dict | None = None
        self.in_range_since: float | None = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()

    # ---------- telemetri ----------
    def start_telemetry(self) -> threading.Thread:
        thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        thread.start()
        return thread

    def _telemetry_loop(self) -> None:
        while not self._stop.is_set():
            self.send(protocol.telemetry(
                distance_cm=self.distance_cm,
                in_forbidden_zone=in_forbidden_zone(self.ctrl.pan, self.ctrl.tilt),
                pan=self.ctrl.pan,
                tilt=self.ctrl.tilt,
            ))
            self._stop.wait(0.2)

    def send(self, payload: dict) -> None:
        line = protocol.encode_line(payload)
        with self._send_lock:
            try:
                self.conn.sendall(line)
            except OSError:
                self._stop.set()

    def stop(self) -> None:
        self._stop.set()

    # ---------- mesaj işleme (rpi/main.py ile aynı mantık) ----------
    def handle(self, msg: dict) -> None:
        mtype = msg.get("type")

        if mtype == protocol.MessageType.MODE:
            self.autonomous = bool(msg["autonomous"])
            self.engage_request = None
            self._log(f"mod: {'OTONOM' if self.autonomous else 'MANUEL'}")
            self.send(protocol.status("OTONOM" if self.autonomous else "MANUEL"))

        elif mtype == protocol.MessageType.MANUAL and not self.autonomous:
            # dx/dy artımdır (protocol.MANUAL_IS_DELTA), mutlak açı değil.
            pan, tilt = self.ctrl.manual(msg["dx"], msg["dy"])
            self._command_angles(pan, tilt)

        elif mtype == protocol.MessageType.TARGET and self.autonomous:
            pan, tilt = self.ctrl.step(msg["cx"], msg["cy"])
            self._command_angles(pan, tilt)
            self._check_engagement(msg.get("class_id"), msg.get("track_id"))

        elif mtype == protocol.MessageType.ENGAGE:
            self.engage_request = msg
            self.in_range_since = None
            self._log(f"angajman talebi: {msg}")

    def _command_angles(self, pan: float, tilt: float) -> None:
        if in_forbidden_zone(pan, tilt):
            self._log(f"YASAK BÖLGE pan={pan:.1f} tilt={tilt:.1f} — iletilmedi")
            return
        self._log(f"servo pan={pan:.1f} tilt={tilt:.1f}")

    def _check_engagement(self, class_id, track_id) -> None:
        if self.engage_request is None or class_id is None:
            return
        if engagement.is_safe_distance(class_id, self.distance_cm):
            if self.in_range_since is None:
                self.in_range_since = time.monotonic()
            elif (time.monotonic() - self.in_range_since
                    >= engagement.ENGAGE_STABLE_SECONDS):
                self._log(f"ATEŞ: class={class_id} dist={self.distance_cm:.0f}cm")
                payload = protocol.event_fired(track_id)
                payload.update(class_id=class_id, distance_cm=self.distance_cm)
                self.send(payload)
                self.engage_request = None
                self.in_range_since = None
        else:
            self.in_range_since = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[SIM] {message}", flush=True)


def serve(host: str, port: int, distance_cm: float, verbose: bool) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[SIM] PC bağlantısı bekleniyor {host}:{port}", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"[SIM] PC bağlandı: {addr}", flush=True)
        rpi = SimulatedRpi(conn, distance_cm, verbose)
        rpi.start_telemetry()
        buffer = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    message = protocol.decode_line(line)
                    if message is not None:
                        rpi.handle(message)
        except OSError:
            pass
        finally:
            rpi.stop()
            conn.close()
            print("[SIM] PC bağlantısı koptu, yeniden bekleniyor", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="GÖKHİSAR RPi simülatörü")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=protocol.COMMAND_PORT)
    parser.add_argument("--distance", type=float, default=500.0,
                        help="Sabit LiDAR mesafesi (cm)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        serve(args.host, args.port, args.distance, args.verbose)
    except KeyboardInterrupt:
        print("\n[SIM] kapatıldı", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
