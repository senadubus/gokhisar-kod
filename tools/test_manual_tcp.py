#!/usr/bin/env python3
"""PC'den RPi'ye tek manuel adım gönder — arayüz olmadan test.

Kullanım:
  python tools/test_manual_tcp.py 192.168.137.133
  python tools/test_manual_tcp.py 192.168.137.133 --dx 5 --dy 0
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description="RPi manuel TCP testi")
    p.add_argument("host", nargs="?", default="192.168.137.133")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--dx", type=float, default=5.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--count", type=int, default=3)
    args = p.parse_args()

    try:
        sock = socket.create_connection((args.host, args.port), timeout=3)
    except OSError as exc:
        print(f"[HATA] Bağlanamadı {args.host}:{args.port} → {exc}")
        return 1

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[OK] Bağlandı {args.host}:{args.port}")

    mode = {"type": "mode", "autonomous": False, "stage": 1}
    sock.sendall((json.dumps(mode) + "\n").encode())
    print(f"[>] {mode}")

    for i in range(args.count):
        msg = {"type": "manual", "dx": args.dx, "dy": args.dy}
        sock.sendall((json.dumps(msg) + "\n").encode())
        print(f"[>] #{i+1} {msg}")
        time.sleep(0.4)

    # Birkaç status satırı oku
    sock.settimeout(1.0)
    buf = b""
    try:
        while len(buf) < 4000:
            chunk = sock.recv(1024)
            if not chunk:
                break
            buf += chunk
            if buf.count(b"\n") >= 3:
                break
    except socket.timeout:
        pass

    sock.close()
    if not buf:
        print("[UYARI] RPi'den status gelmedi (yine de manuel gitmiş olabilir)")
        return 0

    print("[<] RPi cevap:")
    for line in buf.decode("utf-8", errors="replace").splitlines()[:5]:
        print(f"    {line[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
