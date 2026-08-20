# Raspberry Pi 5 — Atış Kontrol

Üst katmandan TCP JSON alır → PID / menzil kararı → STM32’ye UART binary gönderir.

## Kurulum

```bash
cd rpi5
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

UART: `raspi-config` → Serial Port açık, login shell kapalı.

Kablo:
- RPi TXD → STM32 PA10 (RX)
- RPi RXD → STM32 PA9 (TX)
- GND ortak
- LiDAR ayrı UART (`/dev/ttyAMA1`)

## Çalıştırma

```bash
python -m fire_control.main \
  --stm-port /dev/ttyAMA0 \
  --lidar-port "" \
  --tcp-port 5005 \
  --video-host 192.168.137.147 \
  --frame-w 640 \
  --frame-h 480
```

Varsayılan PID preset: **`iyi_yatay`** → `P=0.034 I=0 D=0.010` + tilt yerçekimi FF (`Kg=0.8°`, `cos(elev)`).

```bash
# Preset açık (varsayılan)
python -m fire_control.main --pid-preset iyi_yatay ...

# FF’yi elle ayarla / kapat
python -m fire_control.main --tilt-gravity-kg 1.2 --tilt-gravity-mode cos ...
python -m fire_control.main --tilt-gravity-kg 0 ...

# Preset’siz
python -m fire_control.main --pid-preset none --kp 0.034 --kd 0.010 ...
```

Yerçekimi FF, PID state’ine birikmez; STM’ye giden `tilt_cmd = tilt + Kg·cos(elev)` (droop telafisi). Aksi halde her döngüde Δ’ya eklenen Kg ramp yapardı.

`--video-host` = **PC IP**. Kamera fire_control ile birlikte sürekli UDP:5000’e gider.
Arayüz Başlat = sadece dinler; Durdur = PC tarafını kapatır, Pi yayına devam eder.

Yatay servo tersliği tercihen PC `SERVO_INVERT_PAN` / `SERVO_INVERT_PAN_AUTO` ile.
RPi `--invert-x` yalnız PC invert yoksa.

Örnek giriş mesajları: `../PROTOCOL.md`
