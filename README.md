# HSS — Hava Savunma Sistemi Yazılımı

YOLOv8s tabanlı hedef tespiti, ByteTrack takibi, dost-düşman ayrımı (IFF) ve
PID tabanlı taret yönelimi içeren hibrit mimarili hava savunma sistemi yazılımı.

Sistem üç fiziksel katmanda çalışır:

```
┌─────────────────────────┐   TCP/JSON    ┌──────────────────────┐   UART    ┌──────────────────┐
│  PC (YKİ)               │ ────────────► │  Raspberry Pi 5      │ ────────► │  STM32F411       │
│  Python + PySide6       │  hedef koord. │  Python              │  ANG /    │  C (bare metal)  │
│  ─ YOLOv8s tespit       │  sınıf, kilit │  ─ PID yönelim       │  ATES_ET  │  ─ 50 Hz PWM     │
│  ─ HSV küçük hedef      │  angajman     │  ─ Yasaklı bölge     │           │  ─ Servo sürüş   │
│  ─ Doğrulama + IFF      │               │  ─ LiDAR doğrulama   │ ◄──────── │  ─ MOSFET ateş.  │
│  ─ ByteTrack takip      │               │  ─ Angajman kararı   │  DURUM    │  ─ Geri bildirim │
│  ─ Önceliklendirme      │               │                      │           │                  │
│  ─ Durum makinesi + GUI │               │                      │           │                  │
└─────────────────────────┘               └──────────────────────┘           └──────────────────┘
```

## Klasör Yapısı

```
hss/
├── requirements.txt
├── pc/                          # Yer Kontrol İstasyonu (PC)
│   ├── config.py                # ★ TÜM eşikler/sabitler burada — kod içine sabit gömme!
│   ├── main.py                  # Pipeline: modülleri yaşam döngüsüne göre bağlar + giriş noktası
│   ├── detection/
│   │   ├── yolo_detector.py     # YOLOv8s sarmalayıcı + Detection veri sınıfı + ROI çıkarımı
│   │   └── hsv_detector.py      # HSV küçük hedef tespiti + dinamik ROI + yedek algılama
│   ├── validation/matcher.py    # Maket–balon eşleştirme (2 yöntem)
│   ├── iff/friend_foe.py        # Dost-düşman ayrımı (Hue medyan + zamansal oylama)
│   ├── tracking/tracker.py      # ByteTrack sarmalayıcı + servo için 2. Kalman filtresi
│   ├── evaluation/prioritizer.py# Ağırlıklı öncelik puanı + angajman adayı seçimi
│   ├── lifecycle/state_machine.py # Durum makinesi + kilit + 3 koşullu imha doğrulaması


```

---

## Kurulum

### PC
```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

## Modüller (Kod Haritası)

Veri, her karede şu sırayla akar (hepsi `main.py` → `Pipeline.process`):

### 1. Tespit — `detection/`
- `YoloDetector.detect(frame)` → `list[Detection]` (tam kare YOLOv8s)
- `HsvBalloonDetector.detect(frame)` → küçük/uzak kırmızı balon adayları
  (çift eşikli HSV maskesi → morfoloji → kontur → dairesellik ≥ 0.72)
- Her HSV adayı için `dynamic_roi()` üretilir ve `YoloDetector.detect_in_roi()`
  ile yeniden değerlendirilir (uzak hedefin etkin çözünürlüğünü artırır)
- `detect_backup()`: performans düşüşünde YOLO'suz yedek boru hattı
  (`Pipeline.backup_mode = True` ile devreye girer)

**`Detection`** tüm sistemin ortak veri sınıfıdır: `x1,y1,x2,y2,conf,class_id,source`.
`source` alanı `"yolo" | "hsv" | "yolo_roi"` olabilir; takipte kimlik
sürekliliği bu ortak format sayesinde korunur.

### 2. Doğrulama — `validation/matcher.py`
`TargetMatcher.match(frame, models, balloons)` iki yöntemle maket–balon eşler:
1. Maketin altındaki dinamik bölgeye düşen balon → doğrudan eşleşme.
   Bölge derinliği maket piksel yüksekliğine bağlıdır; hedef küçüldükçe
   kademeli genişler (`MATCH_REGION_*` sabitleri).
2. Eşleşmeyen **HSV** balonlarının üst ROI'sinde YOLO yeniden çalıştırılır;
   maket bulunursa eşleşir. Eşleşmeyenler elenmez, sonraki karede tekrar denenir.

### 3. IFF — `iff/friend_foe.py`
- Kırmızı = `DUSMAN`, camgöbeği = `DOST` (Hue medyanı, düşük doygunluklu
  pikseller dışlanır)
- **Tek kare asla yeterli değildir**: her `track_id` için 15 karelik geçmiş
  tutulur; en az `IFF_VOTE_MIN_FRAMES = 5` tutarlı oy gerekir
- `stage=2` → balonlu tüm maketler doğrudan düşman

### 4. Takip — `tracking/tracker.py`
- `TargetTracker.update(all_dets)`: YOLO + HSV tespitlerinin **birleşik listesi**
  ByteTrack'e verilir; dönen `TrackedTarget` sözlüğü `track_id` anahtarlıdır
- Ölçüm yoksa `misses` artar; `TRACK_BUFFER = 30` kare sonunda hedef düşer
- `ServoKalman`: ByteTrack'ten **bağımsız** ikinci Kalman filtresi; servoya
  giden merkez koordinatını yumuşatır (`update()` ölçümlü, `predict_only()` ölçümsüz)

### 5. Değerlendirme — `evaluation/prioritizer.py`
```
puan = 0.5·boyut + 0.3·takip_kararlılığı + 0.2·servo_kararlılığı
```
Yalnızca DÜŞMAN doğrulanmış hedefler puanlanır; en yüksek puanlı hedef
angajman adayıdır.

### 6. Yaşam Döngüsü — `lifecycle/state_machine.py`
Aşağıdaki durum makinesini yönetir; kilit toleransını ve imha koşullarını denetler.


## Hedef Yaşam Döngüsü

```
                 tespit           maket-balon        DÜŞMAN (IFF          öncelik
                 edildi           eşleşti            oylaması)            birincisi
   ┌────────┐            ┌──────────┐        ┌───────┐         ┌──────────┐         ┌─────────────┐
   │ DETECT ├───────────►│ VALIDATE ├───────►│ TRACK ├────────►│ EVALUATE ├────────►│ TARGET_LOCK │
   └────────┘            └──────────┘        └───▲───┘         └──────────┘         └──────┬──────┘
                                                 │                                         │
                                                 │  tolerans dışına çıkma /                │ merkezde kararlı
                                                 │  imha koşulları sağlanamadı             │ + LiDAR onayı
                                                 └────────────────────────────┐            ▼
                                                                              │      [ATEŞLEME]
                                                                    ┌─────────┴──┐        │
                                                                    │ 3 koşul eş │◄───────┘
                                                                    │ zamanlı mı?│  DESTROY_EVAL_DELAY_S
                                                                    └─────┬──────┘  sonra değerlendirilir
                                                                          │ evet
                                                                          ▼
                                                                    ┌───────────┐
                                                                    │ DESTROYED │
                                                                    └───────────┘
```

İmha üç koşulun **eş zamanlı** sağlanmasını gerektirir:
1. `DESTROY_MISS_FRAMES` boyunca yeniden tespit edilememe
2. Takip zincirinin sonlanması
3. Güven skorunun `DESTROY_CONF_THRESHOLD` altında kalması

Sağlanamazsa hedef TRACK'e döner (yanlış imha kararı önlenir).

---


## Ayar Parametreleri (Tuning)

Sahada ayarlanması gerekenler (hepsi `pc/config.py` ve `rpi/` içinde):

| Parametre | Yer | Açıklama |
|---|---|---|
| `HSV_RED_*`, `HSV_CYAN_*` | config.py | Işık koşuluna göre renk aralıkları — **ilk ayarlanacak şey** |
| `MIN_CIRCULARITY` | config.py | Balon dairesellik eşiği (düşürmek = daha toleranslı) |
| `YOLO_CONF_THRESHOLD` | config.py | Yanlış pozitif/negatif dengesi |
| `LOCK_TOLERANCE_PX`, `LOCK_STABLE_FRAMES` | config.py | Kilit hassasiyeti |
| `W_SIZE / W_STABILITY / W_SERVO` | config.py | Öncelik puan ağırlıkları (toplam 1.0) |
| `kp, ki, kd` | rpi/pid_controller.py | PID katsayıları — önce yalnız Kp, sonra Kd, en son Ki |
| `FORBIDDEN_ZONES` | rpi/main.py | Mekanik montaja göre yasaklı açı bölgeleri |
| `SAFE_ENGAGE_DISTANCES` | rpi/main.py | Sınıf bazlı güvenli angajman mesafeleri (cm) |
| `SERVO_MIN_US / MAX_US` | stm32/main.c | Kullanılan servonun gerçek darbe aralığı |

**PID ayar sırası:** `Ki=Kd=0` ile başlayıp hedefe salınımlı yaklaşana dek `Kp`
artırın → salınımı sönümleyene dek `Kd` ekleyin → kalıcı hata varsa çok küçük `Ki`.

---
