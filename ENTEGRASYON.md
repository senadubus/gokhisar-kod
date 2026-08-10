# Arayüz/Backend ↔ Görüntü İşleme Entegrasyonu

Bu belge, `KaganOzbey/Gokhisar` deposundaki PySide6 yer kontrol arayüzünün bu
depoya (`senadubus/gokhisar-kod`) nasıl taşındığını, hangi köprülerin yazıldığını
ve hangi kararların neden alındığını anlatır.

**Temel kural:** `pc/vision/`, `rpi/`, `stm32/` ve `pc/config.py` altındaki görüntü
işleme kodunun tek satırı değiştirilmedi — dosyalar yalnızca `git mv` ile yeni
yerlerine taşındı, içerikleri bit düzeyinde aynı. Uyum gerektiren her yer
`pc/integration/` altındaki köprü katmanında çözüldü.

Bunu her an doğrulayabilirsiniz:

```bash
git diff HEAD --stat -- 'pc/vision/' 'rpi/' 'stm32/'   # tüm satırlar 0 olmalı
```

---

## 1. Neden bir köprü katmanı gerekti

İki depo birbirinden habersiz geliştirilmişti ve altı noktada uyuşmuyordu:

| # | Uyuşmazlık | Sonucu ne olurdu | Çözüm |
|---|---|---|---|
| 1 | Arayüz ikili (binary) TCP paketi gönderiyor, `rpi/main.py` satır tabanlı JSON okuyor | İki uç birbirini hiç anlamaz | `RpiChannel` + `RpiLinkWorker`, `pc/vision/comms/rpi_link.py`'yi kullanır |
| 2 | Modelin sınıf sırası `pc/config.py` ile farklı | RPi yanlış güvenli mesafe tablosunu seçer | `ClassMap` |
| 3 | `README`'de anlatılan `pc/vision/main.py` orkestratörü depoda yok | Modüller birbirine hiç bağlanmamış | `VisionPipeline` |
| 4 | Arayüz mutlak servo açısı gönderiyor, `PanTiltController.manual()` artım bekliyor | Taret her komutta uçar | `RpiLinkWorker` mutlak→artım dönüşümü |
| 5 | `pc/config.py`'de `RPI_HOST` yer tutucu (`xxxx.xxxx.xxxx.xxxx`) | Bağlantı hiç kurulamaz | `Settings` (ortam değişkeni + makul varsayılan) |
| 6 | KTR 4.4.2'deki **sistem** durum makinesi hiçbir depoda yok | Operatör sistemin ne yaptığını göremez | `SystemStateMachine` |

---

## 2. Dizin yapısı

Ağaç, sistemin **fiziksel düğümlerini** yansıtır: bir dizin bir makinedir.
`shared/` üçünün üzerinde anlaştığı sözleşmedir.

```
gokhisar-kod/
├── main.py                        # YENİ — uygulama giriş noktası
├── conftest.py                    # YENİ — tüm testleri kökten çalıştırır
├── ENTEGRASYON.md                 # bu dosya
│
├── shared/                        # YENİ — düğümler arası sözleşme (yalnız stdlib)
│   ├── classes.py                 #   sınıf kimlikleri + görünen adlar
│   ├── geometry.py                #   kare boyutu ve merkezi
│   ├── protocol.py                #   portlar, mesaj tipleri, telemetri kurucuları
│   └── engagement.py              #   güvenli mesafeler, yasaklı bölgeler, servo aralığı
│
├── pc/                            # ── PC'de tek uygulama, tek süreç ──
│   ├── config.py                  # DOKUNULMADI — görüntü işleme eşikleri
│   ├── vision/                    # DOKUNULMADI — senadubus boru hattı
│   │   ├── detection/  validation/  tracking/  iff/  evaluation/  lifecycle/
│   │   ├── comms/rpi_link.py      #   PC→RPi protokolünün tek kodlayıcısı
│   │   └── tests/test_modules.py
│   ├── ui/                        # KaganOzbey YKİ (furkan_son_kod dalı)
│   │   ├── main_window.py  styles.py  components/
│   │   ├── workers/               #   QThread worker'ları
│   │   │   ├── vision_worker.py   #     YENİ — boru hattını iş parçacığında koşturur
│   │   │   ├── rpi_link_worker.py #     YENİ — JSON komut kanalı
│   │   │   └── gstreamer_video_worker.py, detection_worker.py, target_simulator.py
│   │   └── utils/config.py        #   arayüze özgü ayarlar
│   └── integration/               # YENİ — köprü katmanı
│       ├── bootstrap.py           #   sys.path kurulumu
│       ├── settings.py            #   birleşik ayarlar
│       ├── class_map.py           #   model → sistem sınıf çevirisi
│       ├── vision_pipeline.py     #   eksik orkestratör
│       ├── system_state.py        #   KTR 4.4.2 sistem FSM'i
│       └── rpi_channel.py         #   RpiLink sarmalayıcısı + telemetri okuma
│
├── rpi/                           # DOKUNULMADI — ayrı süreç: PID, LiDAR, güvenlik
├── stm32/                         # DOKUNULMADI — ayrı MCU: 50 Hz PWM, MOSFET
│
├── models/best.pt                 # YOLO ağırlıkları (git'e girmez, .gitignore'da)
├── tests/
│   ├── test_integration.py        # YENİ — köprü katmanı testleri
│   └── test_contract.py           # YENİ — sözleşme sapma testleri
└── tools/rpi_simulator.py         # YENİ — donanımsız uçtan uca test
```

`pc/ui/` içinde ne varsa arayüz uygulamasına, `pc/vision/` içinde ne varsa
görüntü işlemeye ait. İkisinin birbirine değdiği her satır `pc/integration/`
altındadır; yukarı akıştan gelen bir commit bu üç dizinden yalnızca
`pc/vision/`e dokunur, dolayısıyla çakışma yüzeyi sıfırdır.

### 2.1 `shared/` — sözleşme neden ayrı bir paket

Aynı sabit üç yerde bağımsız olarak tanımlanıyordu. En tehlikelisi kare
geometrisiydi: `pc/config.py` 1280x720 diyordu, `rpi/pid_controller.py` de ayrı
bir literal olarak 1280x720 diyordu ve ikisi birbirini tanımıyordu. Kamera
çözünürlüğü değişseydi PC koordinatları yeni uzayda üretmeye başlar, RPi hatayı
hâlâ eski merkeze göre hesaplardı. Ortaya çıkan sabit nişan kaymasını **PID
düzeltemez**, çünkü sapma bir bozucu etki değil referansın kendisindedir.

`shared/` yalnızca standart kütüphaneye bağlıdır; olduğu gibi Raspberry Pi'ye
kopyalanabilir. numpy'a bağımlı `Detection` gibi tipler bilerek dışarıda
bırakıldı — RPi hiçbir zaman bir `Detection` görmez, yalnızca
`{cx, cy, class_id, track_id, locked}` alır.

**Bugünkü sınır.** `pc/config.py` ve `rpi/*.py` yukarı akış dosyaları olduğu ve
değiştirilmedikleri için henüz `shared`dan *okumuyorlar*; kendi literallerini
taşımaya devam ediyorlar. Yani sapma derleme zamanında engellenemiyor.
`tests/test_contract.py` engellenemeyeni görünür kılıyor: her iki taraf da
`shared` ile karşılaştırılıyor ve biri ayrıldığı anda test kırılıp hangi
dosyanın hangi değerinin kaydığını söylüyor. Görüntü işleme ekibi bir gün
`from shared import ...` satırlarını kabul ederse geçiş birkaç satırlık bir
değişiklik olur.

Şu an sözleşmeyi gerçekten tüketen taraflar: `pc/integration/`, `pc/ui/` ve
`tools/rpi_simulator.py`.

---

## 3. Veri akışı

```
     Raspberry Pi kamerası
              │  RTP/JPEG over UDP:5000
              ▼
     GStreamerVideoWorker ──────────────► VideoDisplay (ham görüntü)
              │ frame_received(bytes)
              ▼
     VisionWorker  (QThread, latest-only kuyruk)
              │
              ▼
     VisionPipeline.process(frame)
       1. kareyi 1280x720'a normalize et
       2. YOLO tam kare  +  HSV balon  +  dinamik ROI ikinci geçiş
       3. sınıf kimliklerini config uzayına çevir (ClassMap)
       4. maket–balon eşleştirme doğrulaması (TargetMatcher)
       5. ByteTrack ile takip (TargetTracker)
       6. IFF: renk bandı + zamansal oylama (FriendFoeClassifier)
       7. önceliklendirme (TargetPrioritizer)
       8. hedef yaşam döngüsü + kilit (TargetLifecycleManager)
       9. servo hedefi için Kalman kestirimi (ServoKalman)
      10. imha doğrulaması
              │
              ├── detections_ready(DetectionFrame) ──► VideoDisplay (kutular)
              └── result_ready(PipelineResult) ──────► MainWindow
                                                          │
                          SystemStateMachine ◄────────────┤
                          StatusPanel / LogPanel ◄────────┤
                                                          ▼
                                                  RpiLinkWorker
                                                          │ TCP:5005 JSON
                                                          ▼
                                                  rpi/main.py → STM32
```

---

## 4. Protokol sözleşmesi

Taşıma: TCP, satır sonu ile ayrılmış JSON. Sözleşme `shared/protocol.py`de
beyan edilmiştir; iki yönün sahibi farklıdır.

### 4.1 PC → RPi — kodlayıcı `pc/vision/comms/rpi_link.py`

Bu yönü `RpiLink` kodluyor ve `rpi/main.py` tam olarak onun ürettiğini
bekliyor. `shared/protocol.py` bu yön için **ikinci bir kodlayıcı yazmaz** —
yazsaydı iki ayrı gerçek olurdu ve hangisinin doğru olduğu belirsizleşirdi.
Onun yerine zorunlu alanları `REQUIRED_FIELDS` içinde beyan eder;
`tests/test_contract.py::test_rpi_link_messages_satisfy_declared_schema`
`RpiLink`'in gerçekten bu alanları ürettiğini, bir başka test de `rpi/main.py`nin
gönderdiğimiz her mesaj tipini gerçekten işlediğini doğrular.

Satır tabanlı JSON, her mesaj `\n` ile biter.

```json
{"type":"mode","autonomous":true}
{"type":"manual","dx":-5.0,"dy":2.5}
{"type":"target","t":1738000000.0,"cx":700.0,"cy":400.0,"class_id":2,"track_id":7,"locked":true}
{"type":"engage","track_id":7,"class_id":2}
```

`manual` **artım** taşır, mutlak açı değil. Arayüzün kaydırıcıları mutlak
çalıştığı için dönüşümü `RpiLinkWorker` yapar:

```
pan  = 90 + azimut_kaydırıcı / 2      (kaydırıcı ±180° → servo 0–180°)
tilt = 90 + elevasyon_kaydırıcı       (kaydırıcı  ±90° → servo 0–180°)
artım = yeni_mutlak − son_gönderilen_mutlak
```

### 4.2 RPi → PC — kodlayıcı `shared/protocol.py`

`rpi/main.py`'nin bugünkü hâli PC'ye hiçbir şey göndermiyor; LiDAR mesafesini ve
STM32 geri bildirimini yalnızca kendi konsoluna basıyor. KTR 4.3 ise bu
telemetriyi açıkça vaat ediyor. Bu yönün bir sahibi olmadığı için kodlayıcıyı
`shared` üstlendi (`telemetry()`, `event_fired()`, `event_fail_safe()`,
`status()`). `tools/rpi_simulator.py` bugün bunları kullanıyor; RPi tarafı
yazıldığı gün aynı fonksiyonları çağırması yeterli, PC'de hiçbir değişiklik
gerekmez.

```json
{"type":"telemetry","distance_cm":412.0,"pan":95.3,"tilt":88.1,"in_forbidden_zone":false}
{"type":"event","event":"fired","track_id":7,"class_id":2,"distance_cm":412.0}
{"type":"event","event":"fail_safe","reason":"STM32 yanıt vermiyor"}
{"type":"status","status":"OTONOM"}
```

Bilinmeyen alanlar sessizce yok sayılır; RPi yeni alan eklediğinde arayüz
kırılmaz.

### 4.3 Sınıf kimliği çevirisi

Eğitilmiş modelin sınıf sırası `pc/config.py` ile aynı **değil**:

| Model (`best.pt`) | → | `pc/config.py` | RPi güvenli mesafe |
|---|---|---|---|
| 0 helikopter | → | 1 Helikopter | 300–1500 cm |
| 1 iha | → | 2 İHA | 300–1500 cm |
| 2 jet | → | 3 Savaş Uçağı | 300–1500 cm |
| 3 mini-micro-iha | → | 2 İHA | 300–1500 cm |
| 4 rocket | → | 0 Balistik Füze | 300–1500 cm |
| *(yok)* | | 4 Balon | 200–1500 cm |

Çeviri yapılmasaydı `rocket` tespiti RPi'ye `class_id=4` (balon) olarak gider,
balonun daha yakın angajman mesafesi uygulanırdı. Bu doğrudan bir güvenlik
hatasıdır ve `test_model_class_ids_are_remapped_to_config_space` bunu koruyor.

**Model balon sınıfı içermiyor.** Balon tespiti HSV renk eşiğine düşüyor
(`pc/vision/detection/hsv_detector.py`); boru hattı bunu başlangıçta log'a yazıyor.

---

## 5. Sistem durum makinesi (KTR 4.4.2)

`pc/vision/lifecycle/state_machine.py` **hedef başına** durum tutar. Onun üstünde,
operatöre gösterilen sistem durumu `pc/integration/system_state.py`'de
türetilir:

| Durum | Koşul | Arayüzde |
|---|---|---|
| `IDLE` | sistem durdurulmuş | BEKLEMEDE |
| `SCANNING` | çalışıyor, tespit yok | TARAMA |
| `DETECT` | tespit var, doğrulanmış iz yok | TESPİT |
| `TRACK` | doğrulanmış iz var | TAKİP |
| `EVALUATE` | angajman adayı seçildi | DEĞERLENDİRME |
| `TARGET_LOCK` | kilit koşulları sağlandı | HEDEF KİLİDİ |
| `ENGAGEMENT` | angajman komutu gitti (2 sn görünür) | ANGAJMAN |
| `DESTROYED` | imha doğrulandı (2,5 sn görünür) | İMHA EDİLDİ |
| `LOST` | izler kayboldu (3 sn görünür) | HEDEF KAYIP |
| `FAIL_SAFE` | kritik hata — **yalnızca RESET ile çıkılır** | GÜVENLİ DURUŞ |

Sıralama önemli: kilit varken "TESPİT" göstermek operatörü yanıltırdı. Geçici
durumların (ANGAJMAN, İMHA, KAYIP) görünür kalma süresi var, yoksa tek karede
geçip gözden kaçarlardı.

`FAIL_SAFE` mandallı: hatanın kendiliğinden "geçmesi" sahte bir güven yaratır.

---

## 6. Ateşleme güvenliği

Otonom angajman için **üç koşul birlikte** sağlanmalı:

1. Boru hattı hedef kilidini doğrulamış olmalı (`PipelineResult.locked`).
2. IFF kararı **DÜŞMAN** olmalı — dost ya da bilinmeyen hedefe ateş edilmez.
3. Operatör "KİLİDİ AÇ" düğmesine basmış olmalı. İnsan onayı otonom modda da
   devrededir; kilit kapalıyken sistem hedefi takip eder, kilitler, ama tetiği
   çekmez.

Mesafe ve yasak açı bölgesi denetimi PC'de tekrarlanmıyor: LiDAR verisi RPi'de
ve orada gecikmesiz. Aynı kontrolü iki yerde yapmak, iki yerin zamanla
ayrışması riskini getirir.

---

## 7. Kurulum ve çalıştırma

```bash
# 1) Sanal ortam
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt filterpy

# 2) Model ağırlıkları (git'e girmez)
cp /yol/best.pt models/best.pt
#   veya:  export GOKHISAR_YOLO_WEIGHTS=/yol/best.pt

# 3) Çalıştır
python main.py --rpi-host 192.168.1.100
```

### Donanımsız uçtan uca test

```bash
# Terminal 1 — sahte Raspberry Pi
python tools/rpi_simulator.py --verbose --distance 500

# Terminal 2 — arayüz
GOKHISAR_RPI_HOST=127.0.0.1 python main.py

# Terminal 3 — sahte video (isteğe bağlı, GStreamer gerekir)
gst-launch-1.0 videotestsrc ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5000
```

### Testler

```bash
pytest            # 46 test: 6 görüntü işleme + 20 entegrasyon + 20 sözleşme
```

`tests/test_contract.py` özellikle önemli: `shared/` ile `pc/config.py`,
`rpi/main.py`, `rpi/pid_controller.py` ve `stm32/main.c` değerlerinin hâlâ
örtüştüğünü doğrular. Bu testlerden biri kırılırsa iki düğüm farklı şeye
inanmaya başlamış demektir — kodu değil, hangi tarafın doğru olduğunu tartışın.

---

## 8. Ortam değişkenleri

Hiçbir ayar için kod değiştirmeye gerek yok.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GOKHISAR_RPI_HOST` | `192.168.1.100` | RPi IP (`pc/config.py` yer tutucusunun yerine) |
| `GOKHISAR_RPI_PORT` | `config.RPI_PORT` (5005) | RPi TCP portu |
| `GOKHISAR_UDP_VIDEO_PORT` | 5000 | Video akış portu |
| `GOKHISAR_YOLO_WEIGHTS` | `models/best.pt` | Model dosyası yolu |
| `GOKHISAR_HSV_ASSIST` | açık | HSV balon tespiti |
| `GOKHISAR_ROI_REFINE` | açık | Dinamik ROI ikinci geçişi |
| `GOKHISAR_MAX_ROI_REFINE` | 2 | Kare başına en fazla ROI geçişi |
| `GOKHISAR_TARGET_RATE_HZ` | 30 | RPi'ye hedef gönderim üst sınırı |
| `GOKHISAR_RECONNECT_S` | 2.0 | Yeniden bağlanma periyodu |
| `GOKHISAR_DESTROY_DELAY_S` | 2.0 | Ateşten sonra imha değerlendirme gecikmesi |
| `GOKHISAR_ENGAGE_REPEAT_S` | 1.0 | Angajman komutu tekrar aralığı |

---

## 9. Performans notları

* **Latest-only kuyruk.** Hem `VisionWorker` hem `RpiLinkWorker` yalnızca en
  yeni veriyi tutar, eskisini atar. Nişan alan bir sistemde bayat kutu, yavaş
  kutudan çok daha tehlikelidir.
* **ROI ikinci geçişi pahalı.** Kırpılan bölge de aynı 640 piksele
  ölçeklendiği için her ROI çağrısı neredeyse tam kare çıkarımı kadar sürüyor.
  Ölçüm (CPU): tam kare ~290 ms, sınırsız yenilemede kare süresi 1 sn'yi aşıyor.
  Bu yüzden yalnızca **üzerinde maket tespit edilmemiş** balonlar, en fazla
  `GOKHISAR_MAX_ROI_REFINE` tanesi yenileniyor.
* **Kare normalizasyonu.** `pc/config.py`'deki piksel eşikleri 1280x720'a göre
  yazılmış. Kamera başka çözünürlükte yayın yaparsa eşikler sessizce yanlış
  ölçeğe kayardı; boru hattı girişte tek noktada ölçekliyor.
* **GPU.** `ultralytics` CUDA varsa otomatik kullanır. CPU'da tam boru hattı
  ~3 FPS civarında kalır; yarışma makinesinde GPU şart.

---

## 10. Bilinen boşluklar

Bunların hepsi `rpi/` veya `pc/vision/` tarafında değişiklik gerektirdiği için,
"görüntü işleme koduna dokunma" kuralı gereği yapılmadı.

1. **Manuel modda ateşleme RPi'de tamamlanmıyor.** `rpi/main.py`'de
   `_check_engagement()` yalnızca `target` mesajı işlenirken çağrılıyor ve o dal
   `self.autonomous` şartına bağlı. Manuel modda gönderilen `engage` talebi
   kuyruğa alınır ama tetiklenmez. Arayüz bu durumu log'a yazıyor.
   *Gereken:* `rpi/main.py`'de manuel angajman dalı.
2. **RPi telemetri göndermiyor.** LiDAR mesafesi, yasak bölge durumu ve
   ateşleme onayı şu an yalnızca RPi konsoluna yazılıyor. Arayüzdeki mesafe
   göstergesi ve menzil bantları bu veri gelene kadar boş kalır.
   *Gereken:* `rpi/main.py`'de bölüm 4.2'deki satırların soketten yazılması.
3. **PID katsayı girişleri bağlı değil.** Kontrol panelindeki P/I/D alanları
   okunuyor ama gönderilmiyor; protokolde karşılığı olan bir mesaj tipi yok.
   *Gereken:* `rpi_link.py`'de `send_pid()` ve `rpi/main.py`'de karşılığı.
4. **Balon sınıfı modelde yok.** Balon tespiti HSV'ye bağımlı; farklı ışıkta
   `pc/config.py`'deki HSV eşikleri yeniden ayarlanmalı.
5. **Servo aralığı 180°, KTR 4.1.1 ise yatay eksende 270° vaat ediyor.**
   Hem `rpi/pid_controller.py` hem `stm32/main.c` iki ekseni de 0–180'e kırpıyor.
   `shared/engagement.py` kodun gerçeğini yazıyor ve `test_servo_range_matches_stm32`
   bunu kilitliyor. 270°'ye geçilecekse üçü birlikte değişmeli.
6. **`DetectionWorker` artık ana yolda değil.** Silinmedi: model dosyası dışında
   hiçbir şeye ihtiyaç duymadan hızlı bir "sadece tespit" doğrulaması yapmaya
   yarıyor. Uygulamanın yolu artık `VisionWorker` + `RpiLinkWorker`.
   Buna karşılık ikili TCP protokolü kullanan `network_worker.py` **silindi**:
   `shared/protocol.py` ile çelişen, kullanılmayan ikinci bir protokol
   uygulaması ileride yanlışlıkla referans alınabilecek bir tuzaktı. Aynı
   şekilde `utils/paths.py` ve `utils/serial_port.py` de hiçbir yerden
   çağrılmadıkları ve artık var olmayan bir dizin düzenine atıf yaptıkları için
   kaldırıldı.
7. **`README.md` eski ağacı anlatıyor.** Yukarı akışa ait olduğu için
   değiştirilmedi; `pc/detection/` yazan yerleri `pc/vision/detection/` diye
   okuyun. Güncel ağaç bu belgenin 2. bölümünde.
