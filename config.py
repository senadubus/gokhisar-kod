"""Sistem geneli konfigürasyon sabitleri."""

# ---------------- YOLO ----------------
YOLO_MODEL_PATH = "yolo_modeli.pt"   # eğitilmiş model
YOLO_IMG_SIZE = 640
YOLO_CONF_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45


CLASS_NAMES = {0: "fuze", 1: "helikopter", 2: "iha", 3: "ucak", 4: "balon"}
BALLOON_CLASS_ID = 4
MODEL_CLASS_IDS = {0, 1, 2, 3}

# ---------------- HSV (küçük hedef / yedek tespit) ----------------
# Kırmızı için çift eşik (Hue 0 civarı sarmal olduğu için iki aralık)
HSV_RED_LOWER_1 = (0, 120, 70)
HSV_RED_UPPER_1 = (10, 255, 255)
HSV_RED_LOWER_2 = (170, 120, 70)
HSV_RED_UPPER_2 = (180, 255, 255)

# Dost (camgöbeği/mavi) aralığı — IFF için
HSV_CYAN_LOWER = (80, 100, 70)
HSV_CYAN_UPPER = (130, 255, 255)

MIN_CIRCULARITY = 0.72        # balon adayı kabul eşiği
MIN_CONTOUR_AREA = 12         # piksel^2, çok küçük gürültüyü ele
ROI_SCALE_FACTOR = 6.0        # dinamik ROI = balon boyutu * katsayı

# ---------------- Doğrulama / Eşleştirme ----------------
MATCH_REGION_BASE_RATIO = 1.2     # maket yüksekliğine oranla eşleştirme bölgesi
MATCH_REGION_EXTEND_STEP = 0.4    # hedef küçüldükçe kademeli uzatma katsayısı
SMALL_TARGET_PX_HEIGHT = 40       # bu değerin altı "uzak hedef" sayılır

# ---------------- IFF ----------------
IFF_HISTORY_LEN = 15          # sınıflandırma geçmişi uzunluğu (kare)
IFF_VOTE_MIN_FRAMES = 5       # düşman doğrulaması için asgari tutarlı kare
HUE_RED_RANGES = [(0, 10), (170, 180)]
HUE_CYAN_RANGE = (80, 130)

# ---------------- Takip ----------------
TRACK_HIGH_CONF = 0.5
TRACK_LOW_CONF = 0.1
TRACK_BUFFER = 30             # kayıp hedefin tutulacağı kare sayısı
TRACK_MATCH_IOU = 0.8

# ---------------- Değerlendirme / Önceliklendirme ----------------
W_SIZE = 0.5                  # boyut ağırlığı
W_STABILITY = 0.3             # takip kararlılığı ağırlığı
W_SERVO = 0.2                 # servo düzeltme (az düzeltme = yüksek puan)

# ---------------- Kilitlenme ----------------
LOCK_TOLERANCE_PX = 25        # görüntü merkezine tolerans yarıçapı
LOCK_STABLE_FRAMES = 10       # kilit için gereken kararlı kare sayısı

# ---------------- İmha Değerlendirme ----------------
DESTROY_MISS_FRAMES = 45      # yeniden tespit edilememe süresi (kare)
DESTROY_CONF_THRESHOLD = 0.2  # güven skoru eşiği

# ---------------- Haberleşme (PC -> RPi) ----------------
RPI_HOST = "xxxx.xxxx.xxxx.xxxx"
RPI_PORT = 5005

# ---------------- Kamera ----------------
FRAME_WIDTH = 1280 # ilgili kameraya göre değiştiririz.
FRAME_HEIGHT = 720 # ilgili kameraya göre değiştirilmeli.
FRAME_CENTER = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
