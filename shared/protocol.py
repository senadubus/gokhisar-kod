"""PC ↔ Raspberry Pi haberleşme sözleşmesi — KTR 4.3, Şekil 4.13.

Taşıma: TCP, satır sonu ile ayrılmış JSON nesneleri (newline-delimited JSON).
KTR "veri güvenilirliği gereksinimi nedeniyle TCP/IP" ve "esnekliğin artırılması
amacıyla JSON formatı" diyor; uygulama bunu birebir karşılıyor.

İki yön farklı sahiplere ait
----------------------------
**PC → RPi.** Kodlamayı `pc/vision/comms/rpi_link.py` içindeki `RpiLink`
yapıyor ve `rpi/main.py` tam olarak onun ürettiğini bekliyor. Burada ikinci bir
kodlayıcı yazmıyoruz — iki kodlayıcı iki ayrı gerçek demek olurdu. Bunun yerine
şema **beyan ediliyor** ve `tests/test_contract.py` `RpiLink`'in gerçekten bu
alanları ürettiğini doğruluyor.

**RPi → PC.** Bu yönün henüz bir uygulaması yok: `rpi/main.py` STM32 geri
bildirimini ve LiDAR mesafesini yalnızca kendi konsoluna basıyor, PC'ye hiçbir
şey göndermiyor. KTR 4.3 ise bu telemetriyi açıkça vaat ediyor. Sahipsiz olduğu
için kodlayıcıyı `shared` üstleniyor: `tools/rpi_simulator.py` bugün bunu
kullanıyor, RPi tarafı yazıldığı gün aynı fonksiyonları çağırması yeterli.
"""

import json
from typing import Any

# ----------------------------------------------------------------------
# Portlar
# ----------------------------------------------------------------------

#: PC → RPi komut kanalı (TCP). `rpi/main.py` bu portu dinler.
COMMAND_PORT: int = 5005

#: RPi → PC video akışı (UDP, RTP/JPEG). GStreamer `udpsrc` bu portu dinler.
VIDEO_PORT: int = 5000

#: Saha kurulumunda RPi'nin varsayılan adresi; ortam değişkeniyle ezilebilir.
DEFAULT_RPI_HOST: str = "192.168.1.100"


# ----------------------------------------------------------------------
# Mesaj tipleri
# ----------------------------------------------------------------------

class MessageType:
    """`type` alanının alabileceği değerler."""

    # PC → RPi
    TARGET = "target"
    ENGAGE = "engage"
    MANUAL = "manual"
    MODE = "mode"

    # RPi → PC
    TELEMETRY = "telemetry"
    EVENT = "event"
    STATUS = "status"


class EventName:
    """`event` mesajlarındaki olay adları."""

    FIRED = "fired"
    FAIL_SAFE = "fail_safe"


# ----------------------------------------------------------------------
# Şema beyanı (PC → RPi yönü; kodlayıcı RpiLink'te)
# ----------------------------------------------------------------------

#: Mesaj tipi -> zorunlu alan adları. `t` gibi bilgilendirme amaçlı alanlar
#: zorunlu sayılmaz; RPi onları okumuyor.
REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    MessageType.TARGET: frozenset({"cx", "cy", "class_id", "track_id", "locked"}),
    MessageType.ENGAGE: frozenset({"track_id", "class_id"}),
    MessageType.MANUAL: frozenset({"dx", "dy"}),
    MessageType.MODE: frozenset({"autonomous"}),
}

#: Manuel yönelim komutu **mutlak açı değil artım** taşır: `rpi/main.py`
#: gelen dx/dy'yi `PanTiltController.manual()`'a veriyor, o da mevcut açının
#: üzerine ekliyor. Arayüz mutlak açı gösterdiği için `RpiLinkWorker` çevrimi
#: yapar; bu not sözleşmenin en kolay yanlış anlaşılan maddesi.
MANUAL_IS_DELTA: bool = True


def missing_fields(payload: dict[str, Any]) -> frozenset[str]:
    """Bir mesajda eksik olan zorunlu alanlar; bilinmeyen tipte boş küme."""
    required = REQUIRED_FIELDS.get(payload.get("type", ""))
    if required is None:
        return frozenset()
    return required - payload.keys()


# ----------------------------------------------------------------------
# Satır kodlama / çözme
# ----------------------------------------------------------------------

def encode_line(payload: dict[str, Any]) -> bytes:
    """Bir mesajı hat üzerinde taşınacak bayt satırına çevir."""
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode_line(line: bytes | str) -> dict[str, Any] | None:
    """Tek bir satırı çöz; bozuk satırda `None` (bağlantı düşürülmez).

    Gürültülü bir satır yüzünden akışı kesmek yanlış olur: TCP üzerinde tek
    bir bozuk kare, sonraki geçerli telemetriyi kaybetmeyi gerektirmez.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# ----------------------------------------------------------------------
# RPi → PC telemetri kurucuları (bu yönün sahibi shared)
# ----------------------------------------------------------------------

def telemetry(distance_cm: float | None = None,
              in_forbidden_zone: bool | None = None,
              pan: float | None = None,
              tilt: float | None = None) -> dict[str, Any]:
    """Periyodik durum telemetrisi.

    Alanlar isteğe bağlı: LiDAR okunamadığında `distance_cm` hiç konmaz.
    Arayüz bilmediği alanı sessizce yok sayar, eksik alanı da — böylece RPi
    tarafı yeni alan eklediğinde PC'de kod değişikliği gerekmez.
    """
    payload: dict[str, Any] = {"type": MessageType.TELEMETRY}
    if distance_cm is not None:
        payload["distance_cm"] = round(float(distance_cm), 1)
    if in_forbidden_zone is not None:
        payload["in_forbidden_zone"] = bool(in_forbidden_zone)
    if pan is not None:
        payload["pan"] = round(float(pan), 1)
    if tilt is not None:
        payload["tilt"] = round(float(tilt), 1)
    return payload


def event_fired(track_id: int | None = None) -> dict[str, Any]:
    """STM32'nin "ateşleme gerçekleşti" bildirimi (KTR 4.2.2.8 kapalı çevrim).

    PC bu mesajı aldığı anda imha değerlendirme sayacını başlatır; bu yüzden
    zamanlaması, PC'nin angajman *talebi* gönderdiği andan daha doğrudur.
    """
    payload: dict[str, Any] = {"type": MessageType.EVENT, "event": EventName.FIRED}
    if track_id is not None:
        payload["track_id"] = int(track_id)
    return payload


def event_fail_safe(reason: str) -> dict[str, Any]:
    """Güvenli duruşa geçiş bildirimi (KTR 4.4.2 FAIL_SAFE)."""
    return {"type": MessageType.EVENT, "event": EventName.FAIL_SAFE, "reason": reason}


def status(text: str) -> dict[str, Any]:
    """Serbest metinli durum satırı; arayüzün durum çubuğunda gösterilir."""
    return {"type": MessageType.STATUS, "status": text}
