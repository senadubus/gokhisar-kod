"""`shared/` sözleşmesinin üç düğümde de tuttuğunu doğrulayan sapma testleri.

Neden bu testler var
--------------------
`pc/config.py`, `rpi/main.py`, `rpi/pid_controller.py` ve `stm32/main.c`
yukarı akış dosyaları; entegrasyon kapsamında değiştirilmedikleri için hâlâ
kendi literallerini taşıyorlar ve `shared`dan okumuyorlar. Yani sapma
*derleme zamanında* engellenemiyor.

Bu testler engellenemeyeni **görünür** kılıyor: iki taraftan biri değiştiği
anda kırılıyorlar ve hata mesajı hangi dosyanın hangi değerinin ayrıldığını
söylüyor. Görüntü işleme ekibi bir gün `from shared...` satırlarını kabul
ederse bu testlerin çoğu totolojiye dönüşür — o zaman silinebilirler.

En kritik olanı kare geometrisi: PC ile RPi farklı kare boyutu varsayarsa
nişan sabit bir açı kadar kayar ve PID bunu kapatamaz, çünkü sapma bir bozucu
etki değil referansın kendisindedir.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pc.integration import bootstrap  # noqa: F401  (sys.path kurulumu)
from shared import classes, engagement, geometry, protocol

import config as vision_config

RPI_DIR = bootstrap.PROJECT_ROOT / "rpi"
STM32_MAIN = bootstrap.PROJECT_ROOT / "stm32" / "main.c"


# ----------------------------------------------------------------------
# Yukarı akış dosyalarını *çalıştırmadan* okumak
# ----------------------------------------------------------------------
# `rpi/main.py` içe aktarılamaz: `hardware_links` üzerinden pyserial ve RPi'ye
# özgü GPIO kütüphanelerini çeker. Kaynağı AST ile okuyup yalnızca modül
# düzeyindeki sabit atamalarını değerlendiriyoruz — yan etkisiz ve bağımlılıksız.

def _module_constants(path: Path) -> dict[str, object]:
    """Bir Python dosyasındaki modül düzeyi sabit atamalarını çıkar.

    Çoklu atamayı da açar: `rpi/main.py` portu `HOST, PORT = "0.0.0.0", 5005`
    biçiminde tanımlıyor.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, object] = {}

    def bind(target: ast.expr, value: object) -> None:
        if isinstance(target, ast.Name):
            found[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (tuple, list)) and len(value) == len(target.elts):
                for element, item in zip(target.elts, value):
                    bind(element, item)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in node.targets:
            bind(target, value)
    return found


@pytest.fixture(scope="module")
def rpi_main() -> dict[str, object]:
    return _module_constants(RPI_DIR / "main.py")


@pytest.fixture(scope="module")
def rpi_pid_source() -> str:
    return (RPI_DIR / "pid_controller.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Kare geometrisi
# ----------------------------------------------------------------------

def test_frame_geometry_matches_vision_config():
    """`shared` ile `pc/config.py` aynı kare uzayını tanımlamalı."""
    assert geometry.FRAME_WIDTH == vision_config.FRAME_WIDTH
    assert geometry.FRAME_HEIGHT == vision_config.FRAME_HEIGHT
    assert geometry.FRAME_CENTER == tuple(vision_config.FRAME_CENTER)


def test_frame_geometry_matches_rpi_pid(rpi_pid_source: str):
    """RPi'nin PID'i de aynı kare boyutunu varsaymalı.

    `rpi/pid_controller.py` boyutu bir sınıf içinde varsayılan argüman olarak
    taşıyor, modül sabiti olarak değil; bu yüzden AST yerine düzenli ifadeyle
    okunuyor.
    """
    widths = {int(m) for m in re.findall(r"frame_w(?:idth)?\s*[:=]\s*int\s*=\s*(\d+)",
                                         rpi_pid_source)}
    heights = {int(m) for m in re.findall(r"frame_h(?:eight)?\s*[:=]\s*int\s*=\s*(\d+)",
                                          rpi_pid_source)}
    assert widths, "rpi/pid_controller.py içinde kare genişliği bulunamadı"
    assert widths == {geometry.FRAME_WIDTH}, (
        f"RPi {widths} varsayıyor, sözleşme {geometry.FRAME_WIDTH}"
    )
    assert heights == {geometry.FRAME_HEIGHT}, (
        f"RPi {heights} varsayıyor, sözleşme {geometry.FRAME_HEIGHT}"
    )


# ----------------------------------------------------------------------
# Sınıf kimlikleri
# ----------------------------------------------------------------------

def test_class_ids_match_vision_config():
    """Sınıf kimlikleri ve adları `pc/config.py` ile birebir aynı olmalı."""
    for target in classes.TargetClass:
        assert int(target) in vision_config.CLASS_NAMES, (
            f"{target.name} ({int(target)}) config.CLASS_NAMES'te yok"
        )
    assert classes.BALLOON_CLASS_ID == vision_config.BALLOON_CLASS_ID
    assert set(classes.MODEL_CLASS_IDS) == set(vision_config.MODEL_CLASS_IDS)


def test_display_names_cover_every_class():
    """Her sınıfın operatöre gösterilecek bir adı olmalı."""
    for target in classes.TargetClass:
        assert classes.display_name(int(target)) != f"Sınıf {int(target)}"


def test_balloon_is_not_engageable():
    """Balon bir hedef değil, hedefin işaretidir; angajman adayı olamaz."""
    assert not classes.is_engageable(classes.BALLOON_CLASS_ID)
    for class_id in classes.MODEL_CLASS_IDS:
        assert classes.is_engageable(class_id)


# ----------------------------------------------------------------------
# Portlar
# ----------------------------------------------------------------------

def test_command_port_matches_vision_config():
    assert protocol.COMMAND_PORT == vision_config.RPI_PORT


def test_command_port_matches_rpi_listener(rpi_main: dict[str, object]):
    """RPi'nin dinlediği port ile PC'nin bağlandığı port aynı olmalı."""
    listen_port = rpi_main.get("LISTEN_PORT", rpi_main.get("PORT"))
    assert listen_port is not None, "rpi/main.py içinde dinleme portu bulunamadı"
    assert listen_port == protocol.COMMAND_PORT


# ----------------------------------------------------------------------
# Angajman güvenliği
# ----------------------------------------------------------------------

def test_safe_engage_distances_match_rpi(rpi_main: dict[str, object]):
    """Güvenli angajman mesafeleri sözleşmeyle birebir aynı olmalı.

    Ayrıştıkları anda arayüzün gösterdiği menzil durumu ile RPi'nin ateş
    kapısı farklı şeyler söyler; operatör "menzilde" görürken sistem ateş
    etmez (ya da tersi).
    """
    rpi_table = rpi_main.get("SAFE_ENGAGE_DISTANCES")
    assert rpi_table is not None, "rpi/main.py içinde SAFE_ENGAGE_DISTANCES yok"
    normalized = {int(k): tuple(v) for k, v in rpi_table.items()}
    assert normalized == {int(k): tuple(v)
                          for k, v in engagement.SAFE_ENGAGE_DISTANCES_CM.items()}


def test_forbidden_zones_match_rpi(rpi_main: dict[str, object]):
    rpi_zones = rpi_main.get("FORBIDDEN_ZONES")
    assert rpi_zones is not None, "rpi/main.py içinde FORBIDDEN_ZONES yok"
    assert [tuple(z) for z in rpi_zones] == [tuple(z)
                                             for z in engagement.FORBIDDEN_ZONES]


def test_engage_stable_seconds_match_rpi(rpi_main: dict[str, object]):
    assert rpi_main.get("ENGAGE_STABLE_SECONDS") == engagement.ENGAGE_STABLE_SECONDS


def test_every_class_has_a_safe_distance():
    """Tanımsız mesafeli bir sınıf, sessizce hiç ateş edilemeyen bir sınıftır."""
    for target in classes.TargetClass:
        assert int(target) in engagement.SAFE_ENGAGE_DISTANCES_CM


def test_unknown_distance_is_never_safe():
    """LiDAR okunamadıysa ateş kapısı kapalı kalmalı."""
    assert not engagement.is_safe_distance(classes.TargetClass.IHA, None)
    assert not engagement.is_safe_distance(999, 500.0)


def test_forbidden_zone_boundaries():
    """Yasaklı bölge sınırları beklendiği gibi davranıyor mu."""
    assert engagement.in_forbidden_zone(10.0, 90.0)     # sol güvenlik bölgesi
    assert engagement.in_forbidden_zone(170.0, 90.0)    # sağ güvenlik bölgesi
    assert engagement.in_forbidden_zone(90.0, 160.0)    # aşağı, operatör tarafı
    assert not engagement.in_forbidden_zone(90.0, 90.0)  # merkez serbest


def test_servo_range_matches_stm32():
    """STM32'nin darbe haritası ile sözleşmenin açı aralığı aynı olmalı.

    `angle_to_pulse()` içindeki kırpma değeri değişirse (ör. KTR'nin vaat
    ettiği 270°'ye geçilirse) burası da değişmeli; aksi hâlde RPi STM32'nin
    kabul etmeyeceği açılar üretir.
    """
    source = STM32_MAIN.read_text(encoding="utf-8", errors="replace")
    limits = {float(m) for m in re.findall(r"angle\s*>\s*(\d+(?:\.\d+)?)f?", source)}
    assert limits, "stm32/main.c içinde açı kırpması bulunamadı"
    assert limits == {engagement.SERVO_MAX_ANGLE}, (
        f"STM32 {limits} derecede kırpıyor, sözleşme "
        f"{engagement.SERVO_MAX_ANGLE}"
    )


# ----------------------------------------------------------------------
# Mesaj şeması: RpiLink'in ürettiği JSON beyan edilen şemaya uymalı
# ----------------------------------------------------------------------

class _CapturingLink:
    """`RpiLink._send`i yakalayıp gönderilen sözlükleri biriktirir."""

    def __init__(self):
        from comms.rpi_link import RpiLink

        self.sent: list[dict] = []
        self.link = RpiLink(host="127.0.0.1", port=protocol.COMMAND_PORT)
        self.link._send = self._capture  # type: ignore[method-assign]

    def _capture(self, payload: dict) -> bool:
        self.sent.append(payload)
        return True


def test_rpi_link_messages_satisfy_declared_schema():
    """PC → RPi yönünün tek kodlayıcısı `RpiLink`; şemaya uyduğunu doğrula.

    `shared/protocol.py` bu yön için ikinci bir kodlayıcı yazmaz — yazsaydı iki
    ayrı gerçek olurdu. Bunun yerine şemayı beyan eder ve doğruluğu burada
    sınanır.
    """
    capture = _CapturingLink()
    capture.link.send_target(cx=640.0, cy=360.0, class_id=2,
                             track_id=7, locked=True)
    capture.link.send_engage(track_id=7, class_id=2)
    capture.link.send_manual(dx=-5.0, dy=2.5)
    capture.link.send_mode(autonomous=True)

    kinds = [payload["type"] for payload in capture.sent]
    assert kinds == [protocol.MessageType.TARGET, protocol.MessageType.ENGAGE,
                     protocol.MessageType.MANUAL, protocol.MessageType.MODE]

    for payload in capture.sent:
        missing = protocol.missing_fields(payload)
        assert not missing, f"{payload['type']} mesajında eksik alan: {missing}"


def test_rpi_accepts_every_message_type_we_send(rpi_main: dict[str, object]):
    """Gönderdiğimiz her mesaj tipi `rpi/main.py`de gerçekten işleniyor mu.

    Sessizce yok sayılan bir mesaj tipi, hata vermeden çalışmayan bir özellik
    demek; en pahalı hata türü.
    """
    source = (RPI_DIR / "main.py").read_text(encoding="utf-8")
    for message_type in protocol.REQUIRED_FIELDS:
        assert f'"{message_type}"' in source or f"'{message_type}'" in source, (
            f"rpi/main.py '{message_type}' mesajını işlemiyor"
        )


def test_telemetry_round_trip():
    """RPi → PC yönü: kodlanan telemetri aynen çözülebilmeli."""
    payload = protocol.telemetry(distance_cm=412.35, in_forbidden_zone=False,
                                 pan=91.2, tilt=88.7)
    decoded = protocol.decode_line(protocol.encode_line(payload))
    assert decoded == payload
    assert decoded["distance_cm"] == 412.4  # bir ondalığa yuvarlanır


def test_telemetry_omits_unknown_fields():
    """Ölçülemeyen alan hiç konmamalı; 0 göndermek "0 cm" gibi okunur."""
    payload = protocol.telemetry()
    assert payload == {"type": protocol.MessageType.TELEMETRY}


def test_decode_line_tolerates_garbage():
    """Bozuk satır akışı kesmemeli; `None` dönüp bir sonrakine geçilmeli."""
    assert protocol.decode_line(b"{bozuk json") is None
    assert protocol.decode_line(b"   \n") is None
    assert protocol.decode_line(b"[1, 2, 3]") is None  # nesne değil, dizi
    assert protocol.decode_line(b'{"type":"status","status":"OK"}') == {
        "type": "status", "status": "OK"
    }


def test_shared_has_no_heavy_dependencies():
    """`shared` Raspberry Pi'ye kopyalanabilmeli: yalnızca standart kütüphane.

    numpy/torch/opencv buraya sızarsa paket embedded tarafta içe aktarılamaz
    hâle gelir ve sözleşme fiilen yalnızca PC'ye ait olur.
    """
    heavy = {"numpy", "cv2", "torch", "ultralytics", "supervision",
             "PySide6", "filterpy", "serial"}
    shared_dir = bootstrap.PROJECT_ROOT / "shared"
    for path in sorted(shared_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            offenders = names & heavy
            assert not offenders, f"{path.name} ağır bağımlılık içeriyor: {offenders}"
