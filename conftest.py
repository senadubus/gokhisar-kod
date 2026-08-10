"""Pytest ortak kurulumu — depo kökünden tek komutla tüm testleri çalıştırır.

İki test paketi iki farklı import köküne göre yazılmış:

* ``pc/vision/tests/`` — yukarı akıştan gelen görüntü işleme testleri.
  ``import config`` ve ``from detection.yolo_detector import ...`` gibi düz
  isimler kullanıyorlar; ``cd pc`` yapılarak çalıştırılmak üzere yazılmışlardı.
* ``tests/`` — entegrasyon ve sözleşme testleri. ``from pc.integration import ...``
  ile depo kökünü kök kabul ediyorlar.

Görüntü işleme dosyalarına dokunmama kararı gereği o testlerin import satırları
değiştirilmedi. Kökleri `sys.path`e eklemek aynı sonucu bedelsiz veriyor;
kurulumun tek doğru kaynağı `pc/integration/bootstrap.py`, burada yalnızca o
çağrılıyor.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pc.integration import bootstrap  # noqa: E402  (yol kurulumundan sonra)

bootstrap.install()

# `pc/vision/tests/test_modules.py` içindeki PID testi RPi modülünü
# `dirname(__file__)/../../rpi` diye göreli yoldan buluyor. Bu varsayım dosya
# `pc/tests/` altındayken doğruydu; `pc/vision/tests/` altına taşınınca bir
# dizin kaydı. Testi düzeltmek görüntü işleme dosyasına dokunmak olurdu, o
# yüzden yolu buradan sağlıyoruz.
#
# `insert` değil `append`: `rpi/main.py` ile kökteki `main.py` aynı modül adını
# taşıyor, önek verirsek uygulamanın giriş noktası gölgelenirdi.
_RPI_DIR = str(PROJECT_ROOT / "rpi")
if _RPI_DIR not in sys.path:
    sys.path.append(_RPI_DIR)
