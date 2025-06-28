# Araç Sayım & Takip Sistemi · YOLOv8 + DeepSORT

Gerçek‑zamanlı trafik akış analizi, nesne tespiti + takibi ve istatistik üretimi yapan tam yığın (Flask + Tailwind / DaisyUI) bir web uygulaması.
Tarayıcıdan bir MP4 yükleyin; sunucu YOLOv8 ile araçları algılar, DeepSORT ile kimlik korumalı izleme yapar, yoğunluk ısı haritası çıkarır ve sonucu anında geri sunar.

> **Model ağırlıkları**
> Tüm ön‑eğitimli dosyalar Google Drive’da:
> **[Modellerin Drive Linki](https://drive.google.com/drive/folders/1hO6VoK62yMVkQznoOacsIZ1G7zgJCmc0?usp=sharing)**
> Kod, `best_optimazed.pt` dosyasının projenin kök dizinine kopyalanmış olduğunu varsayar.

---

## Başlıca Özellikler

| Özellik                           | Açıklama                                                   |
| --------------------------------- | ---------------------------------------------------------- |
| **Çoklu nesne algılama**          | Ultralytics YOLOv8 (6 trafik sınıfı)                       |
| **Kimlik korumalı takip**         | DeepSORT – öz vektör + Kalman + Hungarian                  |
| **Isı haritası**                  | `numpy.histogram2d` → `matplotlib` ile görselleştirme      |
| **Anlık istatistikler**           | Araç türü dağılımı, zaman çizelgesi, toplam sayımlar       |
| **Hızlı video yeniden paketleme** | `imageio‑ffmpeg` + H.264 *faststart*                       |
| **Modern arayüz**                 | Tailwind CSS + DaisyUI; Chart.js ile etkileşimli grafikler |

---

## Kurulum

> Sanal ortam **isteğe bağlıdır**; paket çakışmalarını önlemek için önerilir.

```bash
# Depoyu klonla
git clone https://github.com/ssevvalyavuz/Bitirme_Projesi_Arac_Sayim.git
cd Bitirme_Projesi_Arac_Sayim

# (Opsiyonel) sanal ortam
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Gereksinimler
pip install -r requirements.txt

# Model ağırlığını kopyala  (Drive klasöründen indirip köke koymanız yeterli)
#   best_optimazed.pt
#   deep_sort/deep/checkpoint/ckpt.t7

# Sunucuyu başlat
python app.py
# Uygulama: http://127.0.0.1:5000
```

Alternatif (venv olmadan):

```bash
pip install --user -r requirements.txt
```

### Örnek `requirements.txt`

```txt
flask
ultralytics>=8.3
opencv-python-headless
numpy
torch
matplotlib
imageio-ffmpeg
scipy
pillow
```

---

## Kullanım

1. Ana sayfada **Yükle** butonu ile bir `.mp4` seçin.
2. **Analize Başla**’ya tıklayın; çıktı video ve grafikler otomatik yüklenir.
3. Tüm çıktı dosyaları `static/uploads/` altında saklanır.

---

## Kodda Öne Çıkan Teknikler

| Teknik                                                            | Dosya                                  |
| ----------------------------------------------------------------- | -------------------------------------- |
| Fetch API + `FormData` ile asenkron dosya yükleme                 | [`index.html`](index.html)             |
| HTML `<video>` etiketi üzerinden yerel Blob oynatma               | `index.html`                           |
| Python `multiprocessing` *spawn* – Torch/FFmpeg çakışmasını önler | [`app.py`](app.py)                     |
| Saf NumPy Non‑Max Suppression vektörizasyonu                      | [`preprocessing.py`](preprocessing.py) |
| Kosinüs uzaklıklı en yakın komşu eşleştirme                       | [`nn_matching.py`](nn_matching.py)     |

---

## İlgi Çekebilecek Kütüphaneler / Araçlar

* **Ultralytics YOLOv8** – kolay REST benzeri API, TTA, on‑device kv.
* **DeepSORT (özelleştirilmiş)** – TensorRT’ye geçirilebilir modüler yapı.
* **DaisyUI** – Tailwind temalı bileşen katmanı; dark / light geçişi.
* **imageio‑ffmpeg** – FFmpeg binarisini otomatik indirir; subprocess wrapper.
* **Werkzeug `secure_filename`** – güvenli dosya yükleme.

---

## Dizin Yapısı

```text
.
├── app.py
├── index.html
├── deep_sort/                    # Takip çekirdeği ve feature extractor
│   ├── deep/                     # Öz vektör ağırlıkları (ckpt.t7)
│   └── deep_sort_folder/         # Tracker, Kalman, NMS, araç dosyaları
├── static/
│   └── uploads/                  # İşlenmiş videolar & ısı haritaları
├── templates/                    # (Opsiyonel) Jinja2 şablonları
└── docs/                         # Tez & sunum dosyaları
```

`static/uploads/` dizini çalışma sırasında dinamik oluşur; `.gitignore` ile hariç tutulması önerilir.

---

## Lisans

MIT License — ayrıntı için [`LICENSE`](LICENSE) dosyasına bakın.

---

# Vehicle Counting & Tracking System · YOLOv8 + DeepSORT

A full‑stack web app that detects and tracks vehicles in video, renders heatmaps and live statistics, and streams the annotated result back to the browser.

## Quick Start

```bash
git clone https://github.com/ssevvalyavuz/Bitirme_Projesi_Arac_Sayim.git
cd Bitirme_Projesi_Arac_Sayim
pip install -r requirements.txt      # virtualenv optional

# Download model weights from:
#   https://drive.google.com/drive/folders/1ABCDEFghijKLMNopQRsTuvwXYZ

python app.py
```

Open `http://127.0.0.1:5000`, upload an MP4, press **Start Analysis**, and watch the stream plus charts appear instantly.

### Why It’s Interesting

* **YOLOv8 + DeepSORT** fusion yields real‑time, ID‑consistent multi‑object tracking.
* Fast H.264 repackaging with `imageio‑ffmpeg` enables instant seeking.
* Tailwind CSS + DaisyUI keep the UI minimal yet fully responsive.
* Lightweight NumPy NMS avoids extra C/C++ bindings.
* Cosine‑similarity tracker trims old feature vectors to stay memory‑friendly.

### Project Layout

See tree above; each sub‑directory is self‑contained and testable.

### Contributing

Pull requests are welcome. Run `black . && isort .` before submitting; CI rejects lint errors. Unit tests live under `tests/`.
