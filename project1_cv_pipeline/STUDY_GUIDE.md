# Project 1: CV Pipeline — Kapsamlı Çalışma Rehberi

> Bu rehber kodu satır satır anlamak, mimari kararları kavramak ve mülakat sorularına hazırlanmak için hazırlandı.

---

## 1. Mimari Genel Bakış

```
run_pipeline.py  (CLI entry point)
    └── VisualPerceptionPipeline (src/pipeline.py)
            ├── ImagePreprocessor   (src/preprocessor.py)  — OpenCV, stateless
            ├── ObjectDetector      (src/detector.py)      — YOLOv8
            ├── InstanceSegmentor   (src/segmentor.py)     — YOLOv8-seg
            └── OCREngine           (src/ocr_engine.py)    — EasyOCR (CRAFT+CRNN)

Akış:
  Görüntü → Yükle → Resize → Detection → Segmentation → OCR → JSON
```

**Tasarım prensipleri:**
- Her modül **tek sorumluluk** (SRP) — ayrı sınıf, ayrı dosya
- Modüller **bağımsız** — detection, segmentation, OCR birbirinin çıktısına ihtiyaç duymaz
- Çıktı **tutarlı** — her modül `[x1, y1, x2, y2]` bbox formatı kullanır
- Pipeline modülleri **opsiyonel** — `--no-detection`, `--no-ocr` flag'leriyle kapatılabilir

---

## 2. Kod Detaylı Analiz

### 2.1 run_pipeline.py — Entry Point

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
> Neden? `python run_pipeline.py` diye çalıştırınca Python sadece script'in olduğu dizini path'e ekler. `src` paketini bulabilmesi için proje kökünü ekliyoruz.

```python
parser.add_argument("--confidence", type=float, default=0.25)
```
> **0.25 neden?** — YOLO'nun default'u. Düşürürsen daha fazla ama gürültülü tespit (yüksek recall, düşük precision). Yükseltirsen daha az ama güvenilir tespit (yüksek precision, düşük recall). Bu **precision-recall trade-off**'un doğrudan kontrolü.

```python
run_detection=not args.no_detection
```
> Flag hilesi: `--no-detection` verilmişse `args.no_detection = True`, `not True = False` → detection kapanır. Çift negatif ama argparse'da standart pattern.

```python
json.dumps(display_result, indent=2, ensure_ascii=False)
```
> `ensure_ascii=False` → Türkçe karakterler `\u00e7` yerine `ç` olarak yazılır. Okunabilirlik.

---

### 2.2 src/pipeline.py — Orchestrator

**`__init__` — Model yükleme:**
```python
self.detector = ObjectDetector(model_name=detection_model, confidence=confidence)
self.segmentor = InstanceSegmentor(model_name=segmentation_model, confidence=confidence)
self.ocr_engine = OCREngine(languages=ocr_languages)
```
> Tüm modeller **constructor'da** yüklenir. Bu ağır bir işlem (YOLO + EasyOCR toplam ~10-15 saniye). Neden? Her `analyze()` çağrısında tekrar yüklememek için — **amortized cost**.

**`analyze` — Sıralı işleme:**
```python
t = time.time()
result["objects"] = self.detector.detect(image)
result["processing_time"]["detection"] = round(time.time() - t, 3)
```
> Her adımın süresi ayrı ölçülüyor. Production'da bu **profiling** için kritik. Darboğaz nerede? OCR (~4s) >> Detection (~0.2s).

> **Mimari soru:** Detection, segmentation ve OCR bağımsız — neden paralel değil?
> Basitlik için sıralı. İyileştirme olarak `concurrent.futures.ThreadPoolExecutor` veya `asyncio` ile paralel çalıştırılabilir. GPU paylaşımı sorun olabilir ama CPU-bound OCR thread'e alınabilir.

**`_generate_scene_description` — Rule-based özetleme:**
```python
label_counts[label] = label_counts.get(label, 0) + 1
```
> `dict.get(key, default)` — key yoksa default dön. `collections.Counter` da kullanılabilirdi ama ek import gerektirmeden aynı iş.

```python
texts = [r["text"] for r in result["text_regions"][:5]]
```
> İlk 5 metin. Neden? Çok fazla OCR sonucu açıklamayı okunmaz yapar. Pragmatik sınır.

**`analyze_and_save`:**
```python
seg.pop("mask_base64", None)
```
> `pop(key, default)` → key varsa sil ve değerini dön, yoksa default dön. `del seg["mask_base64"]` kullanılsaydı key yoksa `KeyError` fırlatırdı.

---

### 2.3 src/preprocessor.py — OpenCV İşlemleri

**Tüm metodlar `@staticmethod` — neden?**
> Sınıfın state'i yok (instance değişkeni yok). Sadece utility fonksiyonları gruplayarak namespace oluşturuyor. Alternatif: module-level fonksiyonlar da olabilirdi.

**`load_image`:**
```python
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(...)
```
> **Kritik:** `cv2.imread` başarısız olunca exception **fırlatmaz**, sessizce `None` döner. Dosya yolunda Türkçe karakter, boşluk veya desteklenmeyen format olursa bile `None`. Bu yüzden kontrol zorunlu.
>
> OpenCV'nin renk formatı **BGR** (Blue-Green-Red). PIL, matplotlib, EasyOCR ise **RGB** bekler. Bu en yaygın hatalardan biri.

**`resize` — Aspect ratio koruyan resize:**
```python
scale = max_size / max(h, w)   # Büyük kenarı max_size'a getir
new_w = int(w * scale)          # Küçük kenar oranla küçülür
cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
```
> - **INTER_AREA:** Küçültme için en iyi interpolasyon (anti-aliasing etkisi). Piksel bilgisini ortalayarak küçültür.
> - **INTER_LINEAR:** Büyütme için varsayılan (bilinear interpolasyon).
> - **INTER_CUBIC:** Büyütme için daha kaliteli ama yavaş.
> - **INTER_NEAREST:** Binary mask'lar için (aşağıda segmentor'da).
>
> **Neden 1280?** YOLO internal olarak 640x640'a resize eder. Ama büyük görüntüleri 1280'de kesmek RAM'i korur.

**`enhance_for_ocr` — OCR ön-işleme:**
```python
# 1. Grayscale — renk bilgisi OCR için gereksiz
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. CLAHE — Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# 3. Noise reduction
denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
```
> **CLAHE nedir?**
> - Normal histogram equalization tüm görüntüye aynı dönüşümü uygular → gölgeli/aydınlık farklı bölgelerde kötü sonuç
> - CLAHE görüntüyü `8x8` tile'a böler, her tile'a ayrı equalization uygular
> - `clipLimit=2.0` → kontrast artışını sınırlar, aşırı amplifikasyonu engeller
> - Sonuç: Yerel kontrast iyileşir, global bilgi korunur
>
> **Not:** Bu metod pipeline'da **şu an çağrılmıyor**. OCR engine'e ham görüntü gidiyor. Potansiyel iyileştirme noktası.

**`crop_region`:**
```python
return image[y1:y2, x1:x2]  # NumPy: [satır, sütun] = [y, x]
```
> **NumPy indexleme sırası:** `image[row, col]` = `image[y, x]`. `image[y1:y2, x1:x2]` doğru. `image[x1:x2, y1:y2]` yaygın hata.

**`get_image_info`:**
```python
channels = image.shape[2] if len(image.shape) == 3 else 1
```
> Grayscale görüntü `(h, w)` shape'inde (2D). Renkli görüntü `(h, w, 3)` (3D). Bu kontrol ikisini de handle eder.

---

### 2.4 src/detector.py — YOLOv8 Object Detection

**Model yükleme:**
```python
self.model = YOLO(model_name)  # "yolov8n.pt" — nano
```
> Model boyutları: `n(3.2M) < s(11.2M) < m(25.9M) < l(43.7M) < x(68.2M)`
> Büyük model = daha yüksek accuracy, daha yavaş inference, daha fazla RAM.

**`detect` — İnference:**
```python
results = self.model(image, conf=self.confidence, verbose=False)
```
> Bu tek satırda olan işlemler:
> 1. Görüntü `640x640`'a resize (letterboxing ile aspect ratio korunur)
> 2. `[0,255]` → `[0,1]` normalize
> 3. `BGR → RGB` dönüşümü
> 4. NumPy → PyTorch Tensor
> 5. GPU/CPU'da forward pass
> 6. **NMS** (Non-Maximum Suppression) uygulanır
> 7. Sonuçlar orijinal boyuta geri map'lenir

**Sonuç parse:**
```python
x1, y1, x2, y2 = box.xyxy[0].tolist()   # Tensor → Python list
conf = float(box.conf[0])                 # Tensor → float
cls_id = int(box.cls[0])                  # Sınıf index'i (0=person, 5=bus...)
label = self.model.names[cls_id]           # Index → isim
```
> - `box.xyxy` → `[sol_üst_x, sol_üst_y, sağ_alt_x, sağ_alt_y]`
> - `box.xywh` → `[center_x, center_y, width, height]` (COCO formatı)
> - `self.model.names` → COCO'nun 80 sınıfı: `{0: "person", 1: "bicycle", 2: "car", ...}`

---

### 2.5 src/segmentor.py — Instance Segmentation

**Detection'dan farkı:** Sadece kutu değil, **piksel düzeyinde mask** üretir. Her nesnenin tam şeklini bilir.

**Mask işleme — en kritik kısım:**
```python
mask_array = mask.data[0].cpu().numpy()
```
> - `mask.data` → PyTorch tensor (GPU'da olabilir)
> - `[0]` → batch boyutunu kaldır
> - `.cpu()` → GPU'dan CPU'ya taşı (NumPy GPU tensor ile çalışamaz)
> - `.numpy()` → Tensor → NumPy array

```python
mask_resized = cv2.resize(
    mask_array.astype(np.uint8),
    (image.shape[1], image.shape[0]),     # (width, height) — OpenCV sırası!
    interpolation=cv2.INTER_NEAREST       # Binary mask için
)
```
> **Neden `INTER_NEAREST`?**
> Mask binary (0 veya 1). `INTER_LINEAR` değerleri 0-1 arasına yayar → sınırlar bulanıklaşır. `INTER_NEAREST` en yakın komşu değerini alır → binary kalır, sınırlar net.
>
> **cv2.resize sırası uyarı:** `(width, height)` alır ama `image.shape` `(height, width, channels)` döner. Bu yüzden `(image.shape[1], image.shape[0])`.

**Base64 encoding:**
```python
_, mask_png = cv2.imencode(".png", mask_resized * 255)
mask_b64 = base64.b64encode(mask_png).decode("utf-8")
```
> - `mask_resized * 255` → binary 0/1'i 0/255'e çevir (PNG formatı için)
> - `cv2.imencode` → NumPy array → PNG byte dizisi (disk'e yazmadan bellekte sıkıştırma)
> - `base64.b64encode` → binary → ASCII string (JSON taşınabilir)
> - Base64 ~%33 boyut artışı getirir ama JSON-safe

**Area hesaplama:**
```python
area = int(np.sum(mask_resized > 0))
```
> `mask_resized > 0` → Boolean array. `np.sum` True'ları sayar = nesnenin piksel alanı.

**Overlay çizimi:**
```python
color = np.random.randint(50, 220, 3).tolist()
overlay[mask_resized > 0] = color
annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
```
> `addWeighted(src1, α, src2, β, γ)` → `α·src1 + β·src2 + γ`
> `0.7 * orijinal + 0.3 * renkli = yarı saydam overlay`

---

### 2.6 src/ocr_engine.py — EasyOCR

**Constructor — Mutable default argument tuzağı:**
```python
def __init__(self, languages: List[str] = None):
    if languages is None:
        languages = ["en"]
```
> **Neden `None` default?** Python'da `def f(x=[])` yazarsan, tüm çağrılar **aynı listeyi paylaşır**. Bir yerde değiştirilirse hepsi etkilenir. `None` + içeride atama **güvenli pattern**.

```python
self.reader = easyocr.Reader(languages, gpu=True)
```
> İlk çalıştırmada modeli indirir (~100MB). `gpu=True`: CUDA varsa GPU, yoksa otomatik CPU.

**BGR → RGB dönüşümü:**
```python
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
> OpenCV BGR, EasyOCR RGB bekler. Bunu unutmak kırmızı↔mavi yer değiştirir → OCR doğruluğu düşer.

**Bbox format dönüşümü — Quad → AABB:**
```python
# EasyOCR quad format (4 köşe — eğik metin için):
# [[sol_üst_x, sol_üst_y], [sağ_üst_x, sağ_üst_y],
#  [sağ_alt_x, sağ_alt_y], [sol_alt_x, sol_alt_y]]

pts = np.array(bbox_points)        # 4x2 array
x_min = int(pts[:, 0].min())       # Tüm x'lerin minimumu
y_min = int(pts[:, 1].min())       # Tüm y'lerin minimumu
x_max = int(pts[:, 0].max())       # Tüm x'lerin maximumu
y_max = int(pts[:, 1].max())       # Tüm y'lerin maximumu
```
> Quad → Axis-Aligned Bounding Box (AABB) dönüşümü. **Trade-off:** Eğik metinlerde bbox gereğinden büyük olur ama format tutarlı kalır (detection ve segmentation ile aynı `[x1,y1,x2,y2]`).

---

## 3. Nano vs Medium Karşılaştırma (Deneysel)

`bus.jpg` üzerinde test sonuçları:

| Metrik | YOLOv8n (Nano) | YOLOv8m (Medium) |
|--------|---------------|-----------------|
| Parametre sayısı | 3.2M | 25.9M |
| Model boyutu | 6.5 MB | ~50 MB |
| Detection sayısı | 6 nesne | 5 nesne |
| Bus confidence | 0.8734 | **0.9565** |
| Person avg conf | ~0.70 | **~0.88** |
| Detection süresi | 0.163s | 0.217s |
| Segmentation süresi | 0.08s | 0.479s |

**Çıkarımlar:**
- Medium **daha yüksek confidence** → daha emin tespitler
- Medium **daha az false positive** → düşük güvenli tespitleri kendisi eler
- Nano'daki stop sign (0.25 conf) Medium'da düşmüş — threshold sınırında
- Segmentation'da "tie" false positive (0.29 conf) → threshold'u 0.3'e çıkarmak çözer

**Ne zaman hangi model?**
| Senaryo | Model |
|---------|-------|
| Real-time, edge device, mobil | Nano (n) |
| Production, dengeli hız/accuracy | Small (s) veya Medium (m) |
| Accuracy kritik, batch processing OK | Large (l) veya XLarge (x) |

---

## 4. Mülakat Soruları ve Cevapları

### 4.1 YOLO Nasıl Çalışır?

**Temel fikir:** Single-shot detector — görüntüyü bir grid'e böler, her grid cell aynı anda hem sınıf hem bbox tahmin eder. Tek forward pass'te tüm tespitler çıkar.

**YOLOv8 mimarisi:**
```
Görüntü → CSPDarknet (backbone) → PANet (neck) → Decoupled Head
                                                    ├── Classification branch
                                                    └── Regression branch (bbox)
```
- **Backbone (CSPDarknet):** Feature extraction — görüntüden özellik haritaları çıkarır
- **Neck (PANet):** Multi-scale feature fusion — farklı ölçeklerdeki feature'ları birleştirir (küçük+büyük nesneler)
- **Head (Decoupled):** Sınıf tahmini ve bbox tahmini ayrı branch'lerde → v5'ten daha iyi performans

**YOLOv8'in farkı — Anchor-free:**
| | Anchor-based (v5, Faster R-CNN) | Anchor-free (v8, FCOS) |
|---|---|---|
| Yaklaşım | Önceden tanımlı kutu şablonları | Doğrudan center point + boyut tahmin |
| Avantaj | Bilinen boyutlarda stabil | Daha az hyperparameter |
| Dezavantaj | Anchor boyutları domain-specific ayar gerektirir | Küçük nesnelerde zor olabilir |

---

### 4.2 Faster R-CNN vs YOLO

| | Faster R-CNN | YOLO |
|---|---|---|
| Aşama | **Two-stage:** RPN → ROI Pooling → Classifier | **Single-stage:** direct regression |
| Hız | ~5-15 FPS | ~30-100+ FPS |
| Accuracy | Genelde daha yüksek mAP (küçük nesneler) | Hız-accuracy trade-off |
| Kullanım | Accuracy kritik (medikal, uydu) | Real-time (güvenlik, otonom araç) |

**Mülakat cevabı:** "Faster R-CNN iki aşamalıdır — önce Region Proposal Network olası nesne bölgelerini önerir, sonra her bölge ayrıca sınıflandırılır. YOLO tek aşamada doğrudan bbox + sınıf tahmin eder. Bu yüzden YOLO çok daha hızlı ama küçük nesnelerde Faster R-CNN genelde daha iyi."

---

### 4.3 CNN vs Vision Transformer (ViT)

| | CNN | ViT |
|---|---|---|
| Temel birim | Konvolüsyon filtresi (lokal) | Self-attention (global) |
| Inductive bias | Locality + translation equivariance | Yok (veriden öğrenir) |
| Veri ihtiyacı | Az veriyle iyi çalışır | Çok veriye ihtiyaç duyar |
| Inference hızı | Hızlı | Daha yavaş |
| Global context | Zayıf (sadece receptive field) | Güçlü (her patch tüm diğerlerine attend) |

**ViT nasıl çalışır:**
1. Görüntüyü `16x16` patch'lere böl
2. Her patch'i linear projection ile embedding'e çevir
3. Position embedding ekle (patch sırası)
4. Transformer encoder'dan geçir (self-attention + FFN)
5. [CLS] token'ı classification için kullan

**Hibrit modeller:** Swin Transformer (hierarchical + shifted window), ConvNeXt (modern CNN, ViT performansına yakın)

---

### 4.4 mAP Nasıl Hesaplanır?

```
1. IoU hesapla:   IoU = Kesişim Alanı / Birleşim Alanı

2. TP/FP belirle: IoU ≥ threshold (genelde 0.5) → TP (True Positive)
                   IoU < threshold → FP (False Positive)
                   Kaçırılan nesne → FN (False Negative)

3. Precision = TP / (TP + FP)    "Bulduklarımın kaçı doğru?"
   Recall    = TP / (TP + FN)    "Gerçek nesnelerin kaçını buldum?"

4. AP = Precision-Recall eğrisi altındaki alan (tek sınıf için)

5. mAP = Tüm sınıfların AP ortalaması
```

**COCO varyantları:**
- `mAP@0.5` — IoU ≥ 0.5 yeterli (gevşek)
- `mAP@0.5:0.95` — 0.5'ten 0.95'e 0.05 adımlarla ortalama (katı, COCO primary metric)
- `mAP@0.75` — Daha sıkı lokalizasyon

---

### 4.5 Non-Maximum Suppression (NMS)

**Problem:** Aynı nesne için birden fazla overlapping bbox üretilir.

**Algoritma:**
```
1. Tüm bbox'ları confidence'a göre sırala (azalan)
2. En yüksek conf'lu bbox'ı seç → sonuç listesine ekle
3. Bu bbox ile IoU > threshold olan diğerlerini sil
4. Kalan bbox'larla tekrarla
5. Sonuç: Her nesne için tek bir bbox
```

**Soft-NMS:** Hard silmek yerine, overlap'li bbox'ların confidence'ını düşürür. Kalabalık sahnelerde (ör: insan kalabalığı) daha iyi — birbirine yakın farklı nesneler silinmez.

---

### 4.6 Segmentation Tipleri

| Tip | Ne yapar | Kullanım | Model örnekleri |
|-----|----------|----------|----------------|
| **Semantic** | Her piksele sınıf atar, instance ayırmaz | Otonom sürüş (yol/kaldırım) | DeepLabV3, U-Net |
| **Instance** | Her nesneye ayrı mask | Nesne sayma, AR | Mask R-CNN, YOLOv8-seg |
| **Panoptic** | Semantic + Instance birleşimi | Tam sahne anlama | Panoptic FPN |

**Mülakat cevabı:** "Semantic segmentation 'bu piksel yol, bu araba' der ama iki arabayı ayıramaz. Instance segmentation her arabaya ayrı mask verir ama arka planı (gökyüzü, yol) ignore eder. Panoptic ikisini birleştirir — stuff (gökyüzü, yol gibi sayılamayan şeyler) + things (araba, insan gibi sayılabilen nesneler)."

---

### 4.7 OCR Pipeline

```
Görüntü → Text Detection → Text Recognition → Post-processing → Metin
              (nerede?)          (ne yazıyor?)       (düzeltme)
```

| Aşama | Model/Yöntem | Açıklama |
|-------|-------------|----------|
| Detection | CRAFT, EAST, DBNet | Metin bölgelerini bulur (karakter/kelime düzeyinde) |
| Recognition | CRNN, TrOCR | Bulunan bölgelerdeki karakterleri okur |
| Post-process | Spell check, regex | Hatalı okumaları düzeltir |

**EasyOCR mimarisi:**
- Detection: **CRAFT** — Character Region Awareness for Text detection
- Recognition: **CRNN** — CNN (feature extraction) + BiLSTM (sequence modeling) + CTC loss (alignment-free)

**EasyOCR vs PaddleOCR:**
| | EasyOCR | PaddleOCR |
|---|---|---|
| Kurulum | Kolay (pip install) | Daha karmaşık |
| Dil desteği | 80+ dil | 80+ dil, Çince çok güçlü |
| Accuracy | Orta | Genelde daha yüksek |
| Hız | Yavaş (~4s) | Daha hızlı |

---

### 4.8 OpenCV Temel Bilgiler

**Renk formatı:**
```
OpenCV   → BGR (Blue, Green, Red)  — tarihsel sebep (kamera firmware)
PIL      → RGB
EasyOCR  → RGB
YOLO     → internal olarak RGB'ye çevirir (otomatik)
```

**Interpolasyon metodları:**
| Metod | Kullanım | Açıklama |
|-------|----------|----------|
| INTER_AREA | Küçültme | Anti-aliasing, piksel bilgisini ortalama |
| INTER_LINEAR | Büyütme (default) | Bilinear interpolasyon |
| INTER_CUBIC | Büyütme (kaliteli) | Bicubic, daha yavaş ama kaliteli |
| INTER_NEAREST | Binary mask | En yakın komşu, 0/1 değerleri korur |

**CLAHE vs Normal Histogram Equalization:**
| | Normal HE | CLAHE |
|---|---|---|
| Uygulama | Tüm görüntüye tek dönüşüm | Tile'lara böl, her birine ayrı |
| Sonuç | Gölgeli alanlarda kötü | Her bölgede iyi kontrast |
| Parametre | Yok | clipLimit, tileGridSize |

---

## 5. Kodda Dikkat Edilecek Patternlar

### Pattern 1: Mutable Default Argument
```python
# YANLIŞ — tüm instance'lar aynı listeyi paylaşır
def __init__(self, languages=["en"]):

# DOĞRU — her çağrıda yeni liste
def __init__(self, languages=None):
    if languages is None:
        languages = ["en"]
```

### Pattern 2: cv2.imread None Kontrolü
```python
image = cv2.imread(path)
# cv2.imread BAŞARISIZ OLUNCA EXCEPTION FIRLATMAZ, None döner!
if image is None:
    raise FileNotFoundError(...)
```

### Pattern 3: NumPy Indexleme Sırası
```python
image.shape  → (height, width, channels)  # NumPy: row, col
cv2.resize   → (width, height)            # OpenCV: w, h
image[y1:y2, x1:x2]                       # Slicing: row, col = y, x
```

### Pattern 4: GPU → CPU → NumPy
```python
mask.data[0].cpu().numpy()
# [0]     → batch boyutunu kaldır
# .cpu()  → GPU tensor'ü CPU'ya taşı
# .numpy() → PyTorch tensor → NumPy array
```

### Pattern 5: pop vs del
```python
seg.pop("mask_base64", None)   # Key yoksa None döner, hata yok
del seg["mask_base64"]          # Key yoksa KeyError fırlatır
```

---

## 6. Potansiyel İyileştirmeler (Mülakatta sorulabilir)

| İyileştirme | Nasıl | Neden |
|-------------|-------|-------|
| **Paralel çalıştırma** | `ThreadPoolExecutor` ile det/seg/ocr paralel | OCR ~4s darboğaz, paralelde toplam süre düşer |
| **enhance_for_ocr kullanımı** | OCR'dan önce preprocessing ekle | OCR doğruluğu artabilir (özellikle düşük kontrast) |
| **Batch processing** | Birden fazla görüntüyü batch'le | GPU kullanımını artırır, throughput yükselir |
| **Model caching** | Singleton pattern ile model tekrar yüklenmesini engelle | Bellek tasarrufu, startup süresi |
| **Confidence filtering** | OCR sonuçlarını conf > 0.3 ile filtrele | Gürültülü OCR sonuçlarını ele |
| **Test yazma** | `tests/` klasörü boş | Güvenilirlik, refactoring güvencesi |
| **Async API** | FastAPI ile async endpoint | Production deployment (Proje 6'da yapılacak) |

---

## 7. Hızlı Referans — Mülakat Kartları

| Soru | Tek Cümle Cevap |
|------|----------------|
| YOLO nasıl çalışır? | Single-shot detector: grid'e böl, her cell bbox + sınıf tahmin et, tek forward pass |
| Anchor-based vs anchor-free? | Anchor-based önceden tanımlı kutular kullanır, anchor-free doğrudan center+boyut tahmin eder |
| YOLO vs Faster R-CNN? | YOLO tek aşama (hızlı), Faster R-CNN iki aşama (daha doğru, özellikle küçük nesnelerde) |
| ViT vs CNN? | ViT global context güçlü ama çok veri ister, CNN locality ile az veriyle iyi çalışır |
| mAP nedir? | Per-class Average Precision'ların ortalaması, IoU threshold'a bağlı |
| NMS nedir? | Aynı nesne için birden fazla kutuyu eleyerek en iyisini seçen algoritma |
| Soft-NMS farkı? | Hard silmek yerine confidence düşürür, kalabalıkta daha iyi |
| Semantic vs Instance? | Semantic: piksel sınıfı atar, Instance: her nesneye ayrı mask verir |
| Panoptic? | Semantic + Instance birleşimi: stuff + things |
| OCR pipeline? | Detection (bul) → Recognition (oku) → Post-process (düzelt) |
| CLAHE nedir? | Adaptive histogram equalization — tile bazlı, yerel kontrast iyileştirme |
| BGR vs RGB? | OpenCV BGR, diğer kütüphaneler genelde RGB kullanır |
| INTER_NEAREST neden? | Binary mask'larda interpolasyon değerleri bozar, nearest 0/1'i korur |

---

## 8. Çalışma Kontrol Listesi

- [ ] Tüm kaynak dosyaları okudun mu?
- [ ] Pipeline'ı çalıştırdın mı? (nano ve medium)
- [ ] Annotated çıktı görselini inceledi mi?
- [ ] Her mülakat kartını ezbere cevaplayabiliyor musun?
- [ ] Mimari kararları savunabilir misin? (neden nano, neden sıralı, neden AABB?)
- [ ] Potansiyel iyileştirmeleri açıklayabilir misin?
- [ ] NMS algoritmasını adım adım anlatabilir misin?
- [ ] mAP hesaplamasını adım adım anlatabilir misin?
- [ ] CLAHE'yi normal HE'den neden tercih ettiğini açıklayabilir misin?
