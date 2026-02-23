# Phase 1 — Mülakat Notları: Computer Vision

## 1. YOLO Nasıl Çalışır?

**Temel Fikir:** Görüntüyü grid'e böl, her grid cell aynı anda hem sınıf hem bbox tahmin etsin.
- **Single-shot detector** — görüntüyü bir kez feedforward yaparak tüm tespitleri üretir
- **Two-stage** (Faster R-CNN) gibi ayrı region proposal + classification yok
- Bu yüzden çok hızlı (real-time capable)

**YOLOv8 Farkları:**
- **Anchor-free** detection head (v5 anchor-based idi)
- CSPDarknet backbone → feature extraction
- PANet neck → multi-scale feature fusion
- Decoupled head → ayrı classification + regression head'leri

### Anchor-based vs Anchor-free
| | Anchor-based (YOLOv5, Faster R-CNN) | Anchor-free (YOLOv8, FCOS) |
|---|---|---|
| Yaklaşım | Önceden tanımlı anchor box'lar kullanır | Doğrudan center point + boyut tahmin eder |
| Avantaj | Bilinen obje boyutlarında stabil | Daha az hyperparameter, daha esnek |
| Dezavantaj | Anchor boyutları domain-specific ayar gerektirir | Küçük nesnelerde zor olabilir |

---

## 2. Faster R-CNN vs YOLO

| | Faster R-CNN | YOLO |
|---|---|---|
| Mimari | Two-stage: RPN → ROI Pooling → Classifier | Single-stage: direct regression |
| Hız | Yavaş (~5-15 FPS) | Hızlı (~30-100+ FPS) |
| Accuracy | Genelde daha yüksek mAP (küçük nesneler) | Hızla accuracy trade-off |
| Kullanım | Accuracy kritik: medikal, uydu | Real-time: güvenlik kamerası, otonom araç |

**Ne zaman hangisi?**
- **YOLO:** Real-time gerekli, edge deployment, latency kritik
- **Faster R-CNN:** Küçük nesneler önemli, accuracy > speed, batch processing OK

---

## 3. CNN vs Vision Transformer (ViT)

### CNN (Convolutional Neural Network)
- **Inductive bias:** Locality (yakın pikseller ilişkili) + Translation equivariance
- **Mimari:** Conv layers → hierarchical feature learning (edges → textures → objects)
- **Avantaj:** Az veriyle iyi öğrenir, daha hızlı inference
- **Dezavantaj:** Global context yakalamada zayıf

### ViT (Vision Transformer)
- **Yaklaşım:** Görüntüyü 16x16 patch'lere böl, her patch'i token gibi işle (NLP transformer gibi)
- **Self-attention:** Her patch tüm diğer patch'lere attend edebilir → global context
- **Avantaj:** Büyük veriyle çok yüksek performans, global relations
- **Dezavantaj:** Çok veriye ihtiyaç duyar, inference daha yavaş, inductive bias az

### Hibrit Yaklaşımlar
- **Swin Transformer:** Hierarchical + shifted window attention → CNN'in locality + ViT'nin attention avantajı
- **ConvNeXt:** Modern CNN tasarımı, ViT performansına yaklaşıyor

**Mülakat cevabı:** "ViT global context'te güçlü ama çok veriye ihtiyaç duyar. CNN daha az veriyle iyi çalışır ve inference'da hızlı. Production'da genelde hibrit veya domain'e göre seçim yapılır."

---

## 4. mAP (Mean Average Precision) Nasıl Hesaplanır?

**Adımlar:**
1. **IoU (Intersection over Union):** Predicted bbox ile ground truth bbox overlap oranı
   - IoU = (Kesişim Alanı) / (Birleşim Alanı)
   - Genelde threshold: IoU ≥ 0.5 → TP (True Positive)

2. **Precision & Recall:**
   - Precision = TP / (TP + FP)  — "Tespit ettiklerimin kaçı doğru?"
   - Recall = TP / (TP + FN) — "Gerçek nesnelerin kaçını buldum?"

3. **AP (Average Precision):** Tek bir sınıf için Precision-Recall eğrisinin altındaki alan (AUC)

4. **mAP:** Tüm sınıfların AP'lerinin ortalaması

**COCO mAP varyantları:**
- **mAP@0.5:** IoU threshold = 0.5
- **mAP@0.5:0.95:** IoU 0.5'ten 0.95'e kadar 0.05 adımlarla → daha katı

---

## 5. Non-Maximum Suppression (NMS)

**Problem:** Aynı nesne için birden fazla overlapping bbox üretilir.

**Çözüm — NMS Algoritması:**
1. Tüm bbox'ları confidence'a göre sırala
2. En yüksek confidence'lı bbox'ı seç → sonuç listesine ekle
3. Bu bbox ile IoU > threshold olan diğer bbox'ları sil (suppress)
4. Kalan bbox'larla 2-3'ü tekrarla

**Soft-NMS:** Hard silmek yerine, overlap'li bbox'ların confidence'ını IoU oranında düşürür. Kalabalık sahnelerde (crowd) daha iyi çalışır.

---

## 6. Segmentation Tipleri

| Tip | Ne Yapar | Örnek |
|---|---|---|
| **Semantic** | Her piksele sınıf atar, instance ayırmaz | "Yol", "araba", "gökyüzü" alanları |
| **Instance** | Her nesne ayrı mask | "Araba-1", "Araba-2" ayrı ayrı |
| **Panoptic** | Semantic + Instance birleşimi | Tüm pikseller sınıflı + nesneler ayrı |

**Mülakat cevabı:** "Semantic segmentation piksel düzeyinde sınıflandırma yapar ama aynı sınıftaki nesneleri ayırmaz. Instance segmentation her nesneye ayrı mask verir. Panoptic ikisini birleştirir — hem stuff (gökyüzü, yol) hem things (araba, kişi) kapsar."

---

## 7. OCR Pipeline

**3 Aşama:**
1. **Text Detection:** Görüntüde text bölgelerini bul (CRAFT, EAST, DBNet)
2. **Text Recognition:** Bulunan bölgelerdeki karakterleri oku (CRNN, TrOCR)
3. **Post-processing:** Spell correction, format normalization

**EasyOCR vs PaddleOCR:**
- EasyOCR: Kolay kurulum, 80+ dil, orta performans
- PaddleOCR: Daha yüksek accuracy, Çince güçlü, daha karmaşık kurulum

---

## 8. OpenCV Temel İşlemler

```python
# Color Space dönüşümleri
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # API'ler genelde RGB bekler
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     # Renk filtreleme için

# Resize (aspect ratio koruyarak)
h, w = img.shape[:2]
scale = 640 / max(h, w)
resized = cv2.resize(img, (int(w*scale), int(h*scale)))

# Preprocessing for OCR
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)  # Kontrast iyileştirme

# Morphological operations (noise removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

---

## Hızlı Özet — Mülakat Kartları

| Soru | Kısa Cevap |
|---|---|
| YOLO nasıl çalışır? | Single-shot detector, grid-based, anchor-free (v8) |
| YOLO vs Faster R-CNN? | Speed vs accuracy trade-off |
| ViT vs CNN? | Global context vs locality + less data |
| mAP nedir? | Per-class AP ortalaması, IoU threshold'a bağlı |
| NMS nedir? | Overlapping bbox'ları confidence'a göre süz |
| Semantic vs Instance? | Piksel sınıfı vs nesne bazlı mask |
| OCR pipeline? | Detection → Recognition → Post-process |
