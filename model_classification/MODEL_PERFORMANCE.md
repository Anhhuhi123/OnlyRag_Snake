# 🐍 Báo Cáo Hiệu Suất Models Phân Loại Rắn

## 📊 Tổng Quan Dataset
- **Số lượng classes**: 124 loài rắn Việt Nam
- **Dataset structure**: Train / Validation / Test split
- **Image size**: 224×224 pixels
- **Augmentation**: AutoAugment, RandAugment, CutMix, MixUp

---

## 🏆 Bảng So Sánh Hiệu Suất Models

| Model | Test Accuracy | Phương Pháp |
|-------|---------------|-------------|
| **Swin-S + CutMix + MixUp & Swin Tiny** | **97.46%** | Ensemble Model |
| **Swin-S + CutMix + MixUp** | **94.01%** | CutMix + MixUp + AutoAugment + Label Smoothing |
| **Swin-S + CutMix** | **93.72%** 🥇 | CutMix + AutoAugment + Label Smoothing |
| **Swin-S (Baseline)** | **93.00%** 🥈 | Basic Augmentation + Label Smoothing |
| **Swin Tiny** | **92.57%** 🥉 | Basic Augmentation |
| **ConvNeXt Tiny** | **92.13%** | AutoAugment + Label Smoothing |
| **EfficientNetV2** |  **91.33%** | RandAugment |
---

## 📈 Chi Tiết Từng Model

### 1️⃣ **Swin-S + CutMix + MixUp & Swin Tiny (Ensemble)** 🏆 (Test: 97.46%)

**Phương pháp:**
- **Ensemble Model**: Kết hợp dự đoán từ 2 models (Swin-S + CutMix + MixUp và Swin Tiny)
- **Voting/Averaging**: Lấy trung bình hoặc vote từ cả 2 models
- **Complementary Strengths**: Models khác nhau bổ trợ cho nhau

**Ích lợi:**
- ✅ Accuracy cao nhất (97.46%) - Tăng ~3.5% so với single model
- ✅ Robust hơn, giảm sai số nhờ kết hợp nhiều quan điểm
- ✅ Cải thiện đặc biệt tốt trên các classes khó

---

### 2️⃣ **Swin-S + CutMix + MixUp** 🥇 (Test: 94.01%)

**Phương pháp:**
- **CutMix**: Cắt và ghép các phần của 2 ảnh khác nhau
- **MixUp**: Trộn 2 ảnh theo tỉ lệ
- **AutoAugment**: Tự động tìm augmentation tốt nhất
- **Label Smoothing**: Làm mềm labels

**Ích lợi:**
- ✅ Accuracy rất cao (94.01%)
- ✅ Regularization mạnh nhất (CutMix + MixUp)
- ✅ Tránh overfitting xuất sắc
- ✅ Generalization tốt nhất trong single models

---

### 3️⃣ **Swin-S + CutMix** 🥈 (Test: 93.72%)

**Phương pháp:**
- **CutMix**: Cắt và ghép các phần của 2 ảnh khác nhau → Model học được đặc điểm từ nhiều vị trí
- **AutoAugment**: Tự động tìm augmentation tốt nhất
- **Label Smoothing**: Làm mềm labels để tránh model quá tự tin → Giảm overfitting

**Ích lợi:**
- ✅ Accuracy cao (93.72%)
- ✅ Tránh overfitting tốt nhờ CutMix
- ✅ Nhận diện rắn tốt ngay cả khi bị che khuất một phần
- ✅ Đơn giản hơn version có MixUp

---

### 4️⃣ **Swin-S (Baseline)** � (Test: 93.00%)

**Phương pháp:**
- **Swin Transformer**: Kiến trúc hiện đại dùng window attention
- **Label Smoothing**: Làm mềm labels
- **Basic Augmentation**: Chỉ dùng flip, resize cơ bản

**Ích lợi:**
- ✅ Training đơn giản, nhanh hơn
- ✅ Ổn định, dễ reproduce
- ✅ Baseline tốt để so sánh các phương pháp khác

---

### 5️⃣ **Swin Tiny** (Test: 92.57%)

**Phương pháp:**
- **Swin Transformer phiên bản nhỏ**: Giảm số layers và params
- **Basic Augmentation**: Augmentation cơ bản

**Ích lợi:**
- ✅ Nhanh hơn Swin-S (~50% faster)
- ✅ Model nhỏ hơn, dễ deploy
- ✅ Accuracy vẫn tốt (92.57%)
- ✅ Phù hợp khi cần balance giữa tốc độ và chính xác

---

### 6️⃣ **ConvNeXt Tiny** (Test: 92.13%)

**Phương pháp:**
- **ConvNeXt**: CNN hiện đại với thiết kế giống Transformer
- **AutoAugment**: Tự động tìm augmentation
- **Label Smoothing**: Giảm overfitting

**Ích lợi:**
- ✅ Inference nhanh nhất (không cần attention mechanism phức tạp)
- ✅ Dễ deploy hơn Swin
- ✅ Phù hợp khi cần tốc độ

---

### 7️⃣ **EfficientNetV2** (Test: 91.33%)

**Phương pháp:**
- **EfficientNetV2**: Kiến trúc được tối ưu cho tốc độ và hiệu quả
- **RandAugment**: Random augmentation
- **Progressive Learning**: Học từ ảnh nhỏ đến lớn

**Ích lợi:**
- ✅ Model nhỏ gọn nhất
- ✅ Training nhanh
- ✅ Phù hợp cho mobile/edge devices

### 5️⃣ **EfficientNetV2**
**File**: `NB_EfficientNetV2.ipynb`

**Hiệu suất:**
- ✅ **Best Validation Accuracy**: ~87-89% (ước tính)
- ⚡ **Training Speed**: Nhanh nhất (~2h)

**Cấu hình:**
```python
Model: EfficientNetV2-S
Optimizer: AdamW
Loss: CrossEntropyLoss
Augmentation: RandAugment
```

**Ưu điểm:**
- ⚡ **Training + Inference nhanh nhất**
- 💾 **Model size nhỏ nhất** (~21M params)
- 🔋 GPU memory efficient
- 📱 Phù hợp mobile/edge deployment

**Nhược điểm:**
- 📉 Accuracy thấp nhất (~87-89%)
- 🎯 Kém hơn ~5-7% so với Swin-S

**Cải tiến đề xuất:**
1. **EfficientNetV2-M/L**: Tăng model capacity
2. **Progressive learning**: Start với small images, tăng dần
3. **CutMix**: EfficientNet benefit nhiều từ CutMix
4. **Longer training**: Train 40-50 epochs với cosine decay

---

### 6️⃣ **Swin Tiny**
**File**: `NB_SwinTiny.ipynb`

**Hiệu suất:**
- ✅ **Best Validation Accuracy**: ~88-90% (ước tính)
- 📊 Model size: ~28M params

**Ưu điểm:**
- ⚡ Nhanh hơn Swin-S
- 💾 Model size nhỏ hơn
- 🎯 Balance tốt giữa speed và accuracy

**Nhược điểm:**
- 📉 Accuracy thấp hơn Swin-S ~4-5%

**Cải tiến đề xuất:**
1. **CutMix + MixUp**: Có thể đạt 91-92%
2. **Knowledge Distillation**: Learn từ Swin-S teacher
3. **Longer training**: 40-50 epochs

---

## 🎯 Khuyến Nghị Chọn Model

### 🏆 **Production Use (Priority: Accuracy)**
```
→ Swin-S + CutMix (94.32%)
   - Best accuracy
   - Ổn định nhất
   - Worth the computational cost
```

---

## 🎯 Khuyến Nghị Chọn Model

**🏆 Ưu tiên Accuracy cao nhất:**
```
→ Swin-S + CutMix (93.72%)
```

**⚡ Ưu tiên Tốc độ và Deploy dễ:**
```
→ ConvNeXt Tiny (92.13%)
```

**⚖️ Cân bằng giữa Accuracy và Đơn giản:**
```
→ Swin-S Baseline (93.00%)
```

---

## 🚀 Hướng Cải Tiến

**1. Tăng accuracy thêm 0.5-1%:**
- Ensemble nhiều models
- Test Time Augmentation (TTA)
- Train thêm epochs với learning rate thấp hơn

**2. Tối ưu cho deployment:**
- Knowledge Distillation: Chuyển kiến thức từ Swin-S → model nhỏ hơn
- Model quantization: Giảm từ FP32 → INT8
- ONNX export để deploy đa nền tảng

**3. Cải thiện classes khó:**
- Thu thập thêm data cho classes có accuracy thấp
- Focal Loss để tập trung vào samples khó
- Data augmentation mạnh hơn cho classes thiểu số
---

## 📚 Giải Thích Thuật Ngữ

**CutMix**: Cắt 1 phần ảnh A và dán vào ảnh B → Model học được đặc điểm ở nhiều vị trí khác nhau

**MixUp**: Trộn 2 ảnh với nhau theo tỉ lệ → Model không quá tự tin vào 1 class duy nhất

**Label Smoothing**: Thay vì label cứng (0 hoặc 1), dùng label mềm (0.05 và 0.95) → Tránh overfitting

**AutoAugment**: Tự động tìm cách augment ảnh tốt nhất (xoay, lật, thay đổi màu sắc...)

**Swin Transformer**: Kiến trúc AI hiện đại, xử lý ảnh theo từng "cửa sổ" nhỏ → Hiệu quả hơn

**ConvNeXt**: CNN cải tiến, kết hợp ưu điểm của CNN và Transformer → Nhanh mà vẫn chính xác

---

**Last Updated**: 28/10/2025  
**Best Model**: Swin-S + CutMix (93.72% Test Acc)

