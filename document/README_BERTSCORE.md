# BERTScore Evaluation for RAG System

## 📝 Mô tả

File `evaluate_generation_bertscore.py` đánh giá chất lượng câu trả lời của LLM bằng **BERTScore** - một metric đo độ tương đồng ngữ nghĩa dựa trên BERT embeddings.

## 🎯 Mục đích

So sánh câu trả lời của LLM (`answer`) với câu trả lời đúng (`ground_truth`) để đánh giá:
- **Precision**: Độ chính xác - câu trả lời có bao nhiêu thông tin đúng
- **Recall**: Độ đầy đủ - câu trả lời bao phủ bao nhiêu % thông tin cần thiết
- **F1**: Điểm cân bằng giữa Precision và Recall

## 📦 Cài đặt

```bash
pip install bert-score
```

## 🚀 Sử dụng

### Chạy evaluation:

```bash
python evaluate_generation_bertscore.py
```

### Input:
- File: `data/predictions.json`
- Format:
```json
[
  {
    "question": "...",
    "ground_truth": "...",
    "answer": "...",
    "contexts": [...]
  }
]
```

### Output:
- File: `data/bertscore_evaluation_results.json`
- Chứa:
  - Average metrics (Precision, Recall, F1)
  - Per-question scores
  - Configuration info

## 📊 Kết quả mẫu

```
BERTScore Precision: 0.6612 ± 0.0416
BERTScore Recall:    0.7883 ± 0.0751
BERTScore F1:        0.7176 ± 0.0477
```

### Score Distribution:
- **Excellent** (0.9-1.0): Câu trả lời xuất sắc
- **Very Good** (0.8-0.9): Câu trả lời rất tốt
- **Good** (0.7-0.8): Câu trả lời chấp nhận được
- **Fair** (0.6-0.7): Câu trả lời trung bình
- **Poor** (<0.6): Câu trả lời kém

## 🔧 Tùy chỉnh

Trong file `evaluate_generation_bertscore.py`:

```python
# Thay đổi model BERT
MODEL_TYPE = "xlm-roberta-base"  # Tốt cho đa ngôn ngữ
# hoặc
MODEL_TYPE = "bert-base-multilingual-cased"

# Thay đổi ngôn ngữ
LANGUAGE = "en"  # Tiếng Anh
LANGUAGE = "vi"  # Tiếng Việt (mặc định)
```

## 📈 Giải thích Metrics

### BERTScore Precision
- Đo lường: Tỷ lệ tokens trong predicted answer có semantic match với ground truth
- Cao (>0.8): Ít thông tin sai/thừa
- Thấp (<0.6): Nhiều thông tin không liên quan

### BERTScore Recall
- Đo lường: Tỷ lệ tokens trong ground truth được match bởi predicted answer
- Cao (>0.8): Bao phủ đầy đủ thông tin
- Thấp (<0.6): Thiếu nhiều thông tin quan trọng

### BERTScore F1
- Điểm tổng hợp (harmonic mean của P và R)
- Metric chính để đánh giá overall quality

## 🆚 So sánh với metrics khác

| Metric | Ưu điểm | Nhược điểm |
|--------|---------|------------|
| **BLEU** | Nhanh, đơn giản | Chỉ đo n-gram overlap, không hiểu ngữ nghĩa |
| **ROUGE** | Tốt cho summarization | Chỉ đo overlap, không hiểu paraphrase |
| **BERTScore** | Đo ngữ nghĩa, hiểu paraphrase | Chậm hơn, cần GPU |
| **Cosine Similarity** | Đo tương đồng tổng thể | Không chi tiết từng phần |

## ⚙️ Best Practices

1. **Rescale with baseline**: Luôn bật để scores dễ interpret
2. **Model selection**: Chọn model phù hợp với ngôn ngữ
   - Tiếng Việt: `bert-base-multilingual-cased` (auto)
   - Đa ngôn ngữ: `xlm-roberta-base`
3. **Batch processing**: BERTScore xử lý theo batch → nhanh hơn
4. **GPU acceleration**: Nếu có GPU, tốc độ tăng đáng kể

## 📝 Ví dụ kết quả

### Top Answer (F1: 0.8384)
```
Question: Rắn ăn cua có nọc độc được thiết kế cho con mồi nào?
Ground Truth: Nọc độc nhẹ được thiết kế đặc biệt cho cua và cá...
Predicted: Rắn ăn cua có nọc độc được thiết kế đặc biệt cho cua và cá...
→ High similarity! ✅
```

### Low Answer (F1: 0.6338)
```
Question: Rắn ráo nhiều đai có độc không?
Ground Truth: Không, Rắn ráo nhiều đai là loài không có nọc độc.
Predicted: I couldn't find any relevant information...
→ Low similarity! ❌
```

## 🔍 Troubleshooting

### Error: "Model not found"
```bash
# Download lại model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-multilingual-cased')"
```

### Chậm khi chạy
- Giảm batch size trong code
- Sử dụng GPU nếu có
- Chọn model nhẹ hơn

### Scores quá thấp
- Kiểm tra language code đúng chưa
- Thử model khác phù hợp hơn
- Kiểm tra quality của ground_truth và predictions

## 📚 Tham khảo

- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [BERTScore GitHub](https://github.com/Tiiiger/bert_score)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 📞 Support

Nếu có vấn đề:
1. Check logs trong terminal
2. Verify input file format
3. Test với data mẫu nhỏ trước
