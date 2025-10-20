# 🎯 RAGAS Evaluation - Quick Reference

## ⚡ Quick Start (3 bước)

```bash
# 1. Cài đặt
pip install ragas datasets pandas

# 2. Generate predictions  
python generate_predictions.py

# 3. Evaluate
python evaluate_ragas.py
```

## 📁 Files

| File | Mô tả |
|------|-------|
| `data/Eveluate.json` | Input: Câu hỏi + ground truth |
| `generate_predictions.py` | Script bước 1: Generate contexts + answer |
| `data/predictions.json` | Output bước 1: Có đủ 4 trường |
| `evaluate_ragas.py` | Script bước 2: Đánh giá RAGAS |
| `evaluation_results.json` | Output: RAGAS scores |
| `evaluation_results.csv` | Output: Bảng dữ liệu |
| `RAGAS_GUIDE.md` | Hướng dẫn đầy đủ |

## 🎯 RAGAS Metrics

| Metric | Đo lường gì? | Điểm tốt |
|--------|--------------|----------|
| Faithfulness | Trung thực với contexts | > 0.7 |
| Answer Relevancy | Liên quan đến câu hỏi | > 0.7 |
| Context Precision | Chất lượng contexts | > 0.7 |
| Context Recall | Đủ thông tin chưa | > 0.7 |
| Answer Similarity | Giống ground truth | > 0.7 |
| Answer Correctness | Tổng hợp | > 0.7 |

## 💻 Commands

### Generate predictions
```bash
# Basic
python generate_predictions.py

# Custom files
python generate_predictions.py --input data/test.json --output data/test_pred.json
```

### Evaluate
```bash
# All metrics
python evaluate_ragas.py

# Specific metrics
python evaluate_ragas.py --metrics faithfulness answer_correctness

# Custom files
python evaluate_ragas.py --input data/test_pred.json --output results.json
```

## 📊 Output Format

### predictions.json
```json
{
  "question": "...",
  "ground_truth": "...",
  "contexts": ["...", "...", "..."],
  "answer": "..."
}
```

### evaluation_results.json
```json
{
  "ragas_scores": {
    "faithfulness": 0.8234,
    "answer_relevancy": 0.7891,
    ...
  },
  "predictions": [...],
  "summary": {...}
}
```

## ❌ Troubleshooting

| Lỗi | Fix |
|-----|-----|
| No index found | `python main.py` |
| Import ragas failed | `pip install ragas datasets pandas` |
| Empty contexts | Rebuild index: `python main.py` |
| Rate limit | Wait hoặc reduce requests |

## 📈 Cải thiện điểm số

| Metric thấp | Cách cải thiện |
|-------------|----------------|
| Faithfulness | Improve contexts quality |
| Answer Relevancy | Tune LLM prompt |
| Context Precision/Recall | Optimize retrieval (chunk size, top_k) |
| Answer Correctness | Improve contexts + prompt |

## 🔄 Workflow

```
Eveluate.json
    ↓
generate_predictions.py  ← Bước 1 (chạy 1 lần)
    ↓
predictions.json
    ↓
evaluate_ragas.py        ← Bước 2 (có thể chạy nhiều lần)
    ↓
evaluation_results.json
```

## 💡 Pro Tips

✅ Test nhỏ trước: Tạo file 10-20 câu để test

✅ Backup predictions: Không phải generate lại

✅ Compare versions: Lưu results với tên khác nhau

✅ Track experiments: Document changes và scores

## 📚 Documentation

- **RAGAS_GUIDE.md**: Hướng dẫn đầy đủ
- **RAGAS_SUMMARY.md**: Tóm tắt hệ thống
- **predictions_example.json**: Ví dụ output

---

**Need help?** → Đọc `RAGAS_GUIDE.md`
