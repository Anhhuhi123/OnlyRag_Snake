# ✅ Tóm tắt - Hệ thống đánh giá RAG với RAGAS

## 📦 Các file đã tạo

### 1. **generate_predictions.py** ⭐
- **Mục đích**: Bước 1 - Generate contexts và answer từ RAG pipeline
- **Input**: `data/Eveluate.json` (question, ground_truth)
- **Output**: `data/predictions.json` (question, ground_truth, contexts, answer)
- **Chạy**: `python generate_predictions.py`

### 2. **evaluate_ragas.py** ⭐  
- **Mục đích**: Bước 2 - Đánh giá predictions bằng RAGAS
- **Input**: `data/predictions.json`
- **Output**: `evaluation_results.json` + `evaluation_results.csv`
- **Chạy**: `python evaluate_ragas.py`

### 3. **RAGAS_GUIDE.md** 📚
- **Mục đích**: Hướng dẫn đầy đủ về cách sử dụng
- **Nội dung**:
  - Hướng dẫn từng bước chi tiết
  - Giải thích 6 RAGAS metrics
  - Cách phân tích kết quả
  - Troubleshooting
  - Tips & best practices

### 4. **requirements.txt** (đã update)
- Thêm 3 dependencies:
  - `ragas>=0.1.0`
  - `datasets>=2.14.0`
  - `pandas>=2.0.0`

### 5. **README.md** (đã update)
- Thêm section về đánh giá RAGAS
- Link đến RAGAS_GUIDE.md

## 🎯 Workflow

### Luồng hoàn chỉnh:

```
┌─────────────────────────────┐
│ data/Eveluate.json          │
│ (question + ground_truth)   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ python generate_predictions │ ← BƯỚC 1
│ .py                         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ data/predictions.json       │
│ (+ contexts + answer)       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ python evaluate_ragas.py    │ ← BƯỚC 2
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ evaluation_results.json     │
│ evaluation_results.csv      │
│ (RAGAS scores)              │
└─────────────────────────────┘
```

## 🚀 Sử dụng

### Cài đặt:
```bash
pip install ragas datasets pandas
```

### Chạy:
```bash
# Bước 1: Generate predictions
python generate_predictions.py

# Bước 2: Evaluate
python evaluate_ragas.py
```

### Options:
```bash
# Custom input/output
python generate_predictions.py --input data/test.json --output data/test_pred.json
python evaluate_ragas.py --input data/test_pred.json --output test_results.json

# Chọn metrics
python evaluate_ragas.py --metrics faithfulness answer_correctness
```

## 📊 Output

1. **predictions.json** - Có 4 trường cho mỗi câu hỏi:
   - `question`: Câu hỏi
   - `ground_truth`: Câu trả lời đúng
   - `contexts`: List các context passages từ vector search
   - `answer`: Câu trả lời từ LLM

2. **evaluation_results.json** - RAGAS scores + chi tiết
3. **evaluation_results.csv** - Bảng dữ liệu để phân tích

## 🎓 RAGAS Metrics

1. **Faithfulness** - Câu trả lời có trung thực với contexts không?
2. **Answer Relevancy** - Câu trả lời có liên quan đến câu hỏi không?
3. **Context Precision** - Contexts có chất lượng cao không?
4. **Context Recall** - Đã lấy đủ thông tin chưa?
5. **Answer Similarity** - Giống ground truth không?
6. **Answer Correctness** - Tổng hợp: chính xác + tương đồng

## 💡 Ưu điểm của cách làm này

✅ **Tách biệt rõ ràng**: 
- Bước 1: Generate data (tốn thời gian, chạy 1 lần)
- Bước 2: Evaluate (nhanh, có thể chạy nhiều lần với metrics khác nhau)

✅ **Tiết kiệm thời gian**: 
- Không cần generate lại khi muốn thử metrics khác
- Có thể backup predictions

✅ **Dễ debug**:
- Check predictions.json để xem contexts và answer
- Phân tích từng bước riêng biệt

✅ **Linh hoạt**:
- Có thể evaluate nhiều lần với config khác nhau
- Dễ so sánh giữa các versions

## 📖 Đọc thêm

Xem **RAGAS_GUIDE.md** để biết:
- Hướng dẫn chi tiết
- Giải thích metrics
- Cách cải thiện điểm số
- Troubleshooting

---

**Chúc bạn đánh giá thành công! 🎉**
