# 📊 Hướng dẫn đánh giá RAG với RAGAS

## 🎯 Tổng quan

Hệ thống đánh giá RAG gồm 2 bước:

1. **Generate Predictions**: Chạy tất cả câu hỏi qua RAG pipeline → sinh ra contexts và answer
2. **Evaluate with RAGAS**: Đánh giá predictions bằng RAGAS metrics

## 📁 Cấu trúc Files

```
Input:  data/Eveluate.json      → Câu hỏi và ground truth
        ↓
Step 1: generate_predictions.py → Sinh contexts + answer
        ↓
Output: data/predictions.json   → Có đủ 4 trường: question, ground_truth, contexts, answer
        ↓
Step 2: evaluate_ragas.py       → Đánh giá bằng RAGAS
        ↓
Output: evaluation_results.json → RAGAS scores + chi tiết
        evaluation_results.csv  → Bảng dữ liệu
```

## 🚀 Cài đặt

### 1. Cài đặt thư viện RAGAS

```bash
pip install ragas datasets pandas
```

### 2. Kiểm tra file Eveluate.json

File `data/Eveluate.json` cần có định dạng:

```json
[
  {
    "question": "Câu hỏi 1?",
    "ground_truth": "Câu trả lời đúng 1"
  },
  {
    "question": "Câu hỏi 2?",
    "ground_truth": "Câu trả lời đúng 2"
  }
]
```

## 📝 Sử dụng

### BƯỚC 1: Generate Predictions

Chạy tất cả câu hỏi qua RAG pipeline để sinh contexts và answer:

```bash
python generate_predictions.py
```

**Options:**
```bash
# Chỉ định input/output file
python generate_predictions.py --input data/Eveluate.json --output data/predictions.json
```

**Output**: File `data/predictions.json` với cấu trúc:

```json
[
  {
    "question": "Tên khoa học của rắn X là gì?",
    "ground_truth": "Tên khoa học là ABC xyz",
    "contexts": [
      "Context passage 1...",
      "Context passage 2...",
      "Context passage 3..."
    ],
    "answer": "Tên khoa học của rắn X là ABC xyz..."
  }
]
```

**Thời gian**: Tùy số câu hỏi
- 100 câu: ~5-10 phút
- 500 câu: ~20-30 phút

### BƯỚC 2: Evaluate với RAGAS

Đánh giá file predictions bằng RAGAS:

```bash
python evaluate_ragas.py
```

**Options:**

```bash
# Chỉ định input/output
python evaluate_ragas.py --input data/predictions.json --output my_results.json

# Chọn metrics cụ thể
python evaluate_ragas.py --metrics faithfulness answer_correctness

# Tất cả metrics
python evaluate_ragas.py --metrics all
```

**Output**: 
- `evaluation_results.json` - Kết quả đầy đủ
- `evaluation_results.csv` - Bảng dữ liệu

## 🎯 RAGAS Metrics

| Metric | Ý nghĩa | Đánh giá gì? |
|--------|---------|--------------|
| **Faithfulness** | Độ trung thực | Câu trả lời có nhất quán với contexts không? |
| **Answer Relevancy** | Độ liên quan | Câu trả lời có liên quan đến câu hỏi không? |
| **Context Precision** | Độ chính xác context | Contexts có chất lượng cao không? |
| **Context Recall** | Độ phủ context | Đã lấy đủ thông tin cần thiết chưa? |
| **Answer Similarity** | Độ tương đồng | Giống ground truth về mặt ngữ nghĩa? |
| **Answer Correctness** | Độ chính xác | Tổng hợp: chính xác + tương đồng |

### Cách hiểu điểm số

- **> 0.7**: Tốt ✅
- **0.5 - 0.7**: Trung bình ⚠️
- **< 0.5**: Cần cải thiện ❌

## 📊 Phân tích kết quả

### 1. Xem điểm tổng quan

```bash
# Xem trong terminal khi chạy evaluate_ragas.py
RAGAS SCORES
============================================================
faithfulness        : 0.8234
answer_relevancy    : 0.7891
context_precision   : 0.8012
context_recall      : 0.7543
answer_similarity   : 0.8156
answer_correctness  : 0.7923
============================================================
```

### 2. Xem chi tiết trong CSV

Mở `evaluation_results.csv` trong Excel/Google Sheets để:
- Xem từng câu hỏi, câu trả lời
- So sánh answer vs ground_truth
- Tìm các câu trả lời sai
- Phân tích patterns

### 3. Xem full data trong JSON

File `evaluation_results.json` chứa:
```json
{
  "ragas_scores": {
    "faithfulness": 0.8234,
    ...
  },
  "predictions": [...],
  "summary": {...}
}
```

## 🔄 Workflow hoàn chỉnh

### Lần đầu chạy:

```bash
# Bước 0: Đảm bảo đã có vector index
python main.py

# Bước 1: Generate predictions
python generate_predictions.py

# Bước 2: Evaluate
python evaluate_ragas.py
```

### Khi thử nghiệm cải thiện:

```bash
# 1. Thay đổi config (chunk_size, reranking, etc.)
# Edit config/config.py

# 2. Rebuild index nếu cần
python main.py

# 3. Generate predictions mới
python generate_predictions.py --output data/predictions_v2.json

# 4. Evaluate
python evaluate_ragas.py --input data/predictions_v2.json --output results_v2.json

# 5. So sánh với version cũ
```

## 💡 Tips & Best Practices

### 1. Test nhỏ trước

```bash
# Tạo file test nhỏ
head -n 10 data/Eveluate.json > data/test_10.json

# Generate predictions
python generate_predictions.py --input data/test_10.json --output data/test_predictions.json

# Evaluate
python evaluate_ragas.py --input data/test_predictions.json --output test_results.json
```

### 2. Backup predictions

```bash
# Lưu predictions để không phải generate lại
cp data/predictions.json data/predictions_backup_$(date +%Y%m%d).json
```

### 3. Compare versions

Tạo script để so sánh:
```python
import json

with open('results_v1.json') as f:
    v1 = json.load(f)
    
with open('results_v2.json') as f:
    v2 = json.load(f)

for metric in v1['ragas_scores']:
    diff = v2['ragas_scores'][metric] - v1['ragas_scores'][metric]
    print(f"{metric}: {diff:+.4f}")
```

### 4. Track experiments

Tạo file `experiments.md`:
```markdown
# Experiment Log

## Baseline (2025-10-19)
- Config: chunk_size=400, rerank=True
- Faithfulness: 0.8234
- Answer Correctness: 0.7923

## Experiment 1 (2025-10-19)
- Changes: Increased chunk_size to 600
- Faithfulness: 0.8456 (+0.0222) ✅
- Answer Correctness: 0.8012 (+0.0089) ✅
```

## ❌ Troubleshooting

### Lỗi: No index found

```bash
# Fix: Tạo index trước
python main.py
```

### Lỗi: Import ragas failed

```bash
# Fix: Cài đặt lại
pip install ragas datasets pandas --upgrade
```

### Predictions có contexts rỗng

**Nguyên nhân**: Vector store không có data hoặc embedding failed

**Fix**:
1. Check index: `ls -lh faiss_index.index` hoặc check Qdrant dashboard
2. Rebuild index: `python main.py`
3. Generate lại: `python generate_predictions.py`

### RAGAS evaluation quá chậm

**Nguyên nhân**: RAGAS cần gọi LLM nhiều lần

**Solutions**:
1. Evaluate với ít metrics hơn: `--metrics faithfulness answer_correctness`
2. Test với dataset nhỏ trước
3. Sử dụng API key có rate limit cao hơn

### Lỗi: rate limit exceeded

**Fix**: 
1. Wait và retry
2. Thêm delay trong code
3. Upgrade API plan

## 📈 Cải thiện điểm số

### Faithfulness thấp
→ Cải thiện context quality:
- Tăng chunk size
- Improve chunking strategy
- Enable reranking

### Answer Relevancy thấp  
→ Tune LLM prompt:
- Rõ ràng hơn trong prompt
- Add examples
- Điều chỉnh temperature

### Context Precision/Recall thấp
→ Optimize retrieval:
- Tune top_k
- Improve embedding quality
- Better reranking

### Answer Correctness thấp
→ Tổng hợp:
- Check ground_truth quality
- Improve contexts + LLM prompt
- Fine-tune retrieval

## 🎓 Giải thích chi tiết metrics

### Faithfulness (Độ trung thực)

**Đo lường**: Tất cả claims trong answer có được support bởi contexts không?

**Cách tính**: 
```
Faithfulness = (Số claims được support) / (Tổng số claims)
```

**Ví dụ**:
- Answer: "Rắn lục có nọc độc và sống ở rừng"
- Contexts: "Rắn lục là loài có nọc độc..."
- → Claim 1 ✅ (có nọc), Claim 2 ❌ (không mention rừng)
- → Faithfulness = 0.5

### Answer Relevancy (Độ liên quan)

**Đo lường**: Answer có trả lời đúng câu hỏi không?

**Cách tính**: RAGAS generate questions từ answer, so với original question

**Ví dụ**:
- Question: "Rắn lục sống ở đâu?"
- Answer: "Rắn lục có nọc độc mạnh" → ❌ Không relevant
- Answer: "Rắn lục sống ở rừng nhiệt đới" → ✅ Relevant

### Context Precision (Độ chính xác context)

**Đo lường**: Contexts có liên quan có rank cao hơn không?

**Cách tính**: Kiểm tra relevant contexts có ở top không

**Ví dụ**:
```
Contexts: [relevant, irrelevant, relevant, irrelevant, relevant]
→ Precision thấp (relevant bị xen kẽ)

Contexts: [relevant, relevant, relevant, irrelevant, irrelevant]  
→ Precision cao (relevant ở top)
```

### Context Recall (Độ phủ context)

**Đo lường**: Ground truth có được support bởi contexts không?

**Cách tính**: 
```
Recall = (Sentences in ground_truth có trong contexts) / (Total sentences)
```

## 📚 Tham khảo

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)

## 🎯 Checklist

Trước khi chạy evaluation:

- [ ] Đã cài đặt: `pip install ragas datasets pandas`
- [ ] Có file `data/Eveluate.json` với format đúng
- [ ] Đã chạy `python main.py` để tạo index
- [ ] API keys đã configured trong `.env`
- [ ] Test với dataset nhỏ trước

Sau khi có kết quả:

- [ ] Review RAGAS scores
- [ ] Xem CSV để phân tích chi tiết
- [ ] Document trong experiment log
- [ ] Backup predictions và results
- [ ] Plan cải thiện dựa trên scores

---

**Happy Evaluating! 🎉**
