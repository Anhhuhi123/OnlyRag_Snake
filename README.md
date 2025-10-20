# RAG Pipeline - Hệ thống Tra cứu Thông tin Rắn Việt Nam

Hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh sử dụng Google Gemini, Qdrant Cloud và Cross-Encoder Re-ranking để tra cứu thông tin về các loài rắn tại Việt Nam.

## ✨ Tính năng nổi bật

- **🤖 Gemini 2.5 Flash LLM** - Mô hình ngôn ngữ tiên tiến cho việc sinh câu trả lời
- **🔍 Gemini Embedding (3072 chiều)** - Embedding chất lượng cao cho tìm kiếm ngữ nghĩa
- **☁️ Qdrant Cloud Vector Store** - Cơ sở dữ liệu vector trên cloud, thay thế FAISS local
- **🎯 Cross-Encoder Re-ranking** - Sắp xếp lại kết quả với ms-marco-MiniLM-L-12-v2
- **⚡ Rate Limiting thông minh** - Batching tự động và exponential backoff cho Gemini API
- **📄 Document Processing** - Chunking và tiền xử lý văn bản thông minh
- **🏗️ Kiến trúc module hóa** - Code sạch, dễ bảo trì và mở rộng
- **📊 JSON Structured Data** - Hỗ trợ 13 trường dữ liệu chuyên biệt cho từng loài rắn

## 📂 Cấu trúc thư mục

```
RAG/
├── README.md                     # Hướng dẫn chi tiết
├── QDRANT_GUIDE.md              # Hướng dẫn sử dụng Qdrant Cloud
├── TROUBLESHOOTING.md           # Khắc phục sự cố
├── requirements.txt              # Thư viện Python cần thiết
├── .env                         # Biến môi trường (API keys)
├── main.py                      # Điểm vào chính của hệ thống
│
├── config/
│   ├── __init__.py
│   └── config.py                # Quản lý cấu hình toàn hệ thống
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py            # Tạo embedding với Gemini (có batching)
│   ├── vector_store.py          # FAISS vector store (local)
│   ├── qdrant_vector_store.py   # Qdrant Cloud vector store
│   ├── llm.py                   # Gemini 2.5 Flash LLM
│   ├── document_processor.py    # Xử lý và chunking document
│   ├── reranker.py              # Cross-encoder re-ranking
│   └── rag_pipeline.py          # Orchestrator chính của RAG
│
├── data/
│   ├── __init__.py
│   ├── document_RAG.json        # Dữ liệu 45 loài rắn (13 fields/loài)
│   └── json_loader.py           # Load và xử lý JSON data
│
└── examples/
    ├── __init__.py
    └── demo.py                  # Script demo và test
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/Anhhuhi123/Rank-F1.git
cd RAG
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `google-genai` - Gemini API
- `qdrant-client` - Qdrant Cloud vector database
- `sentence-transformers` - Cross-encoder re-ranking
- `torch` - Deep learning framework
- `numpy`, `tqdm` - Utilities

### 3. Cấu hình API Keys

Tạo file `.env` trong thư mục gốc:

```bash
# Google Gemini API Key (Free tier: 15 requests/min, 1500 requests/day)
GOOGLE_API_KEY=your_gemini_api_key_here

# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
```

**Lấy API Keys:**
- **Gemini API**: https://aistudio.google.com/app/apikey
- **Qdrant Cloud**: https://cloud.qdrant.io/ (Đăng ký free tier)

### 4. Cấu hình hệ thống

Chỉnh sửa `config/config.py` nếu cần:

```python
# Vector Store (chọn Qdrant hoặc FAISS)
USE_QDRANT = True  # True = Qdrant Cloud, False = FAISS local

# Re-ranking
USE_RERANKING = True
RERANK_TOP_K = 10      # Lấy 10 kết quả từ vector store
FINAL_TOP_K = 5        # Sau re-ranking chỉ giữ 5 kết quả tốt nhất
RERANK_ALPHA = 0.7     # 70% cross-encoder + 30% original score

# Rate Limiting
EMBEDDING_BATCH_SIZE = 100  # Gemini API giới hạn 100 items/request
EMBEDDING_DELAY = 5.0       # Đợi 5 giây giữa các batch
```

## 🎯 Hướng dẫn sử dụng

### 1️⃣ Ingest dữ liệu (Lần đầu tiên)

**Ingest toàn bộ document:**
```bash
python main.py --ingest data/document_RAG.json
```

**Ingest với trường cụ thể:**
```bash
python main.py --ingest data/document_RAG.json --json-fields 'Tên khoa học và tên phổ thông'
```

**Ingest nhiều trường:**
```bash
python main.py --ingest data/document_RAG.json --json-fields 'Tên khoa học và tên phổ thông' 'Đặc điểm nhận dạng' 'Độc tính'
```

Quá trình này sẽ:
- Đọc file JSON chứa thông tin 124 loài rắn
- Chia nhỏ văn bản thành chunks (chunk_size=40, overlap=7)
- Tạo embeddings với Gemini (batching tự động)
- Lưu vào Qdrant Cloud (hoặc FAISS nếu USE_QDRANT=False)

### 2️⃣ Tra cứu thông tin

**Query đơn giản:**
```bash
python main.py --query "Rắn hổ mang chúa là rắn độc hay không độc?"
```

**Query phức tạp:**
```bash
python main.py --query "Những loài rắn nào ở Việt Nam có màu xanh lá cây?"
```

### 3️⃣ Chế độ interactive

```bash
python main.py --demo
```

Hoặc chạy trực tiếp:
```bash
python examples/demo.py
```

### 4️⃣ Test re-ranking

```bash
python test_reranking.py
```

### 5️⃣ Test Qdrant connection

```bash
python test_qdrant.py
```

## 📊 Luồng hoạt động tổng thể

### 🔹 Giai đoạn 1: Ingest Documents (Chỉ chạy 1 lần)

```
┌─────────────────────┐
│ document_RAG.json   │      
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  JSONLoader.load_from_json()                │
│  - Đọc JSON với 13 fields/loài              │
│  - Chọn fields cụ thể (nếu có --json-fields)│
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  DocumentProcessor.process_documents()       │
│  - Chunk văn bản (size=40, overlap=7)       │
│  - Tổng: ~258 chunks từ 45 documents        │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  EmbeddingGenerator.generate_embeddings()    │
│  - Chia thành batches (100 items/batch)     │
│  - 258 chunks → 3 requests                  │
│  - Đợi 5s giữa các batch                    │
│  - Exponential backoff nếu 429 error        │
│  - Output: 258 vectors × 3072 dimensions    │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  QdrantVectorStore.add_embeddings()          │
│  - Upload lên Qdrant Cloud                  │
│  - Collection: snake_knowledge_base         │
│  - Metric: Cosine similarity                │
└─────────────────────────────────────────────┘
```

**Chi tiết rate limiting:**
- Gemini Free Tier: 15 requests/minute, 1,500 requests/day
- Batching: 258 chunks / 100 per batch = 3 requests
- Delay: 5 giây giữa mỗi batch → tổng ~15 giây
- Nếu lỗi 429: Retry với exponential backoff (2s → 4s → 8s)

### 🔹 Giai đoạn 2: Query Processing (Mỗi lần hỏi)

```
┌─────────────────────┐
│  User Query         │
│  "Rắn hổ mang       │
│   chúa độc không?"  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  EmbeddingGenerator.generate_single_embedding│
│  - Tạo query embedding (3072 dim)           │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  QdrantVectorStore.search()                  │
│  - Tìm kiếm vector similarity               │
│  - Lấy top-10 chunks (RERANK_TOP_K=10)      │
│  - Cosine distance ranking                  │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  CrossEncoderReranker.rerank()               │
│  - Model: ms-marco-MiniLM-L-12-v2           │
│  - Re-score 10 chunks dựa trên query        │
│  - Kết hợp scores: 70% cross-encoder +      │
│                    30% original similarity  │
│  - Chọn top-5 chunks tốt nhất               │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  LLM.generate_response()                     │
│  - Model: Gemini 2.5 Flash                  │
│  - Context: 5 chunks được re-rank           │
│  - Prompt: Question + Context               │
│  - Sinh câu trả lời tự nhiên                │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Final Answer       │
│  + Sources          │
│  + Scores           │
└─────────────────────┘
```

### 🔹 Chi tiết Re-ranking

```
Input: 10 passages từ Qdrant (RERANK_TOP_K=10)

┌─────────────────────────────────────────────┐
│  Cross-Encoder Scoring                       │
├─────────────────────────────────────────────┤
│  For each passage:                           │
│    - Input: [query, passage] pair           │
│    - Model: ms-marco-MiniLM-L-12-v2         │
│    - Output: relevance score (0-1)          │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Score Fusion (alpha=0.7)                    │
├─────────────────────────────────────────────┤
│  final_score = 0.7 × cross_encoder_score    │
│              + 0.3 × original_cosine_score  │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Re-sort và lấy top-5 (FINAL_TOP_K=5)       │
└─────────────────────────────────────────────┘

Output: 5 passages tốt nhất → Gửi vào LLM
```

## 🛠️ Cấu hình nâng cao

### Chuyển đổi giữa Qdrant và FAISS

**Sử dụng Qdrant Cloud (Khuyến nghị):**
```python
# config/config.py
USE_QDRANT = True
```

**Sử dụng FAISS Local:**
```python
# config/config.py
USE_QDRANT = False
```

### Tắt Re-ranking

```python
# config/config.py
USE_RERANKING = False
```

### Điều chỉnh chunking

```python
# config/config.py
CHUNK_SIZE = 40        # Số words/chunk
CHUNK_OVERLAP = 7      # Số words overlap giữa chunks
```

### Điều chỉnh retrieval

```python
# config/config.py
RERANK_TOP_K = 10      # Số chunks lấy từ vector store
FINAL_TOP_K = 5        # Số chunks sau re-ranking
RERANK_ALPHA = 0.7     # Trọng số cross-encoder (0.0 - 1.0)
```

## 📝 Ví dụ sử dụng

### Ví dụ 1: Query về độc tính

```bash
python main.py --query "Rắn lục đuôi đỏ có độc không?"
```

**Output:**
```
Query: Rắn lục đuôi đỏ có độc không?
Processing query...
Generating query embedding...
  Processing batch 1/1 (1 texts)...
Searching vector store...
Found 10 candidate passages
Applying cross-encoder re-ranking...
Re-ranked to top 5 passages
Generating response...

Answer:
Có, rắn lục đuôi đỏ (Trimeresurus albolabris) là loài rắn độc. 
Nọc độc của chúng có thể gây đau, sưng và hoại tử mô tại chỗ cắn...

Sources:
1. Trimeresurus albolabris - Rắn lục đuôi đỏ... (Score: 0.95)
2. Độc tính của rắn lục... (Score: 0.89)
...
```

### Ví dụ 2: Query về phân bố

```bash
python main.py --query "Những loài rắn nào sống ở miền Bắc Việt Nam?"
```

### Ví dụ 3: Ingest chỉ trường Độc tính

```bash
python main.py --ingest data/document_RAG.json --json-fields 'Độc tính' 'Xử lý khi bị cắn'
```

## 🐛 Khắc phục sự cố

### Lỗi: 429 RESOURCE_EXHAUSTED

**Nguyên nhân:** Vượt quota Gemini API (15 requests/minute)

**Giải pháp:**
1. Tăng `EMBEDDING_DELAY` trong `config/config.py`
2. Đợi API quota reset (00:00 UTC mỗi ngày)
3. Upgrade lên paid tier

### Lỗi: 503 Model Overloaded

**Nguyên nhân:** Gemini API quá tải tạm thời

**Giải pháp:**
- Đợi vài phút rồi thử lại
- Hệ thống có exponential backoff tự động

### Model re-ranking tải chậm

**Nguyên nhân:** Lần đầu download model từ HuggingFace

**Giải pháp:**
- Đợi download hoàn tất (~400MB)
- Model được cache tại: `~/.cache/huggingface/hub/`

Xem thêm: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📚 Tài liệu tham khảo

- [QDRANT_GUIDE.md](QDRANT_GUIDE.md) - Hướng dẫn chi tiết về Qdrant Cloud
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Khắc phục sự cố đầy đủ

## 🔗 Links hữu ích

- **Gemini API**: https://aistudio.google.com/
- **Qdrant Cloud**: https://cloud.qdrant.io/
- **Sentence Transformers**: https://www.sbert.net/
- **Cross-Encoder Models**: https://www.sbert.net/docs/pretrained_cross-encoders.html

## 📊 Thống kê hệ thống

- **Số loài rắn:** 124 loài
- **Số chunks:** ~258 chunks (sau chunking)
- **Embedding dimension:** 3072 (Gemini)
- **Vector store:** Qdrant Cloud
- **Re-ranking model:** ms-marco-MiniLM-L-12-v2
- **LLM:** Gemini 2.5 Flash

## 🎓 Kiến trúc kỹ thuật

### Two-Stage Retrieval

1. **Stage 1 - Dense Retrieval (Qdrant)**
   - Tìm kiếm nhanh dựa trên cosine similarity
   - Lấy top-10 candidates (recall cao)

2. **Stage 2 - Cross-Encoder Re-ranking**
   - Đánh giá chính xác từng passage với query
   - Lấy top-5 final results (precision cao)

### Rate Limiting Strategy

```python
# Gemini API Limits
Free Tier: 15 requests/minute, 1,500 requests/day

# Batching Strategy
- Batch size: 100 items/request (max của API)
- Delay: 5 seconds between batches
- Exponential backoff: 2s → 4s → 8s (max 3 retries)

# Example: 258 chunks
258 / 100 = 3 requests
3 requests × 5s delay = 15 seconds total
```

## 👥 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo Pull Request hoặc mở Issue.

## 📄 License

MIT License

---

**Phát triển bởi:** Anh Hu Hi  
**Repository:** https://github.com/Anhhuhi123/Rank-F1  
**Ngày cập nhật:** 16/10/2025

### Command Line Interface

```bash
# Test all components
python main.py --test

# Ingest sample document
python main.py --ingest

# Ask a single question
python main.py --query "What are the five Sacred Temples?"

# Show pipeline statistics
python main.py --stats

# Reset the pipeline
python main.py --reset

# Run comprehensive demo
python main.py --demo
```

### Python API

```python
from src.rag_pipeline import RAGPipeline
from data.sample_story import get_sample_story

# Initialize pipeline
rag = RAGPipeline()

# Test components
test_results = rag.test_components()

# Ingest documents
sample_story = get_sample_story()
stats = rag.ingest_documents([sample_story])

# Query the pipeline
result = rag.query("Who is the main character?")
print(result["response"])
```

## Configuration

The pipeline can be configured through `config/config.py`:

```python
class Config:
    # Model configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_BATCH_SIZE = 100  # Max batch size for Gemini API
    EMBEDDING_DELAY = 2.0       # Delay between batches to avoid rate limits
    
    # RAG configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # Re-ranking configurations
    USE_RERANKING = True
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANK_TOP_K = 10        # Retrieve more candidates for re-ranking
    FINAL_TOP_K = 5          # Final number after re-ranking
    RERANK_ALPHA = 0.7       # Weight: 0.7 cross-encoder + 0.3 original score
    
    # FAISS configurations
    VECTOR_DIMENSION = 768
```

## Components

### 1. Embedding Generator (`src/embeddings.py`)
- Uses Gemini embedding model for text vectorization
- Supports batch and single text embedding generation
- Handles API communication and error management

### 2. Vector Store (`src/vector_store.py`)
- FAISS-based vector database for similarity search
- Supports saving/loading indexes to/from disk
- Implements cosine similarity search
- Provides statistics and management functions

### 3. LLM Interface (`src/llm.py`)
- Gemini 2.5 Flash integration for response generation
- Context-aware response generation
- Configurable thinking budget (disabled for faster responses)
- Error handling and fallback mechanisms

### 4. Document Processor (`src/document_processor.py`)
- Intelligent text chunking with overlap
- Text cleaning and normalization
- Sentence-aware splitting for better context preservation
- Configurable chunk sizes and overlap

### 5. Cross-encoder Re-ranker (`src/reranker.py`) 🆕
- Uses `cross-encoder/ms-marco-MiniLM-L-12-v2` for passage re-ranking
- Combines original similarity scores with cross-encoder scores
- Configurable score combination weights (alpha parameter)
- Significantly improves retrieval quality and relevance

### 6. RAG Pipeline (`src/rag_pipeline.py`)
- Main orchestrator combining all components
- Document ingestion and indexing workflow
- Query processing with optional re-ranking
- Pipeline state management and statistics
- **Enhanced retrieval flow**: Vector search → Cross-encoder re-ranking → LLM generation

## Re-ranking Architecture

The pipeline now supports advanced two-stage retrieval:

```
Query → Dense Retrieval (Top-10) → Cross-encoder Re-ranking (Top-5) → LLM Response
```

**Benefits of Re-ranking:**
- 🎯 **Higher Precision**: Cross-encoder provides more accurate relevance scoring
- 🔄 **Best of Both Worlds**: Combines fast dense retrieval with precise cross-encoder scoring  
- ⚙️ **Configurable**: Adjustable weights between original and cross-encoder scores
- 📊 **Transparent**: Detailed scoring information for analysis

## Sample Data

The pipeline includes a sample fantasy story "The Chronicles of Eldoria: The Last Guardian" for testing and demonstration purposes. The story contains rich narrative content perfect for testing RAG capabilities.

## Demo Features

The interactive demo (`examples/demo.py`) includes:

- **Component Testing** - Verify all components work correctly
- **Document Ingestion** - Process and index the sample story
- **Predefined Queries** - Test with curated questions
- **Interactive Mode** - Ask your own questions
- **Statistics Display** - View pipeline performance metrics

## API Key Setup

1. Get your Google API key from [Google AI Studio](https://aistudio.google.com/)
2. Add it to the `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Performance

- **Embedding Dimension**: 768 (Gemini embedding model)
- **Default Chunk Size**: 1000 characters
- **Default Overlap**: 200 characters
- **Default Top-K**: 5 results
- **Vector Search**: Cosine similarity with FAISS

## Error Handling

The pipeline includes comprehensive error handling:
- API key validation
- Network error recovery
- Malformed input handling
- Graceful degradation when components fail

## 📊 Đánh giá với RAGAS

### Giới thiệu

RAGAS (Retrieval Augmented Generation Assessment) - Framework đánh giá RAG pipeline với 6 metrics:
- **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**, **Answer Similarity**, **Answer Correctness**

### Quick Start

```bash
# 1. Cài đặt RAGAS
pip install ragas datasets pandas

# 2. Bước 1: Generate predictions (contexts + answer)
python generate_predictions.py

# 3. Bước 2: Evaluate với RAGAS
python evaluate_ragas.py
```

### Quy trình 2 bước

**Bước 1**: Generate predictions từ RAG
```
Input:  data/Eveluate.json (question, ground_truth)
        ↓
Run:    python generate_predictions.py
        ↓
Output: data/predictions.json (question, ground_truth, contexts, answer)
```

**Bước 2**: Evaluate bằng RAGAS
```
Input:  data/predictions.json
        ↓
Run:    python evaluate_ragas.py
        ↓
Output: evaluation_results.json + evaluation_results.csv
```

### Chi tiết

Xem **`RAGAS_GUIDE.md`** để biết:
- Hướng dẫn chi tiết từng bước
- Giải thích các metrics
- Cách phân tích kết quả
- Tips & troubleshooting

## Contributing

1. Fork the repository
2. Create a feature branch

**Phát triển bởi:** Anh Hu Hi  
**Repository:** https://github.com/Anhhuhi123/Rank-F1  
**Ngày cập nhật:** 19/10/2025