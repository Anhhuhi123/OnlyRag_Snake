import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for RAG pipeline"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Model configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Local embedding model (384 dimensions)
    EMBEDDING_BATCH_SIZE = 32  # Batch size for local model (adjust based on your GPU/CPU)
    EMBEDDING_DELAY = 0  # No delay needed for local model
    
    # LLM Rate limiting (Gemini Free Tier: 10 requests/minute)
    LLM_REQUESTS_PER_MINUTE = 9  # Stay under 10 to be safe
    LLM_DELAY_BETWEEN_REQUESTS = 7  # Delay in seconds (60/9 ≈ 6.7s)
    
    # RAG configurations
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5  # Legacy parameter (không dùng nếu có reranking)
    
    # Field-specific chunking (bật/tắt chunk size riêng cho từng field)
    USE_FIELD_SPECIFIC_CHUNKING = True
    
    # Chunking mode: "words" hoặc "chars"
    CHUNK_BY = "words"  # "words" = chia theo từ, "chars" = chia theo ký tự
    
    # Chunk size và overlap cho từng field (nếu USE_FIELD_SPECIFIC_CHUNKING = True)
    # Giá trị theo CHUNK_BY: nếu "words" thì là số từ, nếu "chars" thì là số ký tự
    # Chiến lược: chunk_size = 60-70% của average word count, overlap = 25-30%
    FIELD_CHUNK_CONFIG = {
        # Nhóm dài (> 400 words): Chunk size lớn để giữ ngữ cảnh
        "Triệu chứng khi bị cắn": {           # Avg: 471 words
            "chunk_size": 300,                # 64% của avg
            "chunk_overlap": 80               # 27% overlap
        },
        "Cách xử lý": {                       # Avg: 476 words
            "chunk_size": 300,                # 63% của avg
            "chunk_overlap": 80               # 27% overlap
        },
        "Đặc điểm hình thái": {               # Avg: 426 words
            "chunk_size": 280,                # 66% của avg
            "chunk_overlap": 70               # 25% overlap
        },
        
        # Nhóm trung bình dài (250-300 words)
        "Tên khoa học và tên phổ thông": {   # Avg: 292 words
            "chunk_size": 200,                # 68% của avg
            "chunk_overlap": 50               # 25% overlap
        },
        
        # Nhóm trung bình (180-230 words)
        "Phân bố địa lý và môi trường sống": {  # Avg: 224 words
            "chunk_size": 150,                # 67% của avg
            "chunk_overlap": 40               # 27% overlap
        },
        "Hành vi và sinh thái": {             # Avg: 219 words
            "chunk_size": 150,                # 69% của avg
            "chunk_overlap": 40               # 27% overlap
        },
        "Tập tính săn mồi": {                 # Avg: 206 words
            "chunk_size": 140,                # 68% của avg
            "chunk_overlap": 35               # 25% overlap
        },
        "Phân loại học": {                    # Avg: 190 words
            "chunk_size": 130,                # 68% của avg
            "chunk_overlap": 35               # 27% overlap
        },
        "Các quan sát thú vị từ các nhà nghiên cứu": {  # Avg: 186 words
            "chunk_size": 130,                # 70% của avg
            "chunk_overlap": 35               # 27% overlap
        },
        
        # Nhóm ngắn (150-170 words)
        "Độc tính": {                         # Avg: 170 words
            "chunk_size": 120,                # 71% của avg
            "chunk_overlap": 30               # 25% overlap
        },
        "Sinh sản": {                         # Avg: 162 words
            "chunk_size": 110,                # 68% của avg
            "chunk_overlap": 30               # 27% overlap
        },
        
        # Nhóm rất ngắn (< 150 words): Chunk size lớn để tránh chia quá nhỏ
        "Tình trạng bảo tồn": {               # Avg: 119 words
            "chunk_size": 150,                # Lớn hơn avg để ít chunk
            "chunk_overlap": 30               # 20% overlap
        },
        "Giá trị nghiên cứu": {               # Avg: 109 words
            "chunk_size": 150,                # Lớn hơn avg để ít chunk
            "chunk_overlap": 30               # 20% overlap
        },
        "Sự liên quan với con người": {       # Avg: 105 words
            "chunk_size": 150,                # Lớn hơn avg để ít chunk
            "chunk_overlap": 30               # 20% overlap
        }
    }
    
    # Re-ranking configurations
    USE_RERANKING = True
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Với chunk size lớn (110-300 words), mỗi chunk chứa nhiều thông tin
    # → Cần ít chunks hơn để đủ context
    RERANK_TOP_K = 15   # Lấy 15 candidates để rerank (tăng từ 10)
    FINAL_TOP_K = 5     # Giữ top 5 chunks có điểm cao nhất sau rerank
    
    # Giải thích:
    # - RERANK_TOP_K = 15: Đủ rộng để cover nhiều fields khác nhau
    # - FINAL_TOP_K = 5: 5 chunks × 150-300 words = 750-1500 words context
    #   → Đủ để trả lời hầu hết câu hỏi mà không quá dài
    # - Nếu câu hỏi phức tạp: có thể tăng FINAL_TOP_K lên 8-10
    
    RERANK_ALPHA = 0.7  # Weight for cross-encoder score (0.7) vs original score (0.3)
    
    # FAISS configurations
    VECTOR_DIMENSION = 384  # multilingual-e5-small embedding dimension
    FAISS_INDEX_PATH = "faiss_index"
    
    # Qdrant configurations 
    USE_QDRANT = True  # Set to True to use Qdrant instead of FAISS (tạm thời dùng FAISS vì mạng không ổn)
    QDRANT_COLLECTION_NAME = "snake_knowledge_base" # Lưu trữ trong Qdrant
    
    @classmethod
    def validate(cls):
        """Validate that all required configurations are set"""
        # Only validate Google API key for LLM (embedding now runs locally)
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables (needed for LLM)")
        if cls.USE_QDRANT and not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY not found in environment variables")
        return True