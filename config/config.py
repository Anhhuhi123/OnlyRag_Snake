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
    TOP_K_RESULTS = 7
    
    # Field-specific chunking (bật/tắt chunk size riêng cho từng field)
    USE_FIELD_SPECIFIC_CHUNKING = True
    
    # Chunking mode: "words" hoặc "chars"
    CHUNK_BY = "words"  # "words" = chia theo từ, "chars" = chia theo ký tự
    
    # Chunk size và overlap cho từng field (nếu USE_FIELD_SPECIFIC_CHUNKING = True)
    # Giá trị theo CHUNK_BY: nếu "words" thì là số từ, nếu "chars" thì là số ký tự
    FIELD_CHUNK_CONFIG = {
        "Tên khoa học và tên phổ thông": {
            "chunk_size": 80,      # 40 từ
            "chunk_overlap": 30
        },
        "Phân loại học": {
            "chunk_size": 60,
            "chunk_overlap": 20
        },
        "Đặc điểm hình thái": {
            "chunk_size": 80,
            "chunk_overlap": 30
        },
        "Độc tính": {
            "chunk_size": 80,
            "chunk_overlap": 20
        },
        "Tập tính săn mồi": {
            "chunk_size": 70,
            "chunk_overlap": 15
        },
        "Hành vi và sinh thái": {
            "chunk_size": 70,
            "chunk_overlap": 15
        },
        "Phân bố địa lý và môi trường sống": {
            "chunk_size": 70,
            "chunk_overlap": 15
        },
        "Sinh sản": {
            "chunk_size": 60,
            "chunk_overlap": 15
        },
        "Tình trạng bảo tồn": {
            "chunk_size": 50,
            "chunk_overlap": 10
        },
        "Giá trị nghiên cứu": {
            "chunk_size": 100,     # Field dài nên cho nhiều từ hơn
            "chunk_overlap": 20
        },
        "Sự liên quan với con người": {
            "chunk_size": 80,
            "chunk_overlap": 15
        },
        "Các quan sát thú vị từ các nhà nghiên cứu": {
            "chunk_size": 80,
            "chunk_overlap": 15
        }
    }
    
    # Re-ranking configurations
    USE_RERANKING = True
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    RERANK_TOP_K = 7  # Get more candidates for re-ranking
    FINAL_TOP_K = 4    # Final number of passages after re-ranking
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