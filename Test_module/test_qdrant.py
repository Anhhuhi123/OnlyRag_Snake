#!/usr/bin/env python3
"""
Demo script to test Qdrant integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qdrant_vector_store import QdrantVectorStore
from config.config import Config
import numpy as np

def main():
    """Test Qdrant vector store"""
    print("🚀 Testing Qdrant Vector Store Integration")
    print("=" * 60)
    
    # Initialize Qdrant store
    print("\n1️⃣ Initializing Qdrant...")
    qdrant_store = QdrantVectorStore()
    
    # Get initial stats
    print("\n2️⃣ Getting collection stats...")
    stats = qdrant_store.get_stats()
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Backend: {stats['backend']}")
    print(f"  Total vectors: {stats['total_embeddings']}")
    print(f"  Dimension: {stats['dimension']}")
    
    # Test adding embeddings
    print("\n3️⃣ Adding test embeddings...")
    test_texts = [
        "Rắn hổ mang chúa là loài rắn độc lớn nhất thế giới",
        "Rắn lục đuôi đỏ có màu xanh lá cây đặc trưng",
        "Trăn gấm là loài rắn không độc lớn nhất châu Á",
        "Rắn cạp nia có khả năng phun nọc độc",
        "Rắn ri cá sống chủ yếu trong và gần nước"
    ]
    
    # Generate random embeddings for testing (in production, use actual embeddings)
    test_embeddings = np.random.rand(len(test_texts), Config.VECTOR_DIMENSION).astype('float32')
    
    qdrant_store.add_embeddings(test_embeddings, test_texts)
    
    # Get stats after adding
    print("\n4️⃣ Stats after adding data:")
    stats = qdrant_store.get_stats()
    print(f"  Total vectors: {stats['total_embeddings']}")
    
    # Test search
    print("\n5️⃣ Testing search...")
    query_embedding = np.random.rand(Config.VECTOR_DIMENSION).astype('float32')
    similar_texts, scores = qdrant_store.search(query_embedding, k=3)
    
    print(f"  Found {len(similar_texts)} similar texts:")
    for i, (text, score) in enumerate(zip(similar_texts, scores), 1):
        print(f"    {i}. Score: {score:.4f} - {text[:80]}...")
    
    # Test save/load
    print("\n6️⃣ Testing save/load...")
    qdrant_store.save_index()
    
    # Create new instance and load
    print("\n7️⃣ Creating new instance and loading...")
    new_store = QdrantVectorStore()
    success = new_store.load_index()
    
    if success:
        print("  ✅ Successfully loaded from Qdrant!")
        stats = new_store.get_stats()
        print(f"  Loaded {stats['total_embeddings']} vectors")
    else:
        print("  ❌ Failed to load from Qdrant")
    
    print("\n✅ Qdrant integration test completed!")
    print("=" * 60)
    
    # Optional: Clean up (uncomment if you want to delete test data)
    # print("\n🗑️  Cleaning up...")
    # qdrant_store.delete_collection()

if __name__ == "__main__":
    main()
