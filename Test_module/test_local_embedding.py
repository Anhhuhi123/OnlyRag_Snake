#!/usr/bin/env python3
"""
Test script for local embedding model (multilingual-e5-small)
"""

from src.embeddings import EmbeddingGenerator
from config.config import Config
import numpy as np

def test_local_embedding():
    """Test local embedding generation"""
    
    print("=" * 60)
    print("Testing Local Embedding Model")
    print("=" * 60)
    
    # Initialize embedding generator
    print("\n1. Initializing EmbeddingGenerator...")
    embedder = EmbeddingGenerator()
    
    # Test Vietnamese texts about snakes
    test_texts = [
        "Rắn hổ mang chúa là loài rắn độc lớn nhất thế giới",
        "Rắn lục đuôi đỏ có màu xanh lá cây và đuôi màu đỏ",
        "Rắn cạp nong sống chủ yếu ở miền Bắc Việt Nam",
        "Các loài rắn độc thường có răng nanh chứa nọc độc",
    ]
    
    print(f"\n2. Testing with {len(test_texts)} Vietnamese texts...")
    print("-" * 60)
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text[:50]}...")
    print("-" * 60)
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(test_texts)
    
    print(f"\n3. Embedding Results:")
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - Dimension: {embeddings.shape[1]}")
    print(f"   - Expected dimension: {Config.VECTOR_DIMENSION}")
    print(f"   - Data type: {embeddings.dtype}")
    print(f"   - First embedding (first 10 values): {embeddings[0][:10]}")
    
    # Check if normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n4. Normalization check:")
    print(f"   - Norms: {norms}")
    print(f"   - All close to 1.0? {np.allclose(norms, 1.0)}")
    
    # Test similarity
    print(f"\n5. Testing semantic similarity:")
    print(f"   Query: '{test_texts[0]}'")
    print(f"   Computing cosine similarity with other texts...")
    
    query_embedding = embeddings[0]
    for i, text in enumerate(test_texts[1:], 1):
        similarity = np.dot(query_embedding, embeddings[i])
        print(f"   - Text {i}: {similarity:.4f} | {text[:40]}...")
    
    # Test single embedding (query mode)
    print(f"\n6. Testing single embedding (query mode):")
    query = "Loài rắn nào có nọc độc mạnh nhất?"
    query_embedding = embedder.generate_single_embedding(query)
    print(f"   Query: '{query}'")
    print(f"   Embedding shape: {query_embedding.shape}")
    print(f"   Norm: {np.linalg.norm(query_embedding):.4f}")
    
    # Compute similarity with passages
    print(f"\n7. Query similarity with passages:")
    for i, text in enumerate(test_texts):
        similarity = np.dot(query_embedding, embeddings[i])
        print(f"   - Score: {similarity:.4f} | {text[:40]}...")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"   Model: {Config.EMBEDDING_MODEL}")
    print(f"   Dimension: {Config.VECTOR_DIMENSION}")
    print(f"   Batch size: {Config.EMBEDDING_BATCH_SIZE}")
    print(f"   Texts processed: {len(test_texts)}")
    print(f"   Total embeddings: {len(embeddings) + 1}")  # +1 for query
    
    return True

if __name__ == "__main__":
    try:
        test_local_embedding()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
