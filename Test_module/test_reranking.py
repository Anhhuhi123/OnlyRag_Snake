#!/usr/bin/env python3
"""
Demo script to test RAG pipeline with Cross-encoder re-ranking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from config.config import Config
import json

def main():
    """Test RAG pipeline with re-ranking"""
    print("🚀 Testing RAG Pipeline with Cross-encoder Re-ranking")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Display pipeline stats
    stats = pipeline.get_pipeline_stats()
    print("\n📊 Pipeline Configuration:")
    print(f"  - Re-ranking enabled: {stats['reranking']['reranking_enabled']}")
    if stats['reranking']['reranking_enabled']:
        print(f"  - Cross-encoder model: {stats['reranking']['cross_encoder_model']}")
        print(f"  - Retrieval candidates: {stats['reranking']['rerank_top_k']}")
        print(f"  - Final top-k: {stats['reranking']['final_top_k']}")
        print(f"  - Alpha (CE weight): {stats['reranking']['rerank_alpha']}")
        print(f"  - Reranker loaded: {stats['reranking']['reranker_loaded']}")
    
    # Sample documents for testing
    sample_documents = [
        """
        Rắn hổ mang chúa (Ophiophagus hannah) là loài rắn độc lớn nhất thế giới. 
        Chúng có thể dài tới 5.5 mét và sống chủ yếu ở các khu rừng nhiệt đới Đông Nam Á. 
        Nọc độc của chúng rất mạnh và có thể gây tử vong cho con người trong vài giờ.
        Rắn hổ mang chúa chủ yếu ăn các loài rắn khác, kể cả rắn độc.
        """,
        """
        Rắn lục đuôi đỏ (Trimeresurus albolabris) là loài rắn độc phổ biến ở Việt Nam.
        Chúng có màu xanh lá cây đặc trưng và sống trên cây.
        Kích thước trung bình từ 60-90cm. Nọc độc gây hoại tử mô và đau đớn.
        Thức ăn chủ yếu là chim nhỏ, thằn lằn và ếch cây.
        """,
        """
        Rắn cạp nia (Naja kaouthia) hay còn gọi là rắn hổ mang Thái Lan.
        Đây là loài rắn độc nguy hiểm với khả năng phun nọc độc.
        Chiều dài trung bình 1-1.5 mét, màu nâu đen với hoa văn đặc trưng.
        Khi bị đe dọa, chúng sẽ dựng cổ và tạo tư thế đe dọa.
        """,
        """
        Trăn gấm (Python reticulatus) là loài rắn không độc lớn nhất châu Á.
        Chúng có thể dài tới 10 mét và nặng hơn 100kg.
        Săn mồi bằng cách siết chặt và có thể nuốt chửng động vật lớn.
        Sống ở môi trường ẩm ướt, gần nguồn nước và rừng nhiệt đới.
        """,
        """
        Rắn ri cá (Xenochrophis piscator) là loài rắn nước không độc.
        Chúng sống chủ yếu trong và gần nước, ăn cá và ếch.
        Màu sắc thường nâu xám với các đốm tối.
        Kích thước trung bình 80-120cm, hiền lành với con người.
        """
    ]
    
    # Check if we need to ingest documents
    try:
        if not pipeline.load_existing_index():
            print("\n📚 Ingesting sample documents...")
            result = pipeline.ingest_documents(sample_documents)
            print(f"   ✅ Indexed {result['total_chunks']} chunks from {result['total_documents']} documents")
        else:
            print("\n📚 Using existing index")
    except Exception as e:
        print(f"\n📚 No existing index found. Ingesting sample documents...")
        result = pipeline.ingest_documents(sample_documents)
        print(f"   ✅ Indexed {result['total_chunks']} chunks from {result['total_documents']} documents")
    
    # Test queries
    test_queries = [
        "Loài rắn nào lớn nhất thế giới?",
        "Rắn nào có màu xanh lá cây?",
        "Rắn nào không độc nhưng rất to?",
        "Rắn nào có thể phun nọc độc?",
        "Rắn nào sống trong nước?"
    ]
    
    print("\n🔍 Testing queries with re-ranking:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n❓ Query {i}: {query}")
        print("-" * 40)
        
        # Process query
        result = pipeline.query(query)
        
        # Display results
        print(f"📝 Response: {result['response'][:200]}...")
        
        if result.get('rerank_info', {}).get('reranking_used', False):
            rerank_info = result['rerank_info']
            print(f"\n🔄 Re-ranking Details:")
            print(f"   - Original retrieval: {rerank_info['original_retrieval_count']} passages")
            print(f"   - Final after re-rank: {rerank_info['final_count_after_rerank']} passages")
            
            print(f"\n📊 Score Comparison (Top 3):")
            for j in range(min(3, len(rerank_info['combined_scores']))):
                print(f"   {j+1}. Combined: {rerank_info['combined_scores'][j]:.4f} | "
                      f"Cross-encoder: {rerank_info['cross_encoder_scores'][j]:.4f} | "
                      f"Original: {rerank_info['original_scores'][j]:.4f}")
        else:
            print("   ⚠️  Re-ranking not used")
        
        print(f"\n📋 Context chunks used: {result['num_context_chunks']}")
        for j, context in enumerate(result['context'][:2]):  # Show first 2 contexts
            print(f"   {j+1}. {context[:100]}...")
    
    print("\n✅ Re-ranking test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()