"""
Demo: Metadata-level Chunking với Context Prefix
Thêm "ngữ cảnh chủ thể" vào từng chunk
"""

import json
import sys
sys.path.append('..')

from src.document_processor import DocumentProcessor
from config.config import Config

def demo_metadata_chunking():
    """Demo chunking với metadata context"""
    
    print("="*70)
    print("DEMO: METADATA-LEVEL CHUNKING WITH CONTEXT PREFIX")
    print("="*70)
    
    # Load data
    print("\n[1/3] Loading data...")
    with open("data/document_RAG.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = data['documents']
    print(f"✅ Loaded {len(documents)} snake documents")
    
    # Initialize processor với chunk size nhỏ để demo
    print("\n[2/3] Processing with metadata context...")
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    
    # Process chỉ 2 loài đầu tiên để demo
    sample_docs = documents[:2]
    
    # Process với metadata fields cụ thể
    chunks = processor.process_document_with_metadata(
        documents=sample_docs,
        name_field="name_vn",
        metadata_fields=["Độc tính", "Phân bố địa lý và môi trường sống", "Tập tính săn mồi"]
    )
    
    # Display results
    print("\n[3/3] Sample chunks with context prefix:")
    print("="*70)
    
    for i, chunk in enumerate(chunks[:10], 1):
        print(f"\nChunk {i}:")
        print(f"Length: {len(chunk)} chars")
        print(f"Content: {chunk[:200]}...")
        print("-"*70)
    
    print(f"\n✅ Generated {len(chunks)} chunks total")
    
    # Save to file
    output_file = "Test_module/demo_metadata_chunks_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(chunks),
            "sample_chunks": chunks[:20],
            "config": {
                "chunk_size": 300,
                "chunk_overlap": 50,
                "documents_processed": len(sample_docs)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Saved sample to: {output_file}")


def compare_with_without_context():
    """So sánh chunking có và không có context prefix"""
    
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT CONTEXT PREFIX")
    print("="*70)
    
    # Load sample text
    with open("data/document_RAG.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doc = data['documents'][0]  # First snake
    snake_name = doc["name_vn"]
    text = doc["Độc tính"]
    
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    
    # Without context
    print("\n❌ WITHOUT CONTEXT PREFIX:")
    print("-"*70)
    chunks_no_context = processor.chunk_text(text)
    for i, chunk in enumerate(chunks_no_context[:3], 1):
        print(f"\nChunk {i}: {chunk[:150]}...")
    
    # With context
    print("\n\n✅ WITH CONTEXT PREFIX:")
    print("-"*70)
    chunks_with_context = processor.chunk_text_with_metadata_context(
        text=text,
        snake_name=snake_name,
        metadata_key="Độc tính"
    )
    for i, chunk in enumerate(chunks_with_context[:3], 1):
        print(f"\nChunk {i}: {chunk[:150]}...")
    
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY:")
    print(f"Without context: {len(chunks_no_context)} chunks")
    print(f"With context:    {len(chunks_with_context)} chunks")
    print("="*70)


if __name__ == "__main__":
    # Demo 1: Metadata chunking
    demo_metadata_chunking()
    
    # Demo 2: Comparison
    compare_with_without_context()
    
    print("\n" + "="*70)
    print("✨ KEY BENEFITS:")
    print("="*70)
    print("""
1. ✅ Mỗi chunk biết rõ ngữ cảnh: "Rắn gì" + "Metadata gì"
2. ✅ Retrieval chính xác hơn: Search "Protobothrops độc tính" → match chunks có prefix
3. ✅ LLM hiểu ngữ cảnh: Không bị nhầm lẫn giữa các loài rắn
4. ✅ Dễ debug: Nhìn chunk là biết ngay thuộc loài nào, metadata nào

Example output:
- "Protobothrops mucrosquamatus - Độc tính: Protobothrops mucrosquamatus là loài rắn độc mạnh..."
- "Protobothrops mucrosquamatus - Phân bố: Phân bố tại việt nam, thái lan..."
- "Protobothrops mucrosquamatus - Tập tính săn mồi: Săn mồi ban đêm..."
    """)
