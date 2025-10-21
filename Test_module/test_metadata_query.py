"""
Test query với metadata context để xem chunks có prefix không
"""

from src.rag_pipeline import RAGPipeline

def test_metadata_query():
    """Test query and display retrieved contexts"""
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Load existing index
    if not rag.load_existing_index():
        print("❌ No index found!")
        return
    
    # Query
    question = "Rắn lục cườm có độc không?"
    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}\n")
    
    result = rag.query(question)
    
    # Display contexts with metadata prefix
    print(f"\n{'='*70}")
    print(f"RETRIEVED CONTEXTS (with metadata prefix)")
    print(f"{'='*70}\n")
    
    for i, (context, score) in enumerate(zip(result['context'], result['similarity_scores']), 1):
        print(f"Context {i} (Score: {score:.4f}):")
        print(f"{'-'*70}")
        print(context[:500] + "..." if len(context) > 500 else context)
        print(f"\n")
    
    # Display response
    print(f"{'='*70}")
    print(f"LLM RESPONSE:")
    print(f"{'='*70}")
    print(result['response'])
    print(f"\n{'='*70}\n")
    
    # Check if contexts have metadata prefix
    has_prefix = any("- Độc tính:" in ctx or "- Phân bố" in ctx or "- Tập tính" in ctx 
                     for ctx in result['context'])
    
    if has_prefix:
        print("✅ SUCCESS: Contexts contain metadata prefix!")
        print("   Metadata-level chunking is working correctly!")
    else:
        print("⚠️  WARNING: Contexts don't have metadata prefix")
        print("   You may be using old data without metadata context")


if __name__ == "__main__":
    test_metadata_query()
