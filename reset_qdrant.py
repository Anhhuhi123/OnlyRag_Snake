#!/usr/bin/env python3
"""
Script to reset Qdrant collection with new dimension
"""

from qdrant_client import QdrantClient
from config.config import Config

def reset_qdrant_collection():
    """Delete old collection and create new one with correct dimension"""
    
    print("=" * 60)
    print("Resetting Qdrant Collection")
    print("=" * 60)
    
    # Connect to Qdrant
    print(f"\n1. Connecting to Qdrant...")
    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
    )
    print("   ✓ Connected successfully")
    
    # Check existing collections
    print(f"\n2. Checking existing collections...")
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"   Found {len(collections)} collection(s): {collection_names}")
    
    # Delete old collection if exists
    if Config.QDRANT_COLLECTION_NAME in collection_names:
        print(f"\n3. Deleting old collection '{Config.QDRANT_COLLECTION_NAME}'...")
        
        # Get collection info first
        collection_info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
        old_dimension = collection_info.config.params.vectors.size
        vector_count = collection_info.points_count
        
        print(f"   Old collection info:")
        print(f"   - Dimension: {old_dimension}")
        print(f"   - Vector count: {vector_count}")
        
        # Delete
        client.delete_collection(Config.QDRANT_COLLECTION_NAME)
        print(f"   ✓ Collection deleted successfully")
    else:
        print(f"\n3. No existing collection found")
    
    # Create new collection
    from qdrant_client.models import Distance, VectorParams
    
    print(f"\n4. Creating new collection '{Config.QDRANT_COLLECTION_NAME}'...")
    print(f"   - New dimension: {Config.VECTOR_DIMENSION}")
    print(f"   - Distance metric: Cosine")
    
    client.create_collection(
        collection_name=Config.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=Config.VECTOR_DIMENSION,
            distance=Distance.COSINE
        )
    )
    print(f"   ✓ Collection created successfully")
    
    # Verify new collection
    print(f"\n5. Verifying new collection...")
    collection_info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
    print(f"   - Name: {Config.QDRANT_COLLECTION_NAME}")
    print(f"   - Dimension: {collection_info.config.params.vectors.size}")
    print(f"   - Distance: {collection_info.config.params.vectors.distance}")
    print(f"   - Vector count: {collection_info.points_count}")
    
    print("\n" + "=" * 60)
    print("✓ Collection reset completed!")
    print("=" * 60)
    
    print(f"\n📌 Next steps:")
    print(f"   1. Re-ingest your documents with new embeddings:")
    print(f"      python main.py --ingest data/document_RAG.json")
    print(f"")
    print(f"   2. Then query as normal:")
    print(f"      python main.py --query 'Your question here'")
    
    return True

if __name__ == "__main__":
    try:
        reset_qdrant_collection()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
