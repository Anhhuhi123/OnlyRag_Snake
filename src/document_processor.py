import re
from typing import List, Dict, Optional
from config.config import Config

class DocumentProcessor:
    """Handles document processing and text chunking with metadata context"""
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Take last part of current chunk as overlap
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > self.chunk_size:
                        # Split long sentence into smaller parts
                        words = sentence.split()
                        temp_chunk = ""
                        
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def chunk_text_with_metadata_context(self, text: str, 
                                         snake_name: str = None, 
                                         metadata_key: str = None) -> List[str]:
        """
        Split text into overlapping chunks with metadata context prefix
        
        Args:
            text: Text to chunk
            snake_name: Tên rắn (e.g., "Protobothrops mucrosquamatus")
            metadata_key: Metadata key (e.g., "Độc tính", "Phân bố")
            
        Returns:
            List of text chunks with context prefix
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Create context prefix if both snake_name and metadata_key provided
        context_prefix = ""
        if snake_name and metadata_key:
            context_prefix = f"{snake_name} - {metadata_key}: "
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Calculate chunk size including prefix
            prefix_adjusted_size = self.chunk_size - len(context_prefix)
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > prefix_adjusted_size:
                if current_chunk:
                    # Add prefix to chunk before appending
                    final_chunk = context_prefix + current_chunk.strip()
                    chunks.append(final_chunk)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Take last part of current chunk as overlap
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > prefix_adjusted_size:
                        # Split long sentence into smaller parts
                        words = sentence.split()
                        temp_chunk = ""
                        
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= prefix_adjusted_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    final_chunk = context_prefix + temp_chunk.strip()
                                    chunks.append(final_chunk)
                                temp_chunk = word
                        
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk with prefix
        if current_chunk:
            final_chunk = context_prefix + current_chunk.strip()
            chunks.append(final_chunk)
        
        # Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def process_document_with_metadata(self, 
                                      documents: List[Dict], 
                                      name_field: str = "name_vn",
                                      metadata_fields: List[str] = None) -> List[str]:
        """
        Process documents with metadata context
        
        Args:
            documents: List of document dicts with metadata
            name_field: Field name for snake name (default: "name_vn")
            metadata_fields: List of metadata field names to process
                           If None, process all fields except id and name fields
            
        Returns:
            List of processed text chunks with context prefix
        """
        if metadata_fields is None:
            # Default metadata fields to process
            metadata_fields = [
                "Tên khoa học và tên phổ thông",
                "Phân loại học",
                "Đặc điểm hình thái",
                "Độc tính",
                "Tập tính săn mồi",
                "Hành vi và sinh thái",
                "Phân bố địa lý và môi trường sống",
                "Sinh sản",
                "Tình trạng bảo tồn",
                "Giá trị nghiên cứu",
                "Sự liên quan với con người",
                "Các quan sát thú vị từ các nhà nghiên cứu"
            ]
        
        all_chunks = []
        
        for doc in documents:
            # Get snake name
            snake_name = doc.get(name_field) or doc.get("name_en") or "Unknown"
            
            print(f"\n📄 Processing: {snake_name}")
            
            # Process each metadata field
            for metadata_key in metadata_fields:
                if metadata_key in doc and doc[metadata_key]:
                    text = doc[metadata_key]
                    
                    # Chunk with context prefix
                    chunks = self.chunk_text_with_metadata_context(
                        text=text,
                        snake_name=snake_name,
                        metadata_key=metadata_key
                    )
                    
                    all_chunks.extend(chunks)
                    print(f"  ✓ {metadata_key}: {len(chunks)} chunks")
        
        print(f"\n✅ Total processed: {len(all_chunks)} chunks with context")
        
        # Print statistics
        if all_chunks:
            avg_length = sum(len(chunk) for chunk in all_chunks) / len(all_chunks)
            max_length = max(len(chunk) for chunk in all_chunks)
            min_length = min(len(chunk) for chunk in all_chunks)
            
            print(f"\nChunk statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Max length: {max_length} characters")
            print(f"  Min length: {min_length} characters")
        
        return all_chunks
    
    def process_document(self, text: str) -> List[str]:
        """
        Process a document by cleaning and chunking (backward compatible method)
        
        Args:
            text: Raw document text
            
        Returns:
            List of processed text chunks
        """
        chunks = self.chunk_text(text)
        
        if chunks:
            print(f"Document processed into {len(chunks)} chunks")
            
            # Print chunk statistics
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            max_length = max(len(chunk) for chunk in chunks)
            min_length = min(len(chunk) for chunk in chunks)
            
            print(f"Chunk statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Max length: {max_length} characters")
            print(f"  Min length: {min_length} characters")
        
        return chunks
        """
        Process a document by cleaning and chunking
        
        Args:
            text: Raw document text
            
        Returns:
            List of processed text chunks
        """
        chunks = self.chunk_text(text)
        print(f"Document processed into {len(chunks)} chunks")
        
        # Print chunk statistics
        if chunks:
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            max_length = max(len(chunk) for chunk in chunks)
            min_length = min(len(chunk) for chunk in chunks)
            
            print(f"Chunk statistics:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Max length: {max_length} characters")
            print(f"  Min length: {min_length} characters")
        
        return chunks