import json
import os
from typing import List, Dict, Any, Union

class JSONDataLoader:
    """Load and process data from JSON files for RAG pipeline"""
    
    def __init__(self, json_file_path: str = None):
        """Initialize JSON data loader"""
        self.json_file_path = json_file_path or "data/documents.json"
    
    def load_json_data(self, file_path: str = None) -> Union[Dict, List]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        file_path = file_path or self.json_file_path
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded JSON data from: {file_path}")
            return data
            
        except FileNotFoundError:
            print(f"❌ JSON file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            raise
    
    def extract_text_from_json(self, json_data: Union[Dict, List], 
                              text_fields: List[str] = None) -> List[str]:
        """
        Extract text content from JSON data
        
        Args:
            json_data: JSON data (dict or list)
            text_fields: List of field names to extract text from
            
        Returns:
            List of extracted text documents
        """
        if text_fields is None:
            # Default fields to look for
            text_fields = ['content', 'text', 'description', 'body', 'article', 'story']
        
        documents = []
        
        if isinstance(json_data, dict):
            documents.extend(self._extract_from_dict(json_data, text_fields))
        elif isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict):
                    documents.extend(self._extract_from_dict(item, text_fields))
                elif isinstance(item, str):
                    documents.append(item)
        
        print(f"📄 Extracted {len(documents)} documents from JSON")
        return documents
    
    def _extract_from_dict(self, data: Dict, text_fields: List[str]) -> List[str]:
        """Extract text from dictionary recursively"""
        documents = []
        
        for key, value in data.items():
            if key in text_fields and isinstance(value, str):
                documents.append(value)
            elif isinstance(value, dict):
                documents.extend(self._extract_from_dict(value, text_fields))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        documents.extend(self._extract_from_dict(item, text_fields))
                    elif isinstance(item, str) and key in text_fields:
                        documents.append(item)
        
        return documents
    
    def create_sample_json(self, output_path: str = "data/documents.json"):
        """Create a sample JSON file for testing"""
        sample_data = {
            "documents": [
                {
                    "id": 1,
                    "title": "Artificial Intelligence Overview",
                    "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.",
                    "category": "technology",
                    "author": "Tech Writer",
                    "date": "2024-01-15"
                },
                {
                    "id": 2,
                    "title": "Machine Learning Fundamentals",
                    "content": "Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn and make predictions or decisions without being explicitly programmed. It uses statistical techniques to give computers the ability to learn from data.",
                    "category": "technology",
                    "author": "Data Scientist",
                    "date": "2024-01-20"
                },
                {
                    "id": 3,
                    "title": "Natural Language Processing",
                    "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves developing algorithms and models that can understand, interpret, and generate human language.",
                    "category": "AI",
                    "author": "NLP Engineer",
                    "date": "2024-02-01"
                }
            ],
            "metadata": {
                "source": "Knowledge Base",
                "version": "1.0",
                "created": "2024-01-15",
                "total_documents": 3
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Created sample JSON file: {output_path}")
        return output_path

def get_json_documents(json_file_path: str = None, 
                      text_fields: List[str] = None) -> List[str]:
    """
    Convenience function to load documents from JSON file
    
    Args:
        json_file_path: Path to JSON file
        text_fields: List of field names to extract text from
        
    Returns:
        List of document texts
    """
    loader = JSONDataLoader(json_file_path)
    
    try:
        json_data = loader.load_json_data()
        documents = loader.extract_text_from_json(json_data, text_fields)
        return documents
    except Exception as e:
        print(f"❌ Failed to load documents from JSON: {e}")
        return []
