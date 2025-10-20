from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import logging

class CrossEncoderReranker:
    """Cross-encoder based re-ranking for RAG pipeline"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize cross-encoder re-ranker
        
        Args:
            model_name: Hugging Face model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            print(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("Cross-encoder model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder model: {e}")
            raise e
    
    def rerank(self, query: str, passages: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Re-rank passages using cross-encoder
        
        Args:
            query: Search query
            passages: List of passage texts to re-rank
            top_k: Number of top passages to return (if None, return all)
            
        Returns:
            List of tuples (passage, score) sorted by relevance score
        """
        if not passages:
            return []
        
        if self.model is None:
            raise RuntimeError("Cross-encoder model not loaded")
        
        print(f"Re-ranking {len(passages)} passages...")
        
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Combine passages with scores
        passage_scores = list(zip(passages, scores))
        
        # Sort by score (descending)
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            passage_scores = passage_scores[:top_k]
        
        print(f"Re-ranking completed. Top score: {passage_scores[0][1]:.4f}")
        
        return passage_scores
    
    def rerank_with_original_scores(
        self, 
        query: str, 
        passages_with_scores: List[Tuple[str, float]], 
        alpha: float = 0.7,
        top_k: int = None
    ) -> List[Tuple[str, float, float, float]]:
        """
        Re-rank passages combining original retrieval scores with cross-encoder scores
        
        Args:
            query: Search query
            passages_with_scores: List of tuples (passage, original_score)
            alpha: Weight for cross-encoder score (1-alpha for original score)
            top_k: Number of top passages to return
            
        Returns:
            List of tuples (passage, combined_score, cross_encoder_score, original_score)
        """
        if not passages_with_scores:
            return []
        
        passages = [item[0] for item in passages_with_scores]
        original_scores = [item[1] for item in passages_with_scores]
        
        # Get cross-encoder scores
        cross_encoder_scores = self.model.predict([[query, passage] for passage in passages])
        
        # Normalize scores to [0, 1] range
        if len(cross_encoder_scores) > 1:
            ce_min, ce_max = min(cross_encoder_scores), max(cross_encoder_scores)
            if ce_max != ce_min:
                cross_encoder_scores = [(score - ce_min) / (ce_max - ce_min) for score in cross_encoder_scores]
        
        orig_min, orig_max = min(original_scores), max(original_scores)
        if orig_max != orig_min:
            normalized_orig_scores = [(score - orig_min) / (orig_max - orig_min) for score in original_scores]
        else:
            normalized_orig_scores = original_scores
        
        # Combine scores
        combined_results = []
        for i, passage in enumerate(passages):
            combined_score = alpha * cross_encoder_scores[i] + (1 - alpha) * normalized_orig_scores[i]
            combined_results.append((
                passage, 
                combined_score, 
                cross_encoder_scores[i], 
                original_scores[i]
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            combined_results = combined_results[:top_k]
        
        print(f"Combined re-ranking completed. Top combined score: {combined_results[0][1]:.4f}")
        
        return combined_results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "model_type": "cross-encoder"
        }