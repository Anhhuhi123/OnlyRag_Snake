"""
Đánh giá Retrieval Performance với Recall@k và Precision@k
Input: data/predictions.json (có question, ground_truth, contexts)
Output: Retrieval metrics và detailed report
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer


from collections import defaultdict
from sentence_transformers import SentenceTransformer


# Load embedding model globally
print("Loading embedding model for similarity calculation...")
EMBEDDING_MODEL = SentenceTransformer('keepitreal/vietnamese-sbert')
print("✅ Model loaded successfully")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Tính Cosine Similarity giữa 2 vectors
    Returns: score từ -1 đến 1 (thường là 0 đến 1 với embeddings)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def calculate_text_similarity(ground_truth: str, context: str) -> float:
    """
    Tính độ tương đồng ngữ nghĩa giữa ground_truth và context 
    bằng Cosine Similarity trên embeddings
    
    Returns: score từ 0 đến 1
    """
    # Encode texts thành embeddings
    embeddings = EMBEDDING_MODEL.encode([ground_truth, context])
    gt_embedding = embeddings[0]
    ctx_embedding = embeddings[1]
    
    # Tính cosine similarity
    similarity = cosine_similarity(gt_embedding, ctx_embedding)
    
    # Normalize về [0, 1] nếu cần (cosine similarity có thể âm nhưng với embeddings thường dương)
    similarity = max(0.0, min(1.0, similarity))
    
    return similarity

def extract_keywords(text: str, min_length: int = 3) -> set:
    """
    Trích xuất keywords từ text (words dài hơn min_length)
    """
    normalized = normalize_text(text)
    words = normalized.split()
    # Lọc stop words tiếng Việt cơ bản
    stop_words = {'là', 'của', 'và', 'có', 'trong', 'được', 'với', 'cho', 'từ', 'này', 
                  'các', 'một', 'để', 'người', 'những', 'khi', 'như', 'đã', 'bởi', 'về',
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be'}
    
    keywords = {word for word in words if len(word) >= min_length and word not in stop_words}
    return keywords


def calculate_text_overlap(ground_truth: str, context: str) -> float:
    """
    Tính độ overlap giữa ground_truth và context dựa trên keywords
    Returns: score từ 0 đến 1
    """
    gt_keywords = extract_keywords(ground_truth)
    ctx_keywords = extract_keywords(context)
    
    if not gt_keywords:
        return 0.0
    
    # Tính Jaccard similarity
    intersection = gt_keywords & ctx_keywords
    union = gt_keywords | ctx_keywords
    
    if not union:
        return 0.0
    
    jaccard_score = len(intersection) / len(union)
    
    # Tính coverage (bao nhiêu % keywords của ground_truth có trong context)
    coverage = len(intersection) / len(gt_keywords)
    
    # Combined score (trọng số 60% coverage, 40% jaccard)
    combined_score = 0.6 * coverage + 0.4 * jaccard_score
    
    return combined_score


def is_context_relevant(ground_truth: str, context: str, threshold: float = 0.5) -> bool:
    """
    Kiểm tra context có liên quan đến ground_truth không
    Sử dụng Cosine Similarity thay vì overlap
    
    Args:
        ground_truth: Câu trả lời đúng
        context: Context cần kiểm tra
        threshold: Ngưỡng similarity để coi là relevant (0.5 = 50% similarity)
    
    Returns:
        True nếu similarity >= threshold
    """
    similarity_score = calculate_text_similarity(ground_truth, context)
    return similarity_score >= threshold


def calculate_recall_at_k(ground_truth: str, contexts: List[str], k: int = 5, 
                          threshold: float = 0.5) -> float:
    """
    Recall@k = Số relevant contexts được retrieve / Tổng số relevant contexts có thể retrieve
    
    Với RAG, ta giả định ground_truth chứa thông tin cần thiết.
    Recall@k đo lường: trong top-k contexts, có bao nhiêu % thông tin relevant được tìm thấy
    
    Args:
        ground_truth: Câu trả lời đúng (chứa thông tin cần thiết)
        contexts: List contexts được retrieve (top-k)
        k: Số lượng contexts xét (mặc định 5)
        threshold: Ngưỡng cosine similarity để coi là relevant (0.5 = 50%)
    
    Returns:
        recall_score: 0 đến 1
    """
    # Lấy top-k contexts
    top_k_contexts = contexts[:k]
    
    # Đếm số contexts relevant
    relevant_count = sum(1 for ctx in top_k_contexts 
                        if is_context_relevant(ground_truth, ctx, threshold))
    
    # Recall = relevant retrieved / total relevant
    # Vì ta không biết tổng số relevant documents trong corpus,
    # ta giả định k là số maximum relevant docs có thể có
    # Hoặc ta có thể coi recall = relevant_count / k (% relevant trong top-k)
    
    recall = relevant_count / k if k > 0 else 0.0
    return recall


def calculate_precision_at_k(ground_truth: str, contexts: List[str], k: int = 5,
                             threshold: float = 0.5) -> float:
    """
    Precision@k = Số relevant contexts trong top-k / k
    
    Đo lường: trong k contexts được retrieve, có bao nhiêu % là relevant
    
    Args:
        ground_truth: Câu trả lời đúng
        contexts: List contexts được retrieve
        k: Số lượng contexts xét
        threshold: Ngưỡng cosine similarity để coi là relevant (0.5 = 50%)
    
    Returns:
        precision_score: 0 đến 1
    """
    # Lấy top-k contexts
    top_k_contexts = contexts[:k]
    
    # Đếm số contexts relevant
    relevant_count = sum(1 for ctx in top_k_contexts 
                        if is_context_relevant(ground_truth, ctx, threshold))
    
    # Precision = relevant retrieved / total retrieved
    precision = relevant_count / k if k > 0 else 0.0
    return precision


def evaluate_retrieval(predictions_file: str = "data/predictions.json",
                      k_values: List[int] = [1, 3, 5],
                      threshold: float = 0.5) -> Dict:
    """
    Đánh giá retrieval performance cho tất cả questions
    
    Args:
        predictions_file: File predictions.json
        k_values: List các giá trị k cần đánh giá
        threshold: Ngưỡng cosine similarity để coi là relevant (0.5 = 50%)
    
    Returns:
        Dictionary chứa metrics và detailed results
    """
    print("="*70)
    print("RETRIEVAL EVALUATION: Recall@k & Precision@k")
    print("="*70)
    
    # Load predictions
    print(f"\n[1/4] Loading predictions from {predictions_file}...")
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        print(f"✅ Loaded {len(predictions)} predictions")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return {}
    
    # Calculate metrics for each k
    print(f"\n[2/4] Calculating Recall@k and Precision@k...")
    print(f"    Relevance threshold: {threshold:.2f} (cosine similarity)")
    print(f"    Using embedding model: keepitreal/vietnamese-sbert")
    
    results = {
        "metrics": {},
        "per_question_results": [],
        "config": {
            "k_values": k_values,
            "threshold": threshold,
            "total_questions": len(predictions)
        }
    }
    
    # Initialize metric storage
    metrics_by_k = {k: {"recall": [], "precision": []} for k in k_values}
    
    # Evaluate each question
    for idx, pred in enumerate(predictions, 1):
        question = pred["question"]
        ground_truth = pred["ground_truth"]
        contexts = pred["contexts"]
        
        question_result = {
            "question": question,
            "ground_truth": ground_truth,
            "num_contexts": len(contexts),
            "metrics": {}
        }
        
        # Calculate for each k
        for k in k_values:
            if k > len(contexts):
                print(f"⚠️  Warning: k={k} > num_contexts={len(contexts)} for question {idx}")
                k_actual = len(contexts)
            else:
                k_actual = k
            
            recall = calculate_recall_at_k(ground_truth, contexts, k_actual, threshold)
            precision = calculate_precision_at_k(ground_truth, contexts, k_actual, threshold)
            
            metrics_by_k[k]["recall"].append(recall)
            metrics_by_k[k]["precision"].append(precision)
            
            question_result["metrics"][f"k={k}"] = {
                "recall": round(recall, 4),
                "precision": round(precision, 4)
            }
        
        results["per_question_results"].append(question_result)
    
    # Calculate average metrics
    print(f"\n[3/4] Computing average metrics...")
    for k in k_values:
        recall_scores = metrics_by_k[k]["recall"]
        precision_scores = metrics_by_k[k]["precision"]
        
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        
        results["metrics"][f"Recall@{k}"] = round(avg_recall, 4)
        results["metrics"][f"Precision@{k}"] = round(avg_precision, 4)
    
    # Print results
    print(f"\n[4/4] Results:")
    print("\n" + "="*70)
    print("AVERAGE METRICS")
    print("="*70)
    
    for k in k_values:
        recall = results["metrics"][f"Recall@{k}"]
        precision = results["metrics"][f"Precision@{k}"]
        print(f"\nk = {k}:")
        print(f"  Recall@{k}:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  Precision@{k}: {precision:.4f} ({precision*100:.2f}%)")
    
    # Save detailed results
    output_file = "data/Evaluation_documents/retrieval_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    return results


def print_detailed_analysis(results: Dict, top_n: int = 3):
    """
    In ra phân tích chi tiết cho một số câu hỏi
    """
    print("\n" + "="*70)
    print(f"DETAILED ANALYSIS - Top {top_n} Questions")
    print("="*70)
    
    per_question = results["per_question_results"][:top_n]
    
    for idx, item in enumerate(per_question, 1):
        print(f"\n{'─'*70}")
        print(f"Question {idx}: {item['question'][:100]}...")
        print(f"Ground Truth: {item['ground_truth'][:150]}...")
        print(f"Number of contexts: {item['num_contexts']}")
        print(f"\nMetrics:")
        
        for k_metrics in item["metrics"].items():
            k_label, scores = k_metrics
            print(f"  {k_label}:")
            print(f"    Recall:    {scores['recall']:.4f}")
            print(f"    Precision: {scores['precision']:.4f}")


if __name__ == "__main__":
    # Configuration
    PREDICTIONS_FILE = "data/predictions.json"
    K_VALUES = [1, 2]  # Đánh giá với k = 1, 2
    THRESHOLD = 0.55  # Ngưỡng cosine similarity 50% để coi là relevant
    
    # Run evaluation
    results = evaluate_retrieval(
        predictions_file=PREDICTIONS_FILE,
        k_values=K_VALUES,
        threshold=THRESHOLD
    )
    
    # Print detailed analysis for top 3 questions
    if results:
        print_detailed_analysis(results, top_n=3)
        
        print("\n" + "="*70)
        print("INTERPRETATION GUIDE")
        print("="*70)
        print("""
Recall@k:  Tỷ lệ thông tin relevant được tìm thấy trong top-k contexts
          - Cao (>0.8): Hệ thống retrieve được hầu hết thông tin cần thiết
          - Trung bình (0.5-0.8): Retrieve được một phần thông tin
          - Thấp (<0.5): Thiếu nhiều thông tin quan trọng

Precision@k: Tỷ lệ contexts relevant trong top-k contexts retrieved
          - Cao (>0.8): Hầu hết contexts đều chứa thông tin hữu ích
          - Trung bình (0.5-0.8): Một số contexts không liên quan
          - Thấp (<0.5): Nhiều contexts không có giá trị

Threshold: {:.2f} = Context cần có ít nhất {:.0f}% cosine similarity với ground_truth
          để được coi là relevant (sử dụng embeddings để đo độ tương đồng ngữ nghĩa)
        """.format(THRESHOLD, THRESHOLD*100))
