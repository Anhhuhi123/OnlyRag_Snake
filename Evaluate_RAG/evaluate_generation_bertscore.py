"""
Đánh giá Generation Performance với BERTScore
Input: data/predictions.json (có question, ground_truth, answer)
Output: BERTScore metrics (Precision, Recall, F1)
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from bert_score import score
from tqdm import tqdm


def calculate_bertscore(predictions: List[str], 
                       references: List[str],
                       lang: str = "vi",
                       model_type: str = None,
                       verbose: bool = True) -> Dict:
    """
    Tính BERTScore cho batch predictions
    
    Args:
        predictions: List câu trả lời của model
        references: List câu trả lời đúng (ground truth)
        lang: Ngôn ngữ (vi cho tiếng Việt)
        model_type: Model BERT để dùng (None = auto select theo lang)
        verbose: In progress
    
    Returns:
        Dictionary chứa P, R, F1 scores
    """
    if verbose:
        print(f"[BERTScore] Calculating for {len(predictions)} predictions...")
        print(f"    Language: {lang}")
        if model_type:
            print(f"    Model: {model_type}")
    
    # Calculate BERTScore
    # P, R, F1 are tensors
    P, R, F1 = score(
        predictions, 
        references, 
        lang=lang,
        model_type=model_type,
        verbose=verbose,
        rescale_with_baseline=True  # Rescale scores để dễ interpret hơn
    )
    
    # Convert to numpy arrays
    P = P.numpy()
    R = R.numpy()
    F1 = F1.numpy()
    
    return {
        "precision": P,
        "recall": R,
        "f1": F1
    }


def evaluate_generation_bertscore(predictions_file: str = "data/predictions.json",
                                  lang: str = "vi",
                                  model_type: str = None) -> Dict:
    """
    Đánh giá generation performance với BERTScore
    
    Args:
        predictions_file: File predictions.json
        lang: Ngôn ngữ
        model_type: Model BERT cụ thể (optional)
    
    Returns:
        Dictionary chứa metrics và detailed results
    """
    print("="*70)
    print("GENERATION EVALUATION: BERTScore")
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
    
    # Extract answers and ground truths
    print(f"\n[2/4] Extracting answers and ground truths...")
    predicted_answers = []
    ground_truths = []
    questions = []
    
    for pred in predictions:
        questions.append(pred["question"])
        predicted_answers.append(pred["answer"])
        ground_truths.append(pred["ground_truth"])
    
    print(f"✅ Extracted {len(predicted_answers)} answer pairs")
    
    # Calculate BERTScore
    print(f"\n[3/4] Calculating BERTScore...")
    bert_scores = calculate_bertscore(
        predictions=predicted_answers,
        references=ground_truths,
        lang=lang,
        model_type=model_type,
        verbose=True
    )
    
    # Process results
    print(f"\n[4/4] Processing results...")
    
    results = {
        "metrics": {
            "BERTScore_Precision": float(np.mean(bert_scores["precision"])),
            "BERTScore_Recall": float(np.mean(bert_scores["recall"])),
            "BERTScore_F1": float(np.mean(bert_scores["f1"])),
            "BERTScore_Precision_std": float(np.std(bert_scores["precision"])),
            "BERTScore_Recall_std": float(np.std(bert_scores["recall"])),
            "BERTScore_F1_std": float(np.std(bert_scores["f1"])),
        },
        "per_question_results": [],
        "config": {
            "language": lang,
            "model_type": model_type or "auto",
            "total_questions": len(predictions),
            "rescale_with_baseline": True
        }
    }
    
    # Store per-question results
    for idx, (q, pred, ref, p, r, f1) in enumerate(zip(
        questions, 
        predicted_answers, 
        ground_truths,
        bert_scores["precision"],
        bert_scores["recall"],
        bert_scores["f1"]
    )):
        results["per_question_results"].append({
            "question": q,
            "predicted_answer": pred,
            "ground_truth": ref,
            "bertscore": {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1)
            }
        })
    
    # Print summary
    print("\n" + "="*70)
    print("AVERAGE BERTSCORE METRICS")
    print("="*70)
    print(f"\nBERTScore Precision: {results['metrics']['BERTScore_Precision']:.4f} ± {results['metrics']['BERTScore_Precision_std']:.4f}")
    print(f"BERTScore Recall:    {results['metrics']['BERTScore_Recall']:.4f} ± {results['metrics']['BERTScore_Recall_std']:.4f}")
    print(f"BERTScore F1:        {results['metrics']['BERTScore_F1']:.4f} ± {results['metrics']['BERTScore_F1_std']:.4f}")
    
    # Save results
    output_file = "data/Evaluation_documents/bertscore_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Detailed results saved to: {output_file}")
    
    return results


def print_detailed_analysis(results: Dict, 
                           top_n: int = 5, 
                           sort_by: str = "f1") -> None:
    """
    In ra phân tích chi tiết
    
    Args:
        results: Results dictionary
        top_n: Số câu hỏi hiển thị
        sort_by: Sắp xếp theo metric nào (f1, precision, recall)
    """
    print("\n" + "="*70)
    print(f"DETAILED ANALYSIS - Top {top_n} Best & Worst by BERTScore F1")
    print("="*70)
    
    per_question = results["per_question_results"]
    
    # Sort by F1
    sorted_results = sorted(per_question, 
                           key=lambda x: x["bertscore"][sort_by], 
                           reverse=True)
    
    # Top performers
    print(f"\n{'='*70}")
    print(f"TOP {top_n} BEST ANSWERS (Highest BERTScore F1)")
    print(f"{'='*70}")
    
    for idx, item in enumerate(sorted_results[:top_n], 1):
        print(f"\n{idx}. Question: {item['question'][:80]}...")
        print(f"   Ground Truth: {item['ground_truth'][:100]}...")
        print(f"   Predicted:    {item['predicted_answer'][:100]}...")
        print(f"   BERTScore - P: {item['bertscore']['precision']:.4f}, "
              f"R: {item['bertscore']['recall']:.4f}, "
              f"F1: {item['bertscore']['f1']:.4f}")
    
    # Worst performers
    print(f"\n{'='*70}")
    print(f"TOP {top_n} WORST ANSWERS (Lowest BERTScore F1)")
    print(f"{'='*70}")
    
    for idx, item in enumerate(sorted_results[-top_n:], 1):
        print(f"\n{idx}. Question: {item['question'][:80]}...")
        print(f"   Ground Truth: {item['ground_truth'][:100]}...")
        print(f"   Predicted:    {item['predicted_answer'][:100]}...")
        print(f"   BERTScore - P: {item['bertscore']['precision']:.4f}, "
              f"R: {item['bertscore']['recall']:.4f}, "
              f"F1: {item['bertscore']['f1']:.4f}")


def print_score_distribution(results: Dict) -> None:
    """
    In ra phân phối điểm số
    """
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION")
    print("="*70)
    
    f1_scores = [item["bertscore"]["f1"] for item in results["per_question_results"]]
    
    # Calculate quartiles
    q25 = np.percentile(f1_scores, 25)
    q50 = np.percentile(f1_scores, 50)
    q75 = np.percentile(f1_scores, 75)
    
    print(f"\nF1 Score Distribution:")
    print(f"  Min:  {min(f1_scores):.4f}")
    print(f"  Q25:  {q25:.4f}")
    print(f"  Q50 (Median): {q50:.4f}")
    print(f"  Q75:  {q75:.4f}")
    print(f"  Max:  {max(f1_scores):.4f}")
    print(f"  Mean: {np.mean(f1_scores):.4f}")
    print(f"  Std:  {np.std(f1_scores):.4f}")
    
    # Score ranges
    print(f"\nScore Range Analysis:")
    ranges = [
        (0.9, 1.0, "Excellent"),
        (0.8, 0.9, "Very Good"),
        (0.7, 0.8, "Good"),
        (0.6, 0.7, "Fair"),
        (0.0, 0.6, "Poor")
    ]
    
    for low, high, label in ranges:
        count = sum(1 for score in f1_scores if low <= score < high)
        pct = (count / len(f1_scores)) * 100
        print(f"  {label:12} [{low:.1f}-{high:.1f}): {count:3d} answers ({pct:5.1f}%)")


if __name__ == "__main__":
    # Configuration
    PREDICTIONS_FILE = "data/predictions_cleaned.json"
    LANGUAGE = "vi"  # Vietnamese
    MODEL_TYPE = None  # Auto-select best model for Vietnamese
    
    # You can specify a specific model:
    # MODEL_TYPE = "bert-base-multilingual-cased"
    # MODEL_TYPE = "xlm-roberta-base"
    
    print("BERTScore Evaluation Tool")
    print("=" * 70)
    print("This tool evaluates LLM-generated answers against ground truth")
    print("using BERTScore - a semantic similarity metric based on BERT embeddings.")
    print("=" * 70)
    
    # Run evaluation
    results = evaluate_generation_bertscore(
        predictions_file=PREDICTIONS_FILE,
        lang=LANGUAGE,
        model_type=MODEL_TYPE
    )
    
    if results:
        # Print detailed analysis
        print_detailed_analysis(results, top_n=3)
        
        # Print score distribution
        print_score_distribution(results)
        
        print("\n" + "="*70)
        print("INTERPRETATION GUIDE")
        print("="*70)
        print("""
BERTScore sử dụng BERT embeddings để đo độ tương đồng ngữ nghĩa:

Precision: Tỷ lệ nội dung trong câu trả lời predicted có trong ground truth
          - Cao: Ít thông tin sai/thừa
          - Thấp: Nhiều thông tin không liên quan

Recall:    Tỷ lệ nội dung trong ground truth được bao phủ bởi predicted
          - Cao: Bao phủ đầy đủ thông tin cần thiết
          - Thấp: Thiếu nhiều thông tin quan trọng

F1:        Điểm cân bằng giữa Precision và Recall
          - >0.9: Excellent - Câu trả lời rất chính xác và đầy đủ
          - 0.8-0.9: Very Good - Câu trả lời tốt, có thể thiếu một số chi tiết
          - 0.7-0.8: Good - Câu trả lời chấp nhận được
          - 0.6-0.7: Fair - Câu trả lời thiếu thông tin hoặc không chính xác
          - <0.6: Poor - Câu trả lời kém chất lượng

Ưu điểm của BERTScore:
- Đo độ tương đồng ngữ nghĩa, không chỉ từ khóa
- Hiểu được paraphrase (cách diễn đạt khác nhưng cùng ý)
- Tương quan cao với đánh giá của con người
- Robust với các biến thể câu văn
        """)
