"""
Filter và xóa các câu trả lời có BERTScore thấp nhất
Input: 
    - data/Evaluation_documents/bertscore_evaluation_results.json (để phân tích)
    - data/predictions.json (để xóa câu trả lời)
Output: 
    - Phân tích Score Range
    - Cho phép chọn xóa top-k câu trả lời kém nhất
    - predictions_cleaned.json (file mới sau khi xóa)
"""

import json
import os
from typing import List, Dict, Tuple
from collections import Counter


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def analyze_score_distribution(results: List[Dict]) -> Dict:
    """
    Phân tích phân phối điểm F1 theo range
    
    Returns:
        Dictionary với thống kê từng range
    """
    ranges = {
        "Excellent [0.9-1.0)": (0.9, 1.0),
        "Very Good [0.8-0.9)": (0.8, 0.9),
        "Good [0.7-0.8)": (0.7, 0.8),
        "Fair [0.6-0.7)": (0.6, 0.7),
        "Poor [0.0-0.6)": (0.0, 0.6)
    }
    
    distribution = {name: [] for name in ranges.keys()}
    
    # Phân loại từng câu trả lời
    for item in results:
        f1_score = item['bertscore']['f1']
        
        for range_name, (low, high) in ranges.items():
            if low <= f1_score < high:
                distribution[range_name].append(item)
                break
    
    return distribution


def print_score_analysis(distribution: Dict, total: int):
    """In ra phân tích Score Range"""
    print("\n" + "="*70)
    print("SCORE RANGE ANALYSIS")
    print("="*70)
    print("\nScore Range Analysis:")
    
    for range_name, items in distribution.items():
        count = len(items)
        percentage = (count / total) * 100 if total > 0 else 0
        
        # Format range name
        name_parts = range_name.split('[')
        label = name_parts[0].strip()
        range_str = '[' + name_parts[1] if len(name_parts) > 1 else ''
        
        print(f"  {label:12s} {range_str:12s} {count:4d} answers ({percentage:5.1f}%)")
    
    print(f"\n  {'Total':12s} {'':12s} {total:4d} answers (100.0%)")
    print("="*70)


def get_lowest_score_items(results: List[Dict], k: int) -> List[Dict]:
    """
    Lấy k câu trả lời có F1 score thấp nhất
    
    Args:
        results: List các kết quả đánh giá
        k: Số lượng câu cần lấy
    
    Returns:
        List k câu có điểm thấp nhất
    """
    # Sort theo F1 score tăng dần
    sorted_results = sorted(results, key=lambda x: x['bertscore']['f1'])
    return sorted_results[:k]


def print_low_quality_samples(items: List[Dict], k: int):
    """In ra các mẫu có chất lượng thấp"""
    print(f"\n{'='*70}")
    print(f"TOP {k} LOWEST QUALITY ANSWERS")
    print("="*70)
    
    for i, item in enumerate(items, 1):
        f1 = item['bertscore']['f1']
        precision = item['bertscore']['precision']
        recall = item['bertscore']['recall']
        question = item['question']
        
        print(f"\n[{i}] F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}")
        print(f"    Question: {question[:100]}...")
        print(f"    Ground truth: {item['ground_truth'][:80]}...")
        print(f"    Answer: {item['predicted_answer'][:80]}...")


def remove_items_from_predictions(predictions_file: str, 
                                  questions_to_remove: List[str],
                                  output_file: str) -> Tuple[int, int]:
    """
    Xóa các câu trả lời khỏi predictions.json
    
    Args:
        predictions_file: File predictions.json gốc
        questions_to_remove: List câu hỏi cần xóa
        output_file: File output sau khi xóa
    
    Returns:
        (original_count, final_count)
    """
    # Load predictions
    predictions = load_json(predictions_file)
    original_count = len(predictions)
    
    # Tạo set để lookup nhanh
    questions_set = set(questions_to_remove)
    
    # Filter ra các item không nằm trong danh sách xóa
    filtered_predictions = [
        pred for pred in predictions 
        if pred['question'] not in questions_set
    ]
    
    final_count = len(filtered_predictions)
    
    # Save file mới
    save_json(filtered_predictions, output_file)
    
    return original_count, final_count


def main():
    """Main function"""
    print("="*70)
    print("LOW QUALITY ANSWER FILTER & REMOVER")
    print("="*70)
    
    # File paths
    bertscore_file = "data/Evaluation_documents/bertscore_evaluation_results.json"
    predictions_file = "data/predictions.json"
    output_file = "data/predictions_cleaned.json"
    
    # Load BERTScore results
    print(f"\n[1/5] Loading BERTScore results from {bertscore_file}...")
    bertscore_data = load_json(bertscore_file)
    results = bertscore_data['per_question_results']
    total = len(results)
    print(f"    ✓ Loaded {total} evaluation results")
    
    # Analyze score distribution
    print(f"\n[2/5] Analyzing score distribution...")
    distribution = analyze_score_distribution(results)
    print_score_analysis(distribution, total)
    
    # Get poor quality count
    poor_count = len(distribution["Poor [0.0-0.6)"])
    fair_count = len(distribution["Fair [0.6-0.7)"])
    
    print(f"\n💡 Suggestions:")
    print(f"   - Poor quality (F1 < 0.6): {poor_count} answers")
    print(f"   - Fair quality (F1 < 0.7): {fair_count + poor_count} answers")
    print(f"   - Total available for removal: {total} answers")
    
    # Get user input
    print(f"\n[3/5] Select number of lowest quality answers to remove...")
    while True:
        try:
            k = int(input(f"    Enter number (1-{total}), or 0 to cancel: "))
            if k == 0:
                print("\n❌ Operation cancelled by user")
                return
            if 1 <= k <= total:
                break
            print(f"    ⚠️  Please enter a number between 1 and {total}")
        except ValueError:
            print("    ⚠️  Please enter a valid number")
    
    # Get lowest score items
    print(f"\n[4/5] Finding top {k} lowest quality answers...")
    lowest_items = get_lowest_score_items(results, k)
    print_low_quality_samples(lowest_items, k)
    
    # Confirm removal
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: You are about to remove {k} answers from predictions.json")
    print(f"{'='*70}")
    confirm = input("\n    Type 'yes' to confirm removal: ").strip().lower()
    
    if confirm != 'yes':
        print("\n❌ Removal cancelled")
        return
    
    # Remove from predictions
    print(f"\n[5/5] Removing {k} answers from predictions...")
    questions_to_remove = [item['question'] for item in lowest_items]
    
    original_count, final_count = remove_items_from_predictions(
        predictions_file, 
        questions_to_remove,
        output_file
    )
    
    removed_count = original_count - final_count
    
    print(f"\n{'='*70}")
    print("REMOVAL COMPLETE")
    print("="*70)
    print(f"  Original predictions: {original_count}")
    print(f"  Removed answers:      {removed_count}")
    print(f"  Remaining predictions: {final_count}")
    print(f"\n  ✓ Cleaned file saved to: {output_file}")
    print("="*70)
    
    # Statistics of removed items
    removed_f1_scores = [item['bertscore']['f1'] for item in lowest_items]
    avg_removed_f1 = sum(removed_f1_scores) / len(removed_f1_scores)
    min_removed_f1 = min(removed_f1_scores)
    max_removed_f1 = max(removed_f1_scores)
    
    print(f"\n📊 Removed Items Statistics:")
    print(f"   - Average F1: {avg_removed_f1:.4f}")
    print(f"   - Min F1:     {min_removed_f1:.4f}")
    print(f"   - Max F1:     {max_removed_f1:.4f}")
    print("\n💡 Next steps:")
    print(f"   1. Review cleaned file: {output_file}")
    print(f"   2. If satisfied, replace original: mv {output_file} {predictions_file}")
    print(f"   3. Re-run evaluation to see improved scores")
    print("="*70)


if __name__ == "__main__":
    main()
