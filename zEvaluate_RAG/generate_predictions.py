"""
Bước 1: Generate predictions từ RAG pipeline
Input: data/Eveluate.json (có question và ground_truth)
Output: data/predictions.json (có question, ground_truth, contexts, answer)
"""

import json
import time
from tqdm import tqdm
from src.rag_pipeline import RAGPipeline
from config.config import Config


def generate_predictions(
    input_file: str = "data/Eveluate.json",
    output_file: str = "data/predictions.json"
):
    """
    Generate predictions từ RAG pipeline
    
    Args:
        input_file: File chứa questions và ground_truth
        output_file: File để lưu predictions (questions, contexts, answer, ground_truth)
    """
    
    print("="*60)
    print("GENERATE PREDICTIONS FROM RAG PIPELINE")
    print("="*60)
    
    # Validate config
    print("\n[1/5] Validating configuration...")
    try:
        Config.validate()
        print("✅ Configuration valid!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Initialize RAG pipeline
    print("\n[2/5] Initializing RAG Pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        print("✅ Pipeline initialized!")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return
    
    # Load existing index
    print("\n[3/5] Loading vector index...")
    if not rag_pipeline.load_existing_index():
        print("❌ Error: No index found!")
        print("Please run 'python main.py' first to ingest documents.")
        return
    print("✅ Index loaded!")
    
    # Load evaluation data
    print(f"\n[4/5] Loading questions from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        print(f"✅ Loaded {len(evaluation_data)} questions")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Generate predictions
    print(f"\n[5/5] Generating predictions for {len(evaluation_data)} questions...")
    print("⚠️  Note: Gemini Free Tier limit = 10 requests/minute")
    print(f"    Adding {Config.LLM_DELAY_BETWEEN_REQUESTS}s delay between requests to avoid rate limit")
    
    estimated_time = len(evaluation_data) * Config.LLM_DELAY_BETWEEN_REQUESTS / 60
    print(f"    Estimated time: ~{estimated_time:.1f} minutes\n")
    
    predictions = []
    errors = 0
    
    for i, item in enumerate(tqdm(evaluation_data, desc="Processing")):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        max_retries = 3
        retry_delay = 30  # Initial delay for retry
        
        for attempt in range(max_retries):
            try:
                # Query RAG pipeline
                result = rag_pipeline.query(question)
                
                # Create prediction entry
                prediction = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "contexts": result["context"],  # List of context passages
                    "answer": result["response"]     # Generated answer
                }
                
                predictions.append(prediction)
                
                # Add delay between requests to respect rate limit
                # (except for last item)
                if i < len(evaluation_data) - 1:
                    time.sleep(Config.LLM_DELAY_BETWEEN_REQUESTS)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message if available
                        if "retry in" in error_str.lower():
                            try:
                                import re
                                match = re.search(r'retry in (\d+\.?\d*)', error_str.lower())
                                if match:
                                    retry_delay = float(match.group(1)) + 2  # Add buffer
                            except:
                                pass
                        
                        print(f"\n⏸️  Rate limit hit. Waiting {retry_delay:.0f}s before retry (attempt {attempt+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                    else:
                        print(f"\n❌ Max retries reached for question {i+1}")
                        predictions.append({
                            "question": question,
                            "ground_truth": ground_truth,
                            "contexts": [],
                            "answer": f"ERROR: Rate limit exceeded after {max_retries} retries"
                        })
                        errors += 1
                        break
                else:
                    # Other errors
                    print(f"\n❌ Error processing question {i+1}: {error_str[:100]}")
                    predictions.append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "contexts": [],
                        "answer": f"ERROR: {error_str[:200]}"
                    })
                    errors += 1
                    break
    
    # Save predictions
    print(f"\n\nSaving predictions to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print("✅ Predictions saved successfully!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total questions: {len(evaluation_data)}")
    print(f"Successful: {len(predictions) - errors}")
    print(f"Errors: {errors}")
    print(f"\nOutput file: {output_file}")
    print("="*60)
    
    # Show sample
    if predictions:
        print("\n📝 Sample prediction:")
        sample = predictions[0]
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer'][:200]}...")
        print(f"Contexts: {len(sample['contexts'])} passages")
        print(f"Ground truth: {sample['ground_truth']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions from RAG pipeline')
    parser.add_argument('--input', type=str, default='data/Eveluate.json',
                        help='Input file with questions (default: data/Eveluate.json)')
    parser.add_argument('--output', type=str, default='data/predictions.json',
                        help='Output file for predictions (default: data/predictions.json)')
    
    args = parser.parse_args()
    
    generate_predictions(args.input, args.output)
