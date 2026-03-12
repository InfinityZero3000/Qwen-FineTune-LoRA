#!/usr/bin/env python3
"""
Script để merge vietnamese_explanations.json vào unified training data
Thêm task mới: 'explanation' - Model đóng vai trò giáo viên giải thích lỗi ngữ pháp bằng tiếng Việt

Author: LexiLingo Team
Date: 2026-01-27
"""

import json
import random
from pathlib import Path
from typing import Dict, List

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(filepath: Path) -> List[Dict]:
    """Load JSON array file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(data: List[Dict], filepath: Path):
    """Save to JSONL format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_explanation_format(explanation_item: Dict, index: int) -> Dict:
    """
    Convert vietnamese_explanations.json format to unified training format
    
    Input format:
    {
        "input": "Error: '...' → Correct: '...'",
        "output": "Vietnamese explanation...",
        "error_type": "modal_verb",
        "quality_score": 85
    }
    
    Output format (messages-based):
    {
        "task": "explanation",
        "messages": [
            {"role": "user", "content": "Error: '...' → Correct: '...'"},
            {"role": "assistant", "content": "Vietnamese explanation..."}
        ],
        "metadata": {
            "source": "vietnamese_explanations",
            "index": 0,
            "error_type": "modal_verb",
            "quality_score": 85
        }
    }
    """
    return {
        "task": "explanation",
        "messages": [
            {
                "role": "user",
                "content": explanation_item["input"]
            },
            {
                "role": "assistant",
                "content": explanation_item["output"]
            }
        ],
        "metadata": {
            "source": "vietnamese_explanations",
            "index": index,
            "error_type": explanation_item.get("error_type", "unknown"),
            "quality_score": explanation_item.get("quality_score", 0)
        }
    }

def merge_datasets(
    existing_train_path: Path,
    existing_val_path: Path,
    explanation_path: Path,
    output_train_path: Path,
    output_val_path: Path,
    val_split_ratio: float = 0.05,
    quality_threshold: int = 50
):
    """
    Merge vietnamese_explanations.json vào existing train/val data
    
    Args:
        existing_train_path: Path to existing train.jsonl
        existing_val_path: Path to existing val.jsonl
        explanation_path: Path to vietnamese_explanations.json
        output_train_path: Output path for merged train.jsonl
        output_val_path: Output path for merged val.jsonl
        val_split_ratio: Ratio of explanation data to use for validation (default: 5%)
        quality_threshold: Minimum quality_score to include (default: 50)
    """
    
    print("="*70)
    print("MERGING EXPLANATION DATA INTO TRAINING DATASET")
    print("="*70)
    
    # Load existing data
    print("\n1. Loading existing data...")
    existing_train = load_jsonl(existing_train_path)
    existing_val = load_jsonl(existing_val_path)
    print(f"   Existing train: {len(existing_train)} samples")
    print(f"   Existing val: {len(existing_val)} samples")
    
    # Load explanation data
    print("\n2. Loading explanation data...")
    explanation_data = load_json(explanation_path)
    print(f"   Total explanation samples: {len(explanation_data)}")
    
    # Filter by quality threshold
    print(f"\n3. Filtering by quality threshold (>= {quality_threshold})...")
    filtered_explanations = [
        item for item in explanation_data
        if item.get("quality_score", 0) >= quality_threshold
    ]
    print(f"   Filtered: {len(filtered_explanations)} samples")
    
    # Convert to unified format
    print("\n4. Converting to unified format...")
    converted_explanations = [
        convert_explanation_format(item, idx)
        for idx, item in enumerate(filtered_explanations)
    ]
    
    # Shuffle and split explanation data
    print(f"\n5. Splitting explanation data ({val_split_ratio*100:.0f}% for validation)...")
    random.seed(42)
    random.shuffle(converted_explanations)
    
    val_size = int(len(converted_explanations) * val_split_ratio)
    explanation_train = converted_explanations[val_size:]
    explanation_val = converted_explanations[:val_size]
    
    print(f"   Explanation train: {len(explanation_train)} samples")
    print(f"   Explanation val: {len(explanation_val)} samples")
    
    # Merge datasets
    print("\n6. Merging datasets...")
    merged_train = existing_train + explanation_train
    merged_val = existing_val + explanation_val
    
    print(f"   Merged train: {len(merged_train)} samples")
    print(f"   Merged val: {len(merged_val)} samples")
    
    # Shuffle merged data
    print("\n7. Shuffling merged data...")
    random.shuffle(merged_train)
    random.shuffle(merged_val)
    
    # Save merged datasets
    print("\n8. Saving merged datasets...")
    save_jsonl(merged_train, output_train_path)
    save_jsonl(merged_val, output_val_path)
    print(f"   Saved train: {output_train_path}")
    print(f"   Saved val: {output_val_path}")
    
    # Print task distribution
    print("\n9. Task Distribution Analysis:")
    train_task_counts = {}
    for sample in merged_train:
        task = sample.get('task', 'unknown')
        train_task_counts[task] = train_task_counts.get(task, 0) + 1
    
    print("\n   TRAIN SET:")
    for task, count in sorted(train_task_counts.items()):
        percentage = (count / len(merged_train)) * 100
        print(f"   - {task:12s}: {count:6d} samples ({percentage:5.1f}%)")
    
    val_task_counts = {}
    for sample in merged_val:
        task = sample.get('task', 'unknown')
        val_task_counts[task] = val_task_counts.get(task, 0) + 1
    
    print("\n   VAL SET:")
    for task, count in sorted(val_task_counts.items()):
        percentage = (count / len(merged_val)) * 100
        print(f"   - {task:12s}: {count:6d} samples ({percentage:5.1f}%)")
    
    print("\n" + "="*70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("="*70)
    
    # Generate report
    report = {
        "merge_date": "2026-01-27",
        "source_files": {
            "existing_train": str(existing_train_path),
            "existing_val": str(existing_val_path),
            "explanation_data": str(explanation_path)
        },
        "output_files": {
            "train": str(output_train_path),
            "val": str(output_val_path)
        },
        "statistics": {
            "before_merge": {
                "train": len(existing_train),
                "val": len(existing_val)
            },
            "explanation_data": {
                "total": len(explanation_data),
                "filtered": len(filtered_explanations),
                "train": len(explanation_train),
                "val": len(explanation_val),
                "quality_threshold": quality_threshold
            },
            "after_merge": {
                "train": len(merged_train),
                "val": len(merged_val)
            },
            "task_distribution_train": train_task_counts,
            "task_distribution_val": val_task_counts
        }
    }
    
    report_path = output_train_path.parent / "merge_explanation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved: {report_path}")
    
    return report

def main():
    """Main execution"""
    # Define paths
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets" / "datasets"
    
    existing_train = datasets_dir / "train.jsonl"
    existing_val = datasets_dir / "val.jsonl"
    explanation_file = datasets_dir / "vietnamese_explanations.json"
    
    output_train = datasets_dir / "train_with_explanation.jsonl"
    output_val = datasets_dir / "val_with_explanation.jsonl"
    
    # Verify input files exist
    if not existing_train.exists():
        print(f"ERROR: {existing_train} not found!")
        return
    
    if not existing_val.exists():
        print(f"ERROR: {existing_val} not found!")
        return
    
    if not explanation_file.exists():
        print(f"ERROR: {explanation_file} not found!")
        return
    
    # Run merge
    report = merge_datasets(
        existing_train_path=existing_train,
        existing_val_path=existing_val,
        explanation_path=explanation_file,
        output_train_path=output_train,
        output_val_path=output_val,
        val_split_ratio=0.05,  # 5% for validation
        quality_threshold=50   # Minimum quality score
    )
    
    print(f"\n✅ Merge completed!")
    print(f"\nNew training files created:")
    print(f"  - {output_train}")
    print(f"  - {output_val}")
    print(f"\nTo use these files:")
    print(f"  1. Backup your original files (optional)")
    print(f"  2. Rename or use the new files in your training pipeline")
    print(f"  3. Update notebook to load from train_with_explanation.jsonl")

if __name__ == "__main__":
    main()
