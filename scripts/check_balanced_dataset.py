"""
Check Balanced Dataset Statistics
=================================

Quick script to analyze the balanced dataset after running balance_dataset.py
"""

import json
from pathlib import Path
from collections import Counter

# Paths
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "downloaded_datasets"
TRAIN_FILE = DATASET_DIR / "train.jsonl"
VAL_FILE = DATASET_DIR / "val.jsonl"


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_dataset():
    """Analyze dataset statistics"""
    print("="*70)
    print("📊 BALANCED DATASET ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n⏳ Loading data...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    
    print(f"✅ Loaded {len(train_data):,} train samples")
    print(f"✅ Loaded {len(val_data):,} val samples")
    
    # Total
    total = len(train_data) + len(val_data)
    print(f"\n📦 Total Dataset: {total:,} samples")
    print(f"   Train: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
    print(f"   Val:   {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
    
    # Task distribution
    print("\n" + "="*70)
    print("📊 TRAIN TASK DISTRIBUTION")
    print("="*70)
    
    train_tasks = Counter(item.get('task', 'unknown') for item in train_data)
    
    for task, count in sorted(train_tasks.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(train_data)) * 100
        bar = "█" * int(pct / 2)
        print(f"   {task:12s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    # Balance ratio
    counts = list(train_tasks.values())
    max_count = max(counts)
    min_count = min(counts)
    balance_ratio = max_count / min_count
    
    print(f"\n📊 Balance Metrics:")
    print(f"   Max task: {max_count:,} samples")
    print(f"   Min task: {min_count:,} samples")
    print(f"   Ratio:    {balance_ratio:.2f}x")
    
    if balance_ratio < 2.0:
        print("   ✅ Excellent balance! (< 2.0x)")
    elif balance_ratio < 3.0:
        print("   ✅ Good balance (< 3.0x)")
    elif balance_ratio < 4.0:
        print("   ⚠️  Moderate imbalance (< 4.0x)")
    else:
        print("   ❌ High imbalance (≥ 4.0x)")
    
    # Val distribution
    print("\n" + "="*70)
    print("📊 VAL TASK DISTRIBUTION")
    print("="*70)
    
    val_tasks = Counter(item.get('task', 'unknown') for item in val_data)
    
    for task, count in sorted(val_tasks.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(val_data)) * 100
        print(f"   {task:12s}: {count:5} ({pct:5.1f}%)")
    
    # Source distribution
    print("\n" + "="*70)
    print("📊 TOP 10 DATA SOURCES")
    print("="*70)
    
    train_sources = Counter()
    for item in train_data:
        source = item.get('metadata', {}).get('source', 'unknown')
        train_sources[source] += 1
    
    for source, count in train_sources.most_common(10):
        pct = (count / len(train_data)) * 100
        print(f"   {source:20s}: {count:5,} ({pct:5.1f}%)")
    
    # Quality check - check for duplicates
    print("\n" + "="*70)
    print("🔍 QUALITY CHECKS")
    print("="*70)
    
    # Check input text uniqueness
    train_inputs = set()
    duplicates = 0
    
    for item in train_data:
        msgs = item.get('messages', [])
        if msgs:
            input_text = msgs[0].get('content', '').strip().lower()
            if input_text in train_inputs:
                duplicates += 1
            train_inputs.add(input_text)
    
    print(f"   Unique train inputs: {len(train_inputs):,}/{len(train_data):,}")
    if duplicates > 0:
        print(f"   ⚠️  Duplicates found: {duplicates}")
    else:
        print(f"   ✅ No duplicates!")
    
    # Check for empty inputs
    empty_inputs = 0
    short_inputs = 0
    
    for item in train_data:
        msgs = item.get('messages', [])
        if not msgs or not msgs[0].get('content', '').strip():
            empty_inputs += 1
        elif len(msgs[0].get('content', '')) < 10:
            short_inputs += 1
    
    print(f"   Empty inputs: {empty_inputs}")
    print(f"   Short inputs (<10 chars): {short_inputs}")
    
    if empty_inputs == 0 and short_inputs < len(train_data) * 0.01:
        print(f"   ✅ Quality looks good!")
    
    # Sample examples
    print("\n" + "="*70)
    print("📋 SAMPLE EXAMPLES (first 5)")
    print("="*70)
    
    for i, item in enumerate(train_data[:5]):
        task = item.get('task', 'unknown')
        msgs = item.get('messages', [])
        source = item.get('metadata', {}).get('source', 'unknown')
        
        if not msgs or len(msgs) < 1:
            continue
        
        input_text = msgs[0].get('content', '')[:100]
        output_text = msgs[1].get('content', '')[:100] if len(msgs) > 1 else ''
        
        print(f"\n[{i+1}] Task: {task} | Source: {source}")
        print(f"    Input:  {input_text}{'...' if len(msgs[0].get('content', '')) > 100 else ''}")
        if output_text:
            print(f"    Output: {output_text}{'...' if len(msgs[1].get('content', '')) > 100 else ''}")
    
    # Training estimates
    print("\n" + "="*70)
    print("⏱️  TRAINING ESTIMATES")
    print("="*70)
    
    # Assume batch_size=1, grad_accum=24, 5 epochs
    batch_size = 1
    grad_accum = 24
    epochs = 5
    effective_batch = batch_size * grad_accum
    
    steps_per_epoch = len(train_data) // effective_batch
    total_steps = steps_per_epoch * epochs
    
    # Estimate time (P100/T4: ~1.5s per step)
    time_per_step = 1.5  # seconds
    total_time_seconds = total_steps * time_per_step
    total_time_hours = total_time_seconds / 3600
    
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch}")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Total steps (5 epochs): {total_steps:,}")
    print(f"   Estimated time (P100/T4): {total_time_hours:.1f} hours")
    
    if total_time_hours < 5:
        print(f"   ✅ Within Kaggle session limit!")
    elif total_time_hours < 9:
        print(f"   ⚠️  Close to session limit")
    else:
        print(f"   ❌ Exceeds typical session limit")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\n🚀 Next steps:")
    print("   1. Review task distribution (should be balanced)")
    print("   2. Check quality metrics (no duplicates, no empty inputs)")
    print("   3. Upload train.jsonl + val.jsonl to Kaggle Dataset")
    print("   4. Update notebook to use new dataset")
    print("   5. Run training!")


if __name__ == "__main__":
    analyze_dataset()
