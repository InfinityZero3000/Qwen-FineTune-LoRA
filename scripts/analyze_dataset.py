#!/usr/bin/env python3
"""
Analyze training dataset statistics
"""
import json
from pathlib import Path

def analyze_dataset():
    """Analyze dataset size and training estimates"""
    
    # Read split report
    report_path = Path("scripts/downloaded_datasets/split_report.json")
    with open(report_path) as f:
        report = json.load(f)
    
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Extract info from report structure
    raw_samples = report['input']['raw_samples']
    train_samples = report['split']['train_samples']
    val_samples = report['split']['val_samples']
    val_ratio = report['split']['val_ratio']
    
    print(f"\nTotal samples: {raw_samples:,}")
    print(f"Train samples: {train_samples:,} ({(1-val_ratio)*100:.0f}%)")
    print(f"Val samples: {val_samples:,} ({val_ratio*100:.0f}%)")
    
    print(f"\nBreakdown by task (train):")
    task_dist = report['distribution']['task']['train']
    for task, count in sorted(task_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task:15s}: {count:,} samples")
    
    # Read sample entries
    train_path = Path("scripts/downloaded_datasets/train.jsonl")
    samples = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            samples.append(json.loads(line))
    
    print(f"\nSample entries:")
    for i, sample in enumerate(samples, 1):
        input_text = sample['input'][:80] + "..." if len(sample['input']) > 80 else sample['input']
        output_text = sample['output'][:80] + "..." if len(sample['output']) > 80 else sample['output']
        print(f"\n{i}. Input: {input_text}")
        print(f"   Output: {output_text}")
    
    print("\n" + "="*70)
    print("DATASET SIZE ASSESSMENT")
    print("="*70)
    
    # Size categories
    if train_samples < 5000:
        size_category = "VERY SMALL"
        quality = "May underfit - needs data augmentation"
    elif train_samples < 10000:
        size_category = "SMALL"
        quality = "Minimum viable - will work but limited"
    elif train_samples < 50000:
        size_category = "MEDIUM"
        quality = "Good - sufficient for decent performance"
    elif train_samples < 100000:
        size_category = "LARGE"
        quality = "Very good - strong performance expected"
    else:
        size_category = "VERY LARGE"
        quality = "Excellent - high quality results expected"
    
    print(f"\nSize category: {size_category}")
    print(f"Quality assessment: {quality}")
    print(f"\nYour dataset: {train_samples:,} training samples")
    print(f"Status: {'ADEQUATE ✓' if train_samples >= 10000 else 'TOO SMALL ⚠'}")
    
    if train_samples < 10000:
        print("\nRECOMMENDATIONS:")
        print("  - Consider data augmentation (paraphrasing, back-translation)")
        print("  - Use smaller model (0.5B instead of 1.5B)")
        print("  - Reduce LoRA rank (r=16 instead of r=32)")
        print("  - More epochs (7-10 instead of 5)")
    
    print("\n" + "="*70)
    print("TRAINING TIME ESTIMATES")
    print("="*70)
    
    # Estimate for different configs
    configs = [
        ("FAST (0.5B, r=16, 3 epochs)", 1, 8, 3, 2.5),
        ("BALANCED (1.5B, r=32, 5 epochs)", 1, 24, 5, 3.5),
        ("HIGH QUALITY (1.5B, r=32, 7 epochs)", 1, 24, 7, 3.5),
    ]
    
    for name, batch, grad_accum, epochs, time_per_step in configs:
        effective_batch = batch * grad_accum
        steps_per_epoch = train_samples // effective_batch
        total_steps = steps_per_epoch * epochs
        
        # Time estimates (seconds per step)
        hours_p100 = (total_steps * time_per_step) / 3600
        hours_t4 = (total_steps * (time_per_step + 1)) / 3600
        
        print(f"\n{name}:")
        print(f"  Effective batch: {effective_batch}")
        print(f"  Steps/epoch: {steps_per_epoch:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Time on P100: {hours_p100:.1f} hours ({hours_p100/24:.1f} days)")
        print(f"  Time on T4: {hours_t4:.1f} hours ({hours_t4/24:.1f} days)")
        
        if hours_p100 > 30:
            print(f"  WARNING: Exceeds Kaggle's 30hr/week GPU limit!")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Calculate best config for 30hr limit
    max_hours = 30
    best_config = None
    
    for name, batch, grad_accum, epochs, time_per_step in configs:
        effective_batch = batch * grad_accum
        steps_per_epoch = train_samples // effective_batch
        total_steps = steps_per_epoch * epochs
        hours_p100 = (total_steps * time_per_step) / 3600
        
        if hours_p100 <= max_hours:
            best_config = (name, hours_p100)
    
    if best_config:
        print(f"\nBest config for Kaggle (30hr limit):")
        print(f"  {best_config[0]}")
        print(f"  Estimated time: {best_config[1]:.1f} hours")
        print(f"  Will complete within Kaggle's weekly limit ✓")
    else:
        print(f"\nWARNING: All configs exceed 30hr limit!")
        print(f"Consider:")
        print(f"  - Reduce epochs to 3-4")
        print(f"  - Use smaller model (0.5B)")
        print(f"  - Split training across multiple sessions")
    
    print("\n" + "="*70)
    print("DATASET QUALITY CHECK")
    print("="*70)
    
    # Check task balance
    task_counts = report['distribution']['task']['train']
    min_count = min(task_counts.values())
    max_count = max(task_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nTask balance:")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / train_samples) * 100
        print(f"  {task:15s}: {count:5,} ({percentage:5.1f}%)")
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 3:
        print("WARNING: Significant class imbalance detected!")
        print("Consider:")
        print("  - Oversample minority classes")
        print("  - Undersample majority classes")
        print("  - Use class weights in training")
    else:
        print("Balance status: GOOD ✓")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_dataset()
