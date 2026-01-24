#!/usr/bin/env python3
"""
Quick Dataset Inspector
=======================

Công cụ nhanh để inspect datasets sau khi download.
Dùng để verify data quality trước khi upload lên Drive.

Usage:
    python inspect_datasets.py
"""

import json
import sys
import math
import re
import argparse
import hashlib
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except Exception:
    pd = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "downloaded_datasets"


TASK_SCHEMAS = {
    "fluency": {"output_required_keys": {"fluency_score", "reasoning"}},
    "grammar": {"output_required_keys": {"corrected", "explanation"}},
    "vocabulary": {"output_required_keys": {"key_words", "level"}},
    "dialogue": {"output_required_keys": {"response"}},
}


def _safe_pct(n, d):
    return (n / d * 100.0) if d else 0.0


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def word_tokens(text: str):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", normalize_text(text))


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def distinct_n(tokens, n: int) -> float:
    if n <= 0:
        return 0.0
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def simhash64(text: str) -> int:
    """Simple 64-bit simhash over word tokens (no external deps)."""
    tokens = word_tokens(text)
    if not tokens:
        return 0
    weights = [0] * 64
    for t in tokens:
        h = hashlib.md5(t.encode("utf-8")).digest()[:8]
        v = int.from_bytes(h, "big", signed=False)
        for i in range(64):
            bit = (v >> i) & 1
            weights[i] += 1 if bit else -1
    out = 0
    for i, w in enumerate(weights):
        if w >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def estimate_near_duplicates(texts, max_items=3000, max_pairs_per_bucket=2000, distance_threshold=3):
    """Estimate near-duplicate ratio using simhash banding on a sample."""
    if not texts:
        return {"sampled": 0, "near_dupe_pairs": 0, "near_dupe_items_est": 0, "note": "no texts"}

    if len(texts) > max_items:
        step = max(1, len(texts) // max_items)
        sampled = texts[::step][:max_items]
        note = f"sampled {len(sampled)}/{len(texts)} (step={step})"
    else:
        sampled = texts
        note = "full set"

    sigs = [simhash64(t) for t in sampled]
    buckets = {}
    # 4 bands of 16 bits
    for idx, sig in enumerate(sigs):
        for band in range(4):
            shift = band * 16
            key = (band, (sig >> shift) & 0xFFFF)
            buckets.setdefault(key, []).append(idx)

    near_pairs = 0
    flagged = set()
    for _, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        # cap comparisons per bucket
        comparisons = 0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                comparisons += 1
                if comparisons > max_pairs_per_bucket:
                    break
                a = idxs[i]
                b = idxs[j]
                if hamming64(sigs[a], sigs[b]) <= distance_threshold:
                    near_pairs += 1
                    flagged.add(a)
                    flagged.add(b)
            if comparisons > max_pairs_per_bucket:
                break

    return {
        "sampled": len(sampled),
        "near_dupe_pairs": near_pairs,
        "near_dupe_items_est": len(flagged),
        "note": note,
        "distance_threshold": distance_threshold,
    }

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def validate_records(task: str, records: list):
    """Return validation stats for a given task list."""
    required_output_keys = TASK_SCHEMAS.get(task, {}).get("output_required_keys", set())

    missing_fields = Counter()
    type_errors = Counter()
    invalid_output = 0
    empty_input = 0
    empty_output = 0
    too_short_input = 0
    too_short_output = 0
    unchanged_correction = 0
    invalid_fluency_score = 0

    for item in records:
        if not isinstance(item, dict):
            type_errors["record_not_dict"] += 1
            continue

        for k in ("task", "input", "output"):
            if k not in item:
                missing_fields[k] += 1

        inp = item.get("input", "")
        out = item.get("output", {})

        if not isinstance(inp, str):
            type_errors["input_not_str"] += 1
            inp = str(inp)

        if not inp.strip():
            empty_input += 1
        if len(inp.strip()) < 5:
            too_short_input += 1

        if not isinstance(out, dict):
            type_errors["output_not_dict"] += 1
            out = {}

        if not out:
            empty_output += 1

        # required output keys
        if required_output_keys and (not required_output_keys.issubset(out.keys())):
            invalid_output += 1

        # task-specific checks
        if task == "fluency":
            s = out.get("fluency_score")
            try:
                s = float(s)
                if not (0.0 <= s <= 1.0):
                    invalid_fluency_score += 1
            except Exception:
                invalid_fluency_score += 1
        elif task == "grammar":
            corrected = out.get("corrected", "")
            if isinstance(corrected, str) and normalize_text(corrected) == normalize_text(inp):
                unchanged_correction += 1
        elif task == "dialogue":
            resp = out.get("response", "")
            if isinstance(resp, str) and len(resp.strip()) < 10:
                too_short_output += 1
        elif task == "vocabulary":
            lvl = out.get("level")
            if isinstance(lvl, str) and not re.fullmatch(r"[ABC][12]", lvl.strip()):
                invalid_output += 1

    n = len(records)
    return {
        "n": n,
        "missing_fields": dict(missing_fields),
        "type_errors": dict(type_errors),
        "invalid_output": invalid_output,
        "empty_input": empty_input,
        "empty_output": empty_output,
        "too_short_input": too_short_input,
        "too_short_output": too_short_output,
        "unchanged_correction": unchanged_correction,
        "invalid_fluency_score": invalid_fluency_score,
    }


def diversity_metrics(texts: list):
    if not texts:
        return {
            "n": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "unique_ratio_exact": 0.0,
            "type_token_ratio": 0.0,
            "distinct1": 0.0,
            "distinct2": 0.0,
        }

    norm = [normalize_text(t) for t in texts]
    unique_ratio = len(set(norm)) / len(norm)
    lengths = [len(t) for t in norm]
    tokens = []
    for t in norm[:5000]:
        tokens.extend(word_tokens(t))
    ttr = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    d1 = distinct_n(tokens, 1)
    d2 = distinct_n(tokens, 2)
    return {
        "n": len(texts),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "unique_ratio_exact": unique_ratio,
        "type_token_ratio": ttr,
        "distinct1": d1,
        "distinct2": d2,
    }


def duplicate_input_stats(records: list):
    """Exact duplicate analysis by normalized input; checks if duplicates map to multiple outputs."""
    if not records:
        return {
            "n": 0,
            "unique_inputs": 0,
            "dupe_inputs": 0,
            "dupe_rate": 0.0,
            "multi_output_inputs": 0,
            "top": [],
        }

    groups = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        inp = normalize_text(item.get("input", ""))
        out = item.get("output", {})
        try:
            out_sig = json.dumps(out, sort_keys=True, ensure_ascii=False)
        except Exception:
            out_sig = str(out)
        groups.setdefault(inp, []).append(out_sig)

    unique_inputs = len(groups)
    dupe_inputs = sum(1 for _, outs in groups.items() if len(outs) > 1)
    multi_output = sum(1 for _, outs in groups.items() if len(set(outs)) > 1)
    # Top duplicate inputs by frequency (show a short preview)
    top = sorted(((k, len(v), len(set(v))) for k, v in groups.items()), key=lambda x: x[1], reverse=True)[:5]
    top_pretty = []
    for k, count, uniq_outs in top:
        if count <= 1:
            continue
        preview = (k[:90] + "...") if len(k) > 90 else k
        top_pretty.append({"input_preview": preview, "count": count, "unique_outputs": uniq_outs})

    n = len(records)
    dupe_rate = 1.0 - (unique_inputs / n) if n else 0.0
    return {
        "n": n,
        "unique_inputs": unique_inputs,
        "dupe_inputs": dupe_inputs,
        "dupe_rate": dupe_rate,
        "multi_output_inputs": multi_output,
        "top": top_pretty,
    }

def inspect_task_data(task_name, file_path, near_dup=False):
    """Inspect a specific task's dataset"""
    print(f"\n{'─'*70}")
    print(f"🔍 {task_name.upper()}")
    print(f"{'─'*70}")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ File: {file_path.name}")
    print(f"📦 Total samples: {len(data)}")
    
    # File size
    size_mb = file_path.stat().st_size / 1024 / 1024
    print(f"💾 File size: {size_mb:.2f} MB")
    
    # Basic structure
    if len(data) > 0:
        sample = data[0]
        print(f"\nSample structure:")
        print(f"  Keys: {list(sample.keys())}")

    # Cleanliness / schema validation
    task_key = task_name.lower()
    v = validate_records(task_key, data)
    print(f"\n✅ Cleanliness checks:")
    print(f"  Missing fields: {v['missing_fields'] or 'none'}")
    if v["type_errors"]:
        print(f"  Type errors: {v['type_errors']}")
    print(f"  Invalid output schema: {v['invalid_output']} ({_safe_pct(v['invalid_output'], v['n']):.2f}%)")
    print(f"  Empty input: {v['empty_input']} ({_safe_pct(v['empty_input'], v['n']):.2f}%)")
    print(f"  Empty output: {v['empty_output']} ({_safe_pct(v['empty_output'], v['n']):.2f}%)")
    print(f"  Too-short input (<5 chars): {v['too_short_input']} ({_safe_pct(v['too_short_input'], v['n']):.2f}%)")
    if task_key == "grammar":
        print(f"  Unchanged correction: {v['unchanged_correction']} ({_safe_pct(v['unchanged_correction'], v['n']):.2f}%)")
    if task_key == "fluency":
        print(f"  Invalid fluency_score: {v['invalid_fluency_score']} ({_safe_pct(v['invalid_fluency_score'], v['n']):.2f}%)")
    if task_key in ("dialogue",):
        print(f"  Too-short output (<10 chars): {v['too_short_output']} ({_safe_pct(v['too_short_output'], v['n']):.2f}%)")

    # Diversity metrics
    inputs = [item.get("input", "") for item in data if isinstance(item, dict)]
    m = diversity_metrics(inputs)
    print(f"\n📈 Input diversity metrics:")
    print(f"  Unique ratio (exact, normalized): {m['unique_ratio_exact']*100:.2f}%")
    print(f"  Avg length: {m['avg_chars']:.0f} chars (min={m['min_chars']}, max={m['max_chars']})")
    print(f"  Type-token ratio (sampled): {m['type_token_ratio']:.3f}")
    print(f"  Distinct-1/2 (sampled): {m['distinct1']:.3f} / {m['distinct2']:.3f}")

    # Exact duplicates (and whether they have multiple different outputs)
    dup = duplicate_input_stats(data)
    print(f"\n🧾 Duplicate-input analysis:")
    print(f"  Unique inputs: {dup['unique_inputs']} / {dup['n']} (dupe rate ~{dup['dupe_rate']*100:.2f}%)")
    if dup["dupe_inputs"]:
        print(f"  Inputs that repeat: {dup['dupe_inputs']} (some repeats may be OK depending on task)")
    if dup["multi_output_inputs"]:
        print(f"  Repeated inputs with multiple different outputs: {dup['multi_output_inputs']}")
    if dup["top"]:
        print("  Top repeated inputs (preview):")
        for t in dup["top"]:
            print(f"    - count={t['count']}, unique_outputs={t['unique_outputs']}: {t['input_preview']}")

    # Near-duplicate estimate (optional)
    if near_dup:
        est = estimate_near_duplicates(inputs)
        near_rate = _safe_pct(est["near_dupe_items_est"], est["sampled"])
        print(f"\n🧪 Near-duplicate estimate (simhash):")
        print(f"  Note: {est['note']}")
        print(f"  Near-dup items (est): {est['near_dupe_items_est']} / {est['sampled']} ({near_rate:.2f}%)")
        print(f"  Distance threshold: {est['distance_threshold']}")
        
        # Task-specific analysis
        if task_name == "Fluency":
            scores = [item['output']['fluency_score'] for item in data]
            print(f"\nFluency score distribution:")
            print(f"  Min: {min(scores):.2f}")
            print(f"  Max: {max(scores):.2f}")
            print(f"  Avg: {sum(scores)/len(scores):.2f}")
            
            # Score ranges
            ranges = {
                '0.0-0.4': sum(1 for s in scores if s < 0.4),
                '0.4-0.6': sum(1 for s in scores if 0.4 <= s < 0.6),
                '0.6-0.8': sum(1 for s in scores if 0.6 <= s < 0.8),
                '0.8-1.0': sum(1 for s in scores if 0.8 <= s <= 1.0),
            }
            for range_name, count in ranges.items():
                pct = count / len(scores) * 100
                print(f"  {range_name}: {count:>4} ({pct:5.1f}%)")
        
        elif task_name == "Grammar":
            sources = [item.get('metadata', {}).get('source', 'unknown') for item in data]
            source_counts = Counter(sources)
            print(f"\nSource distribution:")
            for source, count in source_counts.most_common():
                pct = count / len(data) * 100
                print(f"  {source:20s}: {count:>5} ({pct:5.1f}%)")
            
            # Error types
            error_types = [item['output']['explanation'][:30] for item in data[:100]]
            print(f"\nSample error types (first 5):")
            for i, error in enumerate(error_types[:5], 1):
                print(f"  {i}. {error}...")
        
        elif task_name == "Vocabulary":
            levels = [item['output']['level'] for item in data]
            level_counts = Counter(levels)
            print(f"\nCEFR level distribution:")
            for level in sorted(level_counts.keys()):
                count = level_counts[level]
                pct = count / len(levels) * 100
                print(f"  {level}: {count:>5} ({pct:5.1f}%)")
        
        elif task_name == "Dialogue":
            response_lengths = [len(item['output']['response']) for item in data]
            print(f"\nResponse length statistics:")
            print(f"  Min: {min(response_lengths)} chars")
            print(f"  Max: {max(response_lengths)} chars")
            print(f"  Avg: {sum(response_lengths)/len(response_lengths):.0f} chars")
        
        # Show 2 samples
        print(f"\nSample data (first 2):")
        for i in range(min(2, len(data))):
            sample = data[i]
            print(f"\n  Sample {i+1}:")
            print(f"    Input: {sample['input'][:70]}...")
            print(f"    Output: {str(sample['output'])[:70]}...")
    
    return data

def inspect_unified_data(near_dup: bool = False):
    """Inspect the unified training dataset"""
    print_section("UNIFIED DATASET ANALYSIS")
    
    unified_file = DATA_DIR / "unified_training_data.json"
    
    if not unified_file.exists():
        print(f"❌ Unified dataset not found: {unified_file}")
        return
    
    with open(unified_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ File: {unified_file.name}")
    print(f"📦 Total samples: {len(data)}")
    
    size_mb = unified_file.stat().st_size / 1024 / 1024
    print(f"💾 File size: {size_mb:.2f} MB")
    
    # Task distribution
    task_counts = Counter([item['task'] for item in data])
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        pct = count / len(data) * 100
        print(f"  {task.capitalize():15s}: {count:>5} samples ({pct:5.1f}%)")
    
    # Source distribution
    sources = [item.get('metadata', {}).get('source', 'unknown') for item in data]
    source_counts = Counter(sources)
    print(f"\nSource distribution:")
    for source, count in source_counts.most_common():
        pct = count / len(data) * 100
        print(f"  {source:20s}: {count:>5} samples ({pct:5.1f}%)")
    
    # Unified cleanliness + diversity
    print(f"\n✅ Unified cleanliness checks:")
    missing_task = sum(1 for item in data if 'task' not in item)
    missing_input = sum(1 for item in data if 'input' not in item)
    missing_output = sum(1 for item in data if 'output' not in item)
    empty_input = sum(1 for item in data if not str(item.get('input', '')).strip())
    empty_output = sum(1 for item in data if not item.get('output', {}))
    print(f"  Missing 'task': {missing_task} ({_safe_pct(missing_task, len(data)):.2f}%)")
    print(f"  Missing 'input': {missing_input} ({_safe_pct(missing_input, len(data)):.2f}%)")
    print(f"  Missing 'output': {missing_output} ({_safe_pct(missing_output, len(data)):.2f}%)")
    print(f"  Empty input: {empty_input} ({_safe_pct(empty_input, len(data)):.2f}%)")
    print(f"  Empty output: {empty_output} ({_safe_pct(empty_output, len(data)):.2f}%)")

    inputs = [item.get('input', '') for item in data]
    md = diversity_metrics(inputs)
    print(f"\n📈 Unified input diversity metrics:")
    print(f"  Unique ratio (exact, normalized): {md['unique_ratio_exact']*100:.2f}%")
    print(f"  Avg length: {md['avg_chars']:.0f} chars (min={md['min_chars']}, max={md['max_chars']})")
    print(f"  Type-token ratio (sampled): {md['type_token_ratio']:.3f}")
    print(f"  Distinct-1/2 (sampled): {md['distinct1']:.3f} / {md['distinct2']:.3f}")

    # Source entropy is a good quick proxy for variety
    src_ent = shannon_entropy(source_counts)
    print(f"  Source entropy: {src_ent:.3f} (higher => more diverse sources)")

    if near_dup:
        est = estimate_near_duplicates(inputs)
        near_rate = _safe_pct(est["near_dupe_items_est"], est["sampled"])
        print(f"\n🧪 Unified near-duplicate estimate (simhash):")
        print(f"  Note: {est['note']}")
        print(f"  Near-dup items (est): {est['near_dupe_items_est']} / {est['sampled']} ({near_rate:.2f}%)")
        print(f"  Distance threshold: {est['distance_threshold']}")
    
    # Check CSV file (optional)
    csv_file = DATA_DIR / "unified_training_data.csv"
    if csv_file.exists() and pd is not None:
        try:
            df = pd.read_csv(csv_file)
            print(f"\n✅ CSV file exists: {csv_file.name}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"\n⚠️  CSV exists but could not be parsed by pandas: {e}")
    elif csv_file.exists() and pd is None:
        print(f"\n⚠️  CSV exists but pandas is not installed; skipping CSV parse.")

def compare_with_targets():
    """Compare actual dataset sizes with architecture.md targets"""
    print_section("TARGET vs ACTUAL COMPARISON")
    
    targets = {
        'fluency': 1500,
        'grammar': 7000,
        'vocabulary': 2500,
        'dialogue': 4000
    }
    
    total_target = sum(targets.values())
    
    # Get actual counts
    unified_file = DATA_DIR / "unified_training_data.json"
    if unified_file.exists():
        with open(unified_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        task_counts = Counter([item['task'] for item in data])
        total_actual = sum(task_counts.values())
        
        print(f"\n{'Task':<15} {'Target':>8} {'Actual':>8} {'Progress':>10} {'Status'}")
        print("─"*60)
        
        for task, target in sorted(targets.items()):
            actual = task_counts.get(task, 0)
            progress = actual / target * 100 if target > 0 else 0
            status = "✅" if actual >= target else "⚠️"
            print(f"{task.capitalize():<15} {target:>8} {actual:>8} {progress:>9.1f}% {status}")
        
        print("─"*60)
        total_progress = total_actual / total_target * 100
        total_status = "✅" if total_actual >= total_target else "⚠️"
        print(f"{'TOTAL':<15} {total_target:>8} {total_actual:>8} {total_progress:>9.1f}% {total_status}")
        
        print(f"\n{'🎯 Target:':<20} {total_target} samples")
        print(f"{'📦 Actual:':<20} {total_actual} samples")
        print(f"{'Progress:':<20} {total_progress:.1f}%")
        
        if total_actual >= total_target:
            print(f"\n✅ TARGET ACHIEVED! Ready for training.")
        else:
            gap = total_target - total_actual
            print(f"\n⚠️  Need {gap} more samples to reach target.")
    else:
        print("❌ Unified dataset not found. Run download script first.")

def main():
    """Main inspection function"""
    global DATA_DIR
    parser = argparse.ArgumentParser(description="LexiLingo dataset inspector (quality + diversity).")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to downloaded_datasets folder (default: scripts/downloaded_datasets)",
    )
    parser.add_argument(
        "--near-dup",
        action="store_true",
        help="Estimate near-duplicates using simhash (slower; uses sampling)",
    )
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).expanduser().resolve()

    print("="*70)
    print("🔍 LexiLingo Dataset Inspector")
    print("="*70)

    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        print("\n💡 Example:")
        print("   python inspect_datasets.py --data-dir ./downloaded_datasets")
        sys.exit(1)

    print(f"\n📁 Inspecting: {DATA_DIR}")
    
    # Inspect individual task files
    print_section("INDIVIDUAL TASK DATASETS")
    
    tasks = [
        ("Fluency", DATA_DIR / "fluency_data.json"),
        ("Grammar", DATA_DIR / "grammar_data.json"),
        ("Vocabulary", DATA_DIR / "vocabulary_data.json"),
        ("Dialogue", DATA_DIR / "dialogue_data.json"),
    ]
    
    for task_name, file_path in tasks:
        inspect_task_data(task_name, file_path, near_dup=args.near_dup)
    
    # Inspect unified dataset
    inspect_unified_data(near_dup=args.near_dup)
    
    # Compare with targets
    compare_with_targets()
    
    # Final summary
    print_section("SUMMARY")
    print("\n✅ Inspection complete!")
    print("\nFiles in data directory:")
    for file in sorted(DATA_DIR.glob("*")):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"  {file.name:35s} ({size_mb:>6.2f} MB)")
    
    # Training readiness verdict (basic)
    unified_file = DATA_DIR / "unified_training_data.json"
    verdict = "UNKNOWN"
    reasons = []
    if unified_file.exists():
        with open(unified_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        task_counts = Counter([item.get('task', 'unknown') for item in data if isinstance(item, dict)])
        targets = {'fluency': 1500, 'grammar': 7000, 'vocabulary': 2500, 'dialogue': 4000}
        meets_targets = all(task_counts.get(t, 0) >= v for t, v in targets.items())

        empty_input = sum(1 for item in data if not str(item.get('input', '')).strip())
        empty_output = sum(1 for item in data if not item.get('output', {}))
        too_short_input = sum(1 for item in data if len(str(item.get('input', '')).strip()) < 5)

        # duplication should be judged within each task (cross-task repeats are normal)
        per_task_dupe = {}
        for t in ("fluency", "grammar", "vocabulary", "dialogue"):
            subset = [x for x in data if isinstance(x, dict) and x.get("task") == t]
            dup = duplicate_input_stats(subset)
            per_task_dupe[t] = dup["dupe_rate"]

        if not meets_targets:
            reasons.append("does not meet per-task target counts")
        if empty_input or empty_output:
            reasons.append("has empty input/output")
        if too_short_input > 0:
            reasons.append("contains very short inputs (<5 chars)")
        high_dupe_tasks = {t: r for t, r in per_task_dupe.items() if r > 0.30}
        if high_dupe_tasks:
            reasons.append(
                "high exact duplicate input rate within task: "
                + ", ".join(f"{t}~{r*100:.1f}%" for t, r in sorted(high_dupe_tasks.items()))
            )

        if meets_targets and not reasons:
            verdict = "READY"
        elif meets_targets:
            verdict = "READY_WITH_WARNINGS"
        else:
            verdict = "NOT_READY"

    print("\n🚦 Training readiness:")
    print(f"  Verdict: {verdict}")
    if reasons:
        print("  Notes:")
        for r in reasons:
            print(f"    - {r}")

    print("\n🚀 Next steps:")
    print("  1. If you want deeper duplication check, re-run with --near-dup")
    print("  2. If exact duplicates are high, dedupe by normalized input before training")
    print("  3. Use a proper train/val split by (source + normalized input) to avoid leakage")

if __name__ == "__main__":
    main()
