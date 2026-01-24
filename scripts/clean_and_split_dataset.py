#!/usr/bin/env python3
"""Clean & split LexiLingo unified dataset (anti-leakage).

Goals
-----
- Clean: drop invalid / empty / too-short samples; optionally normalize whitespace.
- Deduplicate: remove exact duplicate records.
- Split: create train/val JSONL where no (source + normalized_input) appears in both.

Usage
-----
python clean_and_split_dataset.py \
  --data-dir ./downloaded_datasets \
  --val-ratio 0.05 \
  --seed 42 \
  --min-input-chars 5

Outputs
-------
- train.jsonl
- val.jsonl
- split_report.json

JSONL format: one JSON object per line with keys: task, input, output, metadata
"""

from __future__ import annotations

import argparse
import json
import random
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(text: str) -> str:
    # usedforsecurity=False is available on Python 3.9+ (macOS likely).
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_unified_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def clean_record(item: Any, min_input_chars: int) -> Tuple[Dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "record_not_dict"

    task = item.get("task")
    inp = item.get("input")
    out = item.get("output")
    meta = item.get("metadata")

    if not isinstance(task, str) or not task.strip():
        return None, "missing_task"
    if not isinstance(inp, str):
        return None, "input_not_str"
    inp = inp.strip()
    if len(inp) < min_input_chars:
        return None, "input_too_short"
    if not isinstance(out, dict) or not out:
        return None, "output_invalid"
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        return None, "metadata_not_dict"

    # Ensure source exists for grouping
    source = meta.get("source", "unknown")
    if not isinstance(source, str) or not source.strip():
        source = "unknown"

    meta = dict(meta)
    meta.setdefault("source", source)

    cleaned = {
        "task": task.strip(),
        "input": inp,
        "output": out,
        "metadata": meta,
    }
    return cleaned, None


def group_key(record: Dict[str, Any]) -> str:
    src = record.get("metadata", {}).get("source", "unknown")
    norm_inp = normalize_text(record.get("input", ""))
    return stable_hash(f"{src}::{norm_inp}")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json_dumps_stable(r))
            f.write("\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split unified_training_data.json into train/val JSONL without leakage.")
    parser.add_argument("--data-dir", type=str, default=str(SCRIPT_DIR / "downloaded_datasets"), help="Folder containing unified_training_data.json")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio by sample count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-input-chars", type=int, default=5, help="Drop samples with very short input strings")
    parser.add_argument("--out-train", type=str, default="train.jsonl", help="Train JSONL filename (within data-dir)")
    parser.add_argument("--out-val", type=str, default="val.jsonl", help="Val JSONL filename (within data-dir)")
    parser.add_argument("--report", type=str, default="split_report.json", help="Report JSON filename (within data-dir)")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    unified_path = data_dir / "unified_training_data.json"
    if not unified_path.exists():
        raise SystemExit(f"Unified dataset not found: {unified_path}")

    raw = load_unified_json(unified_path)

    drop_reasons = Counter()
    cleaned: List[Dict[str, Any]] = []
    for item in raw:
        rec, reason = clean_record(item, min_input_chars=max(1, int(args.min_input_chars)))
        if rec is None:
            drop_reasons[reason or "unknown"] += 1
            continue
        cleaned.append(rec)

    # Exact dedupe
    seen_record = set()
    deduped: List[Dict[str, Any]] = []
    for rec in cleaned:
        sig = stable_hash(json_dumps_stable(rec))
        if sig in seen_record:
            drop_reasons["exact_duplicate_record"] += 1
            continue
        seen_record.add(sig)
        deduped.append(rec)

    # Group by (source + normalized_input)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in deduped:
        k = group_key(rec)
        groups.setdefault(k, []).append(rec)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    target_val = int(round(len(deduped) * float(args.val_ratio)))

    val: List[Dict[str, Any]] = []
    train: List[Dict[str, Any]] = []

    val_count = 0
    for k in group_keys:
        bucket = groups[k]
        if val_count < target_val:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)

    # Sanity: ensure no leakage
    train_keys = {group_key(r) for r in train}
    val_keys = {group_key(r) for r in val}
    leakage = len(train_keys.intersection(val_keys))

    # Task/source stats
    task_train = Counter(r["task"] for r in train)
    task_val = Counter(r["task"] for r in val)
    src_train = Counter(r.get("metadata", {}).get("source", "unknown") for r in train)
    src_val = Counter(r.get("metadata", {}).get("source", "unknown") for r in val)

    out_train = data_dir / args.out_train
    out_val = data_dir / args.out_val
    report_path = data_dir / args.report

    n_train = write_jsonl(out_train, train)
    n_val = write_jsonl(out_val, val)

    report = {
        "input": {
            "unified_path": str(unified_path),
            "raw_samples": len(raw),
        },
        "cleaning": {
            "min_input_chars": int(args.min_input_chars),
            "kept_after_clean": len(cleaned),
            "kept_after_exact_dedupe": len(deduped),
            "dropped": dict(drop_reasons),
        },
        "split": {
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "target_val_samples": target_val,
            "train_samples": n_train,
            "val_samples": n_val,
            "train_groups": len(train_keys),
            "val_groups": len(val_keys),
            "leakage_groups": leakage,
        },
        "distribution": {
            "task": {
                "train": dict(task_train),
                "val": dict(task_val),
            },
            "source": {
                "train": dict(src_train.most_common(20)),
                "val": dict(src_val.most_common(20)),
            },
        },
        "outputs": {
            "train_jsonl": str(out_train),
            "val_jsonl": str(out_val),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ Clean & split complete")
    print(f"  Train: {n_train} -> {out_train}")
    print(f"  Val:   {n_val} -> {out_val}")
    print(f"  Report: {report_path}")
    print(f"  Leakage groups: {leakage}")


if __name__ == "__main__":
    main()
