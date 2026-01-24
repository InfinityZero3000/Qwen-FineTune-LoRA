"""
Balance Dataset by Crawling Additional Data
===========================================

This script crawls additional data from the same sources used in download_and_inspect_datasets.py
to balance the dataset distribution. It focuses on increasing minority classes (fluency, vocabulary)
while avoiding overfitting through:
1. Diverse source mixing
2. Context variation (1-3 sentences)
3. Deduplication across existing data
4. Quality filtering
"""

import json
import random
import hashlib
import re
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set
from datasets import load_dataset
import pandas as pd

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOADED_DIR = SCRIPT_DIR / "downloaded_datasets"
OUTPUT_DIR = SCRIPT_DIR / "downloaded_datasets"
CEFR_DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "cefr" / "ENGLISH_CERF_WORDS.csv"

# Current distribution (from analysis)
CURRENT_COUNTS = {
    "grammar": 6657,
    "dialogue": 3805,
    "vocabulary": 2362,
    "fluency": 1426,
}

# Target distribution (balanced but not perfectly equal to avoid artificial uniformity)
TARGET_COUNTS = {
    "fluency": 4000,      # +2,574 samples needed (minority class - highest boost)
    "vocabulary": 4000,   # +1,638 samples needed (minority class)
    "dialogue": 4500,     # +695 samples needed (slight boost)
    "grammar": 7000,      # +343 samples needed (already largest, minimal boost)
}

# Calculate how many samples to add per task
SAMPLES_TO_ADD = {
    task: max(0, TARGET_COUNTS[task] - CURRENT_COUNTS[task])
    for task in TARGET_COUNTS.keys()
}


def normalize_text(text: str) -> str:
    """Normalize text for deduplication"""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(text: str) -> str:
    """Create stable hash for deduplication"""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def pick_context(text: str, min_sents: int = 1, max_sents: int = 3) -> str:
    """Pick random sentence window for context variation"""
    sents = split_sentences(text)
    if not sents:
        return ""
    k = random.randint(max(1, min_sents), max(1, max_sents))
    k = min(k, len(sents))
    start = 0
    if len(sents) > k:
        start = random.randint(0, len(sents) - k)
    return " ".join(sents[start : start + k]).strip()


def looks_english(text: str, max_non_ascii_ratio: float = 0.2) -> bool:
    """Check if text is predominantly English"""
    if not text:
        return False
    try:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / max(1, len(text))) <= max_non_ascii_ratio
    except Exception:
        return False


# Global CEFR word lists (loaded once)
CEFR_WORD_DICT = {}

def load_cefr_word_list() -> Dict[str, str]:
    """Load CEFR-labeled word list from Kaggle dataset"""
    global CEFR_WORD_DICT
    
    if CEFR_WORD_DICT:  # Already loaded
        return CEFR_WORD_DICT
    
    if not CEFR_DATA_PATH.exists():
        print(f"⚠️  CEFR dataset not found at {CEFR_DATA_PATH}")
        print("   Download it with: kaggle datasets download -d nezahatkk/10-000-english-words-cerf-labelled")
        return {}
    
    word_dict = {}
    with open(CEFR_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['headword'].lower().strip()
            level = row['CEFR'].strip()
            # Store first occurrence (some words have multiple entries)
            if word not in word_dict:
                word_dict[word] = level
    
    CEFR_WORD_DICT = word_dict
    print(f"✅ Loaded {len(word_dict)} CEFR-labeled words")
    return word_dict

def estimate_vocab_level(text: str, use_real_cefr: bool = True) -> str:
    """Estimate CEFR level (A1/A2/B1/B2/C1/C2) using real word list or heuristic"""
    if use_real_cefr and CEFR_WORD_DICT:
        # Use real CEFR word list
        words = re.findall(r"[A-Za-z']+", (text or "").lower())
        if not words:
            return "A2"
        
        # Count words by CEFR level
        level_counts = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0}
        total_known = 0
        
        for word in words:
            if word in CEFR_WORD_DICT:
                level = CEFR_WORD_DICT[word]
                level_counts[level] = level_counts.get(level, 0) + 1
                total_known += 1
        
        if total_known == 0:
            # Fall back to heuristic if no words found
            return estimate_vocab_level(text, use_real_cefr=False)
        
        # Determine overall level based on majority and highest level present
        # If 30%+ words are B2+, classify as B2
        # If 30%+ words are B1+, classify as B1
        # Otherwise A2
        b2_plus = level_counts.get('B2', 0) + level_counts.get('C1', 0) + level_counts.get('C2', 0)
        b1_plus = b2_plus + level_counts.get('B1', 0)
        
        if b2_plus / total_known >= 0.3:
            return "B2"
        elif b1_plus / total_known >= 0.3:
            return "B1"
        else:
            return "A2"
    
    else:
        # Heuristic estimation (fallback)
        words = re.findall(r"[A-Za-z']+", (text or "").lower())
        if not words:
            return "A2"
        avg_len = sum(len(w) for w in words) / len(words)
        long_words = sum(1 for w in words if len(w) >= 9)
        ratio_long = long_words / len(words)
        if avg_len >= 5.6 or ratio_long >= 0.12:
            return "B2"
        if avg_len >= 4.8 or ratio_long >= 0.06:
            return "B1"
        return "A2"


def load_existing_data() -> Tuple[List[Dict], set]:
    """Load existing training data and create deduplication set"""
    print("📂 Loading existing data...")
    
    train_file = DOWNLOADED_DIR / "train.jsonl"
    if not train_file.exists():
        print("⚠️  train.jsonl not found, starting fresh")
        return [], set()
    
    existing_data = []
    seen_hashes = set()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            existing_data.append(item)
            # Create hash from input text for deduplication
            input_text = item.get('messages', [{}])[0].get('content', '')
            hash_key = stable_hash(normalize_text(input_text))
            seen_hashes.add(hash_key)
    
    task_counts = Counter(item.get('task', 'unknown') for item in existing_data)
    print(f"✅ Loaded {len(existing_data)} existing samples")
    print(f"   Task distribution: {dict(task_counts)}")
    print(f"   Unique hashes: {len(seen_hashes)}")
    
    return existing_data, seen_hashes


def crawl_fluency_data(target: int, seen_hashes: set) -> List[Dict]:
    """
    Crawl additional fluency data from multiple sources
    Sources: CoLA + WikiText + C4 subset
    """
    print(f"\n{'='*70}")
    print(f"📊 CRAWLING FLUENCY DATA (target: +{target} samples)")
    print(f"{'='*70}")
    
    fluency_data = []
    
    # Source 1: CoLA (skip already seen range, take new samples)
    try:
        print("\n⏳ [1/3] Loading CoLA dataset (extended range)...")
        # Original script used [:5000], we'll take [5000:10000] for new samples
        cola_dataset = load_dataset("nyu-mll/glue", "cola", split=f"train[5000:10000]")
        
        added = 0
        for item in cola_dataset:
            if len(fluency_data) >= target:
                break
            
            sentence = str(item.get("sentence", "")).strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Check deduplication
            hash_key = stable_hash(normalize_text(sentence))
            if hash_key in seen_hashes:
                continue
            
            label = int(item.get("label", 0))
            score = 0.85 + (label * 0.12)
            
            fluency_data.append({
                "task": "fluency",
                "messages": [
                    {"role": "user", "content": sentence},
                    {"role": "assistant", "content": json.dumps({"fluency_score": round(score, 2)})}
                ],
                "metadata": {"source": "CoLA", "original_label": label}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from CoLA")
        
    except Exception as e:
        print(f"  ⚠️  Error loading CoLA: {e}")
    
    # Source 2: WikiText-103 (for high fluency examples)
    try:
        print("\n⏳ [2/3] Loading WikiText-103...")
        # Take validation split for diversity
        wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        
        added = 0
        for item in wiki_dataset:
            if len(fluency_data) >= target:
                break
            
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 20:
                continue
            
            # Pick 1-2 sentence context
            context = pick_context(text, min_sents=1, max_sents=2)
            if len(context) < 20 or not looks_english(context):
                continue
            
            hash_key = stable_hash(normalize_text(context))
            if hash_key in seen_hashes:
                continue
            
            # WikiText = high quality, high fluency
            score = random.uniform(0.88, 0.98)
            
            fluency_data.append({
                "task": "fluency",
                "messages": [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": json.dumps({"fluency_score": round(score, 2)})}
                ],
                "metadata": {"source": "WikiText", "quality": "high"}
            })
            seen_hashes.add(hash_key)
            added += 1
            
            if added >= target * 0.4:  # Don't overload from one source
                break
        
        print(f"  ✅ Added {added} samples from WikiText")
        
    except Exception as e:
        print(f"  ⚠️  Error loading WikiText: {e}")
    
    # Source 3: C4 (for diverse web text - both high and low fluency)
    try:
        print("\n⏳ [3/3] Loading C4 dataset (web text)...")
        # Use validation split, small sample
        c4_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        
        added = 0
        for idx, item in enumerate(c4_dataset):
            if len(fluency_data) >= target:
                break
            if idx > 10000:  # Limit iteration
                break
            
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 30:
                continue
            
            # Pick 2-3 sentence context for variety
            context = pick_context(text, min_sents=2, max_sents=3)
            if len(context) < 30 or not looks_english(context):
                continue
            
            hash_key = stable_hash(normalize_text(context))
            if hash_key in seen_hashes:
                continue
            
            # C4 quality varies - estimate from length and complexity
            avg_word_len = sum(len(w) for w in context.split()) / max(1, len(context.split()))
            score = min(0.95, max(0.60, 0.70 + (avg_word_len - 4.5) * 0.05))
            
            fluency_data.append({
                "task": "fluency",
                "messages": [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": json.dumps({"fluency_score": round(score, 2)})}
                ],
                "metadata": {"source": "C4", "quality": "web"}
            })
            seen_hashes.add(hash_key)
            added += 1
            
            if added >= target * 0.3:  # Limit C4 contribution
                break
        
        print(f"  ✅ Added {added} samples from C4")
        
    except Exception as e:
        print(f"  ⚠️  Error loading C4: {e}")
    
    print(f"\n✅ Total fluency samples crawled: {len(fluency_data)}/{target}")
    return fluency_data


def crawl_vocabulary_data(target: int, seen_hashes: set) -> List[Dict]:
    """
    Crawl additional vocabulary data from multiple sources
    Sources: Simple Wikipedia + SNLI + AG News + Books3 subset
    """
    print(f"\n{'='*70}")
    print(f"📊 CRAWLING VOCABULARY DATA (target: +{target} samples)")
    print(f"{'='*70}")
    
    vocab_data = []
    
    # Source 1: Simple Wikipedia (A2-B1 level)
    try:
        print("\n⏳ [1/4] Loading Simple Wikipedia...")
        simple_wiki = load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
        
        added = 0
        for item in simple_wiki:
            if len(vocab_data) >= target * 0.3:
                break
            
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 50:
                continue
            
            # Pick 1-2 sentences
            context = pick_context(text, min_sents=1, max_sents=2)
            if len(context) < 30 or not looks_english(context):
                continue
            
            hash_key = stable_hash(normalize_text(context))
            if hash_key in seen_hashes:
                continue
            
            level = estimate_vocab_level(context)
            
            vocab_data.append({
                "task": "vocabulary",
                "messages": [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": json.dumps({"level": level})}
                ],
                "metadata": {"source": "SimpleWiki", "estimated_level": level}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from Simple Wikipedia")
        
    except Exception as e:
        print(f"  ⚠️  Error loading Simple Wikipedia: {e}")
    
    # Source 2: SNLI (extended range)
    try:
        print("\n⏳ [2/4] Loading SNLI dataset (extended)...")
        snli_dataset = load_dataset("snli", split="train[50000:100000]")
        
        added = 0
        for item in snli_dataset:
            if len(vocab_data) >= target * 0.6:
                break
            
            premise = str(item.get("premise", "")).strip()
            hypothesis = str(item.get("hypothesis", "")).strip()
            
            # Use both premise and hypothesis for variety
            for text in [premise, hypothesis]:
                if not text or len(text) < 15:
                    continue
                
                hash_key = stable_hash(normalize_text(text))
                if hash_key in seen_hashes:
                    continue
                
                level = estimate_vocab_level(text)
                
                vocab_data.append({
                    "task": "vocabulary",
                    "messages": [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": json.dumps({"level": level})}
                    ],
                    "metadata": {"source": "SNLI", "estimated_level": level}
                })
                seen_hashes.add(hash_key)
                added += 1
                
                if len(vocab_data) >= target * 0.6:
                    break
        
        print(f"  ✅ Added {added} samples from SNLI")
        
    except Exception as e:
        print(f"  ⚠️  Error loading SNLI: {e}")
    
    # Source 3: AG News (B2 level - news articles)
    try:
        print("\n⏳ [3/4] Loading AG News...")
        ag_news = load_dataset("ag_news", split="train[10000:20000]")
        
        added = 0
        for item in ag_news:
            if len(vocab_data) >= target * 0.8:
                break
            
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 40:
                continue
            
            # Pick 2-3 sentences for B2 complexity
            context = pick_context(text, min_sents=2, max_sents=3)
            if len(context) < 40 or not looks_english(context):
                continue
            
            hash_key = stable_hash(normalize_text(context))
            if hash_key in seen_hashes:
                continue
            
            # AG News is typically B2 level
            level = "B2"
            
            vocab_data.append({
                "task": "vocabulary",
                "messages": [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": json.dumps({"level": level})}
                ],
                "metadata": {"source": "AGNews", "estimated_level": level}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from AG News")
        
    except Exception as e:
        print(f"  ⚠️  Error loading AG News: {e}")
    
    # Source 4: BookCorpus/Books3 alternative - use ELI5 for varied complexity
    try:
        print("\n⏳ [4/4] Loading ELI5 (varied complexity)...")
        eli5 = load_dataset("eli5", split="train_asks[:3000]")
        
        added = 0
        for item in eli5:
            if len(vocab_data) >= target:
                break
            
            text = str(item.get("title", "")).strip()
            if not text or len(text) < 20:
                continue
            
            hash_key = stable_hash(normalize_text(text))
            if hash_key in seen_hashes:
                continue
            
            level = estimate_vocab_level(text)
            
            vocab_data.append({
                "task": "vocabulary",
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": json.dumps({"level": level})}
                ],
                "metadata": {"source": "ELI5", "estimated_level": level}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from ELI5")
        
    except Exception as e:
        print(f"  ⚠️  Error loading ELI5: {e}")
    
    print(f"\n✅ Total vocabulary samples crawled: {len(vocab_data)}/{target}")
    return vocab_data


def crawl_dialogue_data(target: int, seen_hashes: set) -> List[Dict]:
    """
    Crawl additional dialogue data
    Sources: OpenOrca + Dialogsum (extended ranges)
    """
    print(f"\n{'='*70}")
    print(f"📊 CRAWLING DIALOGUE DATA (target: +{target} samples)")
    print(f"{'='*70}")
    
    dialogue_data = []
    
    # Source 1: OpenOrca (extended range)
    try:
        print("\n⏳ [1/2] Loading OpenOrca (extended)...")
        # Original used [10000:15000], we'll take [15000:20000]
        orca_dataset = load_dataset("Open-Orca/OpenOrca", split="train[15000:20000]")
        
        added = 0
        for item in orca_dataset:
            if len(dialogue_data) >= target * 0.7:
                break
            
            question = str(item.get("question", "")).strip()
            response = str(item.get("response", "")).strip()
            
            if not question or not response or len(question) < 10:
                continue
            
            hash_key = stable_hash(normalize_text(question))
            if hash_key in seen_hashes:
                continue
            
            dialogue_data.append({
                "task": "dialogue",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {"source": "OpenOrca"}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from OpenOrca")
        
    except Exception as e:
        print(f"  ⚠️  Error loading OpenOrca: {e}")
    
    # Source 2: Dialogsum (extended range)
    try:
        print("\n⏳ [2/2] Loading Dialogsum (extended)...")
        dialogsum = load_dataset("knkarthick/dialogsum", split="train[5000:10000]")
        
        added = 0
        for item in dialogsum:
            if len(dialogue_data) >= target:
                break
            
            dialogue = str(item.get("dialogue", "")).strip()
            summary = str(item.get("summary", "")).strip()
            
            if not dialogue or not summary or len(dialogue) < 20:
                continue
            
            # Use dialogue as input, summary as response
            hash_key = stable_hash(normalize_text(dialogue))
            if hash_key in seen_hashes:
                continue
            
            # Format as conversation analysis
            input_text = f"Analyze this conversation: {dialogue[:300]}"
            output_text = f"Summary: {summary}"
            
            dialogue_data.append({
                "task": "dialogue",
                "messages": [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text}
                ],
                "metadata": {"source": "Dialogsum"}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Added {added} samples from Dialogsum")
        
    except Exception as e:
        print(f"  ⚠️  Error loading Dialogsum: {e}")
    
    print(f"\n✅ Total dialogue samples crawled: {len(dialogue_data)}/{target}")
    return dialogue_data


def crawl_grammar_data(target: int, seen_hashes: set) -> List[Dict]:
    """
    Crawl additional grammar data
    Source: Generate synthetic errors from high-quality text
    """
    print(f"\n{'='*70}")
    print(f"📊 CRAWLING GRAMMAR DATA (target: +{target} samples)")
    print(f"{'='*70}")
    
    if target <= 0:
        print("⚠️  No additional grammar samples needed")
        return []
    
    print("\n⏳ Generating synthetic grammar errors from WikiText...")
    
    grammar_data = []
    
    # Error injection patterns
    error_patterns = [
        (r'\b(I|you|we|they)\s+is\b', lambda m: f"{m.group(1)} are", "subject-verb agreement"),
        (r'\b(he|she|it)\s+are\b', lambda m: f"{m.group(1)} is", "subject-verb agreement"),
        (r'\b(has|have)\s+(went|gone)\b', lambda m: f"{m.group(1)} went", "incorrect past participle"),
        (r'\b(do|does)\s+not\s+(\w+ed)\b', lambda m: f"{m.group(1)} not {m.group(2)[:-2]}", "tense error"),
        (r'\ba\s+([aeiou]\w+)\b', lambda m: f"an {m.group(1)}", "article error"),
        (r'\ban\s+([^aeiou]\w+)\b', lambda m: f"a {m.group(1)}", "article error"),
    ]
    
    try:
        # Use WikiText validation set
        wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
        
        added = 0
        for item in wiki_dataset:
            if len(grammar_data) >= target:
                break
            
            text = str(item.get("text", "")).strip()
            if not text or len(text) < 30:
                continue
            
            # Pick 1-2 sentences
            context = pick_context(text, min_sents=1, max_sents=2)
            if len(context) < 30 or not looks_english(context):
                continue
            
            # Inject random error
            pattern, replacement, error_type = random.choice(error_patterns)
            incorrect = re.sub(pattern, replacement, context, count=1)
            
            if incorrect == context:  # No match, skip
                continue
            
            hash_key = stable_hash(normalize_text(incorrect))
            if hash_key in seen_hashes:
                continue
            
            grammar_data.append({
                "task": "grammar",
                "messages": [
                    {"role": "user", "content": incorrect},
                    {"role": "assistant", "content": json.dumps({
                        "corrected": context,
                        "errors": [error_type]
                    })}
                ],
                "metadata": {"source": "synthetic", "error_type": error_type}
            })
            seen_hashes.add(hash_key)
            added += 1
        
        print(f"  ✅ Generated {added} synthetic grammar samples")
        
    except Exception as e:
        print(f"  ⚠️  Error generating grammar data: {e}")
    
    print(f"\n✅ Total grammar samples crawled: {len(grammar_data)}/{target}")
    return grammar_data


def merge_and_save_data(existing_data: List[Dict], new_data: Dict[str, List[Dict]]):
    """Merge existing and new data, create train/val splits"""
    print(f"\n{'='*70}")
    print(f"📦 MERGING AND SAVING DATA")
    print(f"{'='*70}")
    
    # Combine all data
    all_new_samples = []
    for task, samples in new_data.items():
        all_new_samples.extend(samples)
    
    print(f"\n📊 New samples by task:")
    for task, samples in new_data.items():
        print(f"   {task}: +{len(samples)} samples")
    print(f"   Total new: {len(all_new_samples)}")
    
    # Merge with existing
    all_data = existing_data + all_new_samples
    print(f"\n✅ Total dataset size: {len(all_data)} samples")
    
    # Shuffle
    random.shuffle(all_data)
    
    # Split 95% train, 5% val
    split_idx = int(len(all_data) * 0.95)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    
    # Task distribution
    train_tasks = Counter(item.get('task', 'unknown') for item in train_data)
    print(f"\n📊 Final train distribution:")
    for task, count in sorted(train_tasks.items()):
        percentage = (count / len(train_data)) * 100
        print(f"   {task}: {count} ({percentage:.1f}%)")
    
    # Calculate balance ratio (max/min)
    counts = list(train_tasks.values())
    balance_ratio = max(counts) / min(counts) if counts else 1.0
    print(f"\n📊 Balance ratio: {balance_ratio:.2f}x (lower is better)")
    if balance_ratio < 2.0:
        print("   ✅ Excellent balance!")
    elif balance_ratio < 3.0:
        print("   ✅ Good balance")
    else:
        print("   ⚠️  Still some imbalance")
    
    # Save train/val splits
    train_file = OUTPUT_DIR / "train.jsonl"
    val_file = OUTPUT_DIR / "val.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n💾 Saved:")
    print(f"   {train_file}")
    print(f"   {val_file}")
    
    # Save detailed report
    report = {
        "total_samples": len(all_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_distribution": dict(train_tasks),
        "balance_ratio": round(balance_ratio, 2),
        "samples_added": {
            task: len(samples) for task, samples in new_data.items()
        }
    }
    
    report_file = OUTPUT_DIR / "balance_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   {report_file}")
    
    return train_data, val_data


def main():
    """Main execution"""
    print("="*70)
    print("🔄 LexiLingo Dataset Balancer")
    print("="*70)
    
    # Load CEFR word list
    load_cefr_word_list()
    
    print("\n📊 Current distribution:")
    for task, count in CURRENT_COUNTS.items():
        print(f"   {task}: {count}")
    
    print("\n🎯 Target distribution:")
    for task, count in TARGET_COUNTS.items():
        print(f"   {task}: {count}")
    
    print("\n➕ Samples to add:")
    total_to_add = 0
    for task, count in SAMPLES_TO_ADD.items():
        if count > 0:
            print(f"   {task}: +{count}")
            total_to_add += count
    
    if total_to_add == 0:
        print("\n✅ Dataset already balanced!")
        return
    
    print(f"\n📊 Total samples to crawl: {total_to_add}")
    print(f"📊 Final dataset size: ~{sum(TARGET_COUNTS.values())} samples")
    
    response = input("\n❓ Continue? [Y/n]: ")
    if response.lower() == 'n':
        print("❌ Cancelled")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load existing data
    existing_data, seen_hashes = load_existing_data()
    
    # Crawl additional data
    new_data = {}
    
    if SAMPLES_TO_ADD["fluency"] > 0:
        new_data["fluency"] = crawl_fluency_data(SAMPLES_TO_ADD["fluency"], seen_hashes)
    
    if SAMPLES_TO_ADD["vocabulary"] > 0:
        new_data["vocabulary"] = crawl_vocabulary_data(SAMPLES_TO_ADD["vocabulary"], seen_hashes)
    
    if SAMPLES_TO_ADD["dialogue"] > 0:
        new_data["dialogue"] = crawl_dialogue_data(SAMPLES_TO_ADD["dialogue"], seen_hashes)
    
    if SAMPLES_TO_ADD["grammar"] > 0:
        new_data["grammar"] = crawl_grammar_data(SAMPLES_TO_ADD["grammar"], seen_hashes)
    
    # Merge and save
    train_data, val_data = merge_and_save_data(existing_data, new_data)
    
    print(f"\n{'='*70}")
    print("✅ BALANCING COMPLETE!")
    print(f"{'='*70}")
    print("\n🚀 Next steps:")
    print("   1. Review balance_report.json for statistics")
    print("   2. Upload new train.jsonl and val.jsonl to Kaggle")
    print("   3. Run training with balanced dataset")
    print("\n💡 Tips to avoid overfitting:")
    print("   - Use dropout (already set: 0.05)")
    print("   - Monitor validation loss during training")
    print("   - Stop early if val loss plateaus")
    print("   - Dataset now has diverse sources and context variation")


if __name__ == "__main__":
    main()
