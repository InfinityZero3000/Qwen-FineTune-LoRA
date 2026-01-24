"""
Download and Inspect Datasets for LexiLingo Training
======================================================

This script downloads all required datasets from HuggingFace and local sources,
saves them to disk for upload to Google Drive, and provides inspection tools
to verify data quality before training.

Target: 15,000+ samples across 4 tasks
- Fluency:    1,500 samples
- Grammar:    7,000 samples (expanded from 2,000)
- Vocabulary: 2,500 samples
- Dialogue:   4,000 samples
"""

import json
import os
import re
import argparse
import hashlib
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from typing import Dict, List
import random

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "downloaded_datasets"

BASE_TARGETS = {
    "fluency": 1500,
    "grammar": 7000,
    "vocabulary": 2500,
    "dialogue": 4000,
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    # naive sentence split; good enough for diversity/context windows
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def pick_context(text: str, min_sents: int, max_sents: int) -> str:
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
    if not text:
        return False
    try:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / max(1, len(text))) <= max_non_ascii_ratio
    except Exception:
        return False


def apply_m2_edits(sentence: str, edits: List[str]) -> str:
    """Apply all M2 'A ' edits to a sentence and return corrected sentence.

    This is a simplified applier (space-token based). Good enough to produce a full corrected string
    and avoid the previous bug where 'corrected' was only a replacement fragment.
    """
    tokens = sentence.split()
    parsed = []
    for line in edits:
        if not line.startswith("A "):
            continue
        try:
            # A start end|||type|||correction|||...
            head, rest = line[2:].split("|||", 1)
            span = head.split()
            start = int(span[0])
            end = int(span[1])
            parts = line.split("|||")
            err_type = parts[1].strip() if len(parts) > 1 else "unknown"
            corr = parts[2].strip() if len(parts) > 2 else ""
            parsed.append((start, end, err_type, corr))
        except Exception:
            continue

    # apply from right to left so offsets stay valid
    parsed.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for start, end, _, corr in parsed:
        start = max(0, min(start, len(tokens)))
        end = max(0, min(end, len(tokens)))
        if end < start:
            start, end = end, start
        if corr == "-NONE-" or corr == "":
            repl = []
        else:
            repl = corr.split()
        tokens[start:end] = repl
    return " ".join(tokens).strip()


def estimate_vocab_level(text: str) -> str:
    """Heuristic CEFR-ish level estimator (A2/B1/B2)."""
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

class DatasetDownloader:
    """Download and inspect datasets for all 4 tasks"""
    
    def __init__(
        self,
        output_dir: Path,
        targets: Dict[str, int],
        min_context_sentences: int = 1,
        max_context_sentences: int = 1,
        dedupe_global: bool = False,
        allow_exceed: bool = False,
        min_input_chars: int = 5,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)  # Ensure folder exists
        self.stats = {}
        self.targets = targets
        self.min_context_sentences = min_context_sentences
        self.max_context_sentences = max_context_sentences
        self.dedupe_global = dedupe_global
        self.allow_exceed = allow_exceed
        self.min_input_chars = max(1, int(min_input_chars))
        self._seen_by_task = {k: set() for k in targets.keys()}
        self._seen_global = set()

    def _dedupe_key(self, task: str, input_text: str) -> str:
        return stable_hash(f"{task}::{normalize_text(input_text)}")

    def _add_sample(self, out_list: List[Dict], task: str, input_text: str, output: Dict, metadata: Dict) -> bool:
        input_text = (input_text or "").strip()
        if not input_text:
            return False
        if len(input_text) < self.min_input_chars:
            return False
        key = self._dedupe_key(task, input_text)
        if key in self._seen_by_task.get(task, set()):
            return False
        if self.dedupe_global:
            gk = stable_hash(normalize_text(input_text))
            if gk in self._seen_global:
                return False
            self._seen_global.add(gk)

        self._seen_by_task.setdefault(task, set()).add(key)
        md = dict(metadata or {})
        # always keep original/raw text for traceability
        md.setdefault("raw_text", input_text)
        out_list.append({
            "task": task,
            "input": input_text,
            "output": output,
            "metadata": md,
        })
        return True
        
    def download_fluency_data(self) -> List[Dict]:
        """
        Download Fluency Scoring Dataset
        Target: 1,500 samples
        Sources: 
        - CoLA (Corpus of Linguistic Acceptability) - grammaticality judgments
        - GLUE SST-2 (sentiment) - use as fluency proxy
        - Local wi_locness for learner text
        """
        print("\n" + "="*70)
        print("📊 TASK 1: FLUENCY SCORING")
        print("="*70)
        target = int(self.targets.get("fluency", 1500))
        print(f"Target: {target:,} samples")
        print("Sources: CoLA + SST-2 + wi_locness")
        
        fluency_data = []
        
        # Source 1: CoLA dataset (grammaticality judgments as fluency proxy)
        try:
            print("\n⏳ [1/3] Loading CoLA dataset...")
            take = min(5000, max(1000, int(target * 0.6)))
            cola_dataset = load_dataset("nyu-mll/glue", "cola", split=f"train[:{take}]")
            
            for idx, item in enumerate(cola_dataset):
                sentence = item['sentence']
                label = item['label']  # 0 = ungrammatical, 1 = grammatical
                
                # Map grammaticality to fluency score
                if label == 1:
                    fluency_score = round(random.uniform(0.75, 0.95), 2)
                    reasoning = "Grammatically correct with natural structure"
                else:
                    fluency_score = round(random.uniform(0.30, 0.65), 2)
                    reasoning = "Contains grammatical issues affecting fluency"
                
                self._add_sample(
                    fluency_data,
                    "fluency",
                    sentence,
                    {"fluency_score": fluency_score, "reasoning": reasoning},
                    {"source": "CoLA", "grammatical": bool(label), "index": idx, "raw_text": sentence},
                )
            
            print(f"  ✅ Loaded {len(fluency_data)} samples from CoLA")
            
        except Exception as e:
            print(f"  ⚠️  Error loading CoLA: {e}")
        
        # Source 2: SST-2 sentiment dataset (use well-formed sentences as high fluency)
        try:
            print("\n⏳ [2/3] Loading SST-2 dataset...")
            take = min(5000, max(300, int(target * 0.2)))
            sst_dataset = load_dataset("nyu-mll/glue", "sst2", split=f"train[:{take}]")
            
            for idx, item in enumerate(sst_dataset):
                sentence = item['sentence']
                
                # SST-2 sentences are well-formed, so assign high fluency
                fluency_score = round(random.uniform(0.80, 0.95), 2)
                reasoning = "Clear and fluent expression with natural word flow"
                
                self._add_sample(
                    fluency_data,
                    "fluency",
                    sentence,
                    {"fluency_score": fluency_score, "reasoning": reasoning},
                    {"source": "SST-2", "index": idx, "raw_text": sentence},
                )
            
            print(f"  ✅ Loaded {len(sst_dataset)} samples from SST-2")
            
        except Exception as e:
            print(f"  ⚠️  Error loading SST-2: {e}")
        
        # Source 3: Use local wi_locness learner corpus for low-fluency examples
        try:
            print("\n⏳ [3/3] Loading wi_locness for learner text...")
            local_path = Path("../datasets/wi+locness/m2")
            if local_path.exists():
                m2_files = list(local_path.glob("*.dev.*.m2"))[:1]  # Use dev set
                
                for m2_file in m2_files:
                    with open(m2_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    current_sent = None
                    error_count = 0
                    
                    for line in lines[:200]:  # Limit to 200
                        line = line.strip()
                        if line.startswith('S '):
                            if current_sent and error_count > 0:
                                # Calculate fluency based on error count
                                fluency_score = max(0.30, 0.90 - (error_count * 0.15))
                                fluency_score = round(fluency_score, 2)
                                
                                self._add_sample(
                                    fluency_data,
                                    "fluency",
                                    current_sent,
                                    {
                                        "fluency_score": fluency_score,
                                        "reasoning": f"Learner text with {error_count} identified errors",
                                    },
                                    {
                                        "source": "wi_locness_learner",
                                        "error_count": error_count,
                                        "raw_text": current_sent,
                                    },
                                )
                            
                            current_sent = line[2:]
                            error_count = 0
                        elif line.startswith('A ') and current_sent:
                            error_count += 1
                
                print(f"  ✅ Loaded {sum(1 for x in fluency_data if x['metadata']['source']=='wi_locness_learner')} samples from wi_locness")
            else:
                print(f"  ⚠️  Local wi_locness not found")
                
        except Exception as e:
            print(f"  ⚠️  Error loading wi_locness: {e}")
        
        # Fill remaining with synthetic if needed
        # Add real-text passages for longer context diversity (WikiText)
        try:
            if len(fluency_data) < target:
                print("\n⏳ [extra] Loading WikiText for longer-context fluency...")
                wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
                count = 0
                for idx, item in enumerate(wikitext):
                    if len(fluency_data) >= target:
                        break
                    text = (item.get("text") or "").strip()
                    if len(text) < 80 or not looks_english(text, 0.15):
                        continue
                    ctx = pick_context(text, self.min_context_sentences, self.max_context_sentences)
                    if len(ctx.split()) < 8:
                        continue
                    fluency_score = round(random.uniform(0.82, 0.97), 2)
                    reasoning = "Real-world text snippet; generally fluent"
                    if self._add_sample(
                        fluency_data,
                        "fluency",
                        ctx,
                        {"fluency_score": fluency_score, "reasoning": reasoning},
                        {"source": "WikiText", "index": idx, "raw_text": text, "context_sentences": len(split_sentences(ctx))},
                    ):
                        count += 1
                    if count >= max(500, int(target * 0.2)):
                        # don't overdo this source
                        break
                print(f"  ✅ Added {count} samples from WikiText")
        except Exception as e:
            print(f"  ⚠️  Error loading WikiText: {e}")

        remaining = max(0, target - len(fluency_data))
        if remaining > 0:
            print(f"\n⏳ Generating {remaining} synthetic samples to reach target...")
            synthetic = self._create_fallback_fluency_data_partial(remaining)
            for s in synthetic:
                self._add_sample(fluency_data, "fluency", s["input"], s["output"], s.get("metadata", {}))
        
        # Save to JSON
        output_file = self.output_dir / "fluency_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fluency_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Total fluency samples: {len(fluency_data)}")
        print(f"💾 Saved to: {output_file}")
        
        # Show source distribution
        sources = {}
        for item in fluency_data:
            src = item['metadata'].get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        print("\n📊 Source distribution:")
        for src, count in sources.items():
            print(f"  {src}: {count} samples")
        
        # Show samples
        print("\n📋 Sample data (first 3):")
        for i in range(min(3, len(fluency_data))):
            sample = fluency_data[i]
            print(f"\n{i+1}. Input: {sample['input'][:80]}...")
            print(f"   Score: {sample['output']['fluency_score']}")
            print(f"   Source: {sample['metadata']['source']}")
        
        self.stats['fluency'] = len(fluency_data)
        return fluency_data
    
    def _create_fallback_fluency_data(self) -> List[Dict]:
        """Create synthetic fluency data as fallback"""
        return self._create_fallback_fluency_data_partial(1500)
    
    def _create_fallback_fluency_data_partial(self, count: int) -> List[Dict]:
        """Create specific number of synthetic fluency samples"""
        # more diverse synthetic pool to reduce duplicates
        subjects = ["I", "You", "He", "She", "We", "They", "My friend", "The student", "My teacher"]
        verbs = ["go", "study", "like", "want", "need", "enjoy", "visit", "watch", "play", "read"]
        objects = ["to school", "English", "coffee", "a movie", "books", "the park", "the museum", "homework", "music"]
        times = ["yesterday", "today", "every day", "last week", "in the evening", "on weekends"]
        patterns = [
            ("{subj} {verb} {obj} {time}", "Simple sentence"),
            ("{subj} really {verb} {obj} {time}", "Natural adverb placement"),
            ("{subj} {verb} {obj} because {subj2} is busy", "Adds causal clause"),
        ]
        
        fluency_data = []
        for i in range(count):
            sub = random.choice(subjects)
            sub2 = random.choice([s for s in subjects if s != sub])
            verb = random.choice(verbs)
            obj = random.choice(objects)
            time = random.choice(times)
            template, reason = random.choice(patterns)
            sent = template.format(subj=sub, subj2=sub2, verb=verb, obj=obj, time=time).strip()
            # inject small mistakes for lower scores sometimes
            if random.random() < 0.35:
                sent = sent.replace("yesterday", "yesterday")
                if "go" in sent and random.random() < 0.5:
                    sent = sent.replace(" go ", " goes ")
                if sent.lower().startswith("he ") and random.random() < 0.5:
                    sent = sent.replace("He ", "He ").replace("want", "want")
            score = round(random.uniform(0.55, 0.95), 2)
            fluency_data.append({
                "task": "fluency",
                "input": sent,
                "output": {"fluency_score": score, "reasoning": reason},
                "metadata": {"source": "synthetic", "index": i, "raw_text": sent},
            })
        
        return fluency_data
    
    def download_grammar_data(self) -> List[Dict]:
        """
        Download Grammar Correction Dataset
        Target: 7,000 samples (expanded)
        Sources: 
        - wi_locness (local): 2,000 samples
        - CoNLL-2014: 1,500 samples
        - FCE corpus: 2,000 samples
        - Synthetic errors: 1,500 samples
        """
        print("\n" + "="*70)
        print("📊 TASK 2: GRAMMAR CORRECTION")
        print("="*70)
        target = int(self.targets.get("grammar", 7000))
        print(f"Target: {target:,} samples")
        print("Sources: wi_locness (local) + CoNLL-2014 + FCE + Synthetic")
        
        grammar_data = []
        
        # Source 1: Local wi_locness dataset
        print("\n⏳ Loading wi_locness (local)...")
        try:
            local_path = Path("../datasets/wi+locness/m2")
            if local_path.exists():
                m2_files = list(local_path.glob("*.train.*.m2"))
                for m2_file in m2_files[:3]:  # ABC.train files
                    print(f"  Reading: {m2_file.name}")
                    with open(m2_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    # Parse M2 format by sentence blocks
                    current_sent = None
                    current_edits = []
                    for raw_line in lines:
                        line = raw_line.rstrip("\n")
                        if line.startswith('S '):
                            current_sent = line[2:].strip()
                            current_edits = []
                        elif line.startswith('A ') and current_sent:
                            current_edits.append(line.strip())
                        elif line.strip() == "" and current_sent is not None:
                            corrected_full = apply_m2_edits(current_sent, current_edits)
                            if corrected_full and corrected_full != current_sent:
                                self._add_sample(
                                    grammar_data,
                                    "grammar",
                                    current_sent,
                                    {
                                        "corrected": corrected_full,
                                        "explanation": "Learner sentence corrected (M2 edits applied)",
                                    },
                                    {"source": "wi_locness", "file": m2_file.name, "raw_text": current_sent},
                                )
                            current_sent = None
                            current_edits = []

                            if (not self.allow_exceed) and len(grammar_data) >= target:
                                break

                    if (not self.allow_exceed) and len(grammar_data) >= target:
                        break
                
                print(f"  ✅ Loaded from local: {len(grammar_data)} samples")
            else:
                print(f"  ⚠️  Local wi_locness not found at {local_path}")
        except Exception as e:
            print(f"  ❌ Error loading local wi_locness: {e}")
        
        # Source 2: CoNLL-2014 - DISABLED (deprecated dataset scripts)
        # print("\n⏳ Loading CoNLL-2014...")
        # Dataset scripts no longer supported by HuggingFace
        
        # Source 3: FCE corpus - DISABLED (deprecated dataset scripts)
        # print("\n⏳ Loading FCE corpus...")
        # Dataset scripts no longer supported by HuggingFace
        
        # Source 2: Synthetic grammar errors (to fill to 7000)
        print("\n⏳ Generating synthetic grammar errors (diverse, real-text based)...")
        synthetic_target = max(0, target - len(grammar_data)) if not self.allow_exceed else 0
        if synthetic_target > 0:
            # stream real text and inject errors to create diverse incorrect inputs
            error_types = ["DET", "PREP", "SVA", "TENSE", "PLURAL"]
            count = 0
            try:
                stream = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
                for idx, item in enumerate(stream):
                    if count >= synthetic_target:
                        break
                    raw = (item.get("text") or "").strip()
                    if len(raw) < 120 or not looks_english(raw, 0.15):
                        continue
                    base = pick_context(raw, max(1, self.min_context_sentences), max(1, self.max_context_sentences))
                    if len(base.split()) < 10:
                        continue
                    # choose an error injection
                    et = random.choice(error_types)
                    incorrect = base
                    explanation = ""
                    if et == "DET":
                        incorrect = re.sub(r"\b(a|an|the)\b\s+", "", base, count=1, flags=re.IGNORECASE)
                        explanation = "Article (a/an/the) missing"
                    elif et == "PREP":
                        incorrect = re.sub(r"\b(in|on|at|to|for|with)\b", "in", base, count=1, flags=re.IGNORECASE)
                        explanation = "Preposition misuse"
                    elif et == "SVA":
                        incorrect = re.sub(r"\b(he|she|it)\s+have\b", r"\1 have", base, flags=re.IGNORECASE)
                        incorrect = re.sub(r"\b(he|she|it)\s+has\b", r"\1 have", incorrect, flags=re.IGNORECASE)
                        explanation = "Subject-verb agreement error"
                    elif et == "TENSE":
                        incorrect = re.sub(r"\bwent\b", "go", base, flags=re.IGNORECASE)
                        incorrect = re.sub(r"\bwas\b", "is", incorrect, flags=re.IGNORECASE)
                        explanation = "Tense error"
                    elif et == "PLURAL":
                        incorrect = re.sub(r"\b(children|people|men|women)\b", "child", base, flags=re.IGNORECASE)
                        explanation = "Number/pluralization error"

                    incorrect = incorrect.strip()
                    if not incorrect or incorrect == base or len(incorrect.split()) < 6:
                        continue

                    if self._add_sample(
                        grammar_data,
                        "grammar",
                        incorrect,
                        {"corrected": base, "explanation": explanation or f"Synthetic error: {et}"},
                        {"source": "synthetic", "index": idx, "error_type": et, "raw_text": raw, "context_sentences": len(split_sentences(base))},
                    ):
                        count += 1
                print(f"  ✅ Generated synthetic: {count} samples")
            except Exception as e:
                print(f"  ⚠️  Error generating real-text synthetic grammar: {e}")
        
        # Save to JSON
        output_file = self.output_dir / "grammar_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(grammar_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Total grammar samples: {len(grammar_data)}")
        print(f"💾 Saved to: {output_file}")
        
        # Show samples by source
        print("\n📋 Sample data by source:")
        sources = {}
        for item in grammar_data:
            src = item['metadata']['source']
            if src not in sources:
                sources[src] = []
            if len(sources[src]) < 2:
                sources[src].append(item)
        
        for src, samples in sources.items():
            print(f"\n{src.upper()} ({sum(1 for x in grammar_data if x['metadata']['source'] == src)} samples):")
            for i, sample in enumerate(samples):
                print(f"  {i+1}. Input: {sample['input'][:60]}...")
                print(f"     Corrected: {sample['output']['corrected'][:60]}...")
                print(f"     Explanation: {sample['output']['explanation']}")
        
        self.stats['grammar'] = len(grammar_data)
        return grammar_data
    
    def download_vocabulary_data(self) -> List[Dict]:
        """
        Download Vocabulary Classification Dataset
        Target: 2,500 samples
        Sources:
        - Simple Wikipedia (simple English = A2-B1)
        - News articles (complex = B2)
        - SNLI dataset (various complexity levels)
        """
        print("\n" + "="*70)
        print("📊 TASK 3: VOCABULARY CLASSIFICATION")
        print("="*70)
        target = int(self.targets.get("vocabulary", 2500))
        print(f"Target: {target:,} samples")
        print("Sources: Simple Wikipedia + SNLI + News")
        
        vocab_data = []

        def add_from_wikitext(max_to_add: int):
            """Fallback real-text source when wikipedia simple isn't available."""
            try:
                print("\n⏳ [fallback] Loading WikiText for vocabulary passages...")
                stream = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
                added = 0
                for idx, item in enumerate(stream):
                    if added >= max_to_add or len(vocab_data) >= target:
                        break
                    raw = (item.get("text") or "").strip()
                    if len(raw) < 120 or not looks_english(raw, 0.15):
                        continue
                    ctx = pick_context(raw, self.min_context_sentences, self.max_context_sentences)
                    if len(ctx.split()) < 12:
                        continue
                    level = estimate_vocab_level(ctx)
                    if self._add_sample(
                        vocab_data,
                        "vocabulary",
                        ctx,
                        {"level": level, "key_words": f"Real-world text (WikiText); heuristic level={level}"},
                        {"source": "WikiText", "index": idx, "raw_text": raw, "context_sentences": len(split_sentences(ctx))},
                    ):
                        added += 1
                print(f"  ✅ Added {added} samples from WikiText")
            except Exception as e:
                print(f"  ⚠️  Error loading WikiText for vocabulary: {e}")
        
        # Source 1: Simple Wikipedia (A2-B1 level)
        try:
            print("\n⏳ [1/3] Loading Simple Wikipedia...")
            take = min(8000, max(800, int(target * 0.5)))
            wiki_dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{take}]")
            
            for idx, item in enumerate(wiki_dataset):
                text = item['text']
                ctx = pick_context(text, self.min_context_sentences, self.max_context_sentences)
                if len(ctx.split()) > 8:
                    level = estimate_vocab_level(ctx)
                    self._add_sample(
                        vocab_data,
                        "vocabulary",
                        ctx,
                        {"level": level, "key_words": f"Real-world text; heuristic level={level}"},
                        {"source": "Simple Wikipedia", "index": idx, "raw_text": text, "context_sentences": len(split_sentences(ctx))},
                    )
            
            print(f"  ✅ Loaded {sum(1 for x in vocab_data if x['metadata']['source']=='Simple Wikipedia')} samples")
            
        except Exception as e:
            print(f"  ⚠️  Error loading Simple Wikipedia: {e}")
            # Fallback to a supported dataset for real passages
            add_from_wikitext(max_to_add=max(500, int(target * 0.4)))
        
        # Source 2: SNLI dataset (natural language inference - varied complexity)
        try:
            print("\n⏳ [2/3] Loading SNLI dataset...")
            take = min(20000, max(2510, int(target * 0.7)))
            snli_dataset = load_dataset("snli", split=f"train[:{take}]")
            
            for idx, item in enumerate(snli_dataset):
                if len(vocab_data) >= target:
                    break
                premise = item['premise']
                if len(premise.split()) > 6:
                    level = estimate_vocab_level(premise)
                    self._add_sample(
                        vocab_data,
                        "vocabulary",
                        premise,
                        {"level": level, "key_words": f"Caption-like text; heuristic level={level}"},
                        {"source": "SNLI", "index": idx, "word_count": len(premise.split()), "raw_text": premise},
                    )
                
            
            print(f"  ✅ Loaded {sum(1 for x in vocab_data if x['metadata']['source']=='SNLI')} samples")
            
        except Exception as e:
            print(f"  ❌ Error loading SNLI: {e}")
        
        # Add a "real text" news-like source for B2-ish variety (AG News)
        try:
            if len(vocab_data) < target:
                print("\n⏳ [3/3] Loading AG News for longer contexts...")
                ag = load_dataset("ag_news", split="train", streaming=True)
                count = 0
                for idx, item in enumerate(ag):
                    if len(vocab_data) >= target:
                        break
                    text = (item.get("text") or "").strip()
                    if len(text) < 80 or not looks_english(text, 0.1):
                        continue
                    ctx = pick_context(text, max(1, self.min_context_sentences), max(1, self.max_context_sentences))
                    if len(ctx.split()) < 12:
                        continue
                    level = "B2" if estimate_vocab_level(ctx) in ("B1", "B2") else "B1"
                    if self._add_sample(
                        vocab_data,
                        "vocabulary",
                        ctx,
                        {"level": level, "key_words": "News-like text (AG News)"},
                        {"source": "AG News", "index": idx, "raw_text": text, "context_sentences": len(split_sentences(ctx))},
                    ):
                        count += 1
                    if count >= max(300, int(target * 0.2)):
                        break
                print(f"  ✅ Added {count} samples from AG News")
        except Exception as e:
            print(f"  ⚠️  Error loading AG News: {e}")

        # If still short, create synthetic
        if len(vocab_data) < target:
            remaining = target - len(vocab_data)
            print(f"\n⏳ Creating {remaining} synthetic vocabulary samples as fallback...")
            # keep generating until we actually reach target under dedupe
            attempts = 0
            while len(vocab_data) < target and attempts < 10:
                needed = target - len(vocab_data)
                batch = min(needed, max(500, remaining))
                synthetic = self._create_fallback_vocabulary_data_partial(batch)
                before = len(vocab_data)
                for s in synthetic:
                    self._add_sample(vocab_data, "vocabulary", s["input"], s["output"], s.get("metadata", {}))
                gained = len(vocab_data) - before
                attempts += 1
                if gained == 0:
                    # as a last resort, add more real text snippets
                    add_from_wikitext(max_to_add=min(1000, needed))
                    if len(vocab_data) == before:
                        break
        
        print(f"\n✅ Total vocabulary samples: {len(vocab_data)}")
        
        # Save to JSON
        output_file = self.output_dir / "vocabulary_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved to: {output_file}")
        
        self.stats['vocabulary'] = len(vocab_data)
        return vocab_data
    
    def _create_fallback_vocabulary_data_partial(self, count: int) -> List[Dict]:
        """Create specific number of synthetic vocabulary samples"""
        # Diverse templated sentences so dedupe doesn't collapse to ~10 rows
        a2_verbs = ["like", "want", "need", "have", "go", "play", "eat", "watch", "read"]
        a2_nouns = ["apples", "music", "school", "a book", "a movie", "my family", "the park", "coffee"]
        b1_verbs = ["discuss", "decide", "prepare", "improve", "explain", "compare", "organize"]
        b1_nouns = ["the plan", "the idea", "the opportunity", "the problem", "the lesson", "the results"]
        b2_verbs = ["analyze", "implement", "investigate", "demonstrate", "evaluate", "illustrate"]
        b2_nouns = ["the policy", "the implications", "the methodology", "environmental issues", "economic trends"]
        connectors = ["because", "although", "while", "so that", "when"]
        
        vocab_data = []
        for i in range(count):
            r = random.random()
            if r < 0.34:
                level = "A2"
                sentence = f"I {random.choice(a2_verbs)} {random.choice(a2_nouns)} {random.choice(['today','every day','on weekends'])}."
            elif r < 0.67:
                level = "B1"
                sentence = f"We should {random.choice(b1_verbs)} {random.choice(b1_nouns)} {random.choice(connectors)} it is important."
            else:
                level = "B2"
                sentence = f"They will {random.choice(b2_verbs)} {random.choice(b2_nouns)} {random.choice(connectors)} the situation changes."

            words = f"Synthetic template; intended level={level}"
            vocab_data.append({
                "task": "vocabulary",
                "input": sentence,
                "output": {
                    "level": level,
                    "key_words": words
                },
                "metadata": {"source": "synthetic", "index": i}
            })
        
        return vocab_data
        
        # Show source and level distribution
        sources = {}
        levels = {}
        for item in vocab_data:
            src = item['metadata'].get('source', 'unknown')
            level = item['output']['level']
            sources[src] = sources.get(src, 0) + 1
            levels[level] = levels.get(level, 0) + 1
        
        print("\n📊 Source distribution:")
        for src, count in sources.items():
            print(f"  {src}: {count} samples")
        
        print("\n📊 CEFR Level distribution:")
        for level, count in sorted(levels.items()):
            print(f"  {level}: {count} samples ({count/len(vocab_data)*100:.1f}%)")
        
        # Show samples
        print("\n📋 Sample data (first 3):")
        for i in range(min(3, len(vocab_data))):
            sample = vocab_data[i]
            print(f"\n{i+1}. Input: {sample['input'][:70]}...")
            print(f"   Level: {sample['output']['level']}")
            print(f"   Source: {sample['metadata']['source']}")
        
        self.stats['vocabulary'] = len(vocab_data)
        return vocab_data
    
    def download_dialogue_data(self) -> List[Dict]:
        """
        Download Dialogue Generation Dataset
        Target: 4,000 samples
        Sources:
        - OpenOrca (instruction following)
        - Dialogsum (dialogue summarization)
        - Anthropic HH-RLHF (helpful harmless)
        """
        print("\n" + "="*70)
        print("📊 TASK 4: DIALOGUE GENERATION")
        print("="*70)
        target = int(self.targets.get("dialogue", 4000))
        print(f"Target: {target:,} samples")
        print("Sources: OpenOrca + Dialogsum + Anthropic HH")
        
        dialogue_data = []
        
        # Source 1: Open-Orca/OpenOrca dataset (instruction following)
        try:
            print("\n⏳ [1/3] Loading OpenOrca dataset...")
            orca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            
            # Keywords to filter out non-English or translation tasks
            translation_keywords = [
                'translate', 'translation', 'turkish', 'french', 'german', 'spanish', 
                'italian', 'chinese', 'japanese', 'arabic', 'russian', 'portuguese',
                'hindi', 'korean', 'vietnamese', 'polish', 'dutch', 'swedish',
                'translate to', 'translate this', 'said in', 'how do you say',
                'language:', 'Turkish:', 'French:', 'German:', 'Spanish:'
            ]
            
            # allocate roughly 50% of dialogue target to OpenOrca
            limit = max(500, int(target * 0.5))
            count = 0
            for idx, item in enumerate(orca):
                if count >= limit:
                    break
                    
                question = item.get('question', '')
                response = item.get('response', '')
                
                if not question or not response or len(response) < 50:
                    continue
                
                # Filter: Skip translation tasks
                question_lower = question.lower()
                if any(kw in question_lower for kw in translation_keywords):
                    continue
                
                # Filter: Check if English (basic ASCII check + common English words)
                try:
                    # Check for excessive non-ASCII characters (>20% is suspicious)
                    non_ascii = sum(1 for c in question if ord(c) > 127)
                    if non_ascii / len(question) > 0.2:
                        continue
                    
                    non_ascii_resp = sum(1 for c in response if ord(c) > 127)
                    if non_ascii_resp / len(response) > 0.2:
                        continue
                except:
                    continue
                
                # Simulate student context
                fluency = round(random.uniform(0.5, 0.95), 2)
                level = random.choice(['A2', 'B1', 'B2'])
                errors = random.choice(['None', 'Grammar error', 'Vocabulary issue'])
                
                # Increase context length: 200 → 500 for question, 300 → 600 for response
                dialogue_input = f"{question[:500]} | fluency:{fluency} | level:{level} | errors:{errors}"

                self._add_sample(
                    dialogue_data,
                    "dialogue",
                    dialogue_input,
                    {"response": response[:600]},
                    {"source": "OpenOrca", "index": idx, "raw_text": question, "raw_response": response[:1200]},
                )
                count += 1
            
            print(f"  ✅ Loaded {sum(1 for x in dialogue_data if x['metadata']['source']=='OpenOrca')} samples")
            
        except Exception as e:
            print(f"  ⚠️  Error loading OpenOrca: {e}")
        
        # Source 2: knkarthick/dialogsum dataset (dialogue summarization)
        try:
            print("\n⏳ [2/3] Loading Dialogsum dataset...")
            limit = max(300, int(target * 0.35))
            dialogsum = load_dataset("knkarthick/dialogsum", split=f"train[:{limit}]")
            
            for idx, item in enumerate(dialogsum):
                dialogue_text = item.get('dialogue', '')
                summary = item.get('summary', '')
                
                if not dialogue_text or not summary or len(dialogue_text) < 50:
                    continue
                
                # Filter: Check if English (should be mostly ASCII)
                try:
                    non_ascii = sum(1 for c in dialogue_text if ord(c) > 127)
                    if non_ascii / len(dialogue_text) > 0.15:
                        continue
                except:
                    continue
                
                # Extract first turn as student question
                lines = dialogue_text.strip().split('\n')
                student_input = lines[0] if lines else dialogue_text[:150]
                
                # Simulate student context
                fluency = round(random.uniform(0.5, 0.95), 2)
                level = random.choice(['A2', 'B1', 'B2'])
                errors = random.choice(['None', 'Grammar error'])
                
                # Increase context length: 200 → 500, 300 → 600
                dialogue_input = f"{student_input[:500]} | fluency:{fluency} | level:{level} | errors:{errors}"

                self._add_sample(
                    dialogue_data,
                    "dialogue",
                    dialogue_input,
                    {"response": summary[:600]},
                    {"source": "Dialogsum", "index": idx, "raw_text": dialogue_text, "raw_response": summary[:1200]},
                )
            
            print(f"  ✅ Loaded {sum(1 for x in dialogue_data if x['metadata']['source']=='Dialogsum')} samples")
            
        except Exception as e:
            print(f"  ⚠️  Error loading Dialogsum: {e}")
        
        # Source 3: Anthropic/hh-rlhf (helpful harmless)
        try:
            print("\n⏳ [3/3] Loading Anthropic HH-RLHF dataset...")
            limit = max(200, int(target * 0.15))
            hh_rlhf = load_dataset("Anthropic/hh-rlhf", split=f"train[:{limit}]", data_dir="harmless-base")
            
            for idx, item in enumerate(hh_rlhf):
                chosen = item.get('chosen', '')
                
                if not chosen or len(chosen) < 50:
                    continue
                
                # Parse conversation format
                parts = chosen.split('\n\nAssistant:')
                if len(parts) >= 2:
                    human_part = parts[0].replace('\n\nHuman:', '').strip()
                    assistant_part = parts[1].strip()
                    
                    if not human_part or not assistant_part:
                        continue
                    
                    # Filter: Check if English
                    try:
                        non_ascii_human = sum(1 for c in human_part if ord(c) > 127)
                        if non_ascii_human / len(human_part) > 0.15:
                            continue
                    except:
                        continue
                    
                    fluency = round(random.uniform(0.5, 0.95), 2)
                    level = random.choice(['A2', 'B1', 'B2'])
                    errors = random.choice(['None', 'Grammar error'])
                    
                    # Increase context length: 200 → 500, 300 → 600
                    dialogue_input = f"{human_part[:500]} | fluency:{fluency} | level:{level} | errors:{errors}"

                    self._add_sample(
                        dialogue_data,
                        "dialogue",
                        dialogue_input,
                        {"response": assistant_part[:600]},
                        {"source": "Anthropic-HH", "index": idx, "raw_text": human_part, "raw_response": assistant_part[:1200]},
                    )
            
            print(f"  ✅ Loaded {sum(1 for x in dialogue_data if x['metadata']['source']=='Anthropic-HH')} samples")
            
        except Exception as e:
            print(f"  ⚠️  Error loading Anthropic HH: {e}")
        
        # Fill remaining with synthetic if needed
        remaining = max(0, target - len(dialogue_data))
        if remaining > 0:
            print(f"\n⏳ Generating {remaining} synthetic samples to reach target...")
            synthetic = self._create_fallback_dialogue_data_partial(remaining)
            for s in synthetic:
                self._add_sample(dialogue_data, "dialogue", s["input"], s["output"], s.get("metadata", {}))
        
        # Save to JSON
        output_file = self.output_dir / "dialogue_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Total dialogue samples: {len(dialogue_data)}")
        print(f"💾 Saved to: {output_file}")
        
        # Show source distribution
        sources = {}
        for item in dialogue_data:
            src = item['metadata'].get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        print("\n📊 Source distribution:")
        for src, count in sources.items():
            print(f"  {src}: {count} samples")
        
        # Show samples
        print("\n📋 Sample data (first 3):")
        for i in range(min(3, len(dialogue_data))):
            sample = dialogue_data[i]
            print(f"\n{i+1}. Input: {sample['input'][:80]}...")
            print(f"   Response: {sample['output']['response'][:100]}...")
            print(f"   Source: {sample['metadata']['source']}")
        
        self.stats['dialogue'] = len(dialogue_data)
        return dialogue_data
    
    def _create_fallback_dialogue_data_partial(self, count: int) -> List[Dict]:
        """Create specific number of synthetic dialogue samples"""
        examples = [
            ("I like learning English | fluency:0.90 | level:A2 | errors:None",
             "That's wonderful! Your sentence is very clear and natural."),
            ("Yesterday I go to school | fluency:0.65 | level:A2 | errors:Incorrect past tense",
             "Good try! With 'yesterday', we need the past tense: 'went'."),
            ("She don't like coffee | fluency:0.55 | level:A2 | errors:Subject-verb agreement",
             "Almost there! With 'she', we use 'doesn't': 'She doesn't like coffee.'"),
            ("We should discuss the opportunity | fluency:0.92 | level:B1 | errors:None",
             "Excellent! Your sentence structure is perfect and natural."),
            ("The weather is beautiful today | fluency:0.95 | level:A2 | errors:None",
             "Perfect! Grammatically correct and very natural expression."),
            ("I have been study English for two year | fluency:0.58 | level:B1 | errors:Grammar error",
             "Good effort! Let's fix two things: 'studying' (not 'study') and 'years' (not 'year'). Try: 'I have been studying English for two years.'"),
            ("My teacher is very kind and helpful | fluency:0.92 | level:A2 | errors:None",
             "Excellent sentence! You used great adjectives to describe your teacher."),
            ("The book what I read yesterday was interesting | fluency:0.62 | level:B1 | errors:Word choice",
             "Almost perfect! Instead of 'what', use 'that' or 'which': 'The book that I read yesterday was interesting.'"),
            ("Can you help me with my homework | fluency:0.88 | level:A2 | errors:None",
             "Of course! Your question is clear. What do you need help with?"),
            ("I want improve my English speaking | fluency:0.68 | level:B1 | errors:Grammar error",
             "Great goal! Remember to use 'to' before the verb: 'I want to improve my English speaking.'"),
        ]
        
        dialogue_data = []
        for i in range(count):
            input_text, response = examples[i % len(examples)]
            dialogue_data.append({
                "task": "dialogue",
                "input": input_text,
                "output": {
                    "response": response
                },
                "metadata": {"source": "synthetic", "index": i}
            })
        
        return dialogue_data
        for i in range(min(3, len(dialogue_data))):
            sample = dialogue_data[i]
            print(f"\n{i+1}. Input: {sample['input'][:80]}...")
            print(f"   Response: {sample['output']['response'][:100]}...")
            print(f"   Source: {sample['metadata']['source']}")
        
        self.stats['dialogue'] = len(dialogue_data)
        return dialogue_data
    
    def _create_fallback_dialogue_data(self) -> List[Dict]:
        """Create synthetic dialogue data as fallback"""
        examples = [
            ("I like learning English | fluency:0.90 | level:A2 | errors:None",
             "That's wonderful! Your sentence is very clear and natural."),
            ("Yesterday I go to school | fluency:0.65 | level:A2 | errors:Incorrect past tense",
             "Good try! With 'yesterday', we need the past tense: 'went'."),
            ("She don't like coffee | fluency:0.55 | level:A2 | errors:Subject-verb agreement",
             "Almost there! With 'she', we use 'doesn't': 'She doesn't like coffee.'"),
            ("We should discuss the opportunity | fluency:0.92 | level:B1 | errors:None",
             "Excellent! Your sentence structure is perfect and natural."),
            ("The weather is beautiful today | fluency:0.95 | level:A2 | errors:None",
             "Perfect! Grammatically correct and very natural expression."),
        ]
        
        dialogue_data = []
        for i in range(4000):
            input_text, response = examples[i % len(examples)]
            dialogue_data.append({
                "task": "dialogue",
                "input": input_text,
                "output": {
                    "response": response
                },
                "metadata": {"source": "synthetic", "index": i}
            })
        
        output_file = self.output_dir / "dialogue_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created synthetic: {len(dialogue_data)} samples")
        print(f"💾 Saved to: {output_file}")
        self.stats['dialogue'] = len(dialogue_data)
        return dialogue_data
    
    def create_unified_dataset(self, fluency_data, grammar_data, vocab_data, dialogue_data):
        """Combine all tasks into unified training dataset"""
        print("\n" + "="*70)
        print("📦 CREATING UNIFIED DATASET")
        print("="*70)
        
        unified_data = fluency_data + grammar_data + vocab_data + dialogue_data

        # Final dedupe pass (per-task normalized input)
        seen = set()
        deduped = []
        for item in unified_data:
            task = item.get("task", "")
            inp = item.get("input", "")
            k = stable_hash(f"{task}::{normalize_text(inp)}")
            if k in seen:
                continue
            seen.add(k)
            deduped.append(item)
        unified_data = deduped
        
        # Shuffle
        random.shuffle(unified_data)
        
        # Save unified dataset
        output_file = self.output_dir / "unified_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Unified dataset created: {len(unified_data)} samples")
        print(f"💾 Saved to: {output_file}")
        
        # Create CSV for easy inspection
        csv_data = []
        for item in unified_data:
            csv_data.append({
                'task': item['task'],
                'input': item['input'][:100],
                'output': str(item['output'])[:100],
                'source': item['metadata'].get('source', 'unknown')
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "unified_training_data.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"💾 CSV saved to: {csv_file}")
        
        return unified_data
    
    def print_summary(self, unified_data):
        """Print comprehensive summary of downloaded datasets"""
        print("\n" + "="*70)
        print("📊 DOWNLOAD SUMMARY")
        print("="*70)
        
        # Task distribution
        print("\n📋 Task Distribution:")
        print(f"  Fluency:    {self.stats.get('fluency', 0):>5} samples (target: {self.targets.get('fluency', 0):,})")
        print(f"  Grammar:    {self.stats.get('grammar', 0):>5} samples (target: {self.targets.get('grammar', 0):,})")
        print(f"  Vocabulary: {self.stats.get('vocabulary', 0):>5} samples (target: {self.targets.get('vocabulary', 0):,})")
        print(f"  Dialogue:   {self.stats.get('dialogue', 0):>5} samples (target: {self.targets.get('dialogue', 0):,})")
        print(f"  " + "-"*50)
        total = sum(self.stats.values())
        total_target = sum(self.targets.values())
        print(f"  TOTAL:      {total:>5} samples (target: {total_target:,})")
        print(f"  Progress:   {total/max(1,total_target)*100:.1f}% of target")
        
        # Source distribution
        print("\n📊 Source Distribution:")
        sources = {}
        for item in unified_data:
            src = item['metadata'].get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {src:20s}: {count:>5} samples ({count/total*100:.1f}%)")
        
        # Files created
        print("\n📁 Files Created:")
        print(f"  {self.output_dir}/")
        for file in sorted(self.output_dir.glob("*.json")):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"    ├── {file.name:30s} ({size_mb:.2f} MB)")
        for file in sorted(self.output_dir.glob("*.csv")):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"    ├── {file.name:30s} ({size_mb:.2f} MB)")
        
        # Next steps
        print("\n" + "="*70)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*70)
        print("\n🚀 Next Steps:")
        print("  1. Inspect data quality:")
        print(f"     cd {self.output_dir}")
        print("     cat fluency_data.json | jq '.[0:5]'  # View samples")
        print("     cat unified_training_data.csv | head -20  # Quick preview")
        print("")
        print("  2. Upload to Google Drive:")
        print("     - Compress: tar -czf datasets.tar.gz downloaded_datasets/")
        print("     - Upload to Drive: /LexiLingo/training_data/")
        print("")
        print("  3. Load in Colab:")
        print("     from google.colab import drive")
        print("     drive.mount('/content/drive')")
        print("     !tar -xzf '/content/drive/MyDrive/LexiLingo/training_data/datasets.tar.gz'")
        print("     with open('downloaded_datasets/unified_training_data.json') as f:")
        print("         data = json.load(f)")
        print("")
        print("  4. Run training:")
        print("     # Use v1.4 notebook with loaded data")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Download and build LexiLingo finetune datasets (with dedupe + richer context).")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output folder (default: scripts/downloaded_datasets)")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Scale base targets (e.g. 2.0 => ~30k samples)")
    parser.add_argument("--min-context-sentences", type=int, default=1, help="Min sentences per sample context")
    parser.add_argument("--max-context-sentences", type=int, default=2, help="Max sentences per sample context")
    parser.add_argument("--dedupe-global", action="store_true", help="Also dedupe across tasks (stricter; may reduce total)")
    parser.add_argument("--allow-exceed", action="store_true", help="Allow exceeding per-task targets when sources provide more")
    parser.add_argument("--min-input-chars", type=int, default=5, help="Drop samples with very short input strings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {k: max(1, int(v * args.multiplier)) for k, v in BASE_TARGETS.items()}
    total = sum(targets.values())

    print("="*70)
    print("🚀 LexiLingo Dataset Downloader v1.0")
    print("="*70)
    print(f"\nThis run targets ~{total:,} samples:")
    print(f"  • Fluency:    {targets['fluency']:,}")
    print(f"  • Grammar:    {targets['grammar']:,}")
    print(f"  • Vocabulary: {targets['vocabulary']:,}")
    print(f"  • Dialogue:   {targets['dialogue']:,}")
    print("\nSources include: GLUE (CoLA/SST-2), wi_locness (local), WikiText, Wikipedia(Simple), SNLI, AG News, OpenOrca, Dialogsum, Anthropic-HH")
    print("")

    if not args.yes:
        response = input("Continue? [Y/n]: ")
        if response.lower() == 'n':
            print("Aborted.")
            return
    
    # Download all datasets
    downloader = DatasetDownloader(
        out_dir,
        targets=targets,
        min_context_sentences=max(1, args.min_context_sentences),
        max_context_sentences=max(max(1, args.min_context_sentences), args.max_context_sentences),
        dedupe_global=args.dedupe_global,
        allow_exceed=args.allow_exceed,
        min_input_chars=args.min_input_chars,
    )
    
    fluency_data = downloader.download_fluency_data()
    grammar_data = downloader.download_grammar_data()
    vocab_data = downloader.download_vocabulary_data()
    dialogue_data = downloader.download_dialogue_data()
    
    # Create unified dataset
    unified_data = downloader.create_unified_dataset(
        fluency_data, grammar_data, vocab_data, dialogue_data
    )
    
    # Print summary
    downloader.print_summary(unified_data)


if __name__ == "__main__":
    main()
