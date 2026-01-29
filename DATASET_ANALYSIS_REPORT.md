# Dataset Analysis Report
**Generated:** January 27, 2026
**Status:** ⚠️ Data Inconsistency Detected

---

## 📊 Overview

Có 2 folder datasets với dữ liệu khác nhau:

### 1. `datasets/datasets/` (LARGER - 26,880 samples)
```
├── train.jsonl          (26,880 samples)
├── val.jsonl            (1,412 samples)
├── vocabulary_data.json (2.1M)
├── vietnamese_explanations.json (2.5M)
└── unified_training_data.csv
```

### 2. `datasets/downloaded_datasets/` (SMALLER - 14,250 samples)
```
├── train.jsonl          (14,250 samples)
├── val.jsonl            (750 samples)
├── dialogue_data.json   (6.9M)
├── fluency_data.json    (693K)
├── grammar_data.json    (4.0M)
├── vocabulary_data.json (2.1M)
├── unified_training_data.json (14M)
├── unified_training_data.csv
└── split_report.json    (1.7K)
```

---

## 🔍 Task Distribution Comparison

| Task | datasets/datasets | downloaded_datasets | Expected (Architecture) | Status |
|------|-------------------|---------------------|------------------------|--------|
| **dialogue** | 6,649 | 3,805 | 3,500 | ✅ Sufficient |
| **fluency** | 7,255 | 1,426 | 1,500 | ⚠️ Mixed |
| **grammar** | 5,881 | 6,657 | 9,200 | ❌ Insufficient |
| **vocabulary** | 7,095 | 2,362 | 2,500 | ✅ Sufficient |
| **TOTAL** | **26,880** | **14,250** | **16,700** | ⚠️ |

---

## 🚨 Issues Detected

### Issue 1: Two Different Datasets
- **datasets/datasets/**: 26,880 samples (LARGER)
- **downloaded_datasets/**: 14,250 samples (SMALLER)
- **Problem**: Không rõ nên dùng folder nào cho training

### Issue 2: Grammar Data Insufficient
- Expected: 9,200 samples
- Current (datasets): 5,881 samples (64% of expected)
- Current (downloaded): 6,657 samples (72% of expected)
- **Missing**: ~2,500 - 3,300 grammar samples

### Issue 3: Data Format Inconsistency
**datasets/datasets/train.jsonl:**
```json
{"task": "fluency", "messages": [...], "metadata": {...}}
```

**downloaded_datasets/train.jsonl:**
```json
{"task": "vocabulary", "input": "...", "output": {...}, "metadata": {...}}
```

Different formats suggest different processing stages!

---

## 💡 Analysis

### downloaded_datasets appears to be the ORIGINAL source data:
- Has `split_report.json` showing data preparation details
- Has individual task JSON files (dialogue_data.json, fluency_data.json, etc.)
- Consistent with architecture's expected numbers
- Date: Created from 15,000 raw samples → cleaned to 14,250 train + 750 val

### datasets/datasets appears to be AUGMENTED/EXPANDED:
- Almost 2x more samples (26,880 vs 14,250)
- Has Vietnamese explanations data
- May include additional synthetic or augmented data
- No split report or source JSON files

---

## 📝 Recommendations

### Option 1: Use `downloaded_datasets` (SAFER)
✅ **Pros:**
- Clean, documented dataset with split_report
- Closer to architecture specifications
- Has source JSON files for debugging
- Proper train/val split with no leakage

❌ **Cons:**
- Grammar data still insufficient (6,657 vs 9,200 expected)
- Fewer samples overall

### Option 2: Use `datasets/datasets` (MORE DATA)
✅ **Pros:**
- More training samples (26,880 vs 14,250)
- Better for model generalization
- Has Vietnamese explanations

❌ **Cons:**
- Unknown data source/processing
- No documentation of how data was created
- Possible data quality issues
- May contain duplicates or low-quality samples

### Option 3: MERGE & DEDUPLICATE (RECOMMENDED)
1. Use `downloaded_datasets` as base (clean data)
2. Add missing grammar samples from `datasets/datasets`
3. Deduplicate across both sources
4. Document the merged dataset

---

## 🎯 Action Items

### Immediate Actions:
1. ✅ **CEFR data recovered** (2,332 words)
2. ⚠️ **Determine which dataset to use for training**
3. ❌ **Download/generate more grammar correction samples** (~3,000 needed)

### Short-term Actions:
1. Verify data quality in both folders
2. Check for duplicates between folders
3. Document data sources and processing steps
4. Create unified dataset with proper documentation

### Data Sources for Missing Grammar Samples:
- BEA-2019 dataset (W&I+LOCNESS)
- FCE corpus (already have wi+locness folder)
- CoNLL-2014 test set
- NUCLE corpus
- Generate synthetic errors from clean text

---

## 🔧 Suggested Script to Merge & Clean

```python
# merge_datasets.py
import json
from collections import defaultdict

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# Load both datasets
ds1 = load_jsonl('datasets/datasets/train.jsonl')
ds2 = load_jsonl('datasets/downloaded_datasets/train.jsonl')

# Deduplicate by content hash
seen = set()
merged = []
for item in ds1 + ds2:
    # Create hash from input content
    if 'messages' in item:
        content = item['messages'][0]['content']
    else:
        content = item['input']
    
    hash_key = hash(content)
    if hash_key not in seen:
        seen.add(hash_key)
        merged.append(item)

# Group by task
by_task = defaultdict(list)
for item in merged:
    by_task[item['task']].append(item)

# Report
print(f"Total unique samples: {len(merged)}")
for task, items in sorted(by_task.items()):
    print(f"  {task}: {len(items)}")
```

---

## 📌 Conclusion

**Current Status:** ⚠️ **PARTIALLY RECOVERED**

- ✅ CEFR data: Recovered (2,332 words)
- ✅ Training data: Available but inconsistent
- ⚠️ Grammar data: Insufficient (~3,000 samples short)
- ❌ Dataset documentation: Missing

**Recommended Next Steps:**
1. Clarify which dataset folder is authoritative
2. Merge and deduplicate if needed  
3. Download additional grammar correction data
4. Create comprehensive dataset documentation
5. Run validation on final merged dataset before training
