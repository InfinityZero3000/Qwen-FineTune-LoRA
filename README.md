# LexiLingo - Deep Learning Model Support

## 🎯 Overview

LexiLingo is a unified English learning assistant powered by **Qwen2.5-1.5B-Instruct** with **LoRA adapters**. The model supports **5 core tasks** through a single unified adapter:

1. **Fluency Scoring** - Đánh giá độ trôi chảy của câu (0.0-1.0)
2. **Vocabulary Classification** - Phân loại mức độ từ vựng (A1-C2 CEFR)
3. **Grammar Correction** - Sửa lỗi ngữ pháp
4. **Dialogue Generation** - Tạo hội thoại tương tác
5. **Grammar Explanation** (NEW) - Giải thích lỗi ngữ pháp bằng tiếng Việt

## 📊 Model Architecture

```
Qwen2.5-1.5B-Instruct (Base Model)
    ↓
Unified LoRA Adapter (r=16, α=32)
    ↓
Task Router → 5 Tasks:
    ├─ Fluency Scoring
    ├─ Vocabulary Classification  
    ├─ Grammar Correction
    ├─ Dialogue Generation
    └─ Grammar Explanation (NEW) ← Model đóng vai trò giáo viên
```

## 🆕 What's New: Explanation Task

**Added:** January 27, 2026

The new **Explanation Task** transforms the model into a **Vietnamese tutor** that:
- Explains **WHY** a sentence is grammatically incorrect
- Teaches the **correct form** and underlying **rules**
- Uses **friendly, conversational Vietnamese** (not overly formal)
- Builds **confidence** through encouraging explanations

### Example:
```
User: "Error: 'He go to school yesterday.' → Correct: 'He went to school yesterday.'"

Model: "Khi nói về hành động trong quá khứ (yesterday), động từ phải chia 
ở thì quá khứ đơn. 'Go' là hiện tại, phải đổi thành 'went' nhé em."
```

See [EXPLANATION_TASK.md](docs/EXPLANATION_TASK.md) for full documentation.

## 📈 Training Data Statistics

### After Merging Explanation Task (v2.0):
```
Total Training Samples: 30,806
Total Validation Samples: 1,618

Task Distribution:
- dialogue:     6,649 samples (21.6%)
- explanation:  3,926 samples (12.7%) ← NEW
- fluency:      7,255 samples (23.6%)
- grammar:      5,881 samples (19.1%)
- vocabulary:   7,095 samples (23.0%)
```

### Original Dataset (v1.0):
```
Total Training Samples: 26,880
- dialogue:    6,649 samples (24.7%)
- fluency:     7,255 samples (27.0%)
- grammar:     5,881 samples (21.9%)
- vocabulary:  7,095 samples (26.4%)
```

## 🗂️ Project Structure

```
DL-Model-Support/
├── config/                      # Configuration files
│   ├── llm_config.yaml         # LLM settings
│   ├── stt_config.yaml         # Speech-to-text
│   └── tts_config.yaml         # Text-to-speech
│
├── datasets/
│   ├── cefr/
│   │   └── ENGLISH_CERF_WORDS.csv        # 2,332 CEFR words (A1-C2)
│   ├── datasets/
│   │   ├── dialogue_data.json            # 6,649 dialogue samples
│   │   ├── fluency_data.json             # 7,255 fluency samples
│   │   ├── grammar_data.json             # 5,881 grammar samples
│   │   ├── vocabulary_data.json          # 7,095 vocabulary samples
│   │   ├── vietnamese_explanations.json  # 4,869 explanation samples (NEW)
│   │   ├── train.jsonl                   # Original 26,880 samples
│   │   ├── val.jsonl                     # Original 1,412 samples
│   │   ├── train_with_explanation.jsonl  # NEW: 30,806 samples
│   │   └── val_with_explanation.jsonl    # NEW: 1,618 samples
│   └── wi+locness/              # Wi+locness grammar corpus
│
├── docs/
│   ├── EXPLANATION_TASK.md              # NEW: Explanation task documentation
│   ├── Training_Optimization_Guide.md
│   └── Optimization_Summary.md
│
├── model/
│   ├── adapters/
│   │   ├── dialogue_lora_adapter/
│   │   ├── fluency_lora_adapter/
│   │   ├── grammar_lora_adapter/
│   │   └── vocabulary_lora_adapter/
│   └── outputs/                 # Training outputs
│
├── scripts/
│   ├── crawl_cefr_words.py              # Generate CEFR vocabulary
│   ├── merge_explanation_data.py         # NEW: Merge explanation data
│   ├── finetune_qwen_lora_kaggle.v1.0.ipynb  # Kaggle training notebook
│   └── finetune_qwen_lora.v3.0.ipynb    # Local training notebook
│
├── check_datasets.py            # Dataset analysis tool
├── DATASET_ANALYSIS_REPORT.md   # Dataset comparison report
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/InfinityZero3000/LexiLingo.git
cd LexiLingo/DL-Model-Support

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data with Explanation Task
```bash
# Merge explanation data into training set
python scripts/merge_explanation_data.py

# Check merge results
cat datasets/datasets/merge_explanation_report.json
```

### 3. Train Model

#### Option A: Kaggle (Recommended for Free GPU)
```bash
# Upload to Kaggle:
# 1. Create new notebook
# 2. Upload finetune_qwen_lora_kaggle.v1.0.ipynb
# 3. Add dataset: train_with_explanation.jsonl + val_with_explanation.jsonl
# 4. Enable GPU (P100 or T4)
# 5. Enable Internet
# 6. Run all cells
```

#### Option B: Local Training
```bash
# Open Jupyter notebook
jupyter notebook scripts/finetune_qwen_lora.v3.0.ipynb

# Or run directly
python scripts/train_unified_model.py
```

## 📝 Dataset Management

### Generate/Update CEFR Vocabulary
```bash
python scripts/crawl_cefr_words.py
# Output: datasets/cefr/ENGLISH_CERF_WORDS.csv (2,332 words)
```

### Merge Explanation Task
```bash
python scripts/merge_explanation_data.py

# Parameters (edit in script):
# - quality_threshold: 50 (minimum quality score)
# - val_split_ratio: 0.05 (5% for validation)
```

### Analyze Dataset
```bash
python check_datasets.py

# Shows:
# - Task distribution
# - Sample counts
# - Data format validation
```

## 🎓 Task Descriptions

### 1. Fluency Scoring
Đánh giá độ tự nhiên và trôi chảy của câu tiếng Anh (0.0-1.0)

**Example:**
```
Input: "The cat sits on the mat."
Output: "Fluency Score: 0.95"
```

### 2. Vocabulary Classification
Phân loại độ khó từ vựng theo CEFR (A1-C2)

**Example:**
```
Input: "Classify: sophisticated"
Output: "Vocabulary Level: C2"
```

### 3. Grammar Correction
Sửa lỗi ngữ pháp trong câu

**Example:**
```
Input: "He go to school yesterday."
Output: "He went to school yesterday."
```

### 4. Dialogue Generation
Tạo hội thoại tự nhiên, hỗ trợ học viên

**Example:**
```
Input: "User: What's the weather like?"
Output: "I'd be happy to help, but I don't have access to real-time data..."
```

### 5. Grammar Explanation (NEW)
Giải thích lỗi ngữ pháp bằng tiếng Việt như một giáo viên

**Example:**
```
Input: "Error: 'She have a car.' → Correct: 'She has a car.'"
Output: "Với chủ ngữ số ít 'She' (ngôi thứ 3 số ít), động từ 'have' phải 
chia thành 'has' trong thì hiện tại đơn nhé em. Quy tắc: He/She/It + has, 
còn I/You/We/They + have."
```

## 🔧 Configuration

### Training Hyperparameters
```python
# LoRA Configuration
r = 16                    # LoRA rank
lora_alpha = 32          # LoRA alpha
lora_dropout = 0.05      # Dropout rate

# Training Configuration
batch_size = 4           # Per device
gradient_accumulation = 4
learning_rate = 2e-4
num_epochs = 3
warmup_ratio = 0.03

# Quantization
load_in_4bit = True      # 4-bit quantization
bnb_4bit_quant_type = "nf4"
```

### Model Configuration
```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

## 📊 Performance Metrics

### Expected Results (After Full Training):

| Task | Metric | Target |
|------|--------|--------|
| Fluency | MAE | < 0.15 |
| Vocabulary | Accuracy | > 85% |
| Grammar | BLEU | > 0.70 |
| Dialogue | Perplexity | < 10.0 |
| Explanation | Human Eval | > 80% satisfaction |

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)
```bash
# Reduce batch size
batch_size = 2
gradient_accumulation = 8

# Or use gradient checkpointing
gradient_checkpointing = True
```

### Issue: Dataset Not Found
```bash
# Check paths
ls -la datasets/datasets/train_with_explanation.jsonl

# Re-run merge if needed
python scripts/merge_explanation_data.py
```

### Issue: Quality Threshold Too High
```bash
# Edit merge_explanation_data.py
quality_threshold = 40  # Lower threshold

# Re-merge
python scripts/merge_explanation_data.py
```

## 📚 Documentation

- [Explanation Task Documentation](docs/EXPLANATION_TASK.md)
- [Training Optimization Guide](docs/Training_Optimization_Guide.md)
- [Dataset Analysis Report](DATASET_ANALYSIS_REPORT.md)
- [Architecture Diagram](architecture.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is part of the LexiLingo ecosystem.

## 👥 Team

- **Development:** LexiLingo Team
- **Contact:** [GitHub Repository](https://github.com/InfinityZero3000/LexiLingo)

## 🔮 Roadmap

- [x] Unified LoRA adapter for 4 tasks
- [x] CEFR vocabulary integration (2,332 words)
- [x] Wi+locness grammar corpus
- [x] Grammar Explanation task (Vietnamese tutor)
- [ ] Real-time inference API
- [ ] Mobile model export (ONNX)
- [ ] Multi-language explanation support
- [ ] Advanced error type classification
- [ ] Difficulty-based task routing (A1-C2)

---

**Last Updated:** January 27, 2026  
**Version:** 2.0 (with Explanation Task)  
**Model:** Qwen2.5-1.5B-Instruct + Unified LoRA
