# ⚡ Quick Optimization Reference Card

## 🎯 TL;DR - Muốn Training Nhanh?

```python
# Copy-paste config này vào notebook:

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 3x faster than 1.5B
MAX_SEQ_LENGTH = 512  # Sweet spot

UNIFIED_LORA_CONFIG = {
    "r": 16,           # 66% fewer params than r=48
    "lora_alpha": 32,  # 2 × r
    "lora_dropout": 0.1,
}

TRAINING_CONFIG = {
    "num_train_epochs": 4,                  # Enough for 0.5B model
    "per_device_train_batch_size": 4,       # 2x larger
    "gradient_accumulation_steps": 6,       # 2x less
    "learning_rate": 3e-4,                  # Higher for small model
    "optim": "paged_adamw_8bit",           # Faster optimizer
    "gradient_checkpointing": True,         # Save memory
    "save_steps": 150,                      # Less I/O
}
```

**Result**: ~**60-70% faster** training (10h → 3-4h trên T4)

---

## 📊 Performance Matrix

| Scenario | Model | Seq Len | LoRA r | Batch | Time (T4) | Quality |
|----------|-------|---------|--------|-------|-----------|---------|
| **🚀 Fastest** | 0.5B | 384 | 8 | 8 | ~2h | ⭐⭐⭐ |
| **⚡ Fast** | 0.5B | 512 | 16 | 4 | ~3-4h | ⭐⭐⭐⭐ |
| **⚖️ Balanced** | 0.5B | 768 | 24 | 2 | ~5-6h | ⭐⭐⭐⭐ |
| **🎯 Quality** | 1.5B | 512 | 32 | 2 | ~7-8h | ⭐⭐⭐⭐⭐ |
| **💎 Best** | 1.5B | 768 | 48 | 2 | ~10-12h | ⭐⭐⭐⭐⭐ |

---

## 🔧 Quick Fixes

### Problem: Training quá lâu
```python
# ❌ Before
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
num_train_epochs = 7
LORA_r = 48

# ✅ After
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
num_train_epochs = 4
LORA_r = 16
# → 3x faster
```

### Problem: OOM (Out of Memory)
```python
# Try in order:
gradient_checkpointing = True           # +50% memory
per_device_train_batch_size = 2         # ÷2 memory
MAX_SEQ_LENGTH = 512                   # -30% memory
LORA_r = 16                            # -50% params
optim = "paged_adamw_8bit"             # -60% optimizer memory
```

### Problem: Colab disconnect mất checkpoint
```python
# Mount Drive TRƯỚC KHI training
from google.colab import drive
drive.mount('/content/drive')

OUTPUT_DIR = "/content/drive/MyDrive/LexiLingo/models"
# → Checkpoints sẽ persist qua disconnects
```

---

## 💡 Pro Tips

### Tip 1: Batch Size Calculator
```python
# Target effective batch = 24
per_device = 4
grad_accum = 24 // per_device  # = 6

# If OOM, reduce per_device and increase grad_accum:
per_device = 2
grad_accum = 24 // per_device  # = 12
```

### Tip 2: Sequence Length Analyzer
```python
# Analyze dataset first
lengths = [len(tokenizer.encode(x["text"])) for x in dataset]
p95 = sorted(lengths)[int(len(lengths) * 0.95)]
MAX_SEQ_LENGTH = p95  # Cover 95% inputs
```

### Tip 3: Speed Benchmark
```python
# Add to notebook
import time
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"Training took: {elapsed/3600:.2f}h")
```

---

## 🎮 Hardware Guide

### T4 GPU (15GB) - Google Colab Free
```python
# Recommended config
MODEL = "0.5B"
SEQ_LEN = 512
BATCH = 4
LORA_R = 16
# Expected: 3-4h for full training
```

### T4 GPU - Pushing Limits
```python
MODEL = "1.5B"
SEQ_LEN = 512
BATCH = 2
LORA_R = 24
gradient_checkpointing = True
# Expected: 7-8h
```

### A100 GPU (40GB) - Colab Pro
```python
# Go wild!
MODEL = "1.5B"
SEQ_LEN = 1024
BATCH = 8
LORA_R = 64
# Expected: 2-3h
```

### CPU/MPS (Local Mac)
```python
# Don't even try large models
MODEL = "0.5B"
SEQ_LEN = 256
BATCH = 1
LORA_R = 8
num_epochs = 2
# Expected: 10-20h (very slow!)
```

---

## 🚦 Decision Tree

```
Bạn có bao nhiêu thời gian?
│
├─ < 3 hours → Use "🚀 Fastest" preset
│   └─ Quality OK cho testing
│
├─ 3-5 hours → Use "⚡ Fast" preset (RECOMMENDED)
│   └─ Best balance!
│
├─ 5-8 hours → Use "⚖️ Balanced" preset
│   └─ Better quality
│
└─ > 8 hours → Use "🎯 Quality" or "💎 Best"
    └─ Production-ready model
```

---

## ✅ Pre-Training Checklist

Copy vào notebook:

```python
# Run this before training
print("Pre-training checklist:")
print(f"✓ Model: {MODEL_NAME}")
print(f"✓ Seq length: {MAX_SEQ_LENGTH}")
print(f"✓ LoRA rank: {UNIFIED_LORA_CONFIG['r']}")
print(f"✓ Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
print(f"✓ Grad accum: {TRAINING_CONFIG['gradient_accumulation_steps']}")
print(f"✓ Epochs: {TRAINING_CONFIG['num_train_epochs']}")
print(f"✓ Optimizer: {TRAINING_CONFIG['optim']}")
print(f"✓ Checkpointing: {TRAINING_CONFIG['gradient_checkpointing']}")
print(f"✓ Output: {TRAINING_CONFIG['output_dir']}")

# Estimate time
effective_batch = (TRAINING_CONFIG['per_device_train_batch_size'] * 
                   TRAINING_CONFIG['gradient_accumulation_steps'])
total_steps = (18400 / effective_batch) * TRAINING_CONFIG['num_train_epochs']
print(f"\nEstimated steps: {total_steps:.0f}")
print(f"Estimated time: ~{total_steps * 0.5 / 3600:.1f}h on T4")
```

---

## 📱 Mobile App Link

Document này complement with:
- [Training_Optimization_Guide.md](../docs/Training_Optimization_Guide.md) - Chi tiết đầy đủ
- [architecture.md](../architecture.md) - System design
- [finetune_qwen_lora.v3.0.ipynb](./finetune_qwen_lora.v3.0.ipynb) - Notebook chính

---

**Keep this file open** khi training để quick reference!
