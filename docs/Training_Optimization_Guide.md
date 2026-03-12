# 🚀 LexiLingo Training Optimization Guide

## Tổng Quan

Document này cung cấp các chiến lược tối ưu hóa để training model LexiLingo nhanh hơn và hiệu quả hơn, đặc biệt trên hardware hạn chế như Google Colab T4 GPU.

---

## ⚡ Quick Wins - Cải Thiện Ngay

### 1. Chọn Model Size Phù Hợp

| Model | Parameters | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **Qwen2.5-0.5B** | 0.5B | ⚡⚡⚡ | ⭐⭐⭐ | **Prototyping, Fast iteration** |
| Qwen2.5-1.5B | 1.5B | ⚡⚡ | ⭐⭐⭐⭐ | Production (balanced) |
| Qwen2.5-3B | 3B | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

**Recommendation**: Dùng **0.5B** cho development, **1.5B** cho production.

### 2. Tối Ưu LoRA Configuration

```python
# ❌ BAD: High rank = slow training
LORA_CONFIG = {
    "r": 64,           # Too high!
    "lora_alpha": 128,
}

# ✅ GOOD: Low rank = fast training
LORA_CONFIG = {
    "r": 16,           # Sweet spot
    "lora_alpha": 32,  # 2 × r
    "lora_dropout": 0.1,
}
```

**Impact**: Reducing rank từ 64 → 16 giảm **75% trainable parameters** và tăng tốc ~**40-50%**.

### 3. Giảm Sequence Length

```python
# Trade-off analysis
MAX_SEQ_LENGTH = 256  # Very fast, may truncate some inputs
MAX_SEQ_LENGTH = 512  # ✅ Balanced (recommended)
MAX_SEQ_LENGTH = 768  # Slower, covers 95%+ inputs
MAX_SEQ_LENGTH = 1024 # Slow, rarely needed
```

**Rule**: Phân tích input lengths trong dataset → chọn length cover 90-95% data.

### 4. Tăng Batch Size, Giảm Gradient Accumulation

```python
# ❌ BAD: Small batch + high accumulation
per_device_train_batch_size = 1
gradient_accumulation_steps = 24

# ✅ GOOD: Larger batch + less accumulation
per_device_train_batch_size = 4
gradient_accumulation_steps = 6
# Effective batch = 24 (same), but 4× faster per step!
```

**Key**: Maximize `batch_size` để tận dụng GPU parallelism, giữ `effective_batch = batch_size × grad_accum` constant.

---

## 🎯 Advanced Optimizations

### 5. Mixed Precision Training

```python
# fp32: Slowest, most accurate
fp16 = False
bf16 = False

# fp16: Fast, risk of NaN (older GPUs)
fp16 = True
bf16 = False

# bf16: Fast, stable (Ampere+ GPUs like A100)
fp16 = False
bf16 = True
```

**Speedup**: 2-3× faster vs fp32, ít memory hơn ~50%.

### 6. Optimizer Choice

```python
# Ranked by speed (fast → slow)
optim = "paged_adamw_8bit"   # ✅ Fastest, least memory
optim = "paged_adamw_32bit"  # Balanced
optim = "adamw_torch"        # Standard
optim = "adafactor"          # Memory-efficient but slow
```

### 7. Gradient Checkpointing

```python
gradient_checkpointing = True  # ✅ Enable
```

**Trade-off**: Giảm **40-50% memory** nhưng tăng **20% training time**. Worth it nếu bị OOM.

### 8. DataLoader Optimization

```python
# Tune này dựa trên CPU cores
dataloader_num_workers = 2   # Colab/T4 (2 vCPUs)
dataloader_num_workers = 4   # Local workstation (8+ cores)
dataloader_num_workers = 0   # Debug (sequential)

dataloader_prefetch_factor = 2  # Prefetch batches
```

### 9. Giảm Logging & Saving Overhead

```python
# ❌ TOO FREQUENT
logging_steps = 1
save_steps = 10

# ✅ REASONABLE
logging_steps = 10
save_steps = 150  # Every 150 steps (~5-10 minutes)
```

---

## 📊 Configuration Presets

### Preset 1: 🚀 Maximum Speed (Prototyping)

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 384

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
}

TRAINING_CONFIG = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 3,
    "learning_rate": 5e-4,
    "optim": "paged_adamw_8bit",
    "save_steps": 200,
}
```

**Use case**: Quick experiments, testing pipeline.

**Expected time**: ~1-2 hours trên T4.

---

### Preset 2: ⚖️ Balanced (Recommended)

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 512

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

TRAINING_CONFIG = {
    "num_train_epochs": 4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 6,
    "learning_rate": 3e-4,
    "optim": "paged_adamw_8bit",
    "save_steps": 150,
}
```

**Use case**: Production training với good speed/quality trade-off.

**Expected time**: ~3-4 hours trên T4.

---

### Preset 3: 🎯 Maximum Quality (Production)

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 768

LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
}

TRAINING_CONFIG = {
    "num_train_epochs": 6,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 12,
    "learning_rate": 2e-4,
    "optim": "paged_adamw_32bit",
    "save_steps": 100,
}
```

**Use case**: Final production model.

**Expected time**: ~10-12 hours trên T4.

---

## 💾 Memory Optimization Strategies

### Strategy 1: Gradient Accumulation

Khi bị OOM với batch_size lớn:

```python
# Instead of:
per_device_train_batch_size = 8  # OOM!

# Use:
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
# Effective batch = 2 × 4 = 8 (same result)
```

### Strategy 2: Sequence Length Cap

```python
# Analyze your dataset first
from datasets import load_dataset

dataset = load_dataset(...)
lengths = [len(tokenizer.encode(x["text"])) for x in dataset]

# Percentiles
p50 = sorted(lengths)[len(lengths)//2]
p90 = sorted(lengths)[int(len(lengths)*0.9)]
p95 = sorted(lengths)[int(len(lengths)*0.95)]

print(f"50th percentile: {p50}")  # Median
print(f"90th percentile: {p90}")  # Most inputs
print(f"95th percentile: {p95}")  # Safe choice

# Choose MAX_SEQ_LENGTH based on p90 or p95
```

### Strategy 3: Model Quantization

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization (extreme memory saving)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
)
```

**Impact**: Giảm memory ~75%, có thể train 1.5B model trên 8GB GPU!

---

## 📈 Monitoring & Debugging

### Training Speed Metrics

```python
# Add to training loop
import time

start_time = time.time()
# ... training ...
elapsed = time.time() - start_time

samples_per_second = total_samples / elapsed
print(f"Throughput: {samples_per_second:.2f} samples/sec")
```

### Expected Benchmarks (T4 GPU)

| Model | Seq Len | Batch | Samples/sec | Memory |
|-------|---------|-------|-------------|--------|
| 0.5B | 512 | 4 | ~15-20 | 5-6 GB |
| 0.5B | 768 | 2 | ~8-10 | 7-8 GB |
| 1.5B | 512 | 2 | ~6-8 | 9-10 GB |
| 1.5B | 768 | 2 | ~4-5 | 12-13 GB |

### OOM Troubleshooting Checklist

❌ **OOM Error?** Try in order:

1. ✅ Enable `gradient_checkpointing = True`
2. ✅ Reduce `per_device_train_batch_size` (8→4→2→1)
3. ✅ Reduce `MAX_SEQ_LENGTH` (768→512→384)
4. ✅ Reduce LoRA `r` (48→32→16→8)
5. ✅ Use 8-bit optimizer (`paged_adamw_8bit`)
6. ✅ Reduce `dataloader_num_workers` (4→2→0)
7. ✅ Use smaller model (1.5B→0.5B)
8. ✅ Enable 4-bit quantization

---

## 🔧 Colab-Specific Tips

### 1. Prevent Disconnects

```python
# Install colab keep-alive
!pip install -q colabcode

from colabcode import ColabCode
ColabCode(port=10000)
```

### 2. Monitor GPU Usage

```python
# Check GPU memory
!nvidia-smi

# Or in Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### 3. Auto-Save to Drive

```python
# ALWAYS mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# Use Drive path for checkpoints
OUTPUT_DIR = "/content/drive/MyDrive/LexiLingo/models"
```

### 4. Resume Training After Disconnect

```python
# Auto-detect latest checkpoint
checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
if checkpoints:
    latest = str(checkpoints[-1])
    print(f"Resuming from: {latest}")
    trainer.train(resume_from_checkpoint=latest)
else:
    print("Starting fresh training")
    trainer.train()
```

---

## 📚 References & Further Reading

1. **Hugging Face Training Docs**: https://huggingface.co/docs/transformers/training
2. **LoRA Paper**: https://arxiv.org/abs/2106.09685
3. **QLoRA (4-bit)**: https://arxiv.org/abs/2305.14314
4. **Efficient Training**: https://huggingface.co/docs/transformers/perf_train_gpu_one

---

## 📝 Summary Checklist

Trước khi training, check:

- [ ] Đã chọn model size phù hợp (0.5B cho prototyping)?
- [ ] `MAX_SEQ_LENGTH` cover 90%+ inputs?
- [ ] LoRA rank <= 32 (16 recommended)?
- [ ] `per_device_train_batch_size` tối đa GPU cho phép?
- [ ] `gradient_accumulation_steps` tính đúng effective batch?
- [ ] Đã enable `gradient_checkpointing`?
- [ ] Đã chọn optimizer tốt (`paged_adamw_8bit`)?
- [ ] Save path vào Google Drive (nếu Colab)?
- [ ] `save_steps` không quá nhỏ (>= 100)?

---

**Last updated**: January 2026
**Author**: LexiLingo Team
