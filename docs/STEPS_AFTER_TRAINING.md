# 📋 Các Bước Sau Khi Training (After Training Checklist)

Bạn đã cài llama.cpp ✅

**Ngoài training, bạn cần:**

```
1. Merge LoRA Adapter  (Kaggle/Colab)
   ↓
2. Download Merged Model  (Kaggle/Colab → Mac)
   ↓
3. Convert to GGUF F16  (Mac with llama.cpp)
   ↓
4. Quantize to Q4_K_M  (Mac with llama.cpp)
   ↓
5. Test Inference  (Mac with llama.cpp)
   ↓
6. Deploy with Server  (Mac with llama.cpp)
   ↓
7. Use Python Client  (Any machine)
```

---

## ✅ Checklist: Bạn đã có gì?

- ✅ **llama.cpp cài đặt** - Bạn vừa cài
- ✅ **Training đã hoàn thành** - Trên Kaggle/Colab
- ❓ **Merged model** - Need to create
- ❓ **GGUF F16** - Need to convert
- ❓ **GGUF Q4_K_M** - Need to quantize
- ❓ **Server running** - Need to deploy

---

## 🎯 Bước 1: Merge LoRA Adapter (Trên Kaggle/Colab)

### Nếu training vừa xong:

```python
# Chạy cell mới trong notebook
from unsloth import FastLanguageModel

# Merge LoRA adapter với base model
model.save_pretrained_merged(
    "/kaggle/working/lexilingo_qwen25_1.5b_merged",
    tokenizer,
    save_method="merged_16bit"  # ← QUAN TRỌNG
)

print("✅ Merged model saved!")
```

**Nếu training đã kết thúc lâu:**

```python
# Load model từ checkpoint
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./kaggle/working/unified_model/checkpoint-xxx",
    # hoặc từ HuggingFace nếu đã push
)

# Unmerge LoRA để lấy full model
model = model.merge_and_unload()

# Hoặc dùng Unsloth's merge
model.save_pretrained_merged(
    "/kaggle/working/lexilingo_qwen25_1.5b_merged",
    tokenizer,
    save_method="merged_16bit"
)
```

**Output:** `lexilingo_qwen25_1.5b_merged/` (~3GB)

---

## 📥 Bước 2: Download Merged Model (Kaggle Output)

### Option A: HuggingFace (Recommended)

```python
# Chạy trong Kaggle notebook cell
from huggingface_hub import login

# Thêm HF token vào Kaggle Secrets trước
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

login(token=hf_token)

# Push to HuggingFace
model.push_to_hub("your-username/lexilingo-qwen25-1.5b", private=True)
tokenizer.push_to_hub("your-username/lexilingo-qwen25-1.5b", private=True)

print("✅ Model uploaded to HuggingFace!")
print("URL: https://huggingface.co/your-username/lexilingo-qwen25-1.5b")
```

### Option B: Zip từ Kaggle Output

```python
# Kaggle UI:
# 1. Right panel → "Output" section
# 2. Find "lexilingo_qwen25_1.5b_merged"
# 3. Click Download
```

**Hoặc tạo zip:**

```python
import zipfile
import os
from pathlib import Path

zip_path = "/kaggle/working/lexilingo_merged.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in Path("/kaggle/working/lexilingo_qwen25_1.5b_merged").rglob('*'):
        if file_path.is_file():
            arcname = file_path.relative_to("/kaggle/working")
            zipf.write(file_path, arcname)

print(f"✅ Zip created: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path) / (1024**3):.1f} GB")
```

---

## 🖥️ Bước 3: Setup trên Mac (Local)

### 3.1: Download merged model

**Nếu dùng HuggingFace:**

```bash
# Create models directory
cd ~/Projects/llama.cpp
mkdir -p models

# Download từ HuggingFace
huggingface-cli download your-username/lexilingo-qwen25-1.5b \
    --local-dir ./models/lexilingo_merged \
    --local-dir-use-symlinks False

# Verify
ls -lh models/lexilingo_merged/
# Should see:
#   config.json
#   tokenizer.json
#   model.safetensors (or model-*.safetensors)
```

**Nếu dùng Zip từ Kaggle:**

```bash
# Extract zip
cd ~/Projects/llama.cpp/models
unzip ~/Downloads/lexilingo_merged.zip

# Should create: lexilingo_qwen25_1.5b_merged/
ls -lh lexilingo_qwen25_1.5b_merged/
```

### 3.2: Verify model files

```bash
cd ~/Projects/llama.cpp/models/lexilingo_qwen25_1.5b_merged

# Check required files
ls -lh

# Expected output:
# config.json (5-10 KB)
# generation_config.json (1-2 KB)
# model.safetensors (3 GB) ← IMPORTANT!
# tokenizer.json (500 KB)
# tokenizer_config.json (500 B)
```

---

## 🔄 Bước 4: Convert Merged Model to GGUF F16

**Merged model** → **GGUF F16** (lossless conversion)

### 4.1: Run conversion script

```bash
cd ~/Projects/llama.cpp

# Convert to GGUF F16
python3 convert_hf_to_gguf.py \
    ./models/lexilingo_qwen25_1.5b_merged/ \
    --outfile ./models/lexilingo_f16.gguf \
    --outtype f16

# Nếu có lỗi "Unsupported model architecture":
# 1. Update llama.cpp: git pull && make
# 2. Hoặc thêm flag: --model-name qwen
```

### 4.2: Verify GGUF file

```bash
# Check file size
ls -lh models/lexilingo_f16.gguf
# Expected: ~3.0 GB

# Verify file format
file models/lexilingo_f16.gguf
# Should show: data (GGUF format)

# Or use llama.cpp to check:
./llama-cli -m ./models/lexilingo_f16.gguf --version
```

**Output:** `lexilingo_f16.gguf` (~3.0 GB)

---

## 📦 Bước 5: Quantize to Q4_K_M

**GGUF F16** → **GGUF Q4_K_M** (3x compression, <2% quality loss)

```bash
cd ~/Projects/llama.cpp

# Quantize
./llama-quantize \
    ./models/lexilingo_f16.gguf \
    ./models/lexilingo_q4_km.gguf \
    Q4_K_M

# Output:
# main: quantizing './models/lexilingo_f16.gguf' to './models/lexilingo_q4_km.gguf'
# main: model size  = 3000.00 MB
# main: quant size  = 980.00 MB
# main: total time  = 45.23 s
# ✅ Done!
```

### Alternative quantization levels:

| Format | Size | Quality | Speed | Use |
|--------|------|---------|-------|-----|
| Q4_K_M | 1.0 GB | ⭐⭐⭐⭐ | ⚡⚡⚡ | **BEST** |
| Q4_K_S | 0.9 GB | ⭐⭐⭐ | ⚡⚡⚡⚡ | Faster |
| Q5_K_M | 1.2 GB | ⭐⭐⭐⭐⭐ | ⚡⚡ | Higher quality |
| Q8_0 | 2.0 GB | ⭐⭐⭐⭐⭐ | ⚡ | Lossless |

**Output:** `lexilingo_q4_km.gguf` (~1.0 GB)

---

## 🧪 Bước 6: Test Inference

### 6.1: Quick test

```bash
cd ~/Projects/llama.cpp

# Test model
./llama-cli \
    -m ./models/lexilingo_q4_km.gguf \
    -p "Analyze the fluency of this sentence: The cat sat on the mat." \
    -n 64 \
    --temp 0.7

# Expected: Model outputs fluency analysis
```

### 6.2: Test all 5 tasks

```bash
#!/bin/bash
# test_lexilingo.sh

MODEL="./models/lexilingo_q4_km.gguf"

echo "Testing LexiLingo Model"
echo "========================"

# Task 1: Fluency
echo -e "\n[1] Fluency Analysis:"
./llama-cli -m $MODEL \
    -p "Analyze the fluency of this sentence: The cat sat on the mat." \
    -n 32 --log-disable

# Task 2: Vocabulary
echo -e "\n[2] Vocabulary Classification:"
./llama-cli -m $MODEL \
    -p "Classify the vocabulary level: The phenomenon is fascinating." \
    -n 16 --log-disable

# Task 3: Grammar
echo -e "\n[3] Grammar Correction:"
./llama-cli -m $MODEL \
    -p "Correct this sentence: She don't like apples." \
    -n 64 --log-disable

# Task 4: Dialogue
echo -e "\n[4] Dialogue Generation:"
./llama-cli -m $MODEL \
    -p "User: What's the weather like today?" \
    -n 128 --log-disable

# Task 5: Explanation
echo -e "\n[5] Vietnamese Explanation:"
./llama-cli -m $MODEL \
    -p "Error: I goes → Correct: I go. Explain in Vietnamese." \
    -n 256 --log-disable

echo -e "\n========================"
echo "Test complete!"
```

**Run test:**

```bash
chmod +x test_lexilingo.sh
./test_lexilingo.sh
```

---

## 🚀 Bước 7: Deploy with Server

### 7.1: Start llama.cpp server

```bash
cd ~/Projects/llama.cpp

# Start server
./llama-server \
    -m ./models/lexilingo_q4_km.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -n 512 \
    --ctx-size 2048

# Output:
# main: llama server listening on http://0.0.0.0:8080
# ✅ Server ready!
```

### 7.2: Test server with curl

```bash
# In another terminal
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Analyze fluency: The cat sat on the mat."}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }'

# Response:
# {
#   "choices": [
#     {"message": {"content": "Score: 5.0"}}
#   ]
# }
```

---

## 🐍 Bước 8: Use Python Client

### 8.1: Install dependencies

```bash
pip install requests
```

### 8.2: Use high-level client

```python
# import export/lexilingo_client.py
import sys
sys.path.insert(0, '/Users/nguyenhuuthang/Documents/RepoGitHub/LexiLingo/DL-Model-Support')

from export.lexilingo_client import LexiLingoClient

# Connect to server
with LexiLingoClient("models/lexilingo_q4_km.gguf", mode="server") as client:
    
    # Task 1: Fluency
    result = client.analyze_fluency("The cat sat on the mat.")
    print(f"Fluency: {result.score}/5.0")
    
    # Task 2: Vocabulary
    result = client.classify_vocabulary("The phenomenon is fascinating.")
    print(f"Vocabulary: {result.level}")
    
    # Task 3: Grammar
    result = client.correct_grammar("She don't like apples.")
    print(f"Grammar: {result.corrected_sentence}")
    
    # Task 4: Dialogue
    result = client.generate_dialogue("What's the weather?")
    print(f"Dialogue: {result.response}")
    
    # Task 5: Explanation
    result = client.explain_error("I goes", "I go")
    print(f"Explanation: {result.explanation}")
```

### 8.3: Batch processing

```python
from export.lexilingo_client import LexiLingoClient

sentences = [
    "The cat sat on the mat.",
    "I goes to school.",
    "She is very happy today.",
]

with LexiLingoClient("models/lexilingo_q4_km.gguf", mode="server") as client:
    results = client.batch_analyze_fluency(sentences)
    
    for sent, result in zip(sentences, results):
        print(f"{sent:40s} → Score: {result.score:.1f}")
```

---

## 📊 Performance Metrics (Mac Intel i9)

| Step | Duration | Output Size |
|------|----------|-------------|
| Merge (Kaggle) | ~2 min | 3.0 GB |
| Download | ~5-10 min | 3.0 GB |
| Convert to GGUF F16 | ~2-3 min | 3.0 GB |
| Quantize to Q4_K_M | ~1 min | 1.0 GB |
| **Total** | **~10-15 min** | **1.0 GB** |
| **Compression** | - | **3x smaller** |
| **Inference speed** | - | **10-15 tok/s** |

---

## 🔗 Complete Flow Summary

```bash
# 1. Merge (Kaggle) - 2 min
# Run: model.save_pretrained_merged(..., save_method="merged_16bit")
# Output: lexilingo_qwen25_1.5b_merged/ (3GB)

# 2. Download (Kaggle → Mac) - 5-10 min
# Option A: huggingface-cli download your-username/lexilingo-qwen25-1.5b
# Option B: Download zip from Kaggle Output

# 3. Convert to GGUF F16 (Mac) - 2-3 min
cd ~/Projects/llama.cpp
python3 convert_hf_to_gguf.py ./models/lexilingo_merged/ --outfile ./models/lexilingo_f16.gguf --outtype f16
# Output: lexilingo_f16.gguf (3GB)

# 4. Quantize to Q4_K_M (Mac) - 1 min
./llama-quantize ./models/lexilingo_f16.gguf ./models/lexilingo_q4_km.gguf Q4_K_M
# Output: lexilingo_q4_km.gguf (1GB)

# 5. Test Inference (Mac)
./llama-cli -m ./models/lexilingo_q4_km.gguf -p "Test prompt" -n 64

# 6. Start Server (Mac)
./llama-server -m ./models/lexilingo_q4_km.gguf --port 8080

# 7. Use Python Client (Any machine)
from export.lexilingo_client import LexiLingoClient
with LexiLingoClient("models/lexilingo_q4_km.gguf", mode="server") as client:
    result = client.analyze_fluency("The cat sat on the mat.")
```

---

## ❓ Troubleshooting

### Error 1: "Model file not found"

```bash
# Check file exists
ls -lh ~/Projects/llama.cpp/models/lexilingo_merged/
# Must see: model.safetensors or model-*.safetensors
```

### Error 2: "convert_hf_to_gguf.py not found"

```bash
# Update llama.cpp
cd ~/Projects/llama.cpp
git pull
make clean && make
```

### Error 3: "Unsupported model architecture"

```python
# Update convert script with model name
python3 convert_hf_to_gguf.py \
    ./models/lexilingo_merged/ \
    --outfile ./models/lexilingo_f16.gguf \
    --outtype f16 \
    --model-name qwen  # ← Add this
```

### Error 4: "Server failed to start"

```bash
# Check port 8080 is available
lsof -i :8080

# Or use different port
./llama-server -m ./models/lexilingo_q4_km.gguf --port 8081
```

### Error 5: Slow inference

```bash
# Adjust threads (set to CPU core count)
./llama-cli -m ./models/lexilingo_q4_km.gguf -t 8 -p "Test" -n 64

# On Mac Intel i9:
./llama-cli -m ./models/lexilingo_q4_km.gguf -t 10  # i9 has ~10 cores
```

---

## 📝 Cheat Sheet

```bash
# Quick reference commands

# 1. Convert
cd ~/Projects/llama.cpp && python3 convert_hf_to_gguf.py ./models/lexilingo_merged/ --outfile ./models/lexilingo_f16.gguf --outtype f16

# 2. Quantize
./llama-quantize ./models/lexilingo_f16.gguf ./models/lexilingo_q4_km.gguf Q4_K_M

# 3. Test
./llama-cli -m ./models/lexilingo_q4_km.gguf -p "Test prompt" -n 64

# 4. Server
./llama-server -m ./models/lexilingo_q4_km.gguf --port 8080

# 5. Clean (remove large F16 after quantize)
rm ./models/lexilingo_f16.gguf
```

---

## ✅ You're All Set!

Sau khi hoàn thành tất cả bước:
- ✅ Model converted to GGUF
- ✅ Model quantized to Q4_K_M
- ✅ Server running and ready
- ✅ Python client ready to use
- ✅ Ready for production deployment!

**Next:** Use `export/lexilingo_client.py` để tích hợp vào ứng dụng của bạn!

---

**Last Updated:** Jan 27, 2026
**Version:** 1.0
