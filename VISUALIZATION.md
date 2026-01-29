# 📊 LexiLingo Complete Pipeline Architecture

## Training → Deployment Flow (v2.0 Updated)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE PIPELINE OVERVIEW                          │
│                    Training → Merge → Deploy → Use                       │
└──────────────────────────────────────────────────────────────────────────┘


🎓 PHASE 1: TRAINING (Kaggle/Colab with GPU)
═══════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │     Qwen2.5-1.5B-Instruct (Base Model)              │
  │     + 4-bit NF4 Quantization                        │
  │     + Unified LoRA Adapter (r=32, α=64)            │
  │     + Unsloth Optimization (2x faster)             │
  └──────────────────┬──────────────────────────────────┘
                     │
                     │ 30,806 training samples
                     │ (5 tasks unified)
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │   Training Dataset                                  │
  ├─────────────────────────────────────────────────────┤
  │  • Fluency Scoring (23.6%)                          │
  │  • Vocabulary Classification (23.0%)                │
  │  • Grammar Correction (19.1%)                       │
  │  • Dialogue Generation (21.6%)                      │
  │  • Explanation Task 🆕 (12.7%)                      │
  └──────────────────┬──────────────────────────────────┘
                     │
                     │ Training time: 4-5h (P100 + Unsloth)
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │   ✅ Trained Model with LoRA Adapter                │
  │      Location: /kaggle/working/outputs/unified      │
  └──────────────────┬──────────────────────────────────┘
                     │
                     └──────────────────────────────────┐
                                                        │


📦 PHASE 2: MERGE & EXPORT (Kaggle/Colab)
═══════════════════════════════════════════════════════════════════════════

                                                        │
                     ┌──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Merge LoRA Adapter with Base Model                 │
  │  model.save_pretrained_merged(...)                  │
  │  save_method="merged_16bit" (CRITICAL ⭐)           │
  └──────────────────┬──────────────────────────────────┘
                     │
                     │ Output: lexilingo_qwen25_1.5b_merged/
                     │ Size: ~3.0 GB (FP16, lossless)
                     │ Files: model.safetensors, tokenizer.json, config.json
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  ✅ Merged Model (Full Precision)                   │
  │     Ready for GGUF conversion                       │
  └──────────────────┬──────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │                                 │
    ▼ Option A                        ▼ Option B
┌─────────────┐                 ┌──────────────┐
│HuggingFace  │                 │  Zip File    │
│Upload       │                 │  (Kaggle)    │
└─────────────┘                 └──────────────┘
    │                                 │
    │ huggingface-cli download         │ unzip & extract
    │ your-username/lexilingo-...     │ ~/Downloads/...
    │                                 │
    └────────────────┬────────────────┘
                     │


🖥️ PHASE 3: CONVERSION (Local Mac with llama.cpp)
═══════════════════════════════════════════════════════════════════════════

                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Install llama.cpp (One-time setup)                 │
  │  git clone https://github.com/ggerganov/llama.cpp   │
  │  cd llama.cpp && make                               │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Download Merged Model                              │
  │  $ huggingface-cli download or unzip                │
  │                                                     │
  │  Location: ~/Projects/llama.cpp/models/             │
  │            lexilingo_qwen25_1.5b_merged/            │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Convert to GGUF F16                                │
  │  $ python3 convert_hf_to_gguf.py \                  │
  │      ./models/lexilingo_merged/ \                   │
  │      --outfile ./models/lexilingo_f16.gguf \        │
  │      --outtype f16                                  │
  │                                                     │
  │  Time: 2-3 minutes                                  │
  │  Output: lexilingo_f16.gguf (~3.0 GB, lossless)    │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Quantize to Q4_K_M                                 │
  │  $ ./llama-quantize \                               │
  │      ./models/lexilingo_f16.gguf \                  │
  │      ./models/lexilingo_q4_km.gguf \                │
  │      Q4_K_M                                         │
  │                                                     │
  │  Time: 1 minute                                     │
  │  Output: lexilingo_q4_km.gguf (~1.0 GB)            │
  │  Compression: 3x smaller, <2% quality loss         │
  └──────────────────┬──────────────────────────────────┘
                     │


🚀 PHASE 4: DEPLOYMENT (Local Mac)
═══════════════════════════════════════════════════════════════════════════

                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Option 1: CLI - Direct Inference                   │
  │  $ ./llama-cli -m ./models/lexilingo_q4_km.gguf \   │
  │      -p "Test prompt" -n 64                         │
  │                                                     │
  │  Speed: 10-15 tok/s                                 │
  │  Best for: Quick testing                            │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Option 2: Server Mode (RECOMMENDED)                │
  │  $ ./llama-server -m ./models/lexilingo_q4_km.gguf \│
  │      --port 8080 --ctx-size 2048                    │
  │                                                     │
  │  Server ready at: http://localhost:8080             │
  │  API: /v1/chat/completions                          │
  │  Best for: Production use                           │
  └──────────────────┬──────────────────────────────────┘
                     │


💻 PHASE 5: INTEGRATION (Any Machine/Language)
═══════════════════════════════════════════════════════════════════════════

                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Python Client (Recommended)                        │
  │  from export.lexilingo_client import LexiLingoClient│
  │                                                     │
  │  with LexiLingoClient(..., mode="server") as client:│
  │      result = client.analyze_fluency(...)           │
  │      result = client.classify_vocabulary(...)       │
  │      result = client.correct_grammar(...)           │
  │      result = client.generate_dialogue(...)         │
  │      result = client.explain_error(...)             │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  REST API (Any Language)                            │
  │  curl http://localhost:8080/v1/chat/completions \  │
  │      -H "Content-Type: application/json" \          │
  │      -d '{"messages": [...], "max_tokens": 128}'    │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  ✅ READY FOR PRODUCTION USE                        │
  │     • Low latency (100-500ms per request)           │
  │     • Low memory (2-4 GB RAM)                       │
  │     • Cross-platform (Mac/Linux/Windows)            │
  │     • Scalable (can run multiple instances)         │
  └──────────────────────────────────────────────────────┘
```

---

## Model Architecture: Input → Processing → Output

```
┌──────────────────────────────────────────────────────────────────────┐
│                   UNIFIED INFERENCE PIPELINE                         │
└──────────────────────────────────────────────────────────────────────┘

USER REQUEST
    │
    └─────────────────────────────────────────────────────┐
                                                          │
                                       ┌──────────────────▼─────────────┐
                                       │  TASK IDENTIFICATION            │
                                       │  (From input format)            │
                                       └──────────┬─────────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                │                                 │                                 │
                ▼                                 ▼                                 ▼
        ┌──────────────┐              ┌──────────────────┐              ┌──────────────┐
        │  "Analyze    │              │"Classify the     │              │"Correct this │
        │  fluency: ..." │              │ vocabulary: ..."  │              │ sentence: ..." │
        └──────┬───────┘              └────────┬─────────┘              └────────┬─────┘
               │                              │                                 │
               ▼                              ▼                                 ▼
        ┌──────────────┐              ┌──────────────────┐              ┌──────────────┐
        │ FLUENCY TASK │              │VOCABULARY TASK   │              │ GRAMMAR TASK │
        └──────┬───────┘              └────────┬─────────┘              └────────┬─────┘
               │                              │                                 │
               └──────────────────────────────┼─────────────────────────────────┘
                                              │
                              ┌───────────────▼────────────────┐
                              │  UNIFIED LORA ADAPTER          │
                              │  (Qwen2.5-1.5B-Instruct)      │
                              │  LoRA r=32, α=64              │
                              │  Unsloth Optimized            │
                              └───────────────┬────────────────┘
                                              │
                ┌─────────────────────────────┼─────────────────────────────────┐
                │                             │                                 │
                ▼                             ▼                                 ▼
        ┌──────────────┐              ┌──────────────────┐              ┌──────────────┐
        │  Score: 4.5  │              │ Level: B1        │              │ Fixed: "He   │
        │  /5.0        │              │                  │              │ went to..."  │
        └──────────────┘              └──────────────────┘              └──────────────┘

Additional Tasks:
                │
                ▼
        ┌──────────────────────────────────────────────────────────────┐
        │              "User: What's the weather?"                     │
        │                                                              │
        │  ↓ DIALOGUE TASK ↓                                          │
        │                                                              │
        │  Response: "I don't have real-time weather data but..."     │
        └──────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────────┐
        │        "Error: 'He go' → Correct: 'He goes'"                │
        │                                                              │
        │  ↓ EXPLANATION TASK (VIETNAMESE) ↓                          │
        │                                                              │
        │  "Khi chủ từ là 'He' (số ít), động từ phải thêm 's'..."    │
        └──────────────────────────────────────────────────────────────┘
```

---

## Task Processing Details

```
┌────────────────────────────────────────────────────────────────────────┐
│                     DETAILED TASK PROCESSING                           │
└────────────────────────────────────────────────────────────────────────┘

1️⃣ FLUENCY ANALYSIS
   Input:    "The cat sat on the mat."
   Process:  Analyze grammatical correctness, natural flow, clarity
   Output:   Score: 5.0/5.0
   Model:    Regression task (0.0-5.0 range)

2️⃣ VOCABULARY CLASSIFICATION
   Input:    "The phenomenon is fascinating."
   Process:  Determine CEFR level from vocabulary complexity
   Output:   Level: B2
   Model:    Classification task (A1, A2, B1, B2, C1, C2)

3️⃣ GRAMMAR CORRECTION
   Input:    "She don't like apples."
   Process:  Identify errors, apply corrections
   Output:   "She doesn't like apples."
   Model:    Sequence-to-sequence task

4️⃣ DIALOGUE GENERATION
   Input:    "User: What's the weather like?"
   Process:  Generate contextually appropriate response
   Output:   "I don't have access to real-time weather data, but..."
   Model:    Conversational task

5️⃣ EXPLANATION (VIETNAMESE TUTOR) 🆕
   Input:    "Error: 'I goes' → Correct: 'I go'"
   Process:  Explain grammar rule in Vietnamese, friendly tone
   Output:   "Khi chủ từ là 'I' (số ít), động từ không thêm 's' nhé em..."
   Model:    Explanation generation task
```

---

## Quantization & Compression Strategy

```
┌────────────────────────────────────────────────────────────────────────┐
│             MODEL SIZE & QUALITY TRADEOFFS                             │
└────────────────────────────────────────────────────────────────────────┘

TRAINING PHASE:
   Qwen2.5-1.5B-Instruct (Base)
   + 4-bit NF4 Quantization
   = ~1.5 GB VRAM for GPU training
   
EXPORT PHASE:
   Merged Model (FP16)
   Size: ~3.0 GB
   Precision: 100% (no loss)
   Use: GGUF conversion baseline

CONVERSION PHASE:
   GGUF F16
   Size: ~3.0 GB
   Precision: 100% (lossless from FP16)
   Loss: 0%
   
DEPLOYMENT PHASE (3x COMPRESSION):
   Q4_K_M ⭐ RECOMMENDED
   ┌─────────────────────┐
   │ Size: ~1.0 GB       │
   │ Quality Loss: <2%   │
   │ Speed: 10-15 tok/s  │
   │ RAM: 2-4 GB         │
   └─────────────────────┘
   
   Alternative Options:
   
   Q4_K_S (Faster)
   Size: ~0.9 GB | Loss: 3-5% | Speed: 12-18 tok/s
   
   Q5_K_M (Better Quality)
   Size: ~1.2 GB | Loss: <1% | Speed: 8-12 tok/s
   
   Q8_0 (Lossless)
   Size: ~2.0 GB | Loss: <0.5% | Speed: 8-10 tok/s

COMPARISON:
   Original (FP32): 6 GB
   FP16: 3 GB (50% smaller)
   Q4_K_M: 1 GB (5x smaller overall!)
```

---

## Performance & Resource Requirements

```
┌────────────────────────────────────────────────────────────────────────┐
│              TRAINING vs INFERENCE REQUIREMENTS                         │
└────────────────────────────────────────────────────────────────────────┘

TRAINING (Kaggle/Colab GPU)
───────────────────────────────────────────────────────────────────────
   GPU Memory:         8 GB (P100) / 16 GB (V100)
   Batch Size:         8 (with Unsloth)
   Gradient Steps:     4
   Training Time:      4-5 hours
   Dataset:            30,806 samples
   Model:              Qwen2.5-1.5B-Instruct
   Optimization:       Unsloth (2x faster, 70% less VRAM)
   Output:             LoRA adapter + Merged model

INFERENCE - Merged Model (CPU, transformers)
───────────────────────────────────────────────────────────────────────
   Model Size:         3.0 GB (FP16)
   RAM Required:       ~8 GB
   Speed:              3-5 tokens/second
   Latency (per 50 tok): 10-15 seconds
   CPU Cores:          8+ recommended
   Best for:           Development only

INFERENCE - GGUF Q4_K_M (CPU, llama.cpp) ⭐ BEST
───────────────────────────────────────────────────────────────────────
   Model Size:         1.0 GB (quantized)
   RAM Required:       2-4 GB
   Speed:              10-15 tokens/second ⚡ (2-3x faster!)
   Latency (per 50 tok): 3-5 seconds
   CPU Cores:          6+ (works well with i9)
   Setup:              llama.cpp server
   Best for:           Production use

Mac Intel i9 Benchmark
───────────────────────────────────────────────────────────────────────
   CPU: 10 cores @ 2.4 GHz
   RAM: 32 GB
   Model: GGUF Q4_K_M
   
   Task 1 - Fluency (20 tokens):     2-3 seconds
   Task 2 - Vocabulary (10 tokens):  1-2 seconds
   Task 3 - Grammar (50 tokens):     3-5 seconds
   Task 4 - Dialogue (100 tokens):   6-8 seconds
   Task 5 - Explanation (200 tokens): 12-15 seconds
```

---

## Deployment Options Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT OPTIONS MATRIX                            │
└────────────────────────────────────────────────────────────────────────┘

1. LOCAL CLI (Simplest)
   ┌─────────────────────────────────────────────────────────┐
   │ ./llama-cli -m model.gguf -p "prompt" -n 64            │
   ├─────────────────────────────────────────────────────────┤
   │ Pros:   No setup, quick testing                         │
   │ Cons:   No persistence, manual for each request         │
   │ Speed:  10-15 tok/s                                     │
   │ Use:    Development & debugging                         │
   └─────────────────────────────────────────────────────────┘

2. LOCAL SERVER (Recommended) ⭐
   ┌─────────────────────────────────────────────────────────┐
   │ ./llama-server -m model.gguf --port 8080               │
   │ → Accessible via REST API                              │
   ├─────────────────────────────────────────────────────────┤
   │ Pros:   Persistent, concurrent requests, REST API      │
   │ Cons:   Requires server setup                           │
   │ Speed:  10-15 tok/s (same model)                        │
   │ Use:    Production on single machine                    │
   └─────────────────────────────────────────────────────────┘

3. PYTHON CLIENT (Integration)
   ┌─────────────────────────────────────────────────────────┐
   │ from export.lexilingo_client import LexiLingoClient    │
   │ client = LexiLingoClient(model_path, mode="server")    │
   ├─────────────────────────────────────────────────────────┤
   │ Pros:   Easy Python integration, task-specific methods  │
   │ Cons:   Python only                                     │
   │ Speed:  10-15 tok/s                                     │
   │ Use:    Python applications & services                  │
   └─────────────────────────────────────────────────────────┘

4. REST API (Cross-Language)
   ┌─────────────────────────────────────────────────────────┐
   │ curl http://localhost:8080/v1/chat/completions         │
   │     -H "Content-Type: application/json"                │
   │     -d '{...}'                                          │
   ├─────────────────────────────────────────────────────────┤
   │ Pros:   Language-agnostic, any client                  │
   │ Cons:   Standard REST (not optimized for LexiLingo)    │
   │ Speed:  10-15 tok/s                                     │
   │ Use:    Web services, cross-language integration        │
   └─────────────────────────────────────────────────────────┘

5. CLOUD DEPLOYMENT (Optional Future)
   ┌─────────────────────────────────────────────────────────┐
   │ RunPod, Hugging Face Inference, Google Cloud, AWS       │
   ├─────────────────────────────────────────────────────────┤
   │ Pros:   Scalable, managed infrastructure                │
   │ Cons:   Cost, latency                                   │
   │ Speed:  Variable (GPU available)                        │
   │ Use:    Large-scale production                          │
   └─────────────────────────────────────────────────────────┘
```

---

## File Structure & Tools

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PROJECT STRUCTURE (v2.0)                          │
└────────────────────────────────────────────────────────────────────────┘

LexiLingo/DL-Model-Support/
│
├─ 📚 TRAINING & SETUP
│  ├─ scripts/finetune_qwen_lora_kaggle.v1.0.ipynb  (Main training)
│  │   └─ Phase 1: Train with Unsloth
│  │   └─ Phase 2: Merge LoRA + Export (NEW!)
│  │       ├─ Merge: save_pretrained_merged(..., save_method="merged_16bit")
│  │       ├─ Export Option A: HuggingFace upload
│  │       └─ Export Option B: Zip download
│  │
│  ├─ requirements.txt
│  │   └─ All dependencies for training
│  │
│  ├─ config/
│  │   ├─ llm_config.yaml
│  │   ├─ stt_config.yaml
│  │   ├─ tts_config.yaml
│  │   └─ llm_config.dev.yaml
│  │
│  └─ datasets/
│      ├─ cefr/
│      │   └─ ENGLISH_CERF_WORDS.csv (Vocabulary reference)
│      │
│      └─ datasets/
│          ├─ train.jsonl (26,880 samples)
│          ├─ val.jsonl (1,412 samples)
│          ├─ train_with_explanation.jsonl (30,806 samples) 🆕
│          ├─ val_with_explanation.jsonl (1,618 samples) 🆕
│          ├─ vietnamese_explanations.jsonl (4,132 samples)
│          ├─ unified_training_data.json
│          ├─ dialogue_data.json
│          ├─ fluency_data.json
│          ├─ grammar_data.json
│          ├─ vocabulary_data.json
│          └─ merge_explanation_report.json (Statistics)
│
├─ 🔄 CONVERSION & DEPLOYMENT (NEW!)
│  ├─ scripts/deploy_lexilingo.sh  ⭐ AUTOMATION SCRIPT
│  │   ├─ Download merged model (HF or zip)
│  │   ├─ Convert to GGUF F16
│  │   ├─ Quantize to Q4_K_M
│  │   ├─ Test inference
│  │   └─ Start server
│  │
│  ├─ export/lexilingo_client.py  ⭐ PYTHON CLIENT
│  │   ├─ LexiLingoCliClient (CLI mode)
│  │   ├─ LexiLingoServerClient (Server mode)
│  │   └─ LexiLingoClient (High-level API)
│  │       ├─ analyze_fluency()
│  │       ├─ classify_vocabulary()
│  │       ├─ correct_grammar()
│  │       ├─ generate_dialogue()
│  │       └─ explain_error()
│  │
│  └─ model/
│      ├─ logging_middleware.py
│      ├─ adapters/
│      │   ├─ dialogue_lora_adapter/
│      │   ├─ fluency_lora_adapter/
│      │   ├─ grammar_lora_adapter/
│      │   └─ vocabulary_lora_adapter/
│      │
│      └─ outputs/
│          ├─ dialogue/
│          ├─ fluency/
│          ├─ grammar/
│          ├─ vocabulary/
│          └─ unified/  (Main output)
│
├─ 📖 DOCUMENTATION (NEW!)
│  ├─ docs/DEPLOYMENT_FLOW.md  ⭐ Complete guide
│  │   ├─ Phase 1: Training setup
│  │   ├─ Phase 2: Merge & Export
│  │   ├─ Phase 3: Convert to GGUF
│  │   ├─ Phase 4: Deploy
│  │   ├─ Troubleshooting
│  │   └─ Performance metrics
│  │
│  ├─ docs/STEPS_AFTER_TRAINING.md  ⭐ Step-by-step
│  │   ├─ 7 complete steps
│  │   ├─ Code examples
│  │   ├─ Commands reference
│  │   └─ Performance benchmarks
│  │
│  ├─ docs/EXPLANATION_TASK.md
│  │   └─ Vietnamese teaching methodology
│  │
│  ├─ docs/Training_Optimization_Guide.md
│  │   └─ Unsloth optimization details
│  │
│  ├─ docs/UNSLOTH_INTEGRATION_COMPLETE.md
│  │   └─ Unsloth setup & benefits
│  │
│  ├─ DEPLOYMENT_FLOW.md (Main architecture)
│  ├─ MODEL_UPDATE_COMPLETE.md (Summary)
│  ├─ QUICK_REFERENCE.md (Cheat sheet)
│  ├─ README.md (Project overview)
│  └─ VISUALIZATION.md (This file)
│
└─ 🧪 TESTING
   ├─ scripts/test_qwen3_simple.py
   ├─ scripts/test_qwen3_quality.py
   ├─ scripts/test_qwen_mac_intel.py
   ├─ scripts/README_TESTING.md
   └─ scripts/README_DATASETS.md
```

---

## Quick Start Command Reference

```
┌────────────────────────────────────────────────────────────────────────┐
│                      QUICK START COMMANDS                              │
└────────────────────────────────────────────────────────────────────────┘

🎓 STEP 1: TRAIN (Kaggle/Colab)
   1. Upload notebook: scripts/finetune_qwen_lora_kaggle.v1.0.ipynb
   2. Run all cells
   3. Total time: 4-5 hours (GPU)
   Output: unified_model/ folder with LoRA adapter

📦 STEP 2: MERGE & EXPORT (Kaggle/Colab)
   # New cells in notebook:
   model.save_pretrained_merged(
       "/kaggle/working/lexilingo_qwen25_1.5b_merged",
       tokenizer,
       save_method="merged_16bit"
   )
   
   # Option A: Push to HuggingFace
   model.push_to_hub("your-username/lexilingo-qwen25-1.5b")
   
   # Option B: Download zip from Kaggle Output
   Output: ~3.0 GB merged model

🖥️ STEP 3: SETUP LOCAL (One-time)
   $ cd ~/Projects
   $ git clone https://github.com/ggerganov/llama.cpp.git
   $ cd llama.cpp
   $ make
   Output: llama-cli, llama-quantize, llama-server ready

📥 STEP 4: DOWNLOAD MODEL
   # Option A: From HuggingFace
   $ huggingface-cli download your-username/lexilingo-qwen25-1.5b \
       --local-dir ~/Projects/llama.cpp/models/lexilingo_merged
   
   # Option B: Extract Kaggle zip
   $ unzip ~/Downloads/lexilingo_merged.zip -d ~/Projects/llama.cpp/models/

🔄 STEP 5: CONVERT & DEPLOY (Automated!)
   $ cd ~/Documents/RepoGitHub/LexiLingo/DL-Model-Support
   
   # Using automation script (RECOMMENDED):
   $ ./scripts/deploy_lexilingo.sh -m hf -u your-username
   
   # Or manual steps:
   $ cd ~/Projects/llama.cpp
   
   # Convert to GGUF F16
   $ python3 convert_hf_to_gguf.py \
       ./models/lexilingo_merged/ \
       --outfile ./models/lexilingo_f16.gguf \
       --outtype f16
   
   # Quantize to Q4_K_M
   $ ./llama-quantize \
       ./models/lexilingo_f16.gguf \
       ./models/lexilingo_q4_km.gguf \
       Q4_K_M

🚀 STEP 6: RUN SERVER
   $ ./llama-server \
       -m ./models/lexilingo_q4_km.gguf \
       --port 8080 \
       --ctx-size 2048
   
   Server ready at: http://localhost:8080

💻 STEP 7: USE PYTHON CLIENT
   from export.lexilingo_client import LexiLingoClient
   
   with LexiLingoClient("models/lexilingo_q4_km.gguf", mode="server") as client:
       # Fluency
       result = client.analyze_fluency("The cat sat on the mat.")
       print(f"Score: {result.score}")
       
       # Vocabulary
       result = client.classify_vocabulary("The phenomenon is fascinating.")
       print(f"Level: {result.level}")
       
       # Grammar
       result = client.correct_grammar("She don't like apples.")
       print(f"Fixed: {result.corrected_sentence}")
       
       # Dialogue
       result = client.generate_dialogue("What's the weather?")
       print(f"Response: {result.response}")
       
       # Explanation
       result = client.explain_error("I goes", "I go")
       print(f"Explanation: {result.explanation}")

🧪 STEP 8: TEST (Optional)
   $ ./llama-cli -m ./models/lexilingo_q4_km.gguf \
       -p "Analyze fluency: The cat sat on the mat." \
       -n 64

📊 STEP 9: MONITOR
   # Check if server is running
   curl http://localhost:8080/health
   
   # Check model info
   ./llama-cli -m ./models/lexilingo_q4_km.gguf --version
```

---

## Version History & Updates

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VERSION TIMELINE                                │
└────────────────────────────────────────────────────────────────────────┘

v1.0 (Initial Release)
├─ 4 tasks: fluency, vocabulary, grammar, dialogue
├─ Qwen2.5-1.5B-Instruct base model
├─ Single LoRA adapter (unified)
├─ 26,880 training samples
└─ Basic training pipeline

v1.1 (Explanation Task Addition)
├─ Added 5th task: Vietnamese grammar explanation (tutor mode)
├─ +3,926 explanation samples
├─ 30,806 total training samples
├─ Friendly tone (Vietnamese pronouns: em, con, nha)
└─ Training: 4-5 hours with Unsloth

v2.0 (Complete Deployment Pipeline) 🆕 CURRENT
├─ Phase 1: Training with Unsloth (2x faster)
├─ Phase 2: Merge LoRA + Export
│   ├─ save_method="merged_16bit" (lossless)
│   ├─ HuggingFace upload option
│   └─ Zip download option
├─ Phase 3: Convert to GGUF F16 (3.0 GB)
├─ Phase 4: Quantize to Q4_K_M (1.0 GB)
├─ Phase 5: Deploy with llama.cpp server
├─ New Tools:
│   ├─ deploy_lexilingo.sh (automation script)
│   ├─ lexilingo_client.py (Python client)
│   ├─ DEPLOYMENT_FLOW.md (complete guide)
│   └─ STEPS_AFTER_TRAINING.md (step-by-step)
├─ 3x model compression (5x overall from FP32)
├─ 2-3x faster inference on CPU
└─ Production-ready deployment

ROADMAP (Future)
├─ v2.1: Multi-language support
├─ v3.0: Larger models (3B, 7B variants)
├─ v3.1: Fine-tuning on custom data
├─ v4.0: REST API optimization
├─ v5.0: Cloud deployment templates
└─ v6.0: Mobile deployment (ONNX)
```

---

## Summary: What Changed in v2.0

```
┌────────────────────────────────────────────────────────────────────────┐
│                  MAJOR UPDATES IN VERSION 2.0                          │
└────────────────────────────────────────────────────────────────────────┘

❌ BEFORE (v1.0)
   Training Model (LoRA only)
   │
   └─→ Download from Kaggle
       └─→ Can't easily use on local machine
           └─→ Complex conversion process
               └─→ Manual CLI usage only

✅ AFTER (v2.0) - Complete Pipeline
   Training Model with LoRA
   │
   ├─→ Merge with base model (merged_16bit)
   │
   ├─→ Push to HuggingFace OR download zip
   │
   ├─→ Convert to GGUF F16 (lossless)
   │
   ├─→ Quantize to Q4_K_M (3x compression)
   │
   ├─→ Deploy with llama.cpp server
   │
   ├─→ Use Python client (easy integration)
   │
   └─→ Production-ready! 🚀

KEY IMPROVEMENTS:

1. 📦 Export Format
   Before: LoRA adapter only (~50MB)
   After:  Full merged model + GGUF + Quantized (~1GB total)

2. 🔄 Conversion
   Before: Manual convert_hf_to_gguf.py steps
   After:  Automated deploy_lexilingo.sh script

3. 🚀 Deployment
   Before: CLI only, one request at a time
   After:  Server mode (concurrent), Python client

4. ⚡ Performance
   Before: 3-5 tok/s on CPU (transformers)
   After:  10-15 tok/s on CPU (llama.cpp) ← 2-3x faster!

5. 💾 Size
   Before: 3.0 GB merged model
   After:  1.0 GB quantized (3x smaller)

6. 📚 Documentation
   Before: Minimal docs
   After:  Complete guides:
           ├─ DEPLOYMENT_FLOW.md
           ├─ STEPS_AFTER_TRAINING.md
           ├─ deploy_lexilingo.sh
           └─ lexilingo_client.py

7. 🐍 Integration
   Before: Complex manual setup
   After:  Simple Python API:
           ```python
           with LexiLingoClient(...) as client:
               result = client.analyze_fluency("...")
           ```
```

---

**Updated:** 2026-01-28  
**Version:** 2.0 (Complete Pipeline)  
**Status:** ✅ Production Ready

📖 **See also:**
- [docs/DEPLOYMENT_FLOW.md](docs/DEPLOYMENT_FLOW.md) - Full deployment guide
- [docs/STEPS_AFTER_TRAINING.md](docs/STEPS_AFTER_TRAINING.md) - Step-by-step instructions
- [scripts/deploy_lexilingo.sh](scripts/deploy_lexilingo.sh) - Automation script
- [export/lexilingo_client.py](export/lexilingo_client.py) - Python client library
