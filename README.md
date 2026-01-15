# LexiLingo DL Model Support

Deep Learning model training and deployment support for LexiLingo English learning app.

## Project Structure

```
DL-Model-Support/
├── run_pipeline.py                   # Master training script
├── requirements.txt                  # Python dependencies
├── config/                           # Configuration files
│   ├── llm_config.yaml              # LLM training config
│   ├── stt_config.yaml              # STT training config
│   └── tts_config.yaml              # TTS config
├── data/                             # Dataset storage
│   ├── raw/                         # Raw downloaded data
│   └── processed/                   # Processed training data
├── datasets/                         # Dataset scripts
│   └── download_datasets.py         # Download all datasets
├── pipelines/                        # Training pipelines
│   ├── llm/                         # LLM training
│   │   ├── train_qwen_grammar.py    # Qwen grammar training
│   │   ├── train_smolm_conversation.py  # SmolLM conversation
│   │   └── hybrid_router.py         # Intelligent task routing
│   └── stt/                         # STT training
│       └── train_whisper.py         # Whisper fine-tuning
├── export/                           # Model export scripts
│   └── export_mobile.py             # GGUF, CoreML, ONNX export
├── outputs/                          # Trained models
└── exports/                          # Exported mobile models
```

## Quick Start

### Choose Your Mode

- **Development Mode** (Mac 32GB): Use larger models for best quality
- **Production Mode** (Mobile): Use optimized small models

### 1. Install Dependencies

```bash
cd DL-Model-Support
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

#### Development Mode (Mac 32GB RAM) - Recommended for You! 

#### Development Mode (Mac 32GB RAM) - Recommended for You! 

```bash
# Use large, high-quality models (Qwen 1.5B, Llama 1B)
python run_pipeline.py --all --config dev \
  --epochs=5 --batch_size=8

# Interactive testing with dev models
python pipelines/llm/hybrid_router.py --interactive --config dev
```

**Models Used**:
- Grammar: Qwen2.5-1.5B (~900MB Q4, 2GB RAM) ⭐⭐⭐⭐⭐
- Conversation: Llama-3.2-1B (~600MB Q4, 1.5GB RAM) ⭐⭐⭐⭐⭐
- Total: ~7GB RAM usage (comfortable on 32GB Mac)

#### Production Mode (Mobile Deployment)

```bash
# Use optimized small models (Qwen 0.5B, SmolLM 360M)
python run_pipeline.py --all \
  --epochs=3 --batch_size=4

#### Production Mode (Mobile Deployment)

```bash
# Use optimized small models (Qwen 0.5B, SmolLM 360M)
python run_pipeline.py --all \
  --epochs=3 --batch_size=4

# Interactive testing with prod models  
python pipelines/llm/hybrid_router.py --interactive
```

**Models Used**:
- Grammar: Qwen2.5-0.5B (~300MB Q4, 600MB RAM) ⭐⭐⭐⭐
- Conversation: SmolLM2-360M (~200MB Q4, 400MB RAM) ⭐⭐⭐⭐
- Total: ~2.4GB RAM usage (works on 4GB+ phones)

#### Individual Steps

```bash
# Or run individual steps:
python run_pipeline.py --download      # Download datasets only
python run_pipeline.py --train-llm     # Train LLM models only
python run_pipeline.py --train-stt     # Train STT model only
python run_pipeline.py --export        # Export models only

# With custom parameters:
python run_pipeline.py --all --epochs=5 --batch_size=8
```

### 3. Individual Scripts

#### Development Mode

```bash
# Download datasets
python datasets/download_datasets.py --all

# Train grammar model with large Qwen (1.5B + LoRA)
python pipelines/llm/train_qwen_grammar.py --train \
  --config dev --epochs=5 --batch_size=8

# Train conversation model with Llama (1B + LoRA)
python pipelines/llm/train_smolm_conversation.py --train \
  --config dev --epochs=7 --batch_size=12

# Interactive chat with dev models
python pipelines/llm/train_smolm_conversation.py --chat --config dev

# Export (keep F16 precision for dev)
python export/export_mobile.py --config dev --quantization F16
```

#### Production Mode (Default)
# Download datasets
python datasets/download_datasets.py --all
python datasets/download_datasets.py --grammar --vocabulary

# Train grammar model (Qwen2.5-0.5B + LoRA)
python pipelines/llm/train_qwen_grammar.py --train --epochs=3

# Train conversation model (SmolLM2-360M + LoRA)
python pipelines/llm/train_smolm_conversation.py --train --epochs=5

# Interactive chat with trained models
python pipelines/llm/train_smolm_conversation.py --chat

# Train STT model (Whisper Small)
python pipelines/stt/train_whisper.py --train --max_steps=4000

# Export for mobile
python export/export_mobile.py --all
python export/export_mobile.py --model ./outputs/qwen-grammar --format gguf
```

### 4. Test Hybrid Router

```bash
# Classify input task type
python pipelines/llm/hybrid_router.py --classify "She go to school"

# Interactive mode
python pipelines/llm/hybrid_router.py --interactive
```

## Model Overview

### 🖥️ Development Mode (Mac 32GB) - YOUR SETUP 

| Model | Task | Size (Q4) | LoRA | Dataset | Quality |
|-------|------|-----------|------|---------|---------|
| Qwen2.5-1.5B | Grammar/Vocab | ~900MB | r=32, α=64 | CoLA, BEA-2019 | ⭐⭐⭐⭐⭐ |
| Llama-3.2-1B | Conversation | ~600MB | r=16, α=32 | DailyDialog, Persona | ⭐⭐⭐⭐⭐ |
| Whisper Medium | STT | ~780MB | Fine-tune | LibriSpeech | ⭐⭐⭐⭐⭐ |

**Total**: ~2.4GB models + ~7GB RAM = Perfect for your Mac!

**Benefits**:
- Best quality outputs for development
- Faster debugging with clear AI responses
- Train once, export to production models
- Comfortable performance on 32GB RAM

### 📱 Production Mode (Mobile 4-8GB)

| Model | Task | Size (Q4) | LoRA | Dataset | Quality |
|-------|------|-----------|------|---------|---------|
| Qwen2.5-0.5B | Grammar/Vocab | ~300MB | r=16, α=32 | CoLA, BEA-2019 | ⭐⭐⭐⭐ |
| SmolLM2-360M | Conversation | ~200MB | r=8, α=16 | DailyDialog, Persona | ⭐⭐⭐⭐ |
| Whisper Small | STT | ~240MB | Fine-tune | LibriSpeech | ⭐⭐⭐⭐ |

**Total**: ~800MB models + ~2.4GB RAM = Works on mid-range phones

### Speech Models (Both Modes)
| Model | Task | Size | Format |
|-------|------|------|--------|
| Whisper Small | Speech-to-Text | ~240MB | ONNX, CoreML |
| Piper TTS | Text-to-Speech | ~50MB | Native |

### Hybrid Router
The `hybrid_router.py` automatically routes requests to the appropriate model:
- **Grammar tasks** → Qwen2.5-0.5B (corrections, explanations)
- **Vocabulary tasks** → Qwen2.5-0.5B (definitions, examples)
- **Conversation** → SmolLM2-360M (friendly tutoring)

## Export Formats

| Format | Platform | Use Case |
|--------|----------|----------|
| GGUF (Q4_K_M) | Android, iOS | llama.cpp inference |
| Core ML | iOS | NPU acceleration |
| ONNX | Cross-platform | General inference |

## Datasets Used

- **Grammar**: CoLA, BEA-2019, JFLEG, C4_200M
- **Vocabulary**: WordNet, Custom definitions
- **Conversation**: DailyDialog, Persona-Chat, EmpatheticDialogues
- **STT**: LibriSpeech, Common Voice

## 💻 Hardware Requirements

### Training
- **Minimum**: GPU with 8GB VRAM (RTX 3060)
- **Recommended**: GPU with 12GB+ VRAM (RTX 3080+)
- **CPU Training**: Possible but slow (~10x slower)

### Inference (Mobile)
- **Android**: Snapdragon 8 Gen 1+ (NPU) or 4GB+ RAM
- **iOS**: A14+ chip (Neural Engine)

## 🔧 Configuration

Edit config files in `config/` to customize:
- `llm_config.yaml`: LoRA settings, training parameters, prompt templates
- `stt_config.yaml`: Whisper settings, pronunciation assessment
- `tts_config.yaml`: Piper voices, Native TTS fallback

## 📂 Outputs

After training, models are saved to:
```
outputs/
├── qwen-grammar/          # Qwen model + LoRA adapters
├── smolm-conversation/    # SmolLM model + LoRA adapters
└── whisper-english/       # Fine-tuned Whisper

exports/
├── gguf/                  # GGUF quantized models
├── coreml/                # Core ML packages
└── onnx/                  # ONNX models
```
