# LexiLingo DL Model Support

Deep Learning infrastructure for LexiLingo - An AI-powered English learning platform featuring multi-task NLP, speech recognition, and intelligent task orchestration.

## Overview

LexiLingo employs a unified architecture that combines multiple specialized models for comprehensive English language learning support. The system uses LoRA adapters for efficient multi-task learning while maintaining low memory footprint suitable for mobile deployment.

## Technical Stack

### Core Technologies
- **Framework**: PyTorch, Hugging Face Transformers
- **Fine-tuning**: LoRA/QLoRA (Parameter-Efficient Fine-Tuning)
- **Quantization**: 4-bit/8-bit GPTQ, GGUF (Q4_K_M)
- **Inference**: llama.cpp, ONNX Runtime, Core ML
- **Caching**: Redis (context management)
- **Embeddings**: MiniLM-L6 (sentence embeddings)

### Model Architecture

#### NLP Models
| Model | Parameters | Task | Quantization | Memory |
|-------|------------|------|--------------|--------|
| Qwen2.5-1.5B | 1.5B | Grammar, Vocabulary | Q4_K_M | ~900MB |
| Qwen2.5-0.5B | 0.5B | Grammar, Vocabulary | Q4_K_M | ~300MB |
| LLaMA3-8B-VI | 8B | Vietnamese Explanations | Q4_K_M | ~5GB |
| MiniLM-L6 | 22M | Context Embedding | FP16 | ~22MB |

**Unified LoRA Adapter**: Single adapter (r=48, α=96, 80MB) handles 4 tasks:
- Fluency Scoring
- Grammar Correction
- Vocabulary Classification  
- Dialogue Generation

#### Speech Models
| Model | Task | Size | Format |
|-------|------|------|--------|
| Faster-Whisper v3 | Speech-to-Text | 244MB | ONNX |
| HuBERT-large | Pronunciation Assessment | 960MB | PyTorch |
| Piper VITS | Text-to-Speech | 30-60MB | ONNX |

### Architecture Components

**Orchestrator**: Intelligent task routing with lazy loading
- Analyzes input complexity and learner level
- Routes to appropriate model (Qwen vs LLaMA3-VI)
- Manages parallel execution (HuBERT + Qwen)
- Implements attention fusion layer

**Context Manager**: 
- MiniLM-L6 embeddings for semantic search
- Redis caching (45% cache hit rate)
- Conversation history tracking

**Performance Metrics** (v2.0):
- Average latency: 350ms (56% improvement)
- Memory usage: 4.8GB (60% reduction)
- Model switching: <1ms
- Uptime: 99.5%

## Datasets

### Grammar & Error Correction
- **CoLA** (Corpus of Linguistic Acceptability): 10,657 sentences
- **BEA-2019** (W&I+LOCNESS): 34,304 error-annotated sentences
- **JFLEG**: 1,511 grammatically corrected sentences
- **C4_200M**: Web-crawled English corpus (grammar patterns)

### Vocabulary & CEFR
- **WordNet 3.1**: Lexical database with definitions
- **Cambridge CEFR**: A2/B1/B2 vocabulary classification
- **Custom**: 15,000+ word-definition pairs with examples

### Dialogue & Conversation
- **DailyDialog**: 13,118 multi-turn conversations
- **Persona-Chat**: 162,064 persona-based dialogues
- **EmpatheticDialogues**: 24,850 emotional conversations

### Speech & Pronunciation
- **LibriSpeech**: 1,000 hours clean English speech
- **Common Voice**: Multi-accent English corpus
- **TIMIT**: Phonetic speech corpus for assessment

## Deployment

### Export Formats
- **GGUF**: Android/iOS via llama.cpp (cross-platform)
- **Core ML**: iOS Neural Engine acceleration
- **ONNX**: General inference engines

### Hardware Requirements

**Training**:
- Minimum: 16GB RAM, GPU 8GB VRAM (RTX 3060)
- Recommended: 32GB RAM, GPU 12GB+ VRAM (RTX 3080+)

**Mobile Inference**:
- iOS: A14+ chip, 4GB+ RAM
- Android: Snapdragon 8 Gen 1+, 4GB+ RAM

## Documentation

- [Architecture Overview](docs/architecture.md) - Detailed system design
- [Architecture Diagrams](pdf/architecture_diagram.pdf) - Visual architecture
- [References](pdf/References.pdf) - Research papers and datasets
- [Performance Metrics](docs/Thông%20số%20đánh%20giá.md) - Benchmarks

## License

This project contains model configurations and training scripts. Pre-trained models are subject to their respective licenses:
- Qwen2.5: Apache 2.0
- LLaMA3: Meta License
- Whisper: MIT License
