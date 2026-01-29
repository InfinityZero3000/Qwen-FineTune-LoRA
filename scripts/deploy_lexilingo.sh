#!/bin/bash
# ============================================================================
# LexiLingo GGUF Conversion & Deployment Pipeline
# ============================================================================
# 
# This script automates all steps after training:
# 1. Download merged model from HuggingFace (or unzip from Kaggle)
# 2. Convert to GGUF F16
# 3. Quantize to Q4_K_M
# 4. Test inference
# 5. Start server
#
# Usage:
#   ./deploy_lexilingo.sh [options]
#
# Options:
#   -h, --help              Show this help message
#   -m, --mode MODE         'hf' (HuggingFace) or 'zip' (Kaggle zip)
#   -u, --username USER     HuggingFace username (for -m hf)
#   -z, --zip-file FILE     Path to zip file (for -m zip)
#   -q, --quantize TYPE     Q4_K_M (default), Q4_K_S, Q5_K_M, Q8_0
#   -p, --port PORT         Server port (default: 8080)
#   -t, --threads THREADS   CPU threads (default: auto-detect)
#   --skip-download         Don't download/extract model
#   --skip-convert          Don't convert to GGUF F16
#   --skip-quantize         Don't quantize
#   --skip-test             Don't test inference
#   --server-only           Start server without other steps
#
# Examples:
#   # Download from HuggingFace and deploy
#   ./deploy_lexilingo.sh -m hf -u your-username
#
#   # Extract zip from Kaggle
#   ./deploy_lexilingo.sh -m zip -z ~/Downloads/lexilingo_merged.zip
#
#   # Just start server
#   ./deploy_lexilingo.sh --server-only
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Projects/llama.cpp}"
MODEL_DIR="$LLAMA_DIR/models"
DOWNLOAD_MODE="hf"  # hf or zip
HF_USERNAME=""
ZIP_FILE=""
QUANTIZE_TYPE="Q4_K_M"
SERVER_PORT=8080
CPU_THREADS=""
SKIP_DOWNLOAD=false
SKIP_CONVERT=false
SKIP_QUANTIZE=false
SKIP_TEST=false
SERVER_ONLY=false

# Model names
MERGED_MODEL_NAME="lexilingo_qwen25_1.5b_merged"
F16_MODEL="$MODEL_DIR/lexilingo_f16.gguf"
QUANTIZED_MODEL="$MODEL_DIR/lexilingo_q4_km.gguf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "\n${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "\n${RED}✗${NC} $1"
}

print_warning() {
    echo -e "\n${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

show_help() {
    head -n 55 "$SCRIPT_DIR/deploy_lexilingo.sh" | tail -n 50
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--mode)
            DOWNLOAD_MODE="$2"
            shift 2
            ;;
        -u|--username)
            HF_USERNAME="$2"
            shift 2
            ;;
        -z|--zip-file)
            ZIP_FILE="$2"
            shift 2
            ;;
        -q|--quantize)
            QUANTIZE_TYPE="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -t|--threads)
            CPU_THREADS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-quantize)
            SKIP_QUANTIZE=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --server-only)
            SERVER_ONLY=true
            SKIP_DOWNLOAD=true
            SKIP_CONVERT=true
            SKIP_QUANTIZE=true
            SKIP_TEST=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "LexiLingo GGUF Conversion & Deployment"

print_step "Checking prerequisites..."

# Check llama.cpp
if [ ! -d "$LLAMA_DIR" ]; then
    print_error "llama.cpp directory not found: $LLAMA_DIR"
    echo "Please install llama.cpp first:"
    echo "  cd ~/Projects"
    echo "  git clone https://github.com/ggerganov/llama.cpp.git"
    echo "  cd llama.cpp && make"
    exit 1
fi

if [ ! -f "$LLAMA_DIR/llama-cli" ]; then
    print_error "llama-cli not found. Please build llama.cpp:"
    echo "  cd $LLAMA_DIR && make"
    exit 1
fi

if [ ! -f "$LLAMA_DIR/llama-quantize" ]; then
    print_error "llama-quantize not found. Please build llama.cpp:"
    echo "  cd $LLAMA_DIR && make"
    exit 1
fi

print_info "llama.cpp found: $LLAMA_DIR"

# Create model directory
mkdir -p "$MODEL_DIR"

# Auto-detect CPU threads
if [ -z "$CPU_THREADS" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CPU_THREADS=$(sysctl -n hw.ncpu)
    else
        CPU_THREADS=$(nproc)
    fi
    print_info "Auto-detected CPU threads: $CPU_THREADS"
fi

# ============================================================================
# Step 1: Download/Extract Model
# ============================================================================

if [ "$SKIP_DOWNLOAD" = false ] && [ "$SERVER_ONLY" = false ]; then
    print_header "Step 1: Download Merged Model"
    
    if [ "$DOWNLOAD_MODE" = "hf" ]; then
        if [ -z "$HF_USERNAME" ]; then
            print_error "HuggingFace username required for -m hf"
            echo "Usage: $0 -m hf -u your-username"
            exit 1
        fi
        
        HF_REPO="$HF_USERNAME/lexilingo-qwen25-1.5b"
        
        print_step "Downloading from HuggingFace: $HF_REPO"
        
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download "$HF_REPO" \
                --local-dir "$MODEL_DIR/$MERGED_MODEL_NAME" \
                --local-dir-use-symlinks False
            print_step "Model downloaded"
        else
            print_error "huggingface-cli not found"
            echo "Install it with: pip install huggingface_hub"
            exit 1
        fi
        
    elif [ "$DOWNLOAD_MODE" = "zip" ]; then
        if [ -z "$ZIP_FILE" ]; then
            print_error "ZIP file required for -m zip"
            echo "Usage: $0 -m zip -z ~/Downloads/lexilingo_merged.zip"
            exit 1
        fi
        
        if [ ! -f "$ZIP_FILE" ]; then
            print_error "ZIP file not found: $ZIP_FILE"
            exit 1
        fi
        
        print_step "Extracting: $ZIP_FILE"
        unzip -q "$ZIP_FILE" -d "$MODEL_DIR"
        print_step "Model extracted"
        
    else
        print_error "Invalid download mode: $DOWNLOAD_MODE"
        echo "Use 'hf' (HuggingFace) or 'zip' (Kaggle)"
        exit 1
    fi
    
    # Verify merged model
    if [ ! -f "$MODEL_DIR/$MERGED_MODEL_NAME/model.safetensors" ] && \
       [ -z "$(find "$MODEL_DIR/$MERGED_MODEL_NAME" -name "model-*.safetensors" 2>/dev/null)" ]; then
        print_error "Model file not found after download"
        echo "Expected: $MODEL_DIR/$MERGED_MODEL_NAME/model.safetensors"
        exit 1
    fi
    print_info "Model verified"
fi

# ============================================================================
# Step 2: Convert to GGUF F16
# ============================================================================

if [ "$SKIP_CONVERT" = false ] && [ "$SERVER_ONLY" = false ]; then
    print_header "Step 2: Convert to GGUF F16"
    
    MERGED_MODEL_PATH="$MODEL_DIR/$MERGED_MODEL_NAME"
    
    if [ ! -d "$MERGED_MODEL_PATH" ]; then
        print_error "Merged model directory not found: $MERGED_MODEL_PATH"
        exit 1
    fi
    
    if [ -f "$F16_MODEL" ]; then
        print_warning "GGUF F16 already exists: $F16_MODEL"
        read -p "Skip conversion? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping conversion"
        else
            rm "$F16_MODEL"
        fi
    fi
    
    if [ ! -f "$F16_MODEL" ]; then
        print_step "Converting to GGUF F16..."
        print_info "Source: $MERGED_MODEL_PATH"
        print_info "Output: $F16_MODEL"
        print_info "This may take 2-3 minutes..."
        
        cd "$LLAMA_DIR"
        python3 convert_hf_to_gguf.py \
            "$MERGED_MODEL_PATH" \
            --outfile "$F16_MODEL" \
            --outtype f16 \
            --model-name qwen
        
        if [ ! -f "$F16_MODEL" ]; then
            print_error "Conversion failed"
            exit 1
        fi
        print_step "Conversion complete"
    fi
    
    # Check file size
    F16_SIZE=$(du -h "$F16_MODEL" | cut -f1)
    print_info "GGUF F16 size: $F16_SIZE"
fi

# ============================================================================
# Step 3: Quantize to Q4_K_M
# ============================================================================

if [ "$SKIP_QUANTIZE" = false ] && [ "$SERVER_ONLY" = false ]; then
    print_header "Step 3: Quantize to $QUANTIZE_TYPE"
    
    QUANT_MODEL="$MODEL_DIR/lexilingo_${QUANTIZE_TYPE,,}.gguf"
    
    if [ ! -f "$F16_MODEL" ]; then
        print_error "GGUF F16 model not found: $F16_MODEL"
        exit 1
    fi
    
    if [ -f "$QUANT_MODEL" ]; then
        print_warning "Quantized model already exists: $QUANT_MODEL"
        read -p "Skip quantization? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping quantization"
        else
            rm "$QUANT_MODEL"
        fi
    fi
    
    if [ ! -f "$QUANT_MODEL" ]; then
        print_step "Quantizing to $QUANTIZE_TYPE..."
        print_info "Source: $F16_MODEL"
        print_info "Output: $QUANT_MODEL"
        print_info "This may take 1-2 minutes..."
        
        cd "$LLAMA_DIR"
        ./llama-quantize "$F16_MODEL" "$QUANT_MODEL" "$QUANTIZE_TYPE"
        
        if [ ! -f "$QUANT_MODEL" ]; then
            print_error "Quantization failed"
            exit 1
        fi
        print_step "Quantization complete"
    fi
    
    # Check file size
    QUANT_SIZE=$(du -h "$QUANT_MODEL" | cut -f1)
    print_info "Quantized model size: $QUANT_SIZE"
    
    # Calculate compression ratio
    if [ -f "$F16_MODEL" ]; then
        F16_BYTES=$(stat -f%z "$F16_MODEL" 2>/dev/null || stat -c%s "$F16_MODEL")
        QUANT_BYTES=$(stat -f%z "$QUANT_MODEL" 2>/dev/null || stat -c%s "$QUANT_MODEL")
        RATIO=$(echo "scale=1; $F16_BYTES / $QUANT_BYTES" | bc)
        print_info "Compression ratio: ${RATIO}x smaller"
    fi
    
    QUANTIZED_MODEL="$QUANT_MODEL"
fi

# ============================================================================
# Step 4: Test Inference
# ============================================================================

if [ "$SKIP_TEST" = false ] && [ "$SERVER_ONLY" = false ]; then
    print_header "Step 4: Test Inference"
    
    if [ ! -f "$QUANTIZED_MODEL" ]; then
        print_error "Quantized model not found: $QUANTIZED_MODEL"
        exit 1
    fi
    
    print_step "Testing model with sample prompts..."
    
    cd "$LLAMA_DIR"
    
    # Test 1: Fluency
    echo -e "\n${BLUE}[Test 1] Fluency Analysis${NC}"
    ./llama-cli \
        -m "$QUANTIZED_MODEL" \
        -p "Analyze the fluency of this sentence: The cat sat on the mat." \
        -n 32 \
        -t "$CPU_THREADS" \
        --log-disable | head -n 10
    
    # Test 2: Vocabulary
    echo -e "\n${BLUE}[Test 2] Vocabulary Classification${NC}"
    ./llama-cli \
        -m "$QUANTIZED_MODEL" \
        -p "Classify the vocabulary level: The phenomenon is fascinating." \
        -n 16 \
        -t "$CPU_THREADS" \
        --log-disable | head -n 10
    
    print_step "Inference test complete"
fi

# ============================================================================
# Step 5: Start Server
# ============================================================================

print_header "Step 5: Start llama.cpp Server"

if [ ! -f "$QUANTIZED_MODEL" ]; then
    print_error "Quantized model not found: $QUANTIZED_MODEL"
    exit 1
fi

print_step "Starting server..."
print_info "Model: $QUANTIZED_MODEL"
print_info "Port: $SERVER_PORT"
print_info "Threads: $CPU_THREADS"
print_info "Server URL: http://localhost:$SERVER_PORT"

cd "$LLAMA_DIR"

# Display server info
echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}LexiLingo Server Ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "API Endpoint: http://localhost:$SERVER_PORT/v1/chat/completions"
echo ""
echo "Example request:"
echo '  curl http://localhost:8080/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{
echo '      "messages": [
echo '        {"role": "user", "content": "Analyze fluency: The cat sat on the mat."}
echo '      ],
echo '      "temperature": 0.7,
echo '      "max_tokens": 128
echo '    }'"'"''
echo ""
echo "Python client:"
echo "  from export.lexilingo_client import LexiLingoClient"
echo "  with LexiLingoClient('$QUANTIZED_MODEL', mode='server') as client:"
echo "      result = client.analyze_fluency('The cat sat on the mat.')"
echo "      print(f'Score: {result.score}')"
echo ""
echo "Press Ctrl+C to stop server"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"

# Start server
./llama-server \
    -m "$QUANTIZED_MODEL" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    -n 512 \
    --ctx-size 2048 \
    -t "$CPU_THREADS"

# ============================================================================
# End
# ============================================================================

echo -e "\n${GREEN}Server stopped${NC}"
