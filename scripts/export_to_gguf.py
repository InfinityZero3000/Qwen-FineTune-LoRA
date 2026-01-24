"""
Export Model to GGUF Format (Quantized)

Convert merged Hugging Face model to GGUF format với 4-bit quantization.
Giảm model size từ 3GB → 900MB, RAM usage từ 3GB → 1.2GB.

Requirements:
    - llama.cpp (clone from https://github.com/ggerganov/llama.cpp)
    - Merged model (chạy merge_lora_adapters.py trước)

Usage:
    python scripts/export_to_gguf.py
    
Output:
    export/qwen-1.5b-lexilingo-Q4_K_M.gguf (~900MB)
"""

import subprocess
from pathlib import Path
import sys
import shutil

def check_requirements():
    """Check if llama.cpp is available"""
    
    print("🔍 Checking requirements...")
    
    # Check for llama.cpp
    llamacpp = Path("llama.cpp")
    if not llamacpp.exists():
        print("\n❌ llama.cpp not found!")
        print("\nPlease clone llama.cpp first:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print("   make")
        return False
    
    # Check for convert script
    convert_script = llamacpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try old location
        convert_script = llamacpp / "convert.py"
        if not convert_script.exists():
            print(f"❌ Convert script not found in llama.cpp/")
            return False
    
    print(f"   ✓ llama.cpp found at {llamacpp}")
    print(f"   ✓ Convert script: {convert_script.name}")
    
    # Check for merged model
    merged_model = Path("export/qwen-1.5b-lexilingo-merged")
    if not merged_model.exists():
        print(f"\n❌ Merged model not found at {merged_model}")
        print("\nPlease run merge_lora_adapters.py first:")
        print("   python scripts/merge_lora_adapters.py")
        return False
    
    print(f"   ✓ Merged model found")
    
    return True, convert_script


def export_gguf(convert_script):
    """Export to GGUF format"""
    
    print("\n" + "=" * 70)
    print("LexiLingo GGUF Exporter (4-bit Quantization)")
    print("=" * 70)
    
    # Configuration
    MERGED_MODEL = Path("export/qwen-1.5b-lexilingo-merged")
    OUTPUT_FILE = Path("export/qwen-1.5b-lexilingo-Q4_K_M.gguf")
    
    # Create export directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Input: {MERGED_MODEL}")
    print(f"📤 Output: {OUTPUT_FILE}")
    print(f"🔧 Quantization: Q4_K_M (4-bit)")
    
    # Step 1: Convert to FP16 GGUF first
    print(f"\n⏳ Step 1/2: Converting to FP16 GGUF...")
    temp_fp16 = OUTPUT_FILE.parent / "temp_fp16.gguf"
    
    try:
        cmd = [
            "python3",
            str(convert_script),
            str(MERGED_MODEL),
            "--outfile", str(temp_fp16),
            "--outtype", "f16"
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✓ FP16 GGUF created")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Conversion failed!")
        print(f"   Error: {e.stderr}")
        sys.exit(1)
    
    # Step 2: Quantize to Q4_K_M
    print(f"\n⏳ Step 2/2: Quantizing to Q4_K_M...")
    
    try:
        quantize_bin = Path("llama.cpp/build/bin/llama-quantize")
        if not quantize_bin.exists():
            # Try alternative location
            quantize_bin = Path("llama.cpp/llama-quantize")
            if not quantize_bin.exists():
                # Build it
                print("   Building quantize tool...")
                subprocess.run(["make", "-C", "llama.cpp"], check=True)
                quantize_bin = Path("llama.cpp/llama-quantize")
        
        cmd = [
            str(quantize_bin),
            str(temp_fp16),
            str(OUTPUT_FILE),
            "Q4_K_M"
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✓ Quantization complete")
        
        # Remove temp file
        temp_fp16.unlink()
        print(f"   ✓ Cleaned up temp files")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Quantization failed!")
        print(f"   Error: {e.stderr}")
        sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"   ❌ Quantize tool not found!")
        print(f"\n   Please build llama.cpp first:")
        print(f"      cd llama.cpp")
        print(f"      make")
        sys.exit(1)
    
    # Check output
    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        
        print(f"\n📊 Summary:")
        print(f"   Original size: ~3000 MB (float16)")
        print(f"   Quantized size: {size_mb:.0f} MB (Q4_K_M)")
        print(f"   Reduction: {(1 - size_mb/3000) * 100:.1f}%")
        print(f"   Location: {OUTPUT_FILE.absolute()}")
        
        print(f"\n✅ Done! Quantized model ready for deployment.")
        print(f"\n🚀 Next step: Test the model")
        print(f"   python scripts/test_quantized_model.py")
        
        return True
    else:
        print(f"\n❌ Output file not created!")
        return False


def main():
    """Main function"""
    
    # Check requirements
    result = check_requirements()
    if result is False:
        sys.exit(1)
    
    ready, convert_script = result
    
    if not ready:
        sys.exit(1)
    
    # Export
    success = export_gguf(convert_script)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
