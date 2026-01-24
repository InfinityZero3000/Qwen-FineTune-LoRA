"""
Merge LoRA Adapters into Base Model

Script để merge tất cả LoRA adapters vào base Qwen model.
Output: Một model hoàn chỉnh không cần load adapters riêng.

Usage:
    python scripts/merge_lora_adapters.py
    
Output:
    export/qwen-1.5b-lexilingo-merged/
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import sys

def merge_adapters():
    """Merge all LoRA adapters into base model"""
    
    print("=" * 70)
    print("LexiLingo LoRA Adapter Merger")
    print("=" * 70)
    
    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    ADAPTER_DIR = Path("model/adapters")
    OUTPUT_DIR = Path("export/qwen-1.5b-lexilingo-merged")
    
    # Check if adapters exist
    adapters = ["grammar", "vocabulary", "fluency", "dialogue"]
    adapter_paths = [ADAPTER_DIR / f"{name}_lora_adapter" for name in adapters]
    
    missing = [p for p in adapter_paths if not p.exists()]
    if missing:
        print(f"❌ Missing adapters:")
        for p in missing:
            print(f"   - {p}")
        print("\nPlease train adapters first using finetune_qwen_lora.v3.0.ipynb")
        sys.exit(1)
    
    print(f"\n📦 Base model: {BASE_MODEL}")
    print(f"📁 Adapters found:")
    for name, path in zip(adapters, adapter_paths):
        print(f"   ✓ {name}: {path}")
    
    # Load base model
    print(f"\n⏳ Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU to save GPU memory
            trust_remote_code=True
        )
        print(f"   ✓ Loaded (size: {base_model.num_parameters() / 1e9:.2f}B params)")
    except Exception as e:
        print(f"   ❌ Failed to load base model: {e}")
        sys.exit(1)
    
    # Merge each adapter sequentially
    print(f"\n🔗 Merging adapters...")
    
    for i, (adapter_name, adapter_path) in enumerate(zip(adapters, adapter_paths), 1):
        print(f"   [{i}/{len(adapters)}] Merging {adapter_name}...", end=" ")
        try:
            # Load adapter onto current model
            model_with_adapter = PeftModel.from_pretrained(
                base_model, 
                str(adapter_path),
                is_trainable=False
            )
            
            # Merge and unload (combines LoRA weights into base)
            base_model = model_with_adapter.merge_and_unload()
            
            print("✓")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    # Save merged model
    print(f"\n💾 Saving merged model to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        base_model.save_pretrained(
            OUTPUT_DIR,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="2GB"      # Split into 2GB chunks
        )
        print(f"   ✓ Model saved")
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   ✓ Tokenizer saved")
        
    except Exception as e:
        print(f"   ❌ Failed to save: {e}")
        sys.exit(1)
    
    # Check output size
    model_files = list(OUTPUT_DIR.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in model_files)
    print(f"\n📊 Summary:")
    print(f"   Model files: {len(model_files)}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    print(f"   Location: {OUTPUT_DIR.absolute()}")
    
    print(f"\n✅ Done! Merged model ready for quantization.")
    print(f"\n🚀 Next step: Run export_to_gguf.py to create quantized version")
    print(f"   python scripts/export_to_gguf.py")


if __name__ == "__main__":
    merge_adapters()
