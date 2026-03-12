#!/usr/bin/env python3
"""
Export Models to Mobile Formats (GGUF, Core ML, ONNX)
======================================================
Converts trained models to optimized mobile formats:

1. GGUF (llama.cpp): For Android/iOS LLM inference
2. Core ML: For iOS NPU acceleration
3. ONNX: Cross-platform inference

Supports:
- Qwen2.5-0.5B (Grammar)
- SmolLM2-360M (Conversation)
- Whisper Small (STT)
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


class ExportFormat(Enum):
    """Supported export formats."""
    GGUF = "gguf"
    COREML = "coreml"
    ONNX = "onnx"
    MLPACKAGE = "mlpackage"


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    # Paths
    model_path: str = "./outputs"
    output_dir: str = "./exports"
    
    # GGUF settings
    gguf_quantization: str = "Q4_K_M"  # Best balance for mobile
    gguf_available_quants: List[str] = None
    
    # Core ML settings
    coreml_compute_units: str = "ALL"  # CPU, GPU, NPU
    coreml_precision: str = "float16"
    
    # ONNX settings
    onnx_opset_version: int = 17
    onnx_optimize: bool = True
    
    def __post_init__(self):
        if self.gguf_available_quants is None:
            self.gguf_available_quants = [
                "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",  # 4-bit
                "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",  # 5-bit
                "Q8_0",  # 8-bit
                "F16",   # 16-bit float
            ]


# =============================================================================
# GGUF Export (for llama.cpp)
# =============================================================================
def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M",
    model_name: str = "model"
) -> Optional[str]:
    """
    Export model to GGUF format for llama.cpp.
    
    Requires: llama.cpp installed with convert scripts
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output directory
        quantization: Quantization level (Q4_K_M recommended)
        model_name: Name for output file
    
    Returns:
        Path to exported model or None if failed
    """
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_file = output_dir / f"{model_name}-{quantization.lower()}.gguf"
    
    print(f"\nExporting to GGUF format...")
    print(f"   Model: {model_path}")
    print(f"   Quantization: {quantization}")
    print(f"   Output: {output_file}")
    
    # Method 1: Using llama.cpp convert script
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "llama.cpp")
    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
    
    if convert_script.exists():
        try:
            # Step 1: Convert to FP16 GGUF
            fp16_file = output_dir / f"{model_name}-f16.gguf"
            
            cmd = [
                sys.executable,
                str(convert_script),
                model_path,
                "--outfile", str(fp16_file),
                "--outtype", "f16"
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   Conversion failed: {result.stderr}")
                return None
            
            # Step 2: Quantize
            if quantization != "F16" and quantize_bin.exists():
                cmd = [
                    str(quantize_bin),
                    str(fp16_file),
                    str(output_file),
                    quantization
                ]
                
                print(f"   Quantizing to {quantization}...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"   Quantization failed: {result.stderr}")
                    return str(fp16_file)  # Return FP16 version
                
                # Remove FP16 intermediate file
                fp16_file.unlink()
            else:
                output_file = fp16_file
            
            print(f"   Exported to {output_file}")
            print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
            
            return str(output_file)
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Method 2: Using transformers + ctransformers
    print("   llama.cpp not found, trying alternative method...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # This is a placeholder - actual conversion requires llama.cpp
        print("   For GGUF export, please install llama.cpp:")
        print("      git clone https://github.com/ggerganov/llama.cpp")
        print("      cd llama.cpp && make")
        print("      export LLAMA_CPP_PATH=/path/to/llama.cpp")
        
        return None
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


# =============================================================================
# Core ML Export (for iOS)
# =============================================================================
def export_to_coreml(
    model_path: str,
    output_path: str,
    model_name: str = "model",
    compute_units: str = "ALL"
) -> Optional[str]:
    """
    Export model to Core ML format for iOS.
    
    Requires: coremltools
    
    Args:
        model_path: Path to HuggingFace/PyTorch model
        output_path: Output directory
        model_name: Name for output file
        compute_units: ALL, CPU_AND_GPU, CPU_AND_NE (Neural Engine)
    
    Returns:
        Path to exported model or None if failed
    """
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{model_name}.mlpackage"
    
    print(f"\n🍎 Exporting to Core ML format...")
    print(f"   Model: {model_path}")
    print(f"   Compute Units: {compute_units}")
    print(f"   Output: {output_file}")
    
    try:
        import coremltools as ct
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load model
        print("   Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        
        # Create traced model
        print("   Tracing model...")
        
        # Example input for tracing
        example_text = "Hello, how are you?"
        inputs = tokenizer(example_text, return_tensors="pt")
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                (inputs["input_ids"], inputs["attention_mask"])
            )
        
        # Convert to Core ML
        print("   Converting to Core ML...")
        
        # Define compute units
        compute_unit_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            compute_units=compute_unit_map.get(compute_units, ct.ComputeUnit.ALL),
            minimum_deployment_target=ct.target.iOS16
        )
        
        # Save model
        mlmodel.save(str(output_file))
        
        print(f"   Exported to {output_file}")
        
        # Calculate size
        total_size = sum(
            f.stat().st_size for f in output_file.rglob("*") if f.is_file()
        )
        print(f"   Size: {total_size / 1024 / 1024:.1f} MB")
        
        return str(output_file)
        
    except ImportError:
        print("   coremltools not installed. Install with:")
        print("      pip install coremltools")
        return None
    except Exception as e:
        print(f"   Error: {e}")
        return None


# =============================================================================
# ONNX Export (Cross-platform)
# =============================================================================
def export_to_onnx(
    model_path: str,
    output_path: str,
    model_name: str = "model",
    opset_version: int = 17,
    optimize: bool = True
) -> Optional[str]:
    """
    Export model to ONNX format.
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output directory
        model_name: Name for output file
        opset_version: ONNX opset version
        optimize: Whether to optimize the model
    
    Returns:
        Path to exported model or None if failed
    """
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{model_name}.onnx"
    
    print(f"\n🔷 Exporting to ONNX format...")
    print(f"   Model: {model_path}")
    print(f"   Opset: {opset_version}")
    print(f"   Output: {output_file}")
    
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
        
        print("   Loading and converting model...")
        
        # Use Optimum for easy ONNX export
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_path,
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Save model
        ort_model.save_pretrained(str(output_dir / model_name))
        
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(str(output_dir / model_name))
        
        print(f"   Exported to {output_dir / model_name}")
        
        # Optimize if requested
        if optimize:
            print("   Optimizing model...")
            try:
                from onnxruntime.transformers import optimizer
                
                optimized_file = output_dir / f"{model_name}_optimized.onnx"
                
                # Find the main ONNX file
                onnx_files = list((output_dir / model_name).glob("*.onnx"))
                if onnx_files:
                    optimizer.optimize_model(
                        str(onnx_files[0]),
                        str(optimized_file),
                        optimization_options=None
                    )
                    print(f"   Optimized model saved")
            except Exception as e:
                print(f"   Optimization skipped: {e}")
        
        return str(output_dir / model_name)
        
    except ImportError:
        print("   optimum not installed. Install with:")
        print("      pip install optimum[onnxruntime]")
        return None
    except Exception as e:
        print(f"   Error: {e}")
        return None


# =============================================================================
# Whisper Export
# =============================================================================
def export_whisper_to_coreml(
    model_path: str,
    output_path: str,
    model_name: str = "whisper"
) -> Optional[str]:
    """Export Whisper model to Core ML for iOS."""
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎤 Exporting Whisper to Core ML...")
    print(f"   Model: {model_path}")
    
    try:
        # Use whisper.cpp or coremltools
        # whisper.cpp provides pre-built Core ML models
        
        print("  For Whisper Core ML models, use whisper.cpp:")
        print("      git clone https://github.com/ggerganov/whisper.cpp")
        print("      cd whisper.cpp/models")
        print("      ./download-ggml-model.sh small")
        print("      ./generate-coreml-model.sh small")
        
        # Alternative: Use Hugging Face's whisper-coreml
        print("\n   Or download pre-converted models from:")
        print("      https://huggingface.co/apple/whisper-small-coreml")
        
        return None
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


# =============================================================================
# Export All Models
# =============================================================================
def export_all_models(config: ExportConfig):
    """Export all trained models to mobile formats."""
    
    print("=" * 60)
    print("LexiLingo Model Export Pipeline")
    print("=" * 60)
    
    results = {}
    
    # 1. Export Qwen Grammar Model
    qwen_path = Path(config.model_path) / "qwen-grammar"
    if qwen_path.exists():
        print("\nExporting Qwen Grammar Model...")
        
        # GGUF for llama.cpp
        gguf_result = export_to_gguf(
            str(qwen_path),
            str(Path(config.output_dir) / "gguf"),
            config.gguf_quantization,
            "qwen-grammar"
        )
        results["qwen_gguf"] = gguf_result
        
        # Core ML for iOS
        coreml_result = export_to_coreml(
            str(qwen_path),
            str(Path(config.output_dir) / "coreml"),
            "qwen-grammar",
            config.coreml_compute_units
        )
        results["qwen_coreml"] = coreml_result
    else:
        print(f"   Qwen model not found at {qwen_path}")
    
    # 2. Export SmolLM Conversation Model
    smolm_path = Path(config.model_path) / "smolm-conversation"
    if smolm_path.exists():
        print("\nExporting SmolLM Conversation Model...")
        
        # GGUF
        gguf_result = export_to_gguf(
            str(smolm_path),
            str(Path(config.output_dir) / "gguf"),
            config.gguf_quantization,
            "smolm-conversation"
        )
        results["smolm_gguf"] = gguf_result
        
        # Core ML
        coreml_result = export_to_coreml(
            str(smolm_path),
            str(Path(config.output_dir) / "coreml"),
            "smolm-conversation",
            config.coreml_compute_units
        )
        results["smolm_coreml"] = coreml_result
    else:
        print(f"   SmolLM model not found at {smolm_path}")
    
    # 3. Export Whisper STT Model
    whisper_path = Path(config.model_path) / "whisper-english"
    if whisper_path.exists():
        print("\n🎤 Exporting Whisper STT Model...")
        
        # Core ML for iOS
        coreml_result = export_whisper_to_coreml(
            str(whisper_path),
            str(Path(config.output_dir) / "coreml"),
            "whisper-english"
        )
        results["whisper_coreml"] = coreml_result
        
        # ONNX for cross-platform
        onnx_result = export_to_onnx(
            str(whisper_path),
            str(Path(config.output_dir) / "onnx"),
            "whisper-english",
            config.onnx_opset_version
        )
        results["whisper_onnx"] = onnx_result
    else:
        print(f"   Whisper model not found at {whisper_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    
    for name, path in results.items():
        status = "" if path else ""
        print(f"   {status} {name}: {path or 'Failed'}")
    
    # Save results
    results_file = Path(config.output_dir) / "export_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export LexiLingo models to mobile formats"
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Export all models"
    )
    parser.add_argument(
        "--model", type=str,
        help="Path to specific model to export"
    )
    parser.add_argument(
        "--format", type=str,
        choices=["gguf", "coreml", "onnx"],
        help="Export format"
    )
    parser.add_argument(
        "--quantization", type=str,
        default="Q4_K_M",
        help="GGUF quantization level"
    )
    parser.add_argument(
        "--output", type=str,
        default="./exports",
        help="Output directory"
    )
    parser.add_argument(
        "--name", type=str,
        default="model",
        help="Output model name"
    )
    
    args = parser.parse_args()
    
    config = ExportConfig(
        output_dir=args.output,
        gguf_quantization=args.quantization
    )
    
    if args.all:
        export_all_models(config)
    
    elif args.model and args.format:
        if args.format == "gguf":
            export_to_gguf(
                args.model,
                args.output,
                args.quantization,
                args.name
            )
        elif args.format == "coreml":
            export_to_coreml(
                args.model,
                args.output,
                args.name
            )
        elif args.format == "onnx":
            export_to_onnx(
                args.model,
                args.output,
                args.name
            )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
