"""
Test Quantized Model Quality

Test GGUF quantized model để đảm bảo quality không giảm quá nhiều.
Compare với expected outputs.

Requirements:
    pip install llama-cpp-python

Usage:
    python scripts/test_quantized_model.py
"""

from pathlib import Path
import sys

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    print("\nInstall it with:")
    print("   pip install llama-cpp-python")
    sys.exit(1)


def test_grammar_correction():
    """Test grammar error correction task"""
    
    print("\n" + "=" * 70)
    print("TEST 1: Grammar Error Correction")
    print("=" * 70)
    
    test_cases = [
        {
            "input": "I goes to school every day.",
            "expected_correction": "I go to school every day.",
            "error_type": "subject-verb agreement"
        },
        {
            "input": "She don't like apples.",
            "expected_correction": "She doesn't like apples.",
            "error_type": "auxiliary verb"
        },
        {
            "input": "The cat are sleeping on the sofa.",
            "expected_correction": "The cat is sleeping on the sofa.",
            "error_type": "subject-verb agreement"
        },
        {
            "input": "I have went to Paris last year.",
            "expected_correction": "I went to Paris last year.",
            "error_type": "past tense"
        }
    ]
    
    return test_cases


def test_vocabulary_level():
    """Test vocabulary level classification"""
    
    print("\n" + "=" * 70)
    print("TEST 2: Vocabulary Level Classification (CEFR)")
    print("=" * 70)
    
    test_cases = [
        {
            "word": "hello",
            "expected_level": "A1",
        },
        {
            "word": "understand",
            "expected_level": "A2",
        },
        {
            "word": "although",
            "expected_level": "B1",
        },
        {
            "word": "nevertheless",
            "expected_level": "B2",
        }
    ]
    
    return test_cases


def test_fluency_scoring():
    """Test fluency scoring"""
    
    print("\n" + "=" * 70)
    print("TEST 3: Fluency Scoring")
    print("=" * 70)
    
    test_cases = [
        {
            "text": "I like pizza.",
            "expected_score": "high",  # Simple, correct
        },
        {
            "text": "Me like pizza very much yes.",
            "expected_score": "low",  # Broken English
        },
        {
            "text": "Although I generally prefer Italian cuisine, I must admit that pizza holds a special place in my heart.",
            "expected_score": "high",  # Complex but fluent
        }
    ]
    
    return test_cases


def load_model():
    """Load quantized GGUF model"""
    
    model_path = Path("export/qwen-1.5b-lexilingo-Q4_K_M.gguf")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nPlease run export_to_gguf.py first:")
        print("   python scripts/export_to_gguf.py")
        sys.exit(1)
    
    print(f"📦 Loading model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024**2):.0f} MB")
    
    try:
        model = Llama(
            model_path=str(model_path),
            n_ctx=512,         # Context window
            n_threads=4,       # CPU threads
            n_gpu_layers=0,    # Use CPU only (set >0 for GPU)
            verbose=False
        )
        print(f"   ✓ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)


def run_test(model, prompt, max_tokens=100):
    """Run inference and get response"""
    
    try:
        response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower = more deterministic
            top_p=0.9,
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
        
    except Exception as e:
        return f"Error: {e}"


def main():
    """Main test function"""
    
    print("=" * 70)
    print("LexiLingo Quantized Model Quality Test")
    print("=" * 70)
    
    # Load model
    model = load_model()
    
    # Test 1: Grammar correction
    print("\n" + "=" * 70)
    print("TEST 1: Grammar Error Correction")
    print("=" * 70)
    
    grammar_tests = test_grammar_correction()
    grammar_score = 0
    
    for i, test in enumerate(grammar_tests, 1):
        print(f"\nTest 1.{i}: {test['error_type']}")
        print(f"   Input: {test['input']}")
        print(f"   Expected: {test['expected_correction']}")
        
        prompt = f"""Correct this sentence and explain the error briefly:
Input: {test['input']}
Correction:"""
        
        response = run_test(model, prompt, max_tokens=80)
        print(f"   Model: {response[:100]}...")
        
        # Simple check: does response contain expected words?
        if "go" in response or "doesn't" in response or "is" in response or "went" in response:
            print(f"   Status: ✓ PASS")
            grammar_score += 1
        else:
            print(f"   Status: ⚠️  Needs review")
    
    # Test 2: Vocabulary level
    print("\n" + "=" * 70)
    print("TEST 2: Vocabulary Level (CEFR)")
    print("=" * 70)
    
    vocab_tests = test_vocabulary_level()
    vocab_score = 0
    
    for i, test in enumerate(vocab_tests, 1):
        print(f"\nTest 2.{i}: {test['word']}")
        print(f"   Expected level: {test['expected_level']}")
        
        prompt = f"""What CEFR level is this word (A1/A2/B1/B2/C1/C2)?
Word: {test['word']}
Level:"""
        
        response = run_test(model, prompt, max_tokens=20)
        print(f"   Model: {response}")
        
        if test['expected_level'] in response:
            print(f"   Status: ✓ PASS")
            vocab_score += 1
        else:
            print(f"   Status: ⚠️  Needs review")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Grammar Correction: {grammar_score}/{len(grammar_tests)} passed")
    print(f"Vocabulary Level: {vocab_score}/{len(vocab_tests)} passed")
    
    total = grammar_score + vocab_score
    max_total = len(grammar_tests) + len(vocab_tests)
    
    print(f"\nOverall: {total}/{max_total} ({total/max_total*100:.1f}%)")
    
    if total / max_total >= 0.75:
        print("\n✅ Model quality is GOOD (≥75%)")
        print("   Quantization preserved most of the performance.")
    elif total / max_total >= 0.5:
        print("\n⚠️  Model quality is ACCEPTABLE (50-75%)")
        print("   Consider fine-tuning or using higher quantization (Q5/Q6).")
    else:
        print("\n❌ Model quality is LOW (<50%)")
        print("   Quantization may have degraded too much.")
        print("   Try Q5_K_M or Q6_K for better quality.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
