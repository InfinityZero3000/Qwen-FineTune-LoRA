#!/usr/bin/env python3
"""
LexiLingo Client - Gọi model sau khi deploy với llama.cpp

3 cách sử dụng:
1. CLI: Gọi trực tiếp ./llama-cli
2. Server API: Khởi động server và gọi qua REST API
3. Python Library: Wrapper class dễ sử dụng
"""

import requests
import subprocess
import json
from typing import Optional, Dict, Any, List
import time
import os
from dataclasses import dataclass

# ============================================================================
# CÁCH 1: CLI - Command Line Interface (Đơn giản nhất)
# ============================================================================

class LexiLingoCliClient:
    """Gọi model qua llama-cli command"""
    
    def __init__(self, model_path: str, llama_dir: str = "~/Projects/llama.cpp"):
        self.model_path = os.path.expanduser(model_path)
        self.llama_cli = os.path.expanduser(f"{llama_dir}/llama-cli")
        
        if not os.path.exists(self.llama_cli):
            raise FileNotFoundError(f"llama-cli not found at {self.llama_cli}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def query(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Gọi model với prompt
        
        Example:
            client = LexiLingoCliClient("models/lexilingo_q4_km.gguf")
            result = client.query("Analyze fluency: The cat sat on the mat.")
            print(result)
        """
        cmd = [
            self.llama_cli,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", "0.7",
            "--top-p", "0.9",
            "--repeat-penalty", "1.1",
            "--log-disable",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"llama-cli error: {result.stderr}")
        
        # Parse output (remove prompt echo)
        output = result.stdout.strip()
        if prompt in output:
            output = output.split(prompt, 1)[1].strip()
        
        return output


# ============================================================================
# CÁCH 2: Server API - REST API (Production-ready)
# ============================================================================

class LexiLingoServerClient:
    """Gọi model qua llama.cpp server (REST API)"""
    
    def __init__(self, 
                 model_path: str,
                 host: str = "localhost",
                 port: int = 8080,
                 llama_dir: str = "~/Projects/llama.cpp",
                 auto_start: bool = True):
        self.model_path = os.path.expanduser(model_path)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.llama_server = os.path.expanduser(f"{llama_dir}/llama-server")
        self.server_process = None
        
        if auto_start:
            self.start_server()
    
    def start_server(self):
        """Khởi động llama.cpp server"""
        print(f"🚀 Starting llama.cpp server on {self.host}:{self.port}...")
        
        cmd = [
            self.llama_server,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-n", "512",
            "--ctx-size", "2048",
            "--log-disable",
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to start
        for i in range(10):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Server started successfully")
                    return
            except:
                time.sleep(0.5)
        
        raise RuntimeError("Server failed to start")
    
    def stop_server(self):
        """Dừng server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            print("✅ Server stopped")
    
    def query(self, 
              prompt: str,
              max_tokens: int = 256,
              temperature: float = 0.7,
              top_p: float = 0.9) -> str:
        """
        Gọi model qua REST API
        
        Example:
            client = LexiLingoServerClient("models/lexilingo_q4_km.gguf")
            result = client.query("Analyze fluency: The cat sat on the mat.")
            print(result)
        """
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()


# ============================================================================
# CÁCH 3: High-level API - Task-specific methods (Dễ nhất)
# ============================================================================

@dataclass
class FluencyResult:
    """Kết quả phân tích fluency"""
    score: float
    raw_output: str

@dataclass
class VocabularyResult:
    """Kết quả phân loại vocabulary"""
    level: str  # A1, A2, B1, B2, C1, C2
    raw_output: str

@dataclass
class GrammarResult:
    """Kết quả sửa grammar"""
    corrected_sentence: str
    raw_output: str

@dataclass
class DialogueResult:
    """Kết quả tạo dialogue"""
    response: str
    raw_output: str

@dataclass
class ExplanationResult:
    """Kết quả giải thích lỗi"""
    explanation: str
    raw_output: str


class LexiLingoClient:
    """
    High-level client cho LexiLingo model
    Cung cấp methods cho từng task cụ thể
    
    Example:
        # Sử dụng với server (recommended)
        with LexiLingoClient("models/lexilingo_q4_km.gguf", mode="server") as client:
            score = client.analyze_fluency("The cat sat on the mat.")
            level = client.classify_vocabulary("The phenomenon is fascinating.")
            corrected = client.correct_grammar("She don't like apples.")
        
        # Hoặc CLI mode (đơn giản hơn, nhưng chậm hơn)
        client = LexiLingoClient("models/lexilingo_q4_km.gguf", mode="cli")
        score = client.analyze_fluency("The cat sat on the mat.")
    """
    
    def __init__(self, 
                 model_path: str,
                 mode: str = "server",  # "server" or "cli"
                 llama_dir: str = "~/Projects/llama.cpp",
                 host: str = "localhost",
                 port: int = 8080):
        """
        Args:
            model_path: Path to GGUF model file
            mode: "server" (faster, production) or "cli" (simpler, testing)
            llama_dir: Directory chứa llama.cpp
            host: Server host (only for server mode)
            port: Server port (only for server mode)
        """
        self.mode = mode
        
        if mode == "server":
            self.client = LexiLingoServerClient(model_path, host, port, llama_dir)
        elif mode == "cli":
            self.client = LexiLingoCliClient(model_path, llama_dir)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'server' or 'cli'")
    
    # ========================================================================
    # Task 1: Fluency Analysis
    # ========================================================================
    
    def analyze_fluency(self, sentence: str) -> FluencyResult:
        """
        Phân tích fluency score (0.0-5.0)
        
        Args:
            sentence: Câu cần phân tích
            
        Returns:
            FluencyResult với score và raw output
            
        Example:
            >>> result = client.analyze_fluency("The cat sat on the mat.")
            >>> print(f"Fluency score: {result.score}")
            Fluency score: 5.0
        """
        prompt = f"Analyze the fluency of this sentence: {sentence}"
        raw = self.client.query(prompt, max_tokens=32)
        
        # Parse score từ output
        try:
            # Try to extract number from output
            import re
            numbers = re.findall(r'\d+\.?\d*', raw)
            if numbers:
                score = float(numbers[0])
                # Clamp to 0-5 range
                score = max(0.0, min(5.0, score))
            else:
                score = 0.0
        except:
            score = 0.0
        
        return FluencyResult(score=score, raw_output=raw)
    
    # ========================================================================
    # Task 2: Vocabulary Classification
    # ========================================================================
    
    def classify_vocabulary(self, sentence: str) -> VocabularyResult:
        """
        Phân loại vocabulary level theo CEFR (A1-C2)
        
        Args:
            sentence: Câu cần phân loại
            
        Returns:
            VocabularyResult với level và raw output
            
        Example:
            >>> result = client.classify_vocabulary("The phenomenon is fascinating.")
            >>> print(f"Level: {result.level}")
            Level: B2
        """
        prompt = f"Classify the vocabulary level: {sentence}"
        raw = self.client.query(prompt, max_tokens=16)
        
        # Parse level từ output
        import re
        match = re.search(r'\b([ABC][12])\b', raw.upper())
        level = match.group(1) if match else "A1"
        
        return VocabularyResult(level=level, raw_output=raw)
    
    # ========================================================================
    # Task 3: Grammar Correction
    # ========================================================================
    
    def correct_grammar(self, sentence: str) -> GrammarResult:
        """
        Sửa lỗi grammar trong câu
        
        Args:
            sentence: Câu có lỗi grammar
            
        Returns:
            GrammarResult với câu đã sửa và raw output
            
        Example:
            >>> result = client.correct_grammar("She don't like apples.")
            >>> print(f"Corrected: {result.corrected_sentence}")
            Corrected: She doesn't like apples.
        """
        prompt = f"Correct this sentence: {sentence}"
        raw = self.client.query(prompt, max_tokens=128)
        
        # Extract corrected sentence (usually first line)
        lines = raw.strip().split('\n')
        corrected = lines[0].strip()
        
        # Remove common prefixes
        for prefix in ["Corrected:", "Correct:", "Fixed:", "→", "-"]:
            if corrected.startswith(prefix):
                corrected = corrected[len(prefix):].strip()
        
        return GrammarResult(corrected_sentence=corrected, raw_output=raw)
    
    # ========================================================================
    # Task 4: Dialogue Generation
    # ========================================================================
    
    def generate_dialogue(self, user_message: str) -> DialogueResult:
        """
        Tạo phản hồi cho user message
        
        Args:
            user_message: Tin nhắn từ user
            
        Returns:
            DialogueResult với response và raw output
            
        Example:
            >>> result = client.generate_dialogue("What's the weather like today?")
            >>> print(f"Response: {result.response}")
            Response: I don't have access to real-time weather...
        """
        prompt = f"User: {user_message}"
        raw = self.client.query(prompt, max_tokens=256)
        
        # Clean response (remove "Assistant:" prefix if present)
        response = raw.strip()
        for prefix in ["Assistant:", "Bot:", "AI:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return DialogueResult(response=response, raw_output=raw)
    
    # ========================================================================
    # Task 5: Error Explanation (Vietnamese)
    # ========================================================================
    
    def explain_error(self, error_text: str, correct_text: str) -> ExplanationResult:
        """
        Giải thích lỗi grammar bằng tiếng Việt
        
        Args:
            error_text: Text có lỗi
            correct_text: Text đã sửa
            
        Returns:
            ExplanationResult với explanation và raw output
            
        Example:
            >>> result = client.explain_error("I goes", "I go")
            >>> print(f"Explanation: {result.explanation}")
            Explanation: Lỗi: Động từ "goes" không phù hợp với chủ ngữ "I"...
        """
        prompt = f"Error: {error_text} → Correct: {correct_text}\nExplain the grammar error in Vietnamese."
        raw = self.client.query(prompt, max_tokens=512)
        
        return ExplanationResult(explanation=raw.strip(), raw_output=raw)
    
    # ========================================================================
    # Batch Processing
    # ========================================================================
    
    def batch_analyze_fluency(self, sentences: List[str]) -> List[FluencyResult]:
        """Phân tích fluency cho nhiều câu"""
        return [self.analyze_fluency(s) for s in sentences]
    
    def batch_classify_vocabulary(self, sentences: List[str]) -> List[VocabularyResult]:
        """Phân loại vocabulary cho nhiều câu"""
        return [self.classify_vocabulary(s) for s in sentences]
    
    def batch_correct_grammar(self, sentences: List[str]) -> List[GrammarResult]:
        """Sửa grammar cho nhiều câu"""
        return [self.correct_grammar(s) for s in sentences]
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "server":
            self.client.stop_server()
    
    def close(self):
        """Đóng connection (cho server mode)"""
        if self.mode == "server":
            self.client.stop_server()


# ============================================================================
# EXAMPLES & USAGE
# ============================================================================

def example_cli_usage():
    """Example: Sử dụng CLI mode (đơn giản)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: CLI Mode (Simple)")
    print("="*70)
    
    model_path = "~/Projects/llama.cpp/models/lexilingo_qwen25_q4_km.gguf"
    client = LexiLingoCliClient(model_path)
    
    # Test fluency
    result = client.query("Analyze the fluency of this sentence: The cat sat on the mat.")
    print(f"\nFluency analysis: {result}")
    
    # Test vocabulary
    result = client.query("Classify the vocabulary level: The phenomenon is fascinating.")
    print(f"\nVocabulary level: {result}")


def example_server_usage():
    """Example: Sử dụng Server mode (production)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Server Mode (Production)")
    print("="*70)
    
    model_path = "~/Projects/llama.cpp/models/lexilingo_qwen25_q4_km.gguf"
    
    # Use context manager để tự động start/stop server
    with LexiLingoServerClient(model_path, port=8080) as client:
        # Test fluency
        result = client.query("Analyze the fluency of this sentence: The cat sat on the mat.")
        print(f"\nFluency: {result}")
        
        # Test vocabulary
        result = client.query("Classify the vocabulary level: The cat sat on the mat.")
        print(f"\nVocabulary: {result}")


def example_highlevel_usage():
    """Example: Sử dụng High-level API (recommended)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: High-level API (Recommended)")
    print("="*70)
    
    model_path = "~/Projects/llama.cpp/models/lexilingo_qwen25_q4_km.gguf"
    
    # Use high-level client
    with LexiLingoClient(model_path, mode="server") as client:
        
        # Task 1: Fluency
        print("\n[Task 1] Fluency Analysis:")
        result = client.analyze_fluency("The cat sat on the mat.")
        print(f"  Score: {result.score}/5.0")
        
        # Task 2: Vocabulary
        print("\n[Task 2] Vocabulary Classification:")
        result = client.classify_vocabulary("The phenomenon is fascinating.")
        print(f"  Level: {result.level}")
        
        # Task 3: Grammar
        print("\n[Task 3] Grammar Correction:")
        result = client.correct_grammar("She don't like apples.")
        print(f"  Corrected: {result.corrected_sentence}")
        
        # Task 4: Dialogue
        print("\n[Task 4] Dialogue Generation:")
        result = client.generate_dialogue("What's the weather like today?")
        print(f"  Response: {result.response}")
        
        # Task 5: Explanation
        print("\n[Task 5] Error Explanation:")
        result = client.explain_error("I goes", "I go")
        print(f"  Explanation: {result.explanation}")


def example_batch_processing():
    """Example: Batch processing nhiều câu"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    model_path = "~/Projects/llama.cpp/models/lexilingo_qwen25_q4_km.gguf"
    
    with LexiLingoClient(model_path, mode="server") as client:
        
        # Batch fluency analysis
        sentences = [
            "The cat sat on the mat.",
            "I goes to school.",
            "She is very happy today.",
        ]
        
        print("\nBatch Fluency Analysis:")
        results = client.batch_analyze_fluency(sentences)
        for sent, result in zip(sentences, results):
            print(f"  {sent:40s} → Score: {result.score:.1f}")
        
        # Batch vocabulary classification
        print("\nBatch Vocabulary Classification:")
        results = client.batch_classify_vocabulary(sentences)
        for sent, result in zip(sentences, results):
            print(f"  {sent:40s} → Level: {result.level}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    LEXILINGO CLIENT EXAMPLES                         ║
║  Cách gọi model sau khi deploy với llama.cpp                         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    try:
        # Example 1: CLI mode
        # example_cli_usage()
        
        # Example 2: Server mode
        # example_server_usage()
        
        # Example 3: High-level API (RECOMMENDED)
        example_highlevel_usage()
        
        # Example 4: Batch processing
        # example_batch_processing()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. llama.cpp is installed at ~/Projects/llama.cpp")
        print("  2. Model file exists at specified path")
        print("  3. Run 'make' in llama.cpp directory")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Examples completed")
    print("="*70)
