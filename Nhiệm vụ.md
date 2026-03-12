# Phân Chia Nhiệm Vụ Triển Khai Kiến Trúc Deep Learning - LexiLingo

> **Dự án**: LexiLingo - AI English Tutor  
> **Tổng số thành viên**: 4 người  
> **Phạm vi**: Xây dựng module AI Backend (DL-Model-Support)

---

## Tổng Quan Phân Chia Feature

Để đảm bảo tiến độ và chuyên môn hóa, team sẽ được chia thành 3 vai trò kỹ thuật chính (bên cạnh bạn là Lead):

1.  **Nguyên (NLP Specialist)**: Chuyên trách về dữ liệu văn bản, Fine-tuning mô hình Qwen2.5 và đánh giá chất lượng ngôn ngữ.
2.  **Hoàng, Nguyên (Audio AI Engineer)**: Chuyên trách về xử lý tín hiệu âm thanh, STT (Whisper), TTS (Piper) và chấm điểm phát âm (HuBERT).
3.  **Mẫn, Hoàng (System/Backend Engineer)**: Chuyên trách về xây dựng Orchestrator, API, Caching, tối ưu hóa hệ thống và triển khai.

---

## Nguyên, Mẫn: NLP Specialist (Fine-tuning & Data)

**Mục tiêu**: Tạo ra "bộ não" ngôn ngữ thông minh, có khả năng sửa lỗi, đánh giá và dạy học.

### Danh sách nhiệm vụ cụ thể

| ID | Nhiệm vụ | Chi tiết thực hiện | Output mong đợi |
|----|----------|--------------------|-----------------|
| **NLP-01** | **Thu thập & Chuẩn hóa Datasets** | 1. Tải các dataset: EFCAMDAT (Fluency), BEA-2019 (Grammar), AutoTutor (Dialogue).<br>2. Viết script python clean text, loại bỏ nhiễu.<br>3. Chuyển đổi tất cả về định dạng JSONL Instruction: `{"instruction": "...", "input": "...", "output": "..."}`. | File `lexilingo_train.jsonl` (~16k dòng) sạch, đúng format. |
| **NLP-02** | **Setup Qwen2.5 Base** | 1. Tải model `Qwen/Qwen2.5-1.5B-Instruct`.<br>2. Thực hiện lượng tử hóa (Quantization) 4-bit (BNB4) để chạy trên Google Colab/Local GPU. | Script `load_model.py` load thành công model 4-bit. |
| **NLP-03** | **Train Unified LoRA Adapter** | 1. Cấu hình LoRA: `r=48`, `alpha=96`, `modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']`.<br>2. Viết Prompt Template cho 4 task (Fluency, Grammar, Vocab, Dialogue).<br>3. Chạy training loop sử dụng thư viện `peft` và `transformers`. | Folder `adapter_model` chứa file `.bin` và `.json`. train loss < 0.5. |
| **NLP-04** | **Xây dựng Evaluator** | 1. Viết script đánh giá Grammar correction (dùng thư viện `errant`).<br>2. Viết script đánh giá Fluency (so sánh MAE với human score). | Report so sánh hiệu năng trước và sau khi train. |

### Hướng dẫn kỹ thuật & Stack
-   **Ngôn ngữ**: Python 3.10+
-   **Thư viện chính**: `transformers`, `peft`, `bitsandbytes`, `datasets`, `pandas`.
-   **Prompt Template mẫu**:
    ```python
    def create_prompt(instruction, input_text):
        return f"<|im_start|>user\n{instruction}\nInput: {input_text}<|im_end|>\n<|im_start|>assistant\n"
    ```
*Các model dòng Instruct (như Qwen2.5-1.5B-Instruct) được dạy để hiểu hội thoại theo cấu trúc từng vai (Role). Nếu chỉ đưa text thô vào, model sẽ bị "ngáo" và có thể chỉ viết tiếp thay vì trả lời. Template này giúp model phân biệt rõ đâu là lời người dùng hỏi, và đâu là chỗ nó cần bắt đầu trả lời.


### Tài liệu tham khảo
1.  **Qwen2.5 Documentation**: [HuggingFace - Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) - *Đọc phần Quickstart và Flash Attention*.
2.  **LoRA Setup**: [PEFT Documentation](https://huggingface.co/docs/peft/index) - *Xem phần LoRAConfig*.
3.  **Grammar Metrics**: [ERRANT Github](https://github.com/chrisjbryant/errant) - *Chuẩn đánh giá sửa lỗi ngữ pháp*.
4.  **AutoTutor Data**: Tìm kiếm các paper về "AutoTutor dialogue strategies" để hiểu cách annotate dữ liệu dạy học.

---

## Hoàng, Nguyên: Audio AI Engineer (Speech Processing)

**Mục tiêu**: Đảm bảo "tai" và "miệng" của ứng dụng hoạt động mượt mà, chính xác và có độ trễ thấp.

### Danh sách nhiệm vụ cụ thể

| ID | Nhiệm vụ | Chi tiết thực hiện | Output mong đợi |
|----|----------|--------------------|-----------------|
| **AUD-01** | **Tối ưu Whisper STT** | 1. Cài đặt `faster-whisper`.<br>2. Convert model `small.en` sang định dạng CTranslate2 (INT8).<br>3. Tích hợp VAD (Silero) để cắt bỏ khoảng lặng đầu/cuối trước khi đưa vào model. | Module `stt_service.py` nhận file audio -> trả text + timestamp < 100ms. |
| **AUD-02** | **Xây dựng Pronunciation Scorer** | 1. Setup model `facebook/hubert-large-ls960`.<br>2. Implement thuật toán **DTW (Dynamic Time Warping)** để so sánh phoneme người dùng nói vs phoneme chuẩn.<br>3. Code logic tô màu lỗi: Đúng (Xanh), Sai (Đỏ), Gần đúng (Vàng). | Hàm `analyze_pronunciation(user_audio, target_text)` trả về JSON điểm số từng phoneme. |
| **AUD-03** | **Deploy TTS (Piper)** | 1. Build Piper TTS từ source hoặc dùng python wrapper.<br>2. Test giọng `en_US-lessac-medium`.<br>3. Tối ưu caching: Những câu cố định ("Good job", "Try again") nên generate 1 lần rồi lưu file. | Module `tts_service.py` nhận text -> trả binary audio WAV. |
| **AUD-04** | **Audio Preprocessing Pipeline** | Xử lý nhiễu, chuẩn hóa volume (normalization), convert sample rate về 16kHz (chuẩn của Whisper/HuBERT). | Pipeline xử lý đầu vào robust, không lỗi format. |

### Hướng dẫn kỹ thuật & Stack
-   **Ngôn ngữ**: Python, C++ (nếu cần build Piper)
-   **Thư viện chính**: `faster-whisper`, `librosa`, `torch` (cho HuBERT), `scipy` (cho DTW), `silero-vad`.
-   **Key Concept**: "Alignment" - việc khớp word/phoneme của audio với text là quan trọng nhất cho tính năng chấm điểm phát âm.

### Tài liệu tham khảo
1.  **Faster Whisper**: [GitHub - guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) - *Cách dùng CTranslate2*.
2.  **HuBERT Model**: [HuggingFace - HuBERT](https://huggingface.co/facebook/hubert-large-ls960) - *Hiểu về CTC decoding*.
3.  **Piper TTS**: [GitHub - rhasspy/piper](https://github.com/rhasspy/piper) - *Neural TTS engine siêu nhẹ*.
4.  **DTW Algorithm**: [Librosa DTW Docs](https://librosa.org/doc/main/generated/librosa.sequence.dtw.html) - *Giải thuật so khớp chuỗi thời gian*.

---

## Mẫn, Hoàng: Backend Engineer (Orchestrator & System)

**Mục tiêu**: Xây dựng khung xương sống, kết nối các part rời rạc thành hệ thống hoàn chỉnh và tối ưu tài nguyên.

### Danh sách nhiệm vụ cụ thể

| ID | Nhiệm vụ | Chi tiết thực hiện | Output mong đợi |
|----|----------|--------------------|-----------------|
| **SYS-01** | **Xây dựng Orchestrator Class** | 1. Code class `AIOrchestrator` (theo mẫu trong architecture.md).<br>2. Implement **AsyncIO**: Gọi STT và Pronunciation song song.<br>3. Logic **Error Handling**: Try/catch từng component, trả fallback nếu lỗi. | File `orchestrator.py` chạy mượt mà flow từ A-Z. |
| **SYS-02** | **Resource & Model Management** | 1. Viết `ModelManager` class (Singleton).<br>2. Implement logic **Lazy Loading**: Chỉ `model.to("cuda")` khi cần, `model.to("cpu")` khi idle lâu.<br>3. Quản lý RAM: Giới hạn queue request để không OOM. | Hệ thống không crash khi load đồng thời nhiều model. |
| **SYS-03** | **Context & Caching (Redis)** | 1. Setup Redis.<br>2. Code logic lưu/đọc Context cửa sổ trượt (5 turn gần nhất).<br>3. Cache response của AI: Hash(input text) -> Cached output. | Độ trễ giảm 50% cho các câu hỏi lặp lại. |
| **SYS-04** | **API Gateway (FastAPI)** | 1. Xây dựng REST API chuẩn.<br>2. Endpoint `/chat`, `/stt`, `/tts`.<br>3. Middleware logging, đo thời gian xử lý (latency) từng request. | Swagger UI (`/docs`) test được toàn bộ tính năng. |

### Hướng dẫn kỹ thuật & Stack
-   **Ngôn ngữ**: Python 3.10+
-   **Framework**: `FastAPI` (Web), `Redis` (Cache), `Uvicorn` (Server).
-   **Design Pattern**: Singleton (cho Models), Strategy (cho Logic xử lý lỗi), Facade (Orchestrator).
-   **Lưu ý**: Cần phối hợp chặt chẽ với Member 1 & 2 để thống nhất Input/Output JSON format.

### Tài liệu tham khảo
1.  **FastAPI Best Practices**: [FastAPI Docs](https://fastapi.tiangolo.com/tutorial/bigger-applications/) - *Cách tổ chức project lớn*.
2.  **Python AsyncIO**: [Real Python - AsyncIO](https://realpython.com/async-io-python/) - *Cần hiểu rõ `await`, `gather`*.
3.  **Clean Architecture in Python**: Tìm đọc các repo mẫu về Clean Architecture áp dụng cho Python backend.
4.  **Redis Caching Patterns**: [Redis Docs](https://redis.io/docs/manual/client-side-caching/) - *Chiến lược Cache-aside*.

---

## Quy trình Phối hợp (Workflow)

1.  **Tuần 1**: 
    -   Chốt format dữ liệu Input/Output của model NLP.
    -   Chốt format dữ liệu Audio input và Phoneme output.
    -   Dựng khung project (Scaffold), cài đặt thư viện chung.
2.  **Tuần 2**:
    -   Train model & xuất file weight.
    -   Hoàn thiện module STT & TTS.
    -   Code xong Orchestrator logic giả lập (Mock models).
3.  **Tuần 3**:
    -   **Integration**: Nhúng model thật thực hiện trong tuần 2 vào Orchestrator của cuối tuần 2.
    -   Test full flow.

## Định dạng giao tiếp (Contract)

Mọi module phải tuân thủ JSON Schema thống nhất. Ví dụ output của Orchestrator gửi cho Mobile App phải luôn có dạng:

```json
{
  "text": "Câu trả lời của AI...",
  "audio_url": "link_to_wav_file...",
  "analysis": {
    "fluency": 0.85,
    "grammar_errors": [],
    "pronunciation_score": 0.90
  },
  "metadata": {
    "latency_ms": 350,
    "cached": false
  }
}
```
