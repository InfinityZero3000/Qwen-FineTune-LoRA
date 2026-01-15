# Kế Hoạch Triển Khai Module AI Chat - LexiLingo v2.0

> **Tài liệu**: Danh sách nhiệm vụ chi tiết để hiện thực hóa kiến trúc AI Chat  
> **Kiến trúc**: Clean Architecture + Modular Feature-First  
> **Core Engine**: Python AI Orchestrator + Flutter UI  
> **Trạng thái**: ⬜ Chưa bắt đầu | ✅ Hoàn thành | 🚧 Đang thực hiện

---

## Phase 1: Môi Trường & Hạ Tầng AI (AI Infrastructure)

Giai đoạn này tập trung xây dựng phần backend xử lý AI (DL-Model-Support), chuẩn bị các model và dataset.

### 1.1 Môi trường & Dataset chuẩn bị
- [ ] **Setup Python Environment**
    - [ ] Tạo Virtual Environment (`venv` hoặc `conda`) với Python 3.10+
    - [ ] Cài đặt các thư viện core: `torch`, `transformers`, `peft`, `bitsandbytes`, `huggingface_hub`
    - [ ] Cài đặt thư viện xử lý audio: `librosa`, `soundfile`, `faster-whisper`
    - [ ] Cài đặt server framework: `fastapi`, `uvicorn`, `redis`
    - [ ] Tạo file `requirements.txt` cập nhật đầy đủ version
- [ ] **Dataset Collection & Processing**
    - [ ] Tải EFCAMDAT dataset (Fluency scoring)
    - [ ] Tải BEA-2019 / CoNLL-2014 dataset (Grammar correction)
    - [ ] Tải AutoTutor Dialogue Corpus (Pedagogical strategy)
    - [ ] Tải Oxford Graded Readers / CEFR corpus (Vocabulary leveling)
    - [ ] Viết script `processing/data_cleaner.py` để chuẩn hóa định dạng dữ liệu về JSONL instruction format
    - [ ] Chia split Train/Validation/Test (80/10/10)

### 1.2 Model Base & Fine-tuning (LoRA)
- [ ] **Qwen2.5-1.5B Base Setup**
    - [ ] Tải model `Qwen/Qwen2.5-1.5B-Instruct`
    - [ ] Viết script lượng tử hóa (Quantization) về 4-bit (BNB4) để tiết kiệm RAM
- [ ] **Unified Adapter Training**
    - [ ] Cấu hình LoRA config (rank=48, alpha=96, modules=[all linear])
    - [ ] Viết training script `train_unified.py` sử dụng thư viện `peft`
    - [ ] Định nghĩa Prompt Template cho Multi-tasking (Fluency, Grammar, Vocab, Dialogue)
    - [ ] Train Unified Adapter trên dataset tổng hợp (~16.7k samples)
    - [ ] Export Adapter (`adapter_model.bin`) và `adapter_config.json`
- [ ] **Model Evaluation**
    - [ ] Viết script `eval_fluency.py` (Tính MAE, Pearson correlation)
    - [ ] Viết script `eval_grammar.py` (Tính F0.5 score, Precision/Recall)
    - [ ] Chạy benchmark so sánh performance với baseline

### 1.3 Audio Models Setup
- [ ] **STT Module (Whisper)**
    - [ ] Setup `faster-whisper` với model `small` hoặc `distil-small.en`
    - [ ] Tối ưu hóa với CTranslate2 để chạy trên CPU/Mobile
    - [ ] Implement VAD (Voice Activity Detection) với Silero VAD để lọc khoảng lặng
- [ ] **Pronunciation Module (HuBERT)**
    - [ ] Tải model `facebook/hubert-large-ls960`
    - [ ] Implement thuật toán DTW (Dynamic Time Warping) để so khớp phoneme
    - [ ] Xây dựng hàm tính điểm phát âm (Phone-level accuracy map)
- [ ] **TTS Module (Piper)**
    - [ ] Compile Piper TTS engine
    - [ ] Tải voice model `en_US-lessac-medium`
    - [ ] Test latency sinh audio

---

## Phase 2: Xây Dựng AI Orchestrator (Backend Core)

Xây dựng bộ não trung tâm điều phối các model theo kiến trúc đã thiết kế.

### 2.1 Core Components Implementation
- [ ] **Context Manager**
    - [ ] Sử dụng `all-MiniLM-L6-v2` để encode ngữ cảnh hội thoại
    - [ ] Xây dựng Sliding Window Buffer (giữ context của 5 turn gần nhất)
    - [ ] Tích hợp Redis để lưu/đọc `user_level`, `learning_history`
- [ ] **Resource Manager**
    - [ ] Implement Singleton Pattern cho Model Loading
    - [ ] Xây dựng cơ chế Lazy Loading cho LLaMA3-VI (chỉ load khi cần tiếng Việt)
    - [ ] Xây dựng cơ chế Offloading (chuyển model từ GPU về CPU khi RAM đầy)

### 2.2 Orchestrator Logic
- [ ] **Task Analyzer**
    - [ ] Viết logic phân tích intent người dùng (Hỏi ngữ pháp? Chat vu vơ? Luyện tập?)
    - [ ] Logic xác định chiến lược dạy (Socratic, Scaffolding, Feedback) dựa trên lịch sử lỗi
- [ ] **Pipeline Execution**
    - [ ] Xây dựng class `AIOrchestrator` chính
    - [ ] Implement `async` flow để chạy song song Qwen và HuBERT
    - [ ] Xây dựng cơ chế Error Handling & Fallback (như thiết kế trong architecture.md)
    - [ ] Implement logic Fusion & Aggregation để gộp kết quả từ các model

### 2.3 API Gateway (FastAPI)
- [ ] Thiết kế API Endpoint: `POST /v1/chat/completions`
- [ ] Thiết kế API Endpoint: `POST /v1/audio/transcriptions` (STT)
- [ ] Thiết kế API Endpoint: `POST /v1/audio/speech` (TTS)
- [ ] Middleware: Rate limiting, Authentication, Logging Request/Response

---

## Phase 3: Flutter App Integration (Clean Architecture)

Tích hợp AI vào ứng dụng mobile, tuân thủ cấu trúc Feature-First và Clean Architecture.

### 3.1 Domain Layer (Feature: Chat)
- [ ] **Entities**
    - [ ] `ChatMessage`: id, text, role, timestamp, audioUrl, metadata (scores, feedback)
    - [ ] `ChatSession`: id, topic, startTime, currentLevel
    - [ ] `AnalysisResult`: fluencyScore, grammarErrors, pronunciationData
- [ ] **Repositories (Abstract)**
    - [ ] `IChatRepository`: define các hàm `sendMessage`, `getHistory`, `analyzePronunciation`
    - [ ] `STTService`, `TTSService` interfaces
- [ ] **UseCases**
    - [ ] `SendMessageUseCase`: Gửi tin nhắn và nhận phản hồi AI
    - [ ] `GetChatHistoryUseCase`: Lấy lịch sử đoạn chat
    - [ ] `AnalyzeSpeechUseCase`: Xử lý luồng voice input

### 3.2 Data Layer (Feature: Chat)
- [ ] **Models**
    - [ ] `ChatMessageModel`: extend Entity, thêm `fromJson`, `toJson`
    - [ ] `AnalysisResponseModel`: Parse JSON từ Orchestrator API
- [ ] **Data Sources**
    - [ ] `ChatRemoteDataSource`: Gọi API lên AI Orchestrator (sử dụng Retrofit/Dio)
    - [ ] `ChatLocalDataSource`: Cache tin nhắn vào SQLite (Drift/Floor) cho offline mode
- [ ] **Repositories (Implementation)**
    - [ ] `ChatRepositoryImpl`: Implement logic chọn nguồn dữ liệu (Local vs Remote), handle network connection check

### 3.3 Presentation Layer (Feature: Chat)
- [ ] **State Management (Provider/Bloc)**
    - [ ] `ChatProvider`: Quản lý list message, loading state, recording state
    - [ ] Logic xử lý UI updates khi nhận stream response
- [ ] **UI Components**
    - [ ] `ChatScreen`: Màn hình chính
    - [ ] `MessageBubble`: Widget hiển thị tin nhắn (User/AI)
    - [ ] `FeedbackWidget`: Hiển thị điểm Fluency và lỗi ngữ pháp dưới tin nhắn AI
    - [ ] `AudioRecorderButton`: Nút ghi âm với animation sóng
    - [ ] `PronunciationView`: Popup hiển thị chi tiết lỗi phát âm (tô đỏ phoneme sai)

---

## Phase 4: Testing & Optimization

### 4.1 Unit Testing
- [ ] **Backend Tests (`pytest`)**
    - [ ] Test Orchestrator logic (Mock model outputs)
    - [ ] Test LoRA Adapter outputs (Input sample -> Output structure check)
    - [ ] Test API endpoints (Input validation, Response format)
- [ ] **Mobile Tests (`flutter_test`)**
    - [ ] Test Domain UseCases
    - [ ] Test Repository Implementation (Mock DataSources)
    - [ ] Widget Test cho Chat Screen components

### 4.2 Integration Testing
- [ ] Test flow trọn vẹn: User Voice Input -> STT -> Orchestrator -> Response -> TTS -> Mobile Audio Playback
- [ ] Kiểm tra độ trễ (Latency) toàn trình. Target: < 2s cho câu trả lời đầu tiên.

### 4.3 Deployment
- [ ] Đóng gói Docker cho AI Backend Service
- [ ] Setup CI/CD Pipeline (GitHub Actions)
- [ ] Build Flutter App (release mode) cho Android/iOS

---

## Checklists Theo Dõi

### Module: AI Backend (Python)
- [ ] Environment Setup
- [ ] Dataset Preparation
- [ ] Model Training (Unified Adapter)
- [ ] Orchestrator Logic
- [ ] FastAPI Implementation

### Module: Mobile App (Flutter)
- [ ] Domain Layer Setup
- [ ] Data Layer Implementation
- [ ] API Client Integration
- [ ] Chat UI Implementation
- [ ] Audio/Voice Features Integration

---
**Ghi chú**: Thực hiện tuần tự theo các Phase. Luôn cập nhật trạng thái vào file này sau mỗi phiên làm việc.
