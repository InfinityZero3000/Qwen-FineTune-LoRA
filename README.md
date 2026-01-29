# LexiLingo
### AI-Powered Adaptive Language Learning Platform

![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active_Development-green?style=flat-square)

## System Overview
LexiLingo is an enterprise-grade, microservices-based platform designed to revolutionize language acquisition through Artificial Intelligence. By integrating large language models with adaptive learning algorithms, the system provides a hyper-personalized educational experience that evolves with the user's proficiency.

## Core Capabilities
*   **Adaptive Learning Engine:** Dynamically adjusts curriculum difficulty and content based on user performance and retention rates.
*   **Intelligent Conversational Agent:** Leverages Google Gemini and custom-tuned LLMs for natural, context-aware dialogue practice.
*   **Multimodal Interaction:** Supports seamless text-to-speech and speech-to-text processing for pronunciation and listening practice.
*   **Gamified Progression:** Implements engagement mechanisms including XP systems, streaks, and leaderboards to maintain user motivation.

## System Architecture

The solution maps to a distributed architecture comprising specialized services:

### 1. Backend Service (Core)
*   **Technostack:** Python, FastAPI, PostgreSQL, SQLAlchemy.
*   **Function:** Manages user identity, persistence, course progression logic, and gamification state. Acts as the central orchestrator for data consistency.

### 2. AI Service (Intelligence)
*   **Technostack:** Python, PyTorch, Hugging Face Transformers.
*   **Function:** Handles heavy compute tasks including real-time speech processing (TTS/STT), grammar correction, and content generation.

### 3. Client Application (Interaction)
*   **Technostack:** Flutter, Dart, Clean Architecture.
*   **Function:** A cross-platform interface (iOS, Android, Web) delivering a responsive and offline-capable user experience.

### 4. Deep Learning Support (Analytics)
*   **Technostack:** Python, Jupyter, Scikit-learn.
*   **Function:** Offline data processing pipeline for model fine-tuning, dataset analysis, and recommender system optimization.

## Technology Stack

| Component | Technologies |
|-----------|--------------|
| **Frontend** | Flutter, Provider, Clean Architecture |
| **Backend** | Python 3.11+, FastAPI, Docker |
| **Database** | PostgreSQL, Redis (Caching) |
| **AI/ML** | Google Gemini, Qwen-1.5B, Whisper, Coqui TTS |
| **DevOps** | Docker Compose, Shell Scripts, CI/CD |

## Documentation
For detailed configuration and operation, refer to the specific module documentation:

*   [**Backend Service**](backend-service/README.md)
*   [**AI Service**](ai-service/README.md)
*   [**Client Application**](flutter-app/README.md)
*   [**System Scripts**](scripts/README.md)

## License
This project is licensed under the MIT License.
