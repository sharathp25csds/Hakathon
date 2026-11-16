# SignBridge AI
> **AI-Powered Assistive Communication System for Deaf and Hard-of-Hearing Individuals**

SignBridge AI bridges communication barriers between deaf/hard-of-hearing individuals and people unfamiliar with sign language. Using real-time computer vision and machine learning, it tracks 21 hand landmarks, classifies supported gestures into text, accumulates words into coherent sentences, and synthesizes spoken audio aloud.

---

## 🌟 Key Capabilities
- **Real-Time 21-Point Hand Pose Estimation**: Tracks 3D hand joints with scale and translation invariance.
- **Dual Classification Engine**:
  - *Geometric Rule Engine*: High-precision, zero-training recognition for foundational communication signs (`HELLO`, `YES`, `NO`, `HELP`, `THANK YOU`, `STOP`, `PEACE`, `OK`, `I_LOVE_YOU`).
  - *Scikit-Learn ML Model*: Random Forest trained on 73-dimensional normalized feature vectors (`ai/sign_recognition/models/gesture_classifier.joblib`).
- **Temporal Debouncing & Smoothing**: Rolling 5-frame majority voting prevents prediction jitter and duplicate word spamming.
- **WSL & Headless Resilience**: Runs both with native OpenCV webcams and with browser-based webcam streaming via WebRTC/Canvas API.
- **Multi-Person Facial Expression Tracking**: Real-time detection & tracking across frames with unique IDs, 9 emotion categories (`neutral`, `happy`, `surprised`, `focused`, `sad`, `angry`, `fearful`, `disgusted`, `confused`), and honest uncertainty reporting (MediaPipe Face Landmarker, Apache 2.0).
- **Accessible Full-Stack Dashboard**: Modern high-contrast UI with live skeleton HUD, confidence meters, sentence builder, text-to-speech, reverse communication input, and privacy protection.

---

## 📂 Project Architecture

```text
Hakathon/
├── .gitignore                           # Git ignore rules
├── README.md                            # Complete documentation
├── docs/
│   └── API_CONTRACT.md                  # REST & Python API integration contract
├── backend/
│   ├── main.py                          # Dual-mode server (FastAPI & ThreadingHTTPServer)
│   └── requirements.txt                 # Backend dependencies
├── frontend/
│   ├── index.html                       # Accessible dashboard interface
│   ├── styles.css                       # High-contrast, responsive CSS styling
│   └── app.js                           # WebRTC camera capture, HUD renderer & TTS
├── ai/
│   ├── sign_recognition/
│   │   ├── config.py                    # Hyperparameters, vocabulary & paths
│   │   ├── hand_detector.py             # MediaPipe Tasks & OpenCV fallback detector
│   │   ├── landmark_extractor.py        # Normalization (73-dim feature extractor)
│   │   ├── gesture_classifier.py        # Hybrid rule + ML classifier
│   │   ├── prediction.py                # Temporal smoothing & SignRecognitionService
│   │   ├── speech.py                    # Multi-tier text-to-speech engine
│   │   ├── camera_test.py               # Local camera diagnostic & headless simulation
│   │   ├── collect_data.py              # Live / synthetic dataset collection tool
│   │   ├── train_model.py               # Scikit-learn Random Forest training pipeline
│   │   ├── models/                      # Model weights (gesture_classifier.joblib)
│   │   └── data/                        # CSV datasets (gestures.csv)
│   └── facial_analysis/
│       └── __init__.py                  # Integration stub for facial analysis module
└── tests/
    ├── test_sign_recognition.py         # AI pipeline unit & integration tests
    └── test_backend_api.py              # API contract validation tests
```

---

## 🚀 Quickstart & Terminal Execution Commands

### 1. Run Automated Test Suite
```bash
.venv/bin/python -m unittest discover tests
```
*Runs all 14 tests verifying landmark normalization, geometric rules, ML model loading, temporal smoothing, and API contract formats.*

### 2. Run Camera Diagnostic (Local Webcam or Headless WSL)
```bash
.venv/bin/python ai/sign_recognition/camera_test.py
```
*If a physical camera is available, opens live window with landmark HUD. In WSL/headless mode, automatically runs terminal simulation.*

### 3. Start SignBridge AI Application & Web Dashboard
```bash
.venv/bin/python backend/main.py 8000
```
Open **`http://localhost:8000`** in any web browser.
- Click **"Start Camera"** to start real-time recognition via your browser webcam.
- Click **"Simulate Frame"** to test AI inference with pre-configured synthetic hand poses.
- Click **"Speak Sentence"** to hear recognized communication spoken aloud via Web Speech API.

### 4. Collect Additional Dataset Samples & Retrain
```bash
# Capture 60 samples for a gesture (live webcam or synthetic)
.venv/bin/python ai/sign_recognition/collect_data.py --label HELLO --count 60 --synthetic

# Train and evaluate Scikit-learn Random Forest model
.venv/bin/python ai/sign_recognition/train_model.py
```

---

## 🤝 Team Integration Guide

- **Frontend Teammate (`feature/frontend`)**: Refer to [docs/API_CONTRACT.md](docs/API_CONTRACT.md). The `/api/predict` endpoint accepts a base64 JPEG image and returns JSON with `recognized_text`, `confidence`, `landmarks`, and `sentence`.
- **Backend Teammate (`feature/backend`)**: Use `from ai.sign_recognition.prediction import SignRecognitionService`. Call `service.process_frame(numpy_frame)` directly.
- **Facial Analysis Teammate (`feature/facial-analysis`)**: Integrate your facial emotion classifier in `ai/facial_analysis/__init__.py`. Hand recognition operates independently with zero coupling.

---

## 🔒 Privacy Notice
Camera frames are processed strictly in real-time in memory. No video footage is ever stored to disk or uploaded to external servers.
