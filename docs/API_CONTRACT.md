# SignBridge AI — API Contract & Team Integration Guide

**Module:** AI Hand Gesture & Sign Recognition  
**Current Branch:** `feature/sign-recognition`  
**Target Integration Branches:** `feature/backend`, `feature/frontend`, `feature/facial-analysis`

---

## 1. REST Endpoints Overview

All endpoints accept and return JSON. The backend server runs by default on `http://localhost:8000`.

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Service health, active vision backend, and model status. |
| `/api/supported-gestures` | `GET` | Returns list of currently verified gestures and descriptions. |
| `/api/predict` | `POST` | Core vision inference: accepts base64 frame, returns recognized sign. |
| `/api/sentence/add` | `POST` | Manually appends a word to the accumulated sentence buffer. |
| `/api/sentence/undo` | `POST` | Removes the most recently appended word from the sentence buffer. |
| `/api/sentence/clear` | `POST` | Clears all words from the accumulated sentence buffer. |
| `/api/facial-analysis/status`| `GET` | Optional facial emotion/context analysis integration point. |

---

## 2. Core Inference: `POST /api/predict`

### Request Body
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "annotate": false
}
```
- `image` *(string, required)*: Base64-encoded JPEG/PNG image or data URL.
- `annotate` *(boolean, optional, default false)*: If `true`, returns base64 `annotated_image` with skeleton HUD.

---

### Response Schemas

#### A. Successful Gesture Recognition
```json
{
  "success": true,
  "recognized_text": "HELLO",
  "confidence": 0.94,
  "hand_detected": true,
  "gesture_recognized": true,
  "status": "recognized",
  "sentence": "HELLO",
  "handedness": "Right",
  "bbox": [120, 80, 240, 280],
  "landmarks": [
    [0.51, 0.78, 0.0],
    [0.46, 0.72, -0.02]
  ],
  "annotated_image": null
}
```

#### B. Hand Detected, But Unknown / Unsupported Gesture
```json
{
  "success": true,
  "recognized_text": null,
  "confidence": 0.22,
  "hand_detected": true,
  "gesture_recognized": false,
  "status": "unknown_gesture",
  "sentence": "HELLO",
  "handedness": "Right",
  "bbox": [120, 80, 240, 280],
  "landmarks": [...]
}
```

#### C. No Hand Detected in Frame
```json
{
  "success": true,
  "recognized_text": null,
  "confidence": 0.0,
  "hand_detected": false,
  "gesture_recognized": false,
  "status": "no_hand_detected",
  "sentence": "HELLO"
}
```

---

## 3. Python Service Interface (Direct Backend Integration)

If the backend teammate imports the module directly in Python rather than calling REST:

```python
from ai.sign_recognition.prediction import SignRecognitionService

# Initialize once at app startup
service = SignRecognitionService()

# In your frame processing loop / WebSocket endpoint:
result = service.process_frame(bgr_numpy_frame, annotate=False)

if result["gesture_recognized"]:
    print(f"Recognized: {result['recognized_text']} ({result['confidence']})")
    print(f"Sentence: {result['sentence']}")
```

---

## 4. Facial Analysis Integration Point (`POST /api/analyze/face`)

The facial analysis module operates independently from hand recognition, providing supplementary emotional context and real-time multi-person tracking:

### Request Body
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Multi-Person Response Schema
```json
{
  "success": true,
  "face_detected": true,
  "num_faces": 2,
  "faces": [
    {
      "track_id": 1,
      "face_id": 1,
      "bbox": [140, 110, 220, 250],
      "expression": "happy",
      "confidence": 0.88,
      "status": "face_detected",
      "blendshapes_summary": {
        "happy": 0.58,
        "surprised": 0.05,
        "neutral": 0.12
      }
    },
    {
      "track_id": 2,
      "face_id": 2,
      "bbox": [640, 115, 220, 250],
      "expression": "focused",
      "confidence": 0.74,
      "status": "face_detected",
      "blendshapes_summary": {
        "focused": 0.42,
        "neutral": 0.20
      }
    }
  ],
  "expression": "happy",
  "confidence": 0.88,
  "bbox": [140, 110, 220, 250],
  "status": "faces_detected",
  "supported_expressions": [
    "neutral",
    "happy",
    "surprised",
    "focused",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "confused"
  ],
  "disclaimer": "Facial expression analysis provides supplementary communication context only and is not definitive evidence of a person's emotional state."
}
```

- **Supported Categories (9)**: `neutral`, `happy`, `surprised`, `focused`, `sad`, `angry`, `fearful`, `disgusted`, `confused`.
- **Honest Uncertainty**: If confidence is low or blendshapes ambiguous, `expression` is reported as `"uncertain"` with status `"uncertain_expression"`.
- **Tracking IDs**: Persistent integer IDs (`track_id`) tracked across frames via IoU and centroid proximity.
- **Model**: Google MediaPipe Face Landmarker (`face_landmarker.task`, Apache 2.0 license).
- Hand gesture classification functions 100% independently regardless of whether facial analysis is online.

---

## 5. Supported Vocabulary

Currently supported, validated signs:
1. `HELLO` — Open palm, all fingers extended upward
2. `YES` — Fist with thumb pointed upward (affirmative / thumbs-up)
3. `NO` — Index and middle finger pinch to thumb tip
4. `HELP` — Thumb pointing straight up, palm upright
5. `THANK YOU` — Flat hand facing forward
6. `STOP` — Open palm facing forward with fingers held upright
7. `PEACE` — Index and middle fingers extended in a V shape
8. `OK` — Thumb and index tips touching to form a circle
9. `I_LOVE_YOU` — Thumb, index, and pinky extended (ASL ILY sign)
