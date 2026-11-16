
from fer.fer import FER

# Initialize facial expression detector
detector = FER(mtcnn=True)


def detect_expression(frame):

    results = detector.detect_emotions(frame)

    if not results:
        return {
            "expression": "No face detected",
            "confidence": 0
        }

    emotions = results[0]["emotions"]

    expression = max(emotions, key=emotions.get)
    confidence = emotions[expression]

    return {
        "expression": expression,
        "confidence": round(confidence * 100, 2)
    }
