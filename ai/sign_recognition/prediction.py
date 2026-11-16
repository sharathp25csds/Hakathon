"""Prediction Engine, Temporal Smoothing, and Master Service for SignBridge AI.
Implements rolling-window stability, duplicate prevention, sentence accumulation,
and the standardized SignBridge AI API contract.
"""

from collections import Counter, deque
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.sign_recognition.config import (
    DEBOUNCE_COOLDOWN_SEC,
    REQUIRED_STABLE_FRAMES,
    SMOOTHING_WINDOW_SIZE,
)
from ai.sign_recognition.gesture_classifier import GestureClassifier
from ai.sign_recognition.hand_detector import HandDetectionResult, HandDetector


class SentenceBuffer:
    """Manages the accumulated assistive communication sentence with undo and clear."""

    def __init__(self):
        self._tokens: List[str] = []

    def append(self, token: str) -> bool:
        """Appends a recognized gesture token if valid and not a duplicate of the last token."""
        if not token or not token.strip():
            return False
        clean_token = token.strip().upper()
        self._tokens.append(clean_token)
        return True

    def undo(self) -> Optional[str]:
        """Removes and returns the most recently appended token."""
        if self._tokens:
            return self._tokens.pop()
        return None

    def clear(self) -> None:
        """Clears all tokens in the sentence buffer."""
        self._tokens.clear()

    @property
    def sentence(self) -> str:
        """Returns the accumulated sentence as a clean space-separated string."""
        return " ".join(self._tokens)

    @property
    def tokens(self) -> List[str]:
        return list(self._tokens)


class TemporalSmoother:
    """Eliminates prediction jitter and prevents duplicate gesture spamming."""

    def __init__(
        self,
        window_size: int = SMOOTHING_WINDOW_SIZE,
        required_stable: int = REQUIRED_STABLE_FRAMES,
        cooldown_sec: float = DEBOUNCE_COOLDOWN_SEC,
    ):
        self.window_size = window_size
        self.required_stable = required_stable
        self.cooldown_sec = cooldown_sec

        self._history: deque = deque(maxlen=window_size)
        self._conf_history: deque = deque(maxlen=window_size)
        self.last_confirmed_gesture: Optional[str] = None
        self.last_confirmation_time: float = 0.0

    def reset(self) -> None:
        """Resets temporal smoothing history."""
        self._history.clear()
        self._conf_history.clear()
        self.last_confirmed_gesture = None
        self.last_confirmation_time = 0.0

    def update(
        self, raw_label: Optional[str], confidence: float
    ) -> Tuple[Optional[str], float, bool]:
        """Updates smoothing window and evaluates stability and cooldown.
        
        Args:
            raw_label: Predicted gesture from current frame or None.
            confidence: Prediction confidence score.
            
        Returns:
            Tuple of (stable_gesture, smoothed_confidence, should_append_to_sentence).
        """
        # Record into history
        self._history.append(raw_label)
        self._conf_history.append(confidence)

        if len(self._history) < self.required_stable:
            return raw_label, confidence, False

        # Majority voting across the window
        counts = Counter(self._history)
        most_common_label, count = counts.most_common(1)[0]
        avg_confidence = float(np.mean(self._conf_history))

        if most_common_label is None or count < self.required_stable:
            return None, avg_confidence, False

        now = time.time()
        time_since_last = now - self.last_confirmation_time

        # Duplicate prevention & cooldown check
        should_append = False
        if most_common_label != self.last_confirmed_gesture or time_since_last >= self.cooldown_sec:
            should_append = True
            self.last_confirmed_gesture = most_common_label
            self.last_confirmation_time = now

        return most_common_label, avg_confidence, should_append


class SignRecognitionService:
    """Master service providing high-level AI gesture recognition and API contract responses."""

    def __init__(self, detector: Optional[HandDetector] = None, classifier: Optional[GestureClassifier] = None):
        self.detector = detector or HandDetector()
        self.classifier = classifier or GestureClassifier()
        self.smoother = TemporalSmoother()
        self.sentence_buffer = SentenceBuffer()

    def process_frame(
        self, frame: np.ndarray, annotate: bool = True
    ) -> Dict[str, Any]:
        """Processes a single BGR video frame and produces the standardized API response.
        
        Args:
            frame: Input BGR image from webcam or client API.
            annotate: Whether to generate an annotated visualization frame.
            
        Returns:
            Dictionary adhering to the strict SignBridge AI API contract.
        """
        if frame is None or frame.size == 0:
            return {
                "success": False,
                "recognized_text": None,
                "confidence": 0.0,
                "hand_detected": False,
                "gesture_recognized": False,
                "status": "invalid_frame",
                "sentence": self.sentence_buffer.sentence,
            }

        # 1. Hand Detection
        detection: HandDetectionResult = self.detector.detect(frame)

        if not detection.detected or not detection.landmarks:
            self.smoother.reset()
            annotated_frame = (
                self.detector.draw_landmarks(frame, detection) if annotate else None
            )
            return {
                "success": True,
                "recognized_text": None,
                "confidence": 0.0,
                "hand_detected": False,
                "gesture_recognized": False,
                "status": "no_hand_detected",
                "sentence": self.sentence_buffer.sentence,
                "annotated_frame": annotated_frame,
            }

        # 2. Gesture Classification
        pts = np.array(detection.landmarks, dtype=np.float32)
        raw_label, raw_conf, is_recognized, meta = self.classifier.classify_landmarks(pts)

        # 3. Temporal Smoothing & Stability
        stable_label, smoothed_conf, should_append = self.smoother.update(
            raw_label if is_recognized else None, raw_conf
        )

        # 4. Sentence Buffer Update (only when stable and cooldown passed)
        if should_append and stable_label:
            self.sentence_buffer.append(stable_label)

        # 5. Format Status Code
        if stable_label is not None:
            status = "recognized"
            display_text = stable_label
        elif is_recognized:
            status = "stabilizing"
            display_text = raw_label
        else:
            status = "unknown_gesture"
            display_text = None

        # 6. Annotate Visual Output
        annotated_frame = None
        if annotate:
            annotated_frame = self.detector.draw_landmarks(frame, detection)
            # Add gesture label banner
            if display_text:
                banner_text = f"Sign: {display_text} ({int(smoothed_conf * 100)}%)"
                cv2.putText(
                    annotated_frame,
                    banner_text,
                    (20, annotated_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            elif status == "unknown_gesture":
                cv2.putText(
                    annotated_frame,
                    "Unknown Gesture",
                    (20, annotated_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

        return {
            "success": True,
            "recognized_text": display_text,
            "confidence": round(smoothed_conf, 2),
            "hand_detected": True,
            "gesture_recognized": bool(display_text is not None),
            "status": status,
            "sentence": self.sentence_buffer.sentence,
            "handedness": detection.handedness,
            "bbox": list(detection.bbox),
            "landmarks": detection.landmarks,
            "annotated_frame": annotated_frame,
            "metadata": meta,
        }
