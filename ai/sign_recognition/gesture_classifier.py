"""Gesture Classification Engine for SignBridge AI.
Implements dual-mode recognition:
1. Geometric Rule Engine (instant, deterministic, zero-training needed).
2. Machine Learning Classifier (Scikit-Learn Random Forest/MLP on normalized features).
Safely handles unsupported gestures and low-confidence detections.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np

from ai.sign_recognition.config import (
    CLASSIFIER_MODEL_PATH,
    MIN_RECOGNITION_CONFIDENCE,
    SUPPORTED_GESTURES,
    UNKNOWN_GESTURE_THRESHOLD,
)
from ai.sign_recognition.landmark_extractor import (
    LandmarkExtractor,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
    WRIST,
)


class RuleBasedClassifier:
    """Deterministic geometric rule engine for foundational sign language gestures."""

    def __init__(self, extractor: LandmarkExtractor):
        self.extractor = extractor

    def classify(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """Classifies normalized hand landmarks using geometric angle and finger state rules.
        
        Args:
            landmarks: Array of shape (21, 3).
            
        Returns:
            Tuple of (predicted_gesture_or_None, confidence_score, metadata_dict).
        """
        norm_pts = self.extractor.normalize_landmarks(landmarks)
        states = self.extractor.compute_finger_states(norm_pts)

        t = states["thumb"]
        i = states["index"]
        m = states["middle"]
        r = states["ring"]
        p = states["pinky"]

        # Key Euclidean distances between normalized fingertips
        thumb_index_dist = float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[INDEX_TIP]))
        index_middle_dist = float(np.linalg.norm(norm_pts[INDEX_TIP] - norm_pts[MIDDLE_TIP]))
        thumb_wrist_y = norm_pts[THUMB_TIP][1] - norm_pts[WRIST][1]

        metadata = {
            "finger_states": states,
            "thumb_index_dist": thumb_index_dist,
            "index_middle_dist": index_middle_dist,
        }

        # 1. HELLO: All 5 fingers extended spread apart pointing upward
        if t and i and m and r and p:
            # Check spread: thumb to index distance is wide
            if thumb_index_dist > 0.40:
                return "HELLO", 0.94, metadata
            else:
                return "STOP", 0.92, metadata  # Fingers held together upright

        # 2. STOP: 4 fingers extended together upright (thumb can be tucked or neutral)
        if i and m and r and p:
            return "STOP", 0.90, metadata

        # 3. YES / Thumbs-Up: Only thumb extended upward; index, middle, ring, pinky curled
        if t and not i and not m and not r and not p:
            # Check if thumb tip is above wrist (in normalized coords, Y < 0 is upward)
            if thumb_wrist_y < 0:
                return "YES", 0.92, metadata
            return "HELP", 0.85, metadata

        # 4. HELP: Thumb extended straight up with palm facing side/upward
        if t and not m and not r and not p:
            return "HELP", 0.88, metadata

        # 5. NO: Index and middle fingers pinching/touching thumb tip, ring & pinky curled
        if not r and not p:
            # Pinch condition: thumb and index are very close together
            if thumb_index_dist < 0.28:
                return "NO", 0.91, metadata

        # 6. PEACE / V-Sign: Index and middle extended in V-shape, thumb/ring/pinky curled
        if i and m and not r and not p:
            if index_middle_dist > 0.15:
                return "PEACE", 0.95, metadata

        # 7. OK: Thumb and index fingertips touching in circle, middle, ring, pinky extended
        if m and r and p and (thumb_index_dist < 0.25):
            return "OK", 0.93, metadata

        # 8. I_LOVE_YOU (ASL ILY): Thumb, index, and pinky extended; middle and ring curled
        if t and i and not m and not r and p:
            return "I_LOVE_YOU", 0.96, metadata

        # 9. THANK YOU: Flat hand open facing outward (palm orientation)
        if i and m and r:
            return "THANK YOU", 0.82, metadata

        # Unrecognized / unsupported hand posture
        return None, 0.25, metadata


class MLGestureClassifier:
    """Machine Learning gesture classifier using trained Scikit-learn model."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or CLASSIFIER_MODEL_PATH
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> bool:
        """Loads trained joblib model if it exists."""
        if not self.model_path.exists():
            self.is_loaded = False
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            return True
        except Exception:
            self.is_loaded = False
            return False

    def predict(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """Predicts gesture class from 73-dimensional feature vector."""
        if not self.is_loaded or self.model is None:
            return None, 0.0, {"error": "model_not_loaded"}

        features = feature_vector.reshape(1, -1)
        try:
            predicted_class = self.model.predict(features)[0]
            confidence = 0.85
            proba_dict = {}

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                confidence = float(np.max(probs))
                proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}

            return str(predicted_class), confidence, {"probabilities": proba_dict}
        except Exception as e:
            return None, 0.0, {"error": str(e)}


class GestureClassifier:
    """Unified hybrid gesture classifier for SignBridge AI.
    
    Seamlessly orchestrates the Geometric Rule Engine and the Scikit-learn ML Model.
    Enforces minimum confidence threshold and strict rejection of unknown gestures.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.extractor = LandmarkExtractor()
        self.rule_classifier = RuleBasedClassifier(self.extractor)
        self.ml_classifier = MLGestureClassifier(model_path)
        self.min_confidence = MIN_RECOGNITION_CONFIDENCE
        self.unknown_threshold = UNKNOWN_GESTURE_THRESHOLD

    def classify_landmarks(self, landmarks: np.ndarray) -> Tuple[Optional[str], float, bool, Dict]:
        """Classifies hand landmarks into a verified gesture.
        
        Args:
            landmarks: Array of shape (21, 3).
            
        Returns:
            Tuple: (gesture_name_or_None, confidence, is_recognized, metadata)
        """
        feature_vec = self.extractor.extract_features(landmarks)

        # 1. Evaluate with Geometric Rule Engine
        rule_label, rule_conf, rule_meta = self.rule_classifier.classify(landmarks)

        # 2. Evaluate with ML Model if available
        ml_label, ml_conf, ml_meta = None, 0.0, {}
        if self.ml_classifier.is_loaded:
            ml_label, ml_conf, ml_meta = self.ml_classifier.predict(feature_vec)

        # 3. Decision Fusion
        chosen_label = None
        chosen_conf = 0.0
        source = "none"

        # If both agree, boost confidence
        if rule_label is not None and ml_label is not None and rule_label == ml_label:
            chosen_label = rule_label
            chosen_conf = min(0.99, max(rule_conf, ml_conf) + 0.05)
            source = "hybrid_consensus"
        elif rule_label is not None and rule_conf >= self.min_confidence:
            chosen_label = rule_label
            chosen_conf = rule_conf
            source = "rule_engine"
        elif ml_label is not None and ml_conf >= self.min_confidence:
            chosen_label = ml_label
            chosen_conf = ml_conf
            source = "ml_model"
        else:
            # Low confidence or unrecognized gesture
            chosen_label = rule_label or ml_label
            chosen_conf = max(rule_conf, ml_conf)
            source = "unrecognized"

        # Strict confidence validation
        is_recognized = (
            chosen_label is not None
            and chosen_conf >= self.min_confidence
            and chosen_label in SUPPORTED_GESTURES
        )

        final_label = chosen_label if is_recognized else None
        final_conf = chosen_conf if is_recognized else (chosen_conf if chosen_conf < self.unknown_threshold else 0.35)

        metadata = {
            "source": source,
            "rule_prediction": rule_label,
            "rule_confidence": rule_conf,
            "ml_prediction": ml_label,
            "ml_confidence": ml_conf,
            "raw_metadata": {**rule_meta, **ml_meta},
        }

        return final_label, final_conf, is_recognized, metadata
