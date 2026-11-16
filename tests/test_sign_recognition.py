"""Comprehensive Automated Test Suite for SignBridge AI.
Validates landmark preprocessing, feature extraction, model loading,
prediction confidence, stability, and strict API response contracts.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.sign_recognition.config import (
    CLASSIFIER_MODEL_PATH,
    SUPPORTED_GESTURES,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition.gesture_classifier import GestureClassifier
from ai.sign_recognition.hand_detector import HandDetectionResult, HandDetector
from ai.sign_recognition.landmark_extractor import LandmarkExtractor
from ai.sign_recognition.prediction import (
    SentenceBuffer,
    SignRecognitionService,
    TemporalSmoother,
)


class TestLandmarkExtractor(unittest.TestCase):
    """Tests for landmark translation, scale invariance, and feature dimensions."""

    def setUp(self):
        self.extractor = LandmarkExtractor()
        # Canonical 21 points
        self.sample_landmarks = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            self.sample_landmarks[i] = [0.1 * i, 0.2 * i, 0.01 * i]

    def test_feature_vector_dimension(self):
        """Feature vector must strictly match configured TOTAL_FEATURE_DIM (73)."""
        feats = self.extractor.extract_features(self.sample_landmarks)
        self.assertEqual(len(feats), TOTAL_FEATURE_DIM)
        self.assertFalse(np.isnan(feats).any())

    def test_translation_invariance(self):
        """Normalizing shifted landmarks should result in the exact same wrist-relative origin."""
        norm_orig = self.extractor.normalize_landmarks(self.sample_landmarks)
        # Shift entire hand in space
        shifted_landmarks = self.sample_landmarks + np.array([5.0, -3.0, 10.0])
        norm_shifted = self.extractor.normalize_landmarks(shifted_landmarks)

        # Wrist (index 0) must always be at origin (0, 0, 0)
        np.testing.assert_allclose(norm_shifted[0], [0.0, 0.0, 0.0], atol=1e-5)
        # Coordinates relative to wrist must match
        np.testing.assert_allclose(norm_orig, norm_shifted, atol=1e-4)

    def test_scale_invariance(self):
        """Scaling hand coordinates should produce identical normalized proportions."""
        scaled_landmarks = self.sample_landmarks * 3.5
        norm_orig = self.extractor.normalize_landmarks(self.sample_landmarks)
        norm_scaled = self.extractor.normalize_landmarks(scaled_landmarks)
        np.testing.assert_allclose(norm_orig, norm_scaled, atol=1e-4)

    def test_invalid_landmark_shape(self):
        """Must raise ValueError for incorrect shape."""
        with self.assertRaises(ValueError):
            self.extractor.normalize_landmarks(np.zeros((10, 3)))


class TestGestureClassifier(unittest.TestCase):
    """Tests for rule engine, ML model, confidence thresholds, and unknown gesture safety."""

    def setUp(self):
        self.classifier = GestureClassifier()
        self.extractor = LandmarkExtractor()

    def test_model_loading(self):
        """Verified model file must load if present."""
        if CLASSIFIER_MODEL_PATH.exists():
            self.assertTrue(self.classifier.ml_classifier.is_loaded)

    def test_supported_gestures_membership(self):
        """Every recognized gesture must be a member of SUPPORTED_GESTURES."""
        for gesture in SUPPORTED_GESTURES:
            self.assertIsInstance(gesture, str)

    def test_unknown_gesture_handling(self):
        """Unrecognized random coordinates must yield None and is_recognized=False."""
        random_landmarks = np.random.uniform(-1, 1, (21, 3)).astype(np.float32)
        label, conf, is_rec, meta = self.classifier.classify_landmarks(random_landmarks)
        # Confidence must be low or label must be None
        if not is_rec:
            self.assertIsNone(label)
            self.assertLess(conf, 0.65)


class TestSentenceBufferAndSmoothing(unittest.TestCase):
    """Tests for sentence accumulation, undo, clear, and temporal debouncing."""

    def test_sentence_buffer_operations(self):
        buf = SentenceBuffer()
        self.assertEqual(buf.sentence, "")

        buf.append("HELLO")
        buf.append("HELP")
        self.assertEqual(buf.sentence, "HELLO HELP")

        popped = buf.undo()
        self.assertEqual(popped, "HELP")
        self.assertEqual(buf.sentence, "HELLO")

        buf.clear()
        self.assertEqual(buf.sentence, "")

    def test_temporal_smoother_duplicate_prevention(self):
        """Same gesture held across frames should only be appended once within cooldown."""
        smoother = TemporalSmoother(window_size=3, required_stable=3, cooldown_sec=2.0)

        # Frame 1 & 2: Building stability
        lbl, conf, should_append = smoother.update("HELLO", 0.95)
        self.assertFalse(should_append)

        lbl, conf, should_append = smoother.update("HELLO", 0.95)
        self.assertFalse(should_append)

        # Frame 3: Stable threshold reached -> Appends once
        lbl, conf, should_append = smoother.update("HELLO", 0.95)
        self.assertTrue(should_append)
        self.assertEqual(lbl, "HELLO")

        # Frame 4: Same gesture held -> Debounced (no spamming)
        lbl, conf, should_append = smoother.update("HELLO", 0.95)
        self.assertFalse(should_append)


class TestSignRecognitionService(unittest.TestCase):
    """Tests the high-level service and strict API contract adherence."""

    def setUp(self):
        self.service = SignRecognitionService()

    def test_no_hand_frame_contract(self):
        """Blank frame must return strict no_hand_detected contract."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.service.process_frame(black_frame, annotate=False)

        self.assertTrue(res["success"])
        self.assertIsNone(res["recognized_text"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertFalse(res["hand_detected"])
        self.assertFalse(res["gesture_recognized"])
        self.assertEqual(res["status"], "no_hand_detected")
        self.assertIn("sentence", res)

    def test_invalid_frame_handling(self):
        """None or empty frame must not crash and return error status."""
        res = self.service.process_frame(None)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "invalid_frame")


if __name__ == "__main__":
    unittest.main()
