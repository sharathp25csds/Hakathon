"""Comprehensive Automated Test Suite for SignBridge AI Module v2.
Validates all 15 core architectural requirements for ISL Alphabet Fingerspelling,
multi-hand feature extraction, dataset validation & training safety,
letter stabilization, duplicate suppression, repeated letter support (HELLO),
word and sentence formation, TTS contract, and master service integration.
Tests do not require an active webcam.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.sign_recognition_v2.config import (
    CLASSIFIER_MODEL_PATH,
    DATASET_PATH,
    FEATURE_VERSION,
    HAND_FEATURE_DIM,
    INTER_HAND_FEATURE_DIM,
    ISL_ALPHABET,
    ISL_ALPHABET_METADATA,
    ISL_CLASSIFIER_MODEL_PATH,
    ISL_COMPLIANCE_DISCLAIMER,
    ISL_DATASET_PATH,
    MIN_RECOGNITION_CONFIDENCE,
    RecognitionMode,
    SUPPORTED_ALPHABET,
    SUPPORTED_GESTURES,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition_v2.gesture_classifier import (
    GestureClassifierV2,
    ISLMLClassifierV2,
    ISLRuleBasedClassifierV2,
)
from ai.sign_recognition_v2.hand_detector import (
    HandData,
    HandDetectionResultV2,
    HandDetectorV2,
)
from ai.sign_recognition_v2.landmark_extractor import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    LandmarkExtractorV2,
)
from ai.sign_recognition_v2.letter_buffer import LetterBufferV2
from ai.sign_recognition_v2.prediction import SignRecognitionServiceV2
from ai.sign_recognition_v2.sentence_builder import SentenceBuilderV2
from ai.sign_recognition_v2.speech import TextToSpeechEngineV2, tts_engine_v2
from ai.sign_recognition_v2.temporal_recognizer import TemporalRecognizerV2
from ai.sign_recognition_v2.train_model import (
    load_and_validate_dataset,
    train_and_evaluate_model,
)
from ai.sign_recognition_v2.word_builder import WordBuilderV2


class TestSignRecognitionV2ISL(unittest.TestCase):
    """Exhaustive test suite covering all 15 required verification categories."""

    def setUp(self):
        self.extractor = LandmarkExtractorV2()
        self.classifier = GestureClassifierV2()
        self.service = SignRecognitionServiceV2()

        # Synthetic standard 21-landmark array
        self.sample_landmarks = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            self.sample_landmarks[i] = [0.05 * i, 0.10 * i, 0.01 * i]

    # -------------------------------------------------------------
    # 1. One-Hand vs Two-Hand Feature Extraction
    # -------------------------------------------------------------
    def test_01_one_hand_and_two_hand_feature_extraction(self):
        # Single-hand input: Array of (21, 3)
        vec_1h = self.extractor.extract_features(self.sample_landmarks)
        self.assertEqual(len(vec_1h), TOTAL_FEATURE_DIM)  # 162
        self.assertEqual(TOTAL_FEATURE_DIM, 162)

        # In single-hand mode, secondary hand features (76) and inter-hand presence must be zero
        sec_features = vec_1h[HAND_FEATURE_DIM : HAND_FEATURE_DIM * 2]
        self.assertTrue(np.all(sec_features == 0.0))
        self.assertEqual(vec_1h[HAND_FEATURE_DIM * 2], 0.0)  # has_second_hand == 0.0

        # Two-hand input via HandDetectionResultV2
        hand1 = HandData(
            landmarks=self.sample_landmarks.tolist(),
            handedness="Right",
            confidence=0.95,
        )
        sample_landmarks2 = self.sample_landmarks.copy() + 0.2
        hand2 = HandData(
            landmarks=sample_landmarks2.tolist(),
            handedness="Left",
            confidence=0.92,
        )
        det_2h = HandDetectionResultV2(
            detected=True,
            num_hands=2,
            hands=[hand1, hand2],
            right_hand=hand1,
            left_hand=hand2,
        )
        vec_2h = self.extractor.extract_features(det_2h)
        self.assertEqual(len(vec_2h), TOTAL_FEATURE_DIM)
        # Secondary hand features must now be populated (non-zero)
        sec_2h = vec_2h[HAND_FEATURE_DIM : HAND_FEATURE_DIM * 2]
        self.assertFalse(np.all(sec_2h == 0.0))
        self.assertEqual(vec_2h[HAND_FEATURE_DIM * 2], 1.0)  # has_second_hand == 1.0

    # -------------------------------------------------------------
    # 2. Missing Hands Graceful Handling
    # -------------------------------------------------------------
    def test_02_missing_hands_handling(self):
        det_empty = HandDetectionResultV2(detected=False, num_hands=0, hands=[])
        label, conf, is_rec, meta = self.classifier.classify_detection(det_empty)
        self.assertIsNone(label)
        self.assertEqual(conf, 0.0)
        self.assertFalse(is_rec)
        self.assertEqual(meta["status"], "no_hand_detected")

        # Frame with None
        res_none = self.service.process_frame(None)
        self.assertFalse(res_none["success"])
        self.assertEqual(res_none["status"], "invalid_frame")

        # Blank black frame (no hands detected)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res_blank = self.service.process_frame(blank)
        self.assertTrue(res_blank["success"])
        self.assertFalse(res_blank["hand_detected"])
        self.assertIsNone(res_blank["recognized_text"])
        self.assertEqual(res_blank["system_status"], "NO HAND DETECTED")

    # -------------------------------------------------------------
    # 3. Non-Finite Values (NaN / Inf) Rejection
    # -------------------------------------------------------------
    def test_03_non_finite_values_rejection(self):
        # NaN in landmarks
        bad_nan = self.sample_landmarks.copy()
        bad_nan[5, 1] = np.nan
        with self.assertRaises(ValueError):
            self.extractor.normalize_single_hand(bad_nan)

        # Inf in landmarks
        bad_inf = self.sample_landmarks.copy()
        bad_inf[3, 0] = np.inf
        with self.assertRaises(ValueError):
            self.extractor.normalize_single_hand(bad_inf)

    # -------------------------------------------------------------
    # 4. Feature Version Enforcement
    # -------------------------------------------------------------
    def test_04_feature_version_enforcement(self):
        self.assertEqual(FEATURE_VERSION, "v2.3_isl_multihand")
        self.assertEqual(self.extractor.feature_version, FEATURE_VERSION)

        # Dataset with mismatched feature version must be rejected
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            header = ["label", "source_type", "feature_version"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
            writer.writerow(header)
            row = ["A", "live_camera", "v1.0_obsolete"] + [0.5] * TOTAL_FEATURE_DIM
            writer.writerow(row)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(temp_path, allow_synthetic_only=False)
            self.assertIn("Feature version mismatch", str(ctx.exception))
        finally:
            temp_path.unlink()

    # -------------------------------------------------------------
    # 5. Dataset Path Isolation
    # -------------------------------------------------------------
    def test_05_dataset_path_isolation(self):
        self.assertEqual(ISL_DATASET_PATH.name, "isl_alphabet.csv")
        self.assertEqual(CLASSIFIER_MODEL_PATH.name, "gesture_classifier.joblib")
        self.assertEqual(ISL_CLASSIFIER_MODEL_PATH.name, "isl_alphabet_classifier.joblib")
        # Ensure ISL dataset path is separate from legacy gestures.csv
        self.assertNotEqual(str(ISL_DATASET_PATH), str(CLASSIFIER_MODEL_PATH.parent.parent / "data" / "gestures.csv"))

    # -------------------------------------------------------------
    # 6. Synthetic-Only & Mixed Dataset Production Rejection
    # -------------------------------------------------------------
    def test_06_synthetic_only_and_mixed_dataset_rejection(self):
        # 6a: Pure synthetic dataset rejection for production
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            header = ["label", "source_type", "session_id", "signer_id"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
            writer.writerow(header)
            for _ in range(8):
                writer.writerow(["A", "synthetic_benchmark", "s1", "u1"] + [0.3] * TOTAL_FEATURE_DIM)
                writer.writerow(["B", "synthetic_benchmark", "s1", "u1"] + [0.7] * TOTAL_FEATURE_DIM)
            temp_syn = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(temp_syn, allow_synthetic_only=False)
            self.assertIn("SYNTHETIC-ONLY REJECTION", str(ctx.exception))

            # Benchmark mode permits synthetic training for test/CI
            X, y, src, sess, sgn, cnts = load_and_validate_dataset(temp_syn, allow_synthetic_only=True)
            self.assertEqual(len(X), 16)
        finally:
            temp_syn.unlink()

        # 6b: Mixed synthetic + real dataset rejection for production
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            header = ["label", "source_type", "session_id", "signer_id"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
            writer.writerow(header)
            for _ in range(5):
                writer.writerow(["A", "live_camera", "s1", "u1"] + [0.2] * TOTAL_FEATURE_DIM)
                writer.writerow(["B", "synthetic_benchmark", "s2", "u2"] + [0.8] * TOTAL_FEATURE_DIM)
            temp_mix = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(temp_mix, allow_synthetic_only=False)
            self.assertIn("MIXED-DATASET REJECTION", str(ctx.exception))
        finally:
            temp_mix.unlink()

    # -------------------------------------------------------------
    # 7. Unsupported Label Rejection
    # -------------------------------------------------------------
    def test_07_unsupported_label_rejection(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            header = ["label", "source_type"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
            writer.writerow(header)
            # Add an invalid non-alphabet label
            writer.writerow(["HELLO_WORLD", "live_camera"] + [0.5] * TOTAL_FEATURE_DIM)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(temp_path, allow_synthetic_only=False, enforce_isl_vocab=True)
            self.assertIn("not a recognized letter in the ISL Alphabet", str(ctx.exception))
        finally:
            temp_path.unlink()

        # Verify rejection of letters marked unsupported in ISLRTC configuration
        mock_meta = {**ISL_ALPHABET_METADATA["S"], "status": "unsupported"}
        with unittest.mock.patch.dict(ISL_ALPHABET_METADATA, {"S": mock_meta}):
            self.assertEqual(ISL_ALPHABET_METADATA["S"]["status"], "unsupported")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                writer = csv.writer(f)
                header = ["label", "source_type"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
                writer.writerow(header)
                writer.writerow(["S", "live_camera"] + [0.5] * TOTAL_FEATURE_DIM)
                temp_s = Path(f.name)

            try:
                with self.assertRaises(ValueError) as ctx:
                    load_and_validate_dataset(temp_s, allow_synthetic_only=False, enforce_isl_vocab=True)
                self.assertIn("marked as 'unsupported' in ISLRTC configuration", str(ctx.exception))
            finally:
                temp_s.unlink()

    # -------------------------------------------------------------
    # 8. Missing ISL Model Handling
    # -------------------------------------------------------------
    def test_08_missing_isl_model_handling(self):
        # Point to a nonexistent model path
        nonexistent = Path("/tmp/nonexistent_isl_model_12345.joblib")
        classifier = GestureClassifierV2(model_path=nonexistent, mode=RecognitionMode.ISL_ALPHABET)
        self.assertFalse(classifier.ml_classifier.is_loaded)

        # In ISL mode, system must NOT silently fall back to legacy gesture_classifier.joblib
        self.assertNotEqual(classifier.ml_classifier.model_path, CLASSIFIER_MODEL_PATH)

        # Must still operate cleanly using geometric rule engine without raising exceptions
        sample = np.zeros((21, 3), dtype=np.float32)
        label, conf, is_rec, meta = classifier.classify_landmarks(sample)
        # Low confidence or unknown
        self.assertFalse(is_rec)
        self.assertIn("disclaimer", meta)

    # -------------------------------------------------------------
    # 9. Rule vs ML Conflict Resolution (Safely UNKNOWN)
    # -------------------------------------------------------------
    def test_09_rule_vs_ml_conflict_resolution(self):
        classifier = GestureClassifierV2(mode=RecognitionMode.ISL_ALPHABET)

        # Simulate conflicting high-confidence predictions:
        # Rule predicts 'V' (0.92) while ML predicts 'B' (0.88)
        label, conf, is_rec, meta = classifier._arbitrate_predictions(
            rule_label="V",
            rule_conf=0.92,
            rule_meta={"rule": "test_v"},
            ml_label="B",
            ml_conf=0.88,
            ml_meta={"ml": "test_b"},
        )

        # Must reject conflict and return UNKNOWN
        self.assertIsNone(label)
        self.assertFalse(is_rec)
        self.assertEqual(meta["status"], "conflicting_predictions")
        self.assertEqual(meta["source"], "conflict_arbiter")
        self.assertLessEqual(conf, 0.35)

    # -------------------------------------------------------------
    # 10. Dynamic Sign Safeguards (J, Z)
    # -------------------------------------------------------------
    def test_10_dynamic_sign_safeguards(self):
        temporal = TemporalRecognizerV2(buffer_size=25)

        # Simulate 'Z' trajectory (zig-zag: Right -> Down-Left -> Right)
        for x in np.linspace(0.2, 0.6, 5):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[INDEX_TIP] = [x, 0.2, 0.0]
            temporal.add_frame(pts)

        for step in np.linspace(0.0, 1.0, 5):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[INDEX_TIP] = [0.6 - 0.4 * step, 0.2 + 0.3 * step, 0.0]
            temporal.add_frame(pts)

        for x in np.linspace(0.2, 0.6, 5):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[INDEX_TIP] = [x, 0.5, 0.0]
            temporal.add_frame(pts)

        # Dynamic sign requires anatomical handshape verification (index extended)
        letter, conf, meta = temporal.evaluate_dynamic_gesture(finger_states={"index": True})
        self.assertEqual(letter, "Z")
        self.assertGreaterEqual(conf, 0.70)
        self.assertEqual(meta["status"], "experimental_dynamic")

        # If handshape fails (e.g. index is curled), dynamic trigger must be rejected
        temporal.reset()
        for x in np.linspace(0.2, 0.6, 15):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[INDEX_TIP] = [x, 0.2, 0.0]
            temporal.add_frame(pts)
        letter_rej, conf_rej, meta_rej = temporal.evaluate_dynamic_gesture(finger_states={"index": False})
        self.assertIsNone(letter_rej)

    # -------------------------------------------------------------
    # 11. Hand Absence vs Unknown vs Low Confidence
    # -------------------------------------------------------------
    def test_11_hand_absence_vs_unknown_vs_low_confidence(self):
        buffer = LetterBufferV2(min_confidence=0.65)

        # 11a: Hand Absence (explicitly resets and sets hand_was_released = True)
        buffer.record_no_hand()
        self.assertTrue(buffer.hand_was_released)
        self.assertEqual(buffer.current_status, "NO HAND DETECTED")

        # 11b: Unknown Gesture (hand present, but gesture is unknown)
        # CRITICAL: Must NOT count as a hand release!
        buffer.hand_was_released = False
        disp, conf, accepted, status = buffer.process_prediction("UNKNOWN", 0.30)
        self.assertFalse(accepted)
        self.assertEqual(status, "UNKNOWN GESTURE")
        self.assertFalse(buffer.hand_was_released)  # Still False!

        # 11c: Low Confidence (hand present, but below min_confidence)
        # Must NOT count as a hand release!
        disp, conf, accepted, status = buffer.process_prediction("A", 0.50)
        self.assertFalse(accepted)
        self.assertEqual(status, "LOW CONFIDENCE")
        self.assertFalse(buffer.hand_was_released)  # Still False!

    # -------------------------------------------------------------
    # 12. Repeated Letter Support (Spelling 'HELLO' with 'LL')
    # -------------------------------------------------------------
    def test_12_repeated_letter_support_hello(self):
        buffer = LetterBufferV2(stability_threshold=3)
        wb = WordBuilderV2()

        # Step 1: Spell 'H' (3 stable frames)
        for _ in range(3):
            disp, conf, acc, _ = buffer.process_prediction("H", 0.90)
            if acc:
                wb.add_letter(disp)
        self.assertEqual(wb.current_word, "H")

        # Step 2: Spell 'E' (3 stable frames)
        for _ in range(3):
            disp, conf, acc, _ = buffer.process_prediction("E", 0.90)
            if acc:
                wb.add_letter(disp)
        self.assertEqual(wb.current_word, "HE")

        # Step 3: Spell first 'L' (3 stable frames)
        for _ in range(3):
            disp, conf, acc, _ = buffer.process_prediction("L", 0.90)
            if acc:
                wb.add_letter(disp)
        self.assertEqual(wb.current_word, "HEL")

        # Step 4: HOLD 'L' continuously without hand release (10 frames)
        # Must NOT append duplicate 'L's during continuous hold!
        for _ in range(10):
            disp, conf, acc, _ = buffer.process_prediction("L", 0.90)
            if acc:
                wb.add_letter(disp)
        self.assertEqual(wb.current_word, "HEL")  # Still "HEL"!

        # Step 5: Signer GENUINELY releases hand (drops hand)
        buffer.record_no_hand()
        self.assertTrue(buffer.hand_was_released)

        # Step 6: Spell second 'L' (3 stable frames after hand release)
        for _ in range(3):
            disp, conf, acc, _ = buffer.process_prediction("L", 0.90)
            if acc:
                wb.add_letter(disp)
        # Second 'L' successfully accepted!
        self.assertEqual(wb.current_word, "HELL")

        # Step 7: Spell 'O' (3 stable frames)
        for _ in range(3):
            disp, conf, acc, _ = buffer.process_prediction("O", 0.90)
            if acc:
                wb.add_letter(disp)
        # 'HELLO' successfully formed!
        self.assertEqual(wb.current_word, "HELLO")

    # -------------------------------------------------------------
    # 13. Word and Sentence Builder Operations
    # -------------------------------------------------------------
    def test_13_word_and_sentence_builder_operations(self):
        sb = SentenceBuilderV2()

        # Append letters to current word
        sb.add_letter("I")
        sb.add_letter("S")
        sb.add_letter("L")
        self.assertEqual(sb.current_word, "ISL")
        self.assertEqual(sb.full_sentence, "ISL")

        # Space insertion completes word
        finished = sb.insert_space()
        self.assertEqual(finished, "ISL")
        self.assertEqual(sb.current_word, "")
        self.assertEqual(sb.completed_words, ["ISL"])

        # Second word
        sb.add_letter("R")
        sb.add_letter("O")
        sb.add_letter("C")
        sb.add_letter("K")
        sb.add_letter("S")
        self.assertEqual(sb.full_sentence, "ISL ROCKS")

        # Backspace inside current word
        removed = sb.backspace()
        self.assertEqual(removed, "S")
        self.assertEqual(sb.current_word, "ROCK")

        # Backspace on empty current word reopens previous completed word
        sb.word_builder.clear_word()
        removed_prev = sb.backspace()
        self.assertEqual(removed_prev, "L")
        self.assertEqual(sb.current_word, "IS")
        self.assertEqual(sb.completed_words, [])

        # Clear sentence
        sb.clear_sentence()
        self.assertEqual(sb.full_sentence, "")
        self.assertEqual(sb.current_word, "")
        self.assertEqual(sb.completed_words, [])

    # -------------------------------------------------------------
    # 14. TTS Contract (Zero Crash & Clean Status)
    # -------------------------------------------------------------
    def test_14_tts_contract(self):
        tts = TextToSpeechEngineV2()
        # Empty input must return False
        self.assertFalse(tts.speak(""))
        self.assertFalse(tts.speak("   "))

        # If native CLI TTS binary is not installed in the environment:
        # Default speak without browser fallback must return False (clean contract)
        if not tts.has_native_tts:
            res_native = tts.speak("Test sentence", browser_fallback=False)
            self.assertFalse(res_native)

            # With browser fallback enabled, must return True
            res_browser = tts.speak("Test sentence", browser_fallback=True)
            self.assertTrue(res_browser)

        # Standard payload formatting
        payload = tts.get_speech_payload("Hello")
        self.assertEqual(payload["text"], "Hello")
        self.assertIn("backend", payload)

    # -------------------------------------------------------------
    # 15. Master Service End-to-End Consistency
    # -------------------------------------------------------------
    def test_15_master_service_consistency(self):
        service = SignRecognitionServiceV2()

        # Initial clean state
        self.assertIsNone(service.current_letter)
        self.assertEqual(service.current_word, "")
        self.assertEqual(service.completed_sentence, "")
        self.assertEqual(service.recognition_status, "NO HAND DETECTED")

        # Frame processing returns standardized dict keys
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = service.process_frame(frame, annotate=True)

        expected_keys = [
            "success",
            "recognized_text",
            "confidence",
            "hand_detected",
            "num_hands",
            "gesture_recognized",
            "status",
            "system_status",
            "current_letter",
            "current_word",
            "sentence",
            "latency_ms",
            "annotated_frame",
        ]
        for key in expected_keys:
            self.assertIn(key, res)

        self.assertIsInstance(res["latency_ms"], float)
        self.assertGreaterEqual(res["latency_ms"], 0.0)
        self.assertTrue(res["success"])

    # -------------------------------------------------------------
    # 16. Incompatible 73-Feature Generic Gesture Model Rejection
    # -------------------------------------------------------------
    def test_16_rejection_of_existing_73_feature_generic_model(self):
        # The existing model file is 73-features and lacks metadata
        clf = ISLMLClassifierV2(model_path=ISL_CLASSIFIER_MODEL_PATH, mode=RecognitionMode.ISL_ALPHABET)
        self.assertFalse(clf.is_loaded)
        self.assertIsNone(clf.model)
        self.assertIsNotNone(clf.load_error)
        self.assertIn("dimension mismatch", clf.load_error.lower())

    # -------------------------------------------------------------
    # 17. Rejection of Models with Generic Gesture Labels
    # -------------------------------------------------------------
    def test_17_rejection_of_models_with_generic_gesture_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "generic_model.joblib"
            meta_path = Path(tmpdir) / "generic_model.json"

            # Create a 162-feature model with generic gesture classes (HELLO, STOP)
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            X_dummy = np.random.randn(10, TOTAL_FEATURE_DIM).astype(np.float32)
            y_dummy = np.array(["HELLO"] * 5 + ["STOP"] * 5)
            rf.fit(X_dummy, y_dummy)
            joblib.dump(rf, model_path)

            meta_info = {
                "model_type": "RandomForestClassifier",
                "feature_version": FEATURE_VERSION,
                "feature_dimension": TOTAL_FEATURE_DIM,
                "classes": ["HELLO", "STOP"],
                "dataset_path": "dummy.csv",
                "dataset_sha256": "abcdef",
                "source_type_counts": {"live_camera": 10},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_info, f)

            clf = ISLMLClassifierV2(model_path=model_path, mode=RecognitionMode.ISL_ALPHABET)
            self.assertFalse(clf.is_loaded)
            self.assertIsNone(clf.model)
            self.assertIn("invalid or non-isl class labels", clf.load_error.lower())

    # -------------------------------------------------------------
    # 18. Rejection when Metadata JSON is Missing
    # -------------------------------------------------------------
    def test_18_rejection_when_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "no_meta_model.joblib"
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            X_dummy = np.random.randn(10, TOTAL_FEATURE_DIM).astype(np.float32)
            y_dummy = np.array(["A"] * 5 + ["B"] * 5)
            rf.fit(X_dummy, y_dummy)
            joblib.dump(rf, model_path)
            # Intentionally do not create JSON metadata

            clf = ISLMLClassifierV2(model_path=model_path, mode=RecognitionMode.ISL_ALPHABET)
            self.assertFalse(clf.is_loaded)
            self.assertIsNone(clf.model)
            self.assertIn("metadata json sidecar not found", clf.load_error.lower())

    # -------------------------------------------------------------
    # 19. Rejection when Metadata Feature Version is Incorrect
    # -------------------------------------------------------------
    def test_19_rejection_when_metadata_feature_version_incorrect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bad_ver_model.joblib"
            meta_path = Path(tmpdir) / "bad_ver_model.json"
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            X_dummy = np.random.randn(10, TOTAL_FEATURE_DIM).astype(np.float32)
            y_dummy = np.array(["A"] * 5 + ["B"] * 5)
            rf.fit(X_dummy, y_dummy)
            joblib.dump(rf, model_path)

            meta_info = {
                "model_type": "RandomForestClassifier",
                "feature_version": "v1.0_obsolete",
                "feature_dimension": TOTAL_FEATURE_DIM,
                "classes": ["A", "B"],
                "dataset_path": "dummy.csv",
                "dataset_sha256": "abcdef",
                "source_type_counts": {"live_camera": 10},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_info, f)

            clf = ISLMLClassifierV2(model_path=model_path, mode=RecognitionMode.ISL_ALPHABET)
            self.assertFalse(clf.is_loaded)
            self.assertIn("feature_version mismatch", clf.load_error.lower())

    # -------------------------------------------------------------
    # 20. Acceptance of Correct 162-Feature Model with Matching Metadata
    # -------------------------------------------------------------
    def test_20_acceptance_of_valid_162_feature_model_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "valid_model.joblib"
            meta_path = Path(tmpdir) / "valid_model.json"
            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            X_dummy = np.random.randn(10, TOTAL_FEATURE_DIM).astype(np.float32)
            y_dummy = np.array(["A"] * 5 + ["B"] * 5)
            rf.fit(X_dummy, y_dummy)
            joblib.dump(rf, model_path)

            meta_info = {
                "model_type": "RandomForestClassifier",
                "feature_version": FEATURE_VERSION,
                "feature_dimension": TOTAL_FEATURE_DIM,
                "classes": ["A", "B"],
                "dataset_path": "real_isl_alphabet.csv",
                "dataset_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "source_type_counts": {"live_camera": 10},
                "is_benchmark_mode": False,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_info, f)

            clf = ISLMLClassifierV2(model_path=model_path, mode=RecognitionMode.ISL_ALPHABET)
            self.assertTrue(clf.is_loaded)
            self.assertIsNotNone(clf.model)
            self.assertIsNone(clf.load_error)

            # Prediction with valid vector
            pred, conf, meta = clf.predict(np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32))
            self.assertIn(pred, ["A", "B"])
            self.assertGreater(conf, 0.0)

    # -------------------------------------------------------------
    # 21. No Silent Legacy Fallback in ISL Mode
    # -------------------------------------------------------------
    def test_21_no_silent_legacy_fallback_in_isl_mode(self):
        clf = ISLMLClassifierV2(mode=RecognitionMode.ISL_ALPHABET)
        # Must strictly target ISL_CLASSIFIER_MODEL_PATH and not CLASSIFIER_MODEL_PATH
        self.assertNotEqual(clf.model_path.resolve(), CLASSIFIER_MODEL_PATH.resolve())
        self.assertFalse(clf.is_loaded)
        pred, conf, meta = clf.predict(np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32))
        self.assertIsNone(pred)
        self.assertEqual(meta["error"], "model_not_loaded")
        self.assertIn("reason", meta)

    # -------------------------------------------------------------
    # 22. Dataset SHA256 Metadata Generation
    # -------------------------------------------------------------
    def test_22_dataset_sha256_metadata_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample_isl.csv"
            out_model = Path(tmpdir) / "trained_model.joblib"
            out_meta = Path(tmpdir) / "trained_model.json"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["label", "source_type", "session_id", "signer_id"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
                writer.writerow(header)
                for i in range(6):
                    writer.writerow(["A", "live_camera", "s1", "u1"] + [0.1 + i * 0.01] * TOTAL_FEATURE_DIM)
                    writer.writerow(["B", "live_camera", "s1", "u1"] + [0.8 + i * 0.01] * TOTAL_FEATURE_DIM)

            X, y, src, sess, sgnr, counts = load_and_validate_dataset(csv_path, allow_synthetic_only=False)
            train_and_evaluate_model(
                X, y, src, sess, sgnr, out_model, dataset_path=csv_path, is_benchmark_mode=False
            )

            self.assertTrue(out_meta.exists())
            with open(out_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertIn("dataset_sha256", meta)
            self.assertEqual(len(meta["dataset_sha256"]), 64)
            # Verify SHA256 matches actual file bytes
            actual_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            self.assertEqual(meta["dataset_sha256"], actual_hash)

    # -------------------------------------------------------------
    # 23. Legacy Dataset Rejection Safeguard
    # -------------------------------------------------------------
    def test_23_legacy_dataset_rejection_safeguard(self):
        # Attempting to load DATASET_PATH (gestures.csv) for ISL training must raise ValueError
        if DATASET_PATH.exists():
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(DATASET_PATH, enforce_isl_vocab=True)
            self.assertIn("DATASET SEPARATION SAFEGUARD", str(ctx.exception))

    # -------------------------------------------------------------
    # 24. Signer-Independent GroupShuffleSplit Evaluation
    # -------------------------------------------------------------
    def test_24_signer_independent_splitting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "multi_signer.csv"
            out_model = Path(tmpdir) / "model.joblib"
            out_meta = Path(tmpdir) / "model.json"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["label", "source_type", "session_id", "signer_id"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
                writer.writerow(header)
                # Signer 1 (Alice)
                for i in range(8):
                    writer.writerow(["A", "live_camera", "sess_alice", "signer_alice"] + [0.1 + i * 0.01] * TOTAL_FEATURE_DIM)
                    writer.writerow(["B", "live_camera", "sess_alice", "signer_alice"] + [0.8 + i * 0.01] * TOTAL_FEATURE_DIM)
                # Signer 2 (Bob)
                for i in range(8):
                    writer.writerow(["A", "live_camera", "sess_bob", "signer_bob"] + [0.1 + i * 0.01] * TOTAL_FEATURE_DIM)
                    writer.writerow(["B", "live_camera", "sess_bob", "signer_bob"] + [0.8 + i * 0.01] * TOTAL_FEATURE_DIM)

            X, y, src, sess, sgnr, counts = load_and_validate_dataset(csv_path, allow_synthetic_only=False)
            train_and_evaluate_model(
                X, y, src, sess, sgnr, out_model, dataset_path=csv_path, is_benchmark_mode=False
            )

            with open(out_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertTrue(meta["signer_independent_evaluation_available"])
            self.assertEqual(meta["evaluation_split_strategy"], "signer_independent_split")
            self.assertEqual(meta["unique_signer_count"], 2)

    # -------------------------------------------------------------
    # 25. Insufficient Signer Data Honest Reporting
    # -------------------------------------------------------------
    def test_25_insufficient_signer_data_honest_reporting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "single_signer.csv"
            out_model = Path(tmpdir) / "model.joblib"
            out_meta = Path(tmpdir) / "model.json"

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["label", "source_type", "session_id", "signer_id"] + [f"feat_{i}" for i in range(TOTAL_FEATURE_DIM)]
                writer.writerow(header)
                # Only 1 signer
                for i in range(6):
                    writer.writerow(["A", "live_camera", "s1", "signer_solo"] + [0.1 + i * 0.01] * TOTAL_FEATURE_DIM)
                    writer.writerow(["B", "live_camera", "s1", "signer_solo"] + [0.8 + i * 0.01] * TOTAL_FEATURE_DIM)

            # When require_signer_independent is requested, must raise ValueError
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(csv_path, allow_synthetic_only=False, require_signer_independent=True)
            self.assertIn("INSUFFICIENT SIGNERS", str(ctx.exception))

            # In standard training, must report honest split strategy (not claiming unseen signer generalization)
            X, y, src, sess, sgnr, counts = load_and_validate_dataset(csv_path, allow_synthetic_only=False)
            train_and_evaluate_model(
                X, y, src, sess, sgnr, out_model, dataset_path=csv_path, is_benchmark_mode=False
            )

            with open(out_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertFalse(meta["signer_independent_evaluation_available"])
            self.assertIn("single signer/session", meta["evaluation_split_strategy"])

    # -------------------------------------------------------------
    # 26. Degraded Detection Rejection in GestureClassifierV2
    # -------------------------------------------------------------
    def test_26_degraded_detection_rejected_by_gesture_classifier(self):
        hand = HandData(landmarks=self.sample_landmarks.tolist(), handedness="Right", confidence=0.85)
        det_degraded = HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            is_degraded=True,
        )

        label, conf, is_rec, meta = self.classifier.classify_detection(det_degraded)
        self.assertIsNone(label)
        self.assertEqual(conf, 0.0)
        self.assertFalse(is_rec)
        self.assertEqual(meta["status"], "degraded_detection_rejected")
        self.assertEqual(meta["source"], "none")
        self.assertIn("explanation", meta)
        self.assertIn("Approximate fallback landmarks", meta["explanation"])

    # -------------------------------------------------------------
    # 27. Degraded Detection Does Not Invoke Temporal, Rule, or ML
    # -------------------------------------------------------------
    def test_27_degraded_detection_does_not_invoke_subcomponents(self):
        classifier = GestureClassifierV2()
        classifier.temporal_recognizer.add_frame = MagicMock()
        classifier.rule_classifier.classify_detection = MagicMock()
        classifier.ml_classifier.predict = MagicMock()

        hand = HandData(landmarks=self.sample_landmarks.tolist(), handedness="Right", confidence=0.85)
        det_degraded = HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            is_degraded=True,
        )

        label, conf, is_rec, meta = classifier.classify_detection(det_degraded)
        self.assertIsNone(label)
        self.assertFalse(is_rec)

        classifier.temporal_recognizer.add_frame.assert_not_called()
        classifier.rule_classifier.classify_detection.assert_not_called()
        classifier.ml_classifier.predict.assert_not_called()

    # -------------------------------------------------------------
    # 28. Degraded Detection Rejection in ISLRuleBasedClassifierV2
    # -------------------------------------------------------------
    def test_28_degraded_detection_rejected_by_rule_classifier(self):
        rule_clf = ISLRuleBasedClassifierV2(self.extractor)
        hand = HandData(landmarks=self.sample_landmarks.tolist(), handedness="Right", confidence=0.85)
        det_degraded = HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            is_degraded=True,
        )

        label, conf, meta = rule_clf.classify_detection(det_degraded)
        self.assertIsNone(label)
        self.assertEqual(conf, 0.0)
        self.assertEqual(meta["status"], "degraded_detection_rejected")
        self.assertEqual(meta["source"], "none")
        self.assertEqual(meta["error"], "degraded_detection_rejected")
        self.assertIn("explanation", meta)

    # -------------------------------------------------------------
    # 29. Non-Degraded Detection Not Rejected by Degraded Guard
    # -------------------------------------------------------------
    def test_29_non_degraded_detection_not_rejected_by_degraded_guard(self):
        hand = HandData(landmarks=self.sample_landmarks.tolist(), handedness="Right", confidence=0.85)
        det_clean = HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            is_degraded=False,
        )

        label, conf, is_rec, meta = self.classifier.classify_detection(det_clean)
        self.assertNotEqual(meta.get("status"), "degraded_detection_rejected")

        rule_clf = ISLRuleBasedClassifierV2(self.extractor)
        r_label, r_conf, r_meta = rule_clf.classify_detection(det_clean)
        self.assertNotEqual(r_meta.get("status"), "degraded_detection_rejected")

    # -------------------------------------------------------------
    # 30. MediaPipe Zero-Hand Inference Does Not Invoke OpenCV Fallback
    # -------------------------------------------------------------
    def test_30_mediapipe_zero_hand_inference_does_not_invoke_fallback(self):
        detector = HandDetectorV2()
        mock_landmarker = MagicMock()
        mock_result = MagicMock()
        mock_result.hand_landmarks = []
        mock_result.handedness = []
        mock_landmarker.detect.return_value = mock_result

        detector.mediapipe_landmarker = mock_landmarker
        detector.backend = "mediapipe_tasks"
        detector.fallback_detector.detect = MagicMock()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        detector.fallback_detector.detect.assert_not_called()
        self.assertFalse(res.detected)
        self.assertEqual(res.num_hands, 0)
        self.assertEqual(res.backend, "mediapipe_tasks")
        self.assertFalse(res.is_degraded)
        self.assertEqual(res.hands, [])

    # -------------------------------------------------------------
    # 31. MediaPipe Successful Hand Detection Preserves Full Behavior
    # -------------------------------------------------------------
    def test_31_mediapipe_successful_detection_preserves_behavior(self):
        detector = HandDetectorV2()
        mock_landmarker = MagicMock()

        mock_lm = MagicMock(x=0.5, y=0.4, z=-0.1)
        mock_hand_landmarks = [mock_lm] * 21
        mock_category = MagicMock(category_name="Right", score=0.96)
        mock_result = MagicMock()
        mock_result.hand_landmarks = [mock_hand_landmarks]
        mock_result.handedness = [[mock_category]]
        mock_landmarker.detect.return_value = mock_result

        detector.mediapipe_landmarker = mock_landmarker
        detector.backend = "mediapipe_tasks"
        detector.fallback_detector.detect = MagicMock()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        detector.fallback_detector.detect.assert_not_called()
        self.assertTrue(res.detected)
        self.assertEqual(res.num_hands, 1)
        self.assertEqual(res.backend, "mediapipe_tasks")
        self.assertFalse(res.is_degraded)
        self.assertEqual(res.handedness, "Right")
        self.assertEqual(len(res.landmarks), 21)
        self.assertEqual(res.confidence, 0.96)
        self.assertIsNotNone(res.right_hand)
        self.assertEqual(res.bbox, (320, 192, 1, 1))

    # -------------------------------------------------------------
    # 32. MediaPipe Inference Exception Uses Degraded Fallback
    # -------------------------------------------------------------
    def test_32_mediapipe_inference_exception_uses_degraded_fallback(self):
        detector = HandDetectorV2()
        mock_landmarker = MagicMock()
        mock_landmarker.detect.side_effect = RuntimeError("MediaPipe internal runtime error")

        detector.mediapipe_landmarker = mock_landmarker
        detector.backend = "mediapipe_tasks"
        detector.fallback_detector.detect = MagicMock(
            return_value=HandDetectionResultV2(
                detected=False,
                backend="opencv_degraded_fallback",
                is_degraded=True,
            )
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        detector.fallback_detector.detect.assert_called_once_with(frame)
        self.assertEqual(res.backend, "opencv_degraded_fallback")
        self.assertTrue(res.is_degraded)

    # -------------------------------------------------------------
    # 33. MediaPipe Unavailable Uses Degraded Fallback
    # -------------------------------------------------------------
    def test_33_mediapipe_unavailable_uses_degraded_fallback(self):
        detector = HandDetectorV2()
        detector.mediapipe_landmarker = None
        detector.fallback_detector.detect = MagicMock(
            return_value=HandDetectionResultV2(
                detected=False,
                backend="opencv_degraded_fallback",
                is_degraded=True,
            )
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.detect(frame)

        detector.fallback_detector.detect.assert_called_once_with(frame)
        self.assertEqual(res.backend, "opencv_degraded_fallback")
        self.assertTrue(res.is_degraded)

    # -------------------------------------------------------------
    # Test Landmark Synthesis Helper
    # -------------------------------------------------------------
    @staticmethod
    def _synthesize_hand(finger_ext=(0, 0, 0, 0, 0), wrist=(0.5, 0.7, 0.0), scale=0.2, flip_y=False):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = wrist
        sign = 1 if flip_y else -1
        pts[1] = [wrist[0] - 0.05, wrist[1] + sign * 0.05, 0]
        pts[2] = [wrist[0] - 0.08, wrist[1] + sign * 0.08, 0]
        pts[3] = [wrist[0] - 0.10, wrist[1] + sign * 0.11, 0]
        pts[4] = [wrist[0] - 0.12, wrist[1] + sign * 0.14, 0] if finger_ext[0] else [wrist[0] - 0.06, wrist[1] + sign * 0.08, 0]

        for f_idx, ext in enumerate(finger_ext[1:], start=1):
            mcp_idx = 1 + f_idx * 4
            base_x = wrist[0] + (f_idx - 2.5) * 0.04
            pts[mcp_idx] = [base_x, wrist[1] + sign * scale * 0.5, 0]
            pts[mcp_idx + 1] = [base_x, wrist[1] + sign * scale * 0.7, 0]
            pts[mcp_idx + 2] = [base_x, wrist[1] + sign * scale * 0.85, 0]
            if ext:
                pts[mcp_idx + 3] = [base_x, wrist[1] + sign * scale * 1.05, 0]
            else:
                pts[mcp_idx + 3] = [base_x, wrist[1] + sign * scale * 0.6, 0]
        return pts

    # -------------------------------------------------------------
    # 34. Symmetric Two-Handed Letters (G, S, X, W, Q)
    # -------------------------------------------------------------
    def test_34_two_handed_letters_G_S_X_W_Q(self):
        rule_clf = ISLRuleBasedClassifierV2(self.extractor)

        # G: Two closed fists stacked vertically
        g1 = self._synthesize_hand((0, 0, 0, 0, 0), wrist=(0.5, 0.35, 0.0))
        g2 = self._synthesize_hand((0, 0, 0, 0, 0), wrist=(0.5, 0.65, 0.0))
        lbl, conf, _ = rule_clf.classify_geometry(g1, g2)
        self.assertEqual(lbl, "G")
        self.assertGreaterEqual(conf, 0.90)

        # S: Two pinkies interlocked
        s1 = self._synthesize_hand((0, 0, 0, 0, 1), wrist=(0.48, 0.6, 0.0))
        s2 = self._synthesize_hand((0, 0, 0, 0, 1), wrist=(0.52, 0.6, 0.0))
        lbl, conf, _ = rule_clf.classify_geometry(s1, s2)
        self.assertEqual(lbl, "S")
        self.assertGreaterEqual(conf, 0.90)

        # X: Two index fingers crossed into an X
        x1 = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.48, 0.6, 0.0))
        x2 = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.52, 0.6, 0.0))
        lbl, conf, _ = rule_clf.classify_geometry(x1, x2)
        self.assertEqual(lbl, "X")
        self.assertGreaterEqual(conf, 0.90)

        # W: Two hands with spread fingers interlocking
        w1 = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.48, 0.6, 0.0))
        w2 = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.52, 0.6, 0.0))
        lbl, conf, _ = rule_clf.classify_geometry(w1, w2)
        self.assertEqual(lbl, "W")
        self.assertGreaterEqual(conf, 0.90)

        # Q: Dominant thumb-index circle hooking into non-dominant circle
        q1 = self._synthesize_hand((1, 1, 0, 0, 0), wrist=(0.48, 0.60, 0.0), flip_y=False)
        q1[THUMB_TIP] = q1[INDEX_TIP].copy()
        q2 = self._synthesize_hand((1, 1, 0, 0, 0), wrist=(0.52, 0.35, 0.0), flip_y=True)
        q2[THUMB_TIP] = q2[INDEX_TIP].copy()
        shift_q = q1[THUMB_TIP] - q2[INDEX_TIP]
        q2 += shift_q
        lbl, conf, _ = rule_clf.classify_geometry(q1, q2)
        self.assertEqual(lbl, "Q")
        self.assertGreaterEqual(conf, 0.90)

    # -------------------------------------------------------------
    # 35. Asymmetric Two-Handed Letters (K, P, M, N, R, T)
    # -------------------------------------------------------------
    def test_35_two_handed_letters_K_P_M_N_R_T(self):
        rule_clf = ISLRuleBasedClassifierV2(self.extractor)

        # K: Upright index knuckle contacted by hooked index
        k_base = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.6, 0.0))
        k_dom = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.55, 0.55, 0.0))
        k_dom[INDEX_TIP] = k_base[INDEX_PIP].copy()
        lbl, conf, _ = rule_clf.classify_geometry(k_dom, k_base)
        self.assertEqual(lbl, "K")
        self.assertGreaterEqual(conf, 0.90)

        # P: Upright index contacted by downward thumb-index loop
        p_base = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.6, 0.0))
        p_dom = self._synthesize_hand((1, 1, 0, 0, 0), wrist=(0.45, 0.45, 0.0), flip_y=True)
        p_dom[INDEX_TIP] = p_dom[THUMB_TIP] = p_base[INDEX_TIP].copy()
        lbl, conf, _ = rule_clf.classify_geometry(p_dom, p_base)
        self.assertEqual(lbl, "P")
        self.assertGreaterEqual(conf, 0.90)

        # Palm letters: Non-dominant flat open palm
        palm_base = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.4, 0.6, 0.0), flip_y=False)

        # M: 3 fingers resting on palm
        m_dom = self._synthesize_hand((0, 1, 1, 1, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl, conf, _ = rule_clf.classify_geometry(m_dom, palm_base)
        self.assertEqual(lbl, "M")
        self.assertGreaterEqual(conf, 0.90)

        # N: 2 fingers resting on palm
        n_dom = self._synthesize_hand((0, 1, 1, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl, conf, _ = rule_clf.classify_geometry(n_dom, palm_base)
        self.assertEqual(lbl, "N")
        self.assertGreaterEqual(conf, 0.90)

        # R: 1 finger resting on palm center
        r_dom = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl, conf, _ = rule_clf.classify_geometry(r_dom, palm_base)
        self.assertEqual(lbl, "R")
        self.assertGreaterEqual(conf, 0.90)

        # T: 1 finger touching palm side edge
        t_dom = r_dom.copy()
        shift_t = palm_base[17] - t_dom[INDEX_TIP]  # PINKY_MCP
        t_dom += shift_t
        lbl, conf, _ = rule_clf.classify_geometry(t_dom, palm_base)
        self.assertEqual(lbl, "T")
        self.assertGreaterEqual(conf, 0.90)

    # -------------------------------------------------------------
    # 36. Palm Contact Family Differentiation (M, N, R, T, H)
    # -------------------------------------------------------------
    def test_36_palm_contact_family_differentiation(self):
        rule_clf = ISLRuleBasedClassifierV2(self.extractor)
        palm_base = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.4, 0.6, 0.0), flip_y=False)

        # 4 fingers (flat palm) -> H
        h_dom = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl_h, _, _ = rule_clf.classify_geometry(h_dom, palm_base)
        self.assertEqual(lbl_h, "H")

        # 3 fingers -> M
        m_dom = self._synthesize_hand((0, 1, 1, 1, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl_m, _, _ = rule_clf.classify_geometry(m_dom, palm_base)
        self.assertEqual(lbl_m, "M")

        # 2 fingers -> N
        n_dom = self._synthesize_hand((0, 1, 1, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl_n, _, _ = rule_clf.classify_geometry(n_dom, palm_base)
        self.assertEqual(lbl_n, "N")

        # 1 finger on palm center -> R
        r_dom = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        lbl_r, _, _ = rule_clf.classify_geometry(r_dom, palm_base)
        self.assertEqual(lbl_r, "R")

        # 1 finger on palm edge -> T
        t_dom = r_dom.copy()
        shift_t = palm_base[17] - t_dom[INDEX_TIP]
        t_dom += shift_t
        lbl_t, _, _ = rule_clf.classify_geometry(t_dom, palm_base)
        self.assertEqual(lbl_t, "T")

        # Ensure all 5 letters produce distinct, mutually exclusive labels
        labels = [lbl_h, lbl_m, lbl_n, lbl_r, lbl_t]
        self.assertEqual(len(set(labels)), 5)
        self.assertEqual(sorted(labels), ["H", "M", "N", "R", "T"])

    # -------------------------------------------------------------
    # 37. Dominant Hand Invariance for Asymmetric Letters
    # -------------------------------------------------------------
    def test_37_dominant_hand_invariance_for_new_letters(self):
        rule_clf = ISLRuleBasedClassifierV2(self.extractor)
        palm_base = self._synthesize_hand((1, 1, 1, 1, 1), wrist=(0.4, 0.6, 0.0), flip_y=False)
        m_dom = self._synthesize_hand((0, 1, 1, 1, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        n_dom = self._synthesize_hand((0, 1, 1, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)
        r_dom = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.35, 0.0), flip_y=True)

        k_base = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.6, 0.0))
        k_dom = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.55, 0.55, 0.0))
        k_dom[INDEX_TIP] = k_base[INDEX_PIP].copy()

        p_base = self._synthesize_hand((0, 1, 0, 0, 0), wrist=(0.4, 0.6, 0.0))
        p_dom = self._synthesize_hand((1, 1, 0, 0, 0), wrist=(0.45, 0.45, 0.0), flip_y=True)
        p_dom[INDEX_TIP] = p_dom[THUMB_TIP] = p_base[INDEX_TIP].copy()

        # Swapping primary and secondary hands must yield identical classification
        for expected, h1, h2 in [
            ("M", m_dom, palm_base),
            ("N", n_dom, palm_base),
            ("R", r_dom, palm_base),
            ("K", k_dom, k_base),
            ("P", p_dom, p_base),
        ]:
            lbl_normal, conf_norm, _ = rule_clf.classify_geometry(h1, h2)
            lbl_swapped, conf_swap, _ = rule_clf.classify_geometry(h2, h1)
            self.assertEqual(lbl_normal, expected)
            self.assertEqual(lbl_swapped, expected)
            self.assertAlmostEqual(conf_norm, conf_swap, places=2)

    # -------------------------------------------------------------
    # 38. Complete 26 Alphabet Coverage
    # -------------------------------------------------------------
    def test_38_complete_26_alphabet_coverage(self):
        # All 26 letters A through Z must be supported
        self.assertEqual(len(SUPPORTED_ALPHABET), 26)
        self.assertEqual(SUPPORTED_ALPHABET, [chr(i) for i in range(ord('A'), ord('Z') + 1)])

        # Metadata must have exactly 24 experimental and 2 experimental_dynamic, 0 unsupported
        counts = {}
        for letter in SUPPORTED_ALPHABET:
            meta = ISL_ALPHABET_METADATA[letter]
            status = meta["status"]
            counts[status] = counts.get(status, 0) + 1
            self.assertIn(status, ("experimental", "experimental_dynamic"))
            self.assertIn("dominant_hand", meta)
            self.assertIn("hands_required", meta)

        self.assertEqual(counts.get("experimental", 0), 24)
        self.assertEqual(counts.get("experimental_dynamic", 0), 2)
        self.assertEqual(counts.get("unsupported", 0), 0)


if __name__ == "__main__":
    unittest.main()
