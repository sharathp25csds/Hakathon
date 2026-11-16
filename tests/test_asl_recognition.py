"""SignBridge AI - Comprehensive American Sign Language (ASL) Alphabet Recognition Tests.
Validates the complete 26-letter ASL fingerspelling alphabet:
- 24 static letters (A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y)
- 2 dynamic letters (J, Z) with strict anatomical handshape constraints
- Confusion pair differentiation (A vs S vs E vs T vs M vs N, U vs V vs R vs K, G vs H, P vs Q, D vs L, F vs D)
- Handedness invariance (right vs left hand mirroring)
- Degraded fallback rejection guard
- Language isolation and service switching
"""

from collections import deque
from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.sign_recognition_v2.config import (
    ASL_ALPHABET,
    ASL_ALPHABET_METADATA,
    SUPPORTED_ASL_ALPHABET,
    SignLanguage,
)
from ai.sign_recognition_v2.landmark_extractor import (
    WRIST,
    THUMB_CMC,
    THUMB_MCP,
    THUMB_IP,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    RING_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
    LandmarkExtractorV2,
)
from ai.sign_recognition_v2.hand_detector import HandData, HandDetectionResultV2
from ai.sign_recognition_v2.asl_rules import ASLRuleBasedClassifierV2, align_hand_x
from ai.sign_recognition_v2.gesture_classifier import GestureClassifierV2
from ai.sign_recognition_v2.temporal_recognizer import TemporalRecognizerV2
from ai.sign_recognition_v2.prediction import SignRecognitionServiceV2


class TestASLRecognition(unittest.TestCase):
    """Exhaustive test suite for ASL alphabet fingerspelling recognition."""

    def setUp(self):
        self.extractor = LandmarkExtractorV2()
        self.classifier = ASLRuleBasedClassifierV2(self.extractor)
        self.service = SignRecognitionServiceV2()

    @staticmethod
    def _create_base_hand(wrist=(0.5, 0.7, 0.0), scale=0.15):
        """Creates a standardized anatomical right hand with curled fingers."""
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[WRIST] = wrist
        pts[INDEX_MCP] = [wrist[0] - 0.03, wrist[1] - scale * 0.9, 0]
        pts[MIDDLE_MCP] = [wrist[0] - 0.00, wrist[1] - scale * 1.0, 0]
        pts[RING_MCP] = [wrist[0] + 0.025, wrist[1] - scale * 0.9, 0]
        pts[PINKY_MCP] = [wrist[0] + 0.05, wrist[1] - scale * 0.8, 0]
        pts[THUMB_CMC] = [wrist[0] - 0.04, wrist[1] - scale * 0.3, 0]
        pts[THUMB_MCP] = [wrist[0] - 0.06, wrist[1] - scale * 0.5, 0]
        pts[THUMB_IP] = [wrist[0] - 0.03, wrist[1] - scale * 0.55, 0]
        pts[THUMB_TIP] = [wrist[0] + 0.01, wrist[1] - scale * 0.60, 0]

        for bm, p, d, t in [
            (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]:
            m = pts[bm]
            pts[p] = [m[0], m[1] - scale * 0.15, 0]
            pts[d] = [m[0], m[1] - scale * 0.05, 0]
            pts[t] = [m[0], m[1] + scale * 0.10, 0]
        return pts

    @staticmethod
    def _set_finger(pts, bm, p, d, t, ext=True, scale=0.15, dx=0.0):
        """Helper to extend or curl a specific finger."""
        m = pts[bm]
        if ext:
            pts[p] = [m[0] + dx * 0.3, m[1] - scale * 0.4, 0]
            pts[d] = [m[0] + dx * 0.7, m[1] - scale * 0.7, 0]
            pts[t] = [m[0] + dx * 1.0, m[1] - scale * 1.0, 0]
        else:
            pts[p] = [m[0], m[1] - scale * 0.15, 0]
            pts[d] = [m[0], m[1] - scale * 0.05, 0]
            pts[t] = [m[0], m[1] + scale * 0.10, 0]

    def _build_asl_hand(self, letter: str) -> np.ndarray:
        """Constructs synthetic landmark array for any of the 24 static ASL letters."""
        h = self._create_base_hand()

        if letter == "A":
            h[THUMB_TIP] = [h[MIDDLE_MCP][0] - 0.05, h[INDEX_MCP][1] + 0.02, 0]

        elif letter == "B":
            for bm, p, d, t in [
                (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
                (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
                (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
                (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
            ]:
                self._set_finger(h, bm, p, d, t, ext=True)

        elif letter == "C":
            h[INDEX_PIP] = [0.45, 0.50, 0]
            h[INDEX_TIP] = [0.45, 0.54, 0]
            h[MIDDLE_PIP] = [0.48, 0.50, 0]
            h[MIDDLE_TIP] = [0.48, 0.54, 0]
            h[THUMB_TIP] = [0.45, 0.62, 0]

        elif letter == "D":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True)
            h[THUMB_TIP] = h[MIDDLE_TIP].copy()

        elif letter == "E":
            h[THUMB_TIP] = [h[MIDDLE_MCP][0], h[INDEX_MCP][1] + 0.06, 0]

        elif letter == "F":
            for bm, p, d, t in [
                (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
                (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
                (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
            ]:
                self._set_finger(h, bm, p, d, t, ext=True)
            h[THUMB_TIP] = h[INDEX_TIP].copy()

        elif letter == "G":
            h[WRIST] = [0.4, 0.6, 0]
            h[MIDDLE_MCP] = [0.55, 0.6, 0]
            h[INDEX_MCP] = [0.55, 0.57, 0]
            h[RING_MCP] = [0.55, 0.63, 0]
            h[PINKY_MCP] = [0.55, 0.66, 0]
            h[INDEX_PIP] = [0.62, 0.57, 0]
            h[INDEX_TIP] = [0.70, 0.57, 0]
            h[THUMB_MCP] = [0.48, 0.62, 0]
            h[THUMB_TIP] = [0.60, 0.60, 0]
            h[MIDDLE_TIP] = [0.53, 0.60, 0]
            h[RING_TIP] = [0.53, 0.63, 0]
            h[PINKY_TIP] = [0.53, 0.66, 0]

        elif letter == "H":
            h[WRIST] = [0.4, 0.6, 0]
            h[MIDDLE_MCP] = [0.55, 0.6, 0]
            h[INDEX_MCP] = [0.55, 0.57, 0]
            h[RING_MCP] = [0.55, 0.63, 0]
            h[PINKY_MCP] = [0.55, 0.66, 0]
            h[INDEX_PIP] = [0.62, 0.57, 0]
            h[INDEX_TIP] = [0.70, 0.57, 0]
            h[MIDDLE_PIP] = [0.62, 0.60, 0]
            h[MIDDLE_TIP] = [0.70, 0.60, 0]
            h[RING_TIP] = [0.53, 0.63, 0]
            h[PINKY_TIP] = [0.53, 0.66, 0]
            h[THUMB_MCP] = [0.48, 0.62, 0]
            h[THUMB_TIP] = [0.52, 0.62, 0]

        elif letter == "I":
            self._set_finger(h, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, ext=True)

        elif letter == "K":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True, dx=-0.01)
            self._set_finger(h, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, ext=True, dx=0.02)
            h[THUMB_TIP] = h[MIDDLE_PIP].copy()

        elif letter == "L":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True)
            h[THUMB_TIP] = [h[WRIST][0] - 0.12, h[THUMB_MCP][1], 0]

        elif letter == "M":
            h[THUMB_TIP] = [h[MIDDLE_MCP][0] + 0.038, h[INDEX_MCP][1] - 0.02, 0]

        elif letter == "N":
            h[THUMB_TIP] = [h[MIDDLE_MCP][0] + 0.012, h[INDEX_MCP][1] - 0.02, 0]

        elif letter == "O":
            h[INDEX_PIP] = [0.47, 0.53, 0]
            h[INDEX_TIP] = [0.47, 0.58, 0]
            h[MIDDLE_PIP] = [0.48, 0.53, 0]
            h[MIDDLE_TIP] = [0.48, 0.58, 0]
            h[RING_PIP] = [0.49, 0.53, 0]
            h[RING_TIP] = [0.49, 0.58, 0]
            h[PINKY_PIP] = [0.50, 0.53, 0]
            h[PINKY_TIP] = [0.50, 0.58, 0]
            h[THUMB_TIP] = [0.47, 0.58, 0]

        elif letter == "P":
            h[WRIST] = [0.5, 0.4, 0]
            h[INDEX_MCP] = [0.47, 0.55, 0]
            h[MIDDLE_MCP] = [0.50, 0.55, 0]
            h[RING_MCP] = [0.53, 0.53, 0]
            h[PINKY_MCP] = [0.55, 0.50, 0]
            h[INDEX_PIP] = [0.47, 0.62, 0]
            h[INDEX_TIP] = [0.47, 0.70, 0]
            h[MIDDLE_PIP] = [0.50, 0.62, 0]
            h[MIDDLE_TIP] = [0.50, 0.70, 0]
            h[RING_TIP] = [0.53, 0.55, 0]
            h[PINKY_TIP] = [0.55, 0.52, 0]
            h[THUMB_MCP] = [0.45, 0.48, 0]
            h[THUMB_TIP] = [0.48, 0.62, 0]

        elif letter == "Q":
            h[WRIST] = [0.5, 0.4, 0]
            h[INDEX_MCP] = [0.47, 0.55, 0]
            h[MIDDLE_MCP] = [0.50, 0.55, 0]
            h[RING_MCP] = [0.53, 0.53, 0]
            h[PINKY_MCP] = [0.55, 0.50, 0]
            h[INDEX_PIP] = [0.47, 0.62, 0]
            h[INDEX_TIP] = [0.47, 0.70, 0]
            h[MIDDLE_PIP] = [0.50, 0.56, 0]
            h[MIDDLE_TIP] = [0.50, 0.55, 0]
            h[RING_TIP] = [0.53, 0.53, 0]
            h[PINKY_TIP] = [0.55, 0.50, 0]
            h[THUMB_MCP] = [0.45, 0.48, 0]
            h[THUMB_TIP] = [0.46, 0.65, 0]

        elif letter == "R":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True, dx=0.035)
            self._set_finger(h, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, ext=True, dx=-0.02)

        elif letter == "S":
            h[THUMB_TIP] = h[MIDDLE_PIP].copy()

        elif letter == "T":
            h[THUMB_TIP] = [h[MIDDLE_MCP][0] - 0.015, h[INDEX_MCP][1] - 0.02, 0]

        elif letter == "U":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True, dx=0.01)
            self._set_finger(h, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, ext=True, dx=-0.01)

        elif letter == "V":
            self._set_finger(h, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, ext=True, dx=-0.02)
            self._set_finger(h, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, ext=True, dx=0.02)

        elif letter == "W":
            for bm, p, d, t, dx in [
                (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, -0.02),
                (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 0.0),
                (RING_MCP, RING_PIP, RING_DIP, RING_TIP, 0.02),
            ]:
                self._set_finger(h, bm, p, d, t, ext=True, dx=dx)

        elif letter == "X":
            h[INDEX_PIP] = [h[INDEX_MCP][0], h[INDEX_MCP][1] - 0.08, 0]
            h[INDEX_TIP] = [h[INDEX_MCP][0], h[INDEX_MCP][1] - 0.02, 0]

        elif letter == "Y":
            h[THUMB_TIP] = [h[WRIST][0] - 0.12, h[THUMB_MCP][1], 0]
            self._set_finger(h, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, ext=True, dx=0.03)

        return h

    # -------------------------------------------------------------------------
    # 1. Complete Coverage of all 24 Static ASL Letters
    # -------------------------------------------------------------------------
    def test_01_all_24_static_asl_letters(self):
        """Verifies accurate classification of all 24 static ASL letters."""
        static_letters = [ltr for ltr in ASL_ALPHABET if ltr not in ("J", "Z")]
        self.assertEqual(len(static_letters), 24)

        for ltr in static_letters:
            pts = self._build_asl_hand(ltr)
            pred, conf, meta = self.classifier.classify_geometry(pts)
            self.assertEqual(
                pred,
                ltr,
                f"Failed recognition for ASL letter '{ltr}': got '{pred}' (conf={conf:.2f}, meta={meta})",
            )
            self.assertGreaterEqual(conf, 0.90)

    # -------------------------------------------------------------------------
    # 2. Dynamic Letters (J, Z)
    # -------------------------------------------------------------------------
    def test_02_dynamic_letters_j_and_z(self):
        """Verifies dynamic J and Z strokes with strict anatomical constraints in ASL mode."""
        temporal = TemporalRecognizerV2(buffer_size=25)

        # Dynamic Z: index finger zig-zag stroke
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

        # ASL Z valid when index is extended and pinky is curled
        letter, conf, meta = temporal.evaluate_dynamic_gesture(
            finger_states={"index": True, "pinky": False}, language="ASL"
        )
        self.assertEqual(letter, "Z")
        self.assertGreaterEqual(conf, 0.70)
        self.assertEqual(meta["language"], "ASL")

        # In ASL, Z must be rejected if pinky is extended
        temporal.reset()
        temporal.last_trigger_time = 0.0
        for x in np.linspace(0.2, 0.6, 15):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[INDEX_TIP] = [x, 0.2, 0.0]
            temporal.add_frame(pts)

        letter_bad, _, _ = temporal.evaluate_dynamic_gesture(
            finger_states={"index": True, "pinky": True}, language="ASL"
        )
        self.assertIsNone(letter_bad)

        # Dynamic J: pinky downward hook stroke
        temporal.reset()
        temporal.last_trigger_time = 0.0
        for y in np.linspace(0.2, 0.55, 10):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[PINKY_TIP] = [0.5, y, 0.0]
            temporal.add_frame(pts)

        for y in np.linspace(0.55, 0.48, 4):
            pts = np.zeros((21, 3), dtype=np.float32)
            pts[PINKY_TIP] = [0.45, y, 0.0]
            temporal.add_frame(pts)

        # ASL J valid when pinky is extended and index is curled
        letter_j, conf_j, meta_j = temporal.evaluate_dynamic_gesture(
            finger_states={"pinky": True, "index": False}, language="ASL"
        )
        self.assertEqual(letter_j, "J")
        self.assertGreaterEqual(conf_j, 0.70)
        self.assertEqual(meta_j["language"], "ASL")

        # In ASL, J must be rejected if index is extended
        letter_j_bad, _, _ = temporal.evaluate_dynamic_gesture(
            finger_states={"pinky": True, "index": True}, language="ASL"
        )
        self.assertIsNone(letter_j_bad)

    # -------------------------------------------------------------------------
    # 3. Fist Family Differentiation
    # -------------------------------------------------------------------------
    def test_03_fist_family_confusion_differentiation(self):
        """Verifies mutual exclusion and precision across fist letters (A, S, E, T, M, N)."""
        fist_letters = ["A", "S", "E", "T", "M", "N"]
        predictions = {}
        for ltr in fist_letters:
            pts = self._build_asl_hand(ltr)
            pred, conf, _ = self.classifier.classify_geometry(pts)
            predictions[ltr] = pred
            self.assertEqual(pred, ltr, f"Fist confusion: expected '{ltr}', got '{pred}'")
            self.assertGreaterEqual(conf, 0.90)

        # Verify all predictions are unique
        self.assertEqual(len(set(predictions.values())), len(fist_letters))

    # -------------------------------------------------------------------------
    # 4. Upright Two-Finger Family Differentiation
    # -------------------------------------------------------------------------
    def test_04_upright_two_finger_confusion_differentiation(self):
        """Verifies mutual exclusion between U (touching), V (spread), R (crossed), K (thumb at PIP)."""
        letters = ["U", "V", "R", "K"]
        for ltr in letters:
            pts = self._build_asl_hand(ltr)
            pred, conf, _ = self.classifier.classify_geometry(pts)
            self.assertEqual(pred, ltr, f"Two-finger confusion: expected '{ltr}', got '{pred}'")
            self.assertGreaterEqual(conf, 0.90)

    # -------------------------------------------------------------------------
    # 5. Horizontal and Downward Families
    # -------------------------------------------------------------------------
    def test_05_horizontal_and_downward_family(self):
        """Verifies horizontal (G, H) and downward (P, Q) letter recognition."""
        for ltr in ["G", "H", "P", "Q"]:
            pts = self._build_asl_hand(ltr)
            pred, conf, meta = self.classifier.classify_geometry(pts)
            self.assertEqual(pred, ltr, f"Orientation confusion: expected '{ltr}', got '{pred}'")
            self.assertGreaterEqual(conf, 0.90)
            if ltr in ("G", "H"):
                self.assertTrue(meta["is_horizontal"])
            else:
                self.assertTrue(meta["is_downward"])

    # -------------------------------------------------------------------------
    # 6. Extended and Loop Shapes
    # -------------------------------------------------------------------------
    def test_06_extended_and_loop_shapes(self):
        """Verifies D vs L, F vs D, Y vs I, and B vs W differentiation."""
        # D vs L
        d_pts = self._build_asl_hand("D")
        l_pts = self._build_asl_hand("L")
        self.assertEqual(self.classifier.classify_geometry(d_pts)[0], "D")
        self.assertEqual(self.classifier.classify_geometry(l_pts)[0], "L")

        # F vs D
        f_pts = self._build_asl_hand("F")
        self.assertEqual(self.classifier.classify_geometry(f_pts)[0], "F")

        # Y vs I
        y_pts = self._build_asl_hand("Y")
        i_pts = self._build_asl_hand("I")
        self.assertEqual(self.classifier.classify_geometry(y_pts)[0], "Y")
        self.assertEqual(self.classifier.classify_geometry(i_pts)[0], "I")

        # B vs W
        b_pts = self._build_asl_hand("B")
        w_pts = self._build_asl_hand("W")
        self.assertEqual(self.classifier.classify_geometry(b_pts)[0], "B")
        self.assertEqual(self.classifier.classify_geometry(w_pts)[0], "W")

    # -------------------------------------------------------------------------
    # 7. Handedness Invariance
    # -------------------------------------------------------------------------
    def test_07_handedness_invariance(self):
        """Verifies that right-handed and mirrored left-handed inputs yield identical predictions."""
        for ltr in ["A", "B", "L", "Y", "V", "W"]:
            pts_r = self._build_asl_hand(ltr)
            pts_l = pts_r.copy()
            # Mirror horizontally around center x=0.5
            pts_l[:, 0] = 1.0 - pts_l[:, 0]

            pred_r, conf_r, _ = self.classifier.classify_geometry(pts_r)
            pred_l, conf_l, _ = self.classifier.classify_geometry(pts_l)

            self.assertEqual(pred_r, ltr)
            self.assertEqual(pred_l, ltr, f"Handedness invariance failed for letter '{ltr}'")
            self.assertAlmostEqual(conf_r, conf_l, places=1)

    # -------------------------------------------------------------------------
    # 8. Degraded Detection Guard
    # -------------------------------------------------------------------------
    def test_08_degraded_fallback_rejection(self):
        """Verifies that degraded detections (OpenCV fallback) are rejected immediately."""
        pts = self._build_asl_hand("A")
        hand = HandData(
            landmarks=[tuple(p) for p in pts],
            handedness="Right",
            confidence=0.95,
            bbox=(0, 0, 100, 100),
        )
        detection = HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            backend="opencv_color_fallback",
            is_degraded=True,
        )

        pred, conf, meta = self.classifier.classify_detection(detection)
        self.assertIsNone(pred)
        self.assertEqual(conf, 0.0)
        self.assertEqual(meta["status"], "degraded_detection_rejected")

    # -------------------------------------------------------------------------
    # 9. Language Isolation
    # -------------------------------------------------------------------------
    def test_09_language_isolation(self):
        """Verifies that ASL and ISL rule engines remain completely isolated."""
        gc = GestureClassifierV2()

        # Set to ASL
        gc.set_language("ASL")
        self.assertEqual(gc.language, SignLanguage.ASL)

        # In ASL mode, single hand 'A' yields ASL classification
        pts_a = self._build_asl_hand("A")
        hand = HandData(landmarks=[tuple(p) for p in pts_a], handedness="Right", confidence=0.95, bbox=(0, 0, 100, 100))
        det = HandDetectionResultV2(detected=True, num_hands=1, hands=[hand], backend="mediapipe_tasks", is_degraded=False)
        pred, conf, is_rec, meta = gc.classify_detection(det)
        self.assertEqual(pred, "A")
        self.assertEqual(meta["language"], "ASL")

        # Switch to ISL
        gc.set_language("ISL")
        self.assertEqual(gc.language, SignLanguage.ISL)
        pred_isl, _, _, meta_isl = gc.classify_detection(det)
        self.assertEqual(meta_isl["language"], "ISL")

    # -------------------------------------------------------------------------
    # 10. Service Integration & Supported Gestures Metadata
    # -------------------------------------------------------------------------
    def test_10_service_language_switching(self):
        """Verifies SignRecognitionServiceV2 language switching and metadata contracts."""
        # Initial language is ISL
        self.assertEqual(self.service.current_language, "ISL")

        # Switch to ASL
        self.service.set_language("ASL")
        self.assertEqual(self.service.current_language, "ASL")

        # Supported ASL alphabet contract
        gestures = self.service.get_supported_gestures("ASL")
        self.assertEqual(gestures["language"], "ASL")
        self.assertEqual(gestures["count"], 26)
        self.assertEqual(len(gestures["gestures"]), 26)
        self.assertIn("counts_by_status", gestures)
        self.assertEqual(gestures["counts_by_status"]["experimental"], 24)
        self.assertEqual(gestures["counts_by_status"]["experimental_dynamic"], 2)
        self.assertEqual(gestures["counts_by_status"]["unsupported"], 0)
        self.assertIn("disclaimer", gestures)

        # Switch back to ISL
        self.service.set_language("ISL")
        self.assertEqual(self.service.current_language, "ISL")
        gestures_isl = self.service.get_supported_gestures("ISL")
        self.assertEqual(gestures_isl["language"], "ISL")
        self.assertEqual(gestures_isl["count"], 26)


if __name__ == "__main__":
    unittest.main()
