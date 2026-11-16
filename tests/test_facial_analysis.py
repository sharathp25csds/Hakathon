"""SignBridge AI - Unit Tests for Multi-Person Facial Expression Analysis.
Verifies multi-person face detection, real-time tracking, 9 expression categories,
uncertain/low confidence handling, and contract compliance.
"""

from pathlib import Path
import unittest
import cv2
import numpy as np

from ai.facial_analysis import (
    FACIAL_ANALYSIS_DISCLAIMER,
    SUPPORTED_EXPRESSIONS,
    FacialAnalysisService,
    FaceTracker,
    TrackedFace,
    classify_expression_from_blendshapes,
    compute_iou,
)


class TestFacialAnalysisService(unittest.TestCase):
    """Comprehensive test suite for Facial Analysis and Multi-Person Tracking."""

    @classmethod
    def setUpClass(cls):
        cls.service = FacialAnalysisService()
        cls.sample_face_path = Path(
            ".venv/lib/python3.14/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"
        )
        cls.has_sample_image = cls.sample_face_path.exists()
        if cls.has_sample_image:
            cls.single_face_img = cv2.imread(str(cls.sample_face_path))
            h, w = cls.single_face_img.shape[:2]
            # Create dual-face image by horizontally placing original and flipped face
            cls.dual_face_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
            cls.dual_face_img[:, :w] = cls.single_face_img
            cls.dual_face_img[:, w:] = cv2.flip(cls.single_face_img, 1)

    def setUp(self):
        self.service.set_active(True)
        self.service.reset_tracker()

    def test_supported_expressions_count_and_content(self):
        """Verifies exactly 9 supported facial expression categories are declared."""
        self.assertEqual(len(SUPPORTED_EXPRESSIONS), 9)
        expected = [
            "neutral",
            "happy",
            "surprised",
            "focused",
            "sad",
            "angry",
            "fearful",
            "disgusted",
            "confused",
        ]
        self.assertEqual(SUPPORTED_EXPRESSIONS, expected)

    def test_invalid_and_disabled_inputs(self):
        """Verifies handling of None, empty frames, and disabled state."""
        # None frame
        res_none = self.service.analyze_face(None)
        self.assertFalse(res_none["success"])
        self.assertFalse(res_none["face_detected"])
        self.assertEqual(res_none["num_faces"], 0)
        self.assertEqual(res_none["faces"], [])
        self.assertEqual(res_none["status"], "invalid_frame")
        self.assertIn("disclaimer", res_none)

        # Empty array
        res_empty = self.service.analyze_face(np.array([], dtype=np.uint8))
        self.assertFalse(res_empty["success"])
        self.assertEqual(res_empty["status"], "invalid_frame")

        # Disabled module
        self.service.set_active(False)
        self.assertEqual(self.service.status, "module_disabled")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res_dis = self.service.analyze_face(blank)
        self.assertFalse(res_dis["success"])
        self.assertEqual(res_dis["status"], "module_disabled")

    def test_blank_frame_no_faces(self):
        """Verifies zero-face detection on empty/blank background."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.service.analyze_face(blank)
        self.assertFalse(res["success"])
        self.assertFalse(res["face_detected"])
        self.assertEqual(res["num_faces"], 0)
        self.assertEqual(res["faces"], [])
        self.assertIsNone(res["expression"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertIsNone(res["bbox"])
        self.assertEqual(res["status"], "no_face_detected")

    def test_single_face_detection_and_contract(self):
        """Verifies detection of a single human face with tracking ID and bounding box."""
        if not self.has_sample_image:
            self.skipTest("Sample face image not available")

        res = self.service.analyze_face(self.single_face_img)
        self.assertTrue(res["success"])
        self.assertTrue(res["face_detected"])
        self.assertEqual(res["num_faces"], 1)
        self.assertEqual(len(res["faces"]), 1)

        face0 = res["faces"][0]
        self.assertEqual(face0["track_id"], 1)
        self.assertEqual(len(face0["bbox"]), 4)
        self.assertGreater(face0["bbox"][2], 30)  # width
        self.assertGreater(face0["bbox"][3], 30)  # height
        self.assertIn(face0["expression"], SUPPORTED_EXPRESSIONS + ["uncertain"])
        self.assertGreater(face0["confidence"], 0.0)

        # Backward compatible top-level fields
        self.assertEqual(res["expression"], face0["expression"])
        self.assertEqual(res["confidence"], face0["confidence"])
        self.assertEqual(res["bbox"], face0["bbox"])
        self.assertEqual(res["status"], "face_detected")

    def test_multi_person_simultaneous_detection(self):
        """Verifies detection of multiple simultaneous faces in one image frame."""
        if not self.has_sample_image:
            self.skipTest("Sample face image not available")

        res = self.service.analyze_face(self.dual_face_img)
        self.assertTrue(res["success"])
        self.assertTrue(res["face_detected"])
        self.assertGreaterEqual(res["num_faces"], 2)
        self.assertGreaterEqual(len(res["faces"]), 2)

        # Verify distinct tracking IDs
        track_ids = [f["track_id"] for f in res["faces"]]
        self.assertEqual(len(track_ids), len(set(track_ids)), "Tracking IDs must be unique")

        # Verify distinct non-overlapping bounding boxes
        box0 = res["faces"][0]["bbox"]
        box1 = res["faces"][1]["bbox"]
        iou = compute_iou(box0, box1)
        self.assertLess(iou, 0.10, "Multiple persons must occupy distinct spatial regions")

        self.assertEqual(res["status"], "faces_detected")

    def test_tracker_persistence_across_frames(self):
        """Verifies that moving faces retain their persistent tracking IDs across frames."""
        tracker = FaceTracker()
        frame_shape = (480, 640)

        # Frame 1: Person 1 at (100, 100), Person 2 at (400, 100)
        f1_detections = [
            {"bbox": [100, 100, 80, 90], "expression": "happy", "confidence": 0.85},
            {"bbox": [400, 100, 80, 90], "expression": "neutral", "confidence": 0.70},
        ]
        f1_res = tracker.update(f1_detections, frame_shape)
        self.assertEqual([r["track_id"] for r in f1_res], [1, 2])

        # Frame 2: Both persons shift slightly (5-10 pixels)
        f2_detections = [
            {"bbox": [106, 104, 80, 90], "expression": "happy", "confidence": 0.86},
            {"bbox": [404, 102, 80, 90], "expression": "neutral", "confidence": 0.72},
        ]
        f2_res = tracker.update(f2_detections, frame_shape)
        # IDs must be preserved without swap
        self.assertEqual([r["track_id"] for r in f2_res], [1, 2])
        self.assertEqual(f2_res[0]["track_id"], 1)
        self.assertEqual(f2_res[1]["track_id"], 2)

    def test_tracker_entering_and_leaving_dynamics(self):
        """Verifies that exiting persons are deregistered and entering persons get unique new IDs."""
        tracker = FaceTracker(max_missed_frames=2)
        frame_shape = (480, 640)

        # Frame 1: Person 1 and Person 2
        f1 = [
            {"bbox": [100, 100, 80, 90], "expression": "happy", "confidence": 0.85},
            {"bbox": [400, 100, 80, 90], "expression": "neutral", "confidence": 0.70},
        ]
        tracker.update(f1, frame_shape)

        # Frame 2: Person 1 leaves, Person 2 remains
        f2 = [
            {"bbox": [402, 102, 80, 90], "expression": "neutral", "confidence": 0.70},
        ]
        res2 = tracker.update(f2, frame_shape)
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0]["track_id"], 2)

        # Frame 3 & 4: Person 1 remains absent (exceeds max_missed_frames=2)
        tracker.update(f2, frame_shape)
        tracker.update(f2, frame_shape)
        self.assertNotIn(1, tracker.tracks, "Person 1 should be purged after timeout")

        # Frame 5: Person 3 enters at new position (250, 300)
        f5 = [
            {"bbox": [402, 102, 80, 90], "expression": "neutral", "confidence": 0.70},
            {"bbox": [250, 300, 80, 90], "expression": "focused", "confidence": 0.75},
        ]
        res5 = tracker.update(f5, frame_shape)
        assigned_ids = [r["track_id"] for r in res5]
        self.assertIn(2, assigned_ids)
        self.assertIn(3, assigned_ids, "New entering person must receive unique incremented ID (3)")
        self.assertNotIn(1, assigned_ids, "Deregistered ID must not be prematurely duplicated")

    def test_all_nine_expression_classifications(self):
        """Verifies that all 9 emotional categories can be classified from blendshapes."""
        cases = {
            "happy": {"mouthSmileLeft": 0.75, "mouthSmileRight": 0.75, "cheekSquintLeft": 0.50},
            "surprised": {"jawOpen": 0.65, "browInnerUp": 0.55, "eyeWideLeft": 0.60, "eyeWideRight": 0.60},
            "sad": {"mouthFrownLeft": 0.60, "mouthFrownRight": 0.60, "browInnerUp": 0.45},
            "angry": {"browDownLeft": 0.70, "browDownRight": 0.70, "mouthPressLeft": 0.50},
            "fearful": {"eyeWideLeft": 0.70, "eyeWideRight": 0.70, "browInnerUp": 0.60, "mouthStretchLeft": 0.50},
            "disgusted": {"noseSneerLeft": 0.65, "noseSneerRight": 0.65, "mouthUpperUpLeft": 0.55},
            "confused": {"browDownLeft": 0.65, "browDownRight": 0.05, "browOuterUpRight": 0.55},
            "focused": {"eyeSquintLeft": 0.60, "eyeSquintRight": 0.60, "browDownLeft": 0.20, "browDownRight": 0.20},
            "neutral": {},
        }

        for expected_expr, blend in cases.items():
            expr, conf, status, summary = classify_expression_from_blendshapes(blend)
            self.assertEqual(
                expr,
                expected_expr,
                f"Failed for {expected_expr}: got {expr} (conf={conf}, status={status})",
            )
            self.assertGreaterEqual(conf, 0.40)
            self.assertEqual(status, "face_detected")

    def test_uncertain_reported_for_ambiguous_and_weak_inputs(self):
        """Verifies that ambiguous or low-confidence expressions are reported as 'uncertain'."""
        # 1. Weak activation: smile and frown are weak (~0.24 score)
        weak_blend = {"mouthSmileLeft": 0.30, "mouthSmileRight": 0.30}
        expr, conf, status, _ = classify_expression_from_blendshapes(weak_blend)
        self.assertEqual(expr, "uncertain")
        self.assertEqual(status, "uncertain_expression")

        # 2. Conflicting/tied activations: equal smile and frown
        tied_blend = {
            "mouthSmileLeft": 0.45,
            "mouthSmileRight": 0.45,
            "mouthFrownLeft": 0.43,
            "mouthFrownRight": 0.43,
        }
        expr_tied, conf_tied, status_tied, _ = classify_expression_from_blendshapes(tied_blend)
        self.assertEqual(expr_tied, "uncertain")
        self.assertEqual(status_tied, "uncertain_expression")


if __name__ == "__main__":
    unittest.main()
