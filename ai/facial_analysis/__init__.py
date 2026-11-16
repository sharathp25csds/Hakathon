"""SignBridge AI - Multi-Person Facial Expression Analysis Module.
Provides real-time multi-person face detection, tracking, and contextual facial expression analysis.

Expression categories (9 supported):
    - neutral, happy, surprised, focused, sad, angry, fearful, disgusted, confused
    - Ambiguous or low-confidence detections are explicitly reported as 'uncertain'.

Facial expression analysis is strictly supplementary communication context and is never definitive proof of emotion.
Uses Google MediaPipe Face Landmarker (Apache 2.0 License).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import cv2
import numpy as np

FACIAL_ANALYSIS_DISCLAIMER = (
    "Facial expression analysis provides supplementary communication context only "
    "and is not definitive evidence of a person's emotional state."
)

SUPPORTED_EXPRESSIONS: List[str] = [
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

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = float(area_a + area_b - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class TrackedFace:
    """Represents an individual tracked face with persistent ID and historical state."""

    def __init__(self, track_id: int, bbox: List[int], centroid: Tuple[float, float]):
        self.track_id: int = track_id
        self.bbox: List[int] = list(bbox)
        self.centroid: Tuple[float, float] = centroid
        self.missed_frames: int = 0
        self.age_frames: int = 1
        self.last_expression: str = "neutral"
        self.last_confidence: float = 0.70
        self.history_expressions: List[str] = []
        self.history_confidences: List[float] = []

    def update(
        self,
        bbox: List[int],
        centroid: Tuple[float, float],
        expression: str,
        confidence: float,
    ) -> None:
        self.bbox = list(bbox)
        self.centroid = centroid
        self.missed_frames = 0
        self.age_frames += 1
        self.last_expression = expression
        self.last_confidence = confidence

        self.history_expressions.append(expression)
        self.history_confidences.append(confidence)
        if len(self.history_expressions) > 10:
            self.history_expressions.pop(0)
            self.history_confidences.pop(0)


class FaceTracker:
    """Associates face detections across consecutive frames to maintain consistent tracking IDs."""

    def __init__(
        self,
        max_missed_frames: int = 15,
        iou_threshold: float = 0.15,
        max_distance_ratio: float = 0.25,
    ):
        self._next_id: int = 1
        self.tracks: Dict[int, TrackedFace] = {}
        self.max_missed_frames: int = max_missed_frames
        self.iou_threshold: float = iou_threshold
        self.max_distance_ratio: float = max_distance_ratio

    def reset(self) -> None:
        """Resets all active tracks and resets ID counter."""
        self._next_id = 1
        self.tracks.clear()

    def update(
        self,
        detected_faces: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Matches detections to existing tracks and assigns unique IDs."""
        h, w = frame_shape
        frame_diag = math.hypot(w, h)

        matched_tracks = set()
        matched_dets = set()

        candidates = []
        for det_idx, det in enumerate(detected_faces):
            bx, by, bw, bh = det["bbox"]
            det_cx = bx + bw / 2.0
            det_cy = by + bh / 2.0
            det_centroid = (det_cx, det_cy)
            det_size = (bw + bh) / 2.0

            for track_id, track in self.tracks.items():
                iou = compute_iou(track.bbox, det["bbox"])
                dist = math.hypot(
                    track.centroid[0] - det_cx, track.centroid[1] - det_cy
                )
                norm_dist = dist / max(frame_diag, 1.0)
                track_size = (track.bbox[2] + track.bbox[3]) / 2.0
                avg_size = max(1.0, (det_size + track_size) / 2.0)

                # Match if overlapping bounding boxes OR very close proximity for fast movements
                is_match = False
                match_score = 0.0
                if iou >= self.iou_threshold:
                    is_match = True
                    match_score = 2.0 + iou
                elif dist <= avg_size * 0.75 and norm_dist <= self.max_distance_ratio:
                    is_match = True
                    match_score = 1.0 - (dist / (avg_size * 0.75))

                if is_match:
                    candidates.append((match_score, track_id, det_idx, det_centroid))

        # Greedy match on highest similarity score
        candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_results = []
        for score, track_id, det_idx, det_centroid in candidates:
            if track_id in matched_tracks or det_idx in matched_dets:
                continue

            matched_tracks.add(track_id)
            matched_dets.add(det_idx)
            det = detected_faces[det_idx]

            self.tracks[track_id].update(
                det["bbox"], det_centroid, det["expression"], det["confidence"]
            )
            assigned_results.append({
                "track_id": track_id,
                "face_id": track_id,
                "bbox": det["bbox"],
                "expression": det["expression"],
                "confidence": det["confidence"],
                "status": det.get("status", "face_detected"),
                "blendshapes_summary": det.get("blendshapes_summary", {}),
            })

        # Register unmatched detections as new tracks
        for det_idx, det in enumerate(detected_faces):
            if det_idx not in matched_dets:
                new_id = self._next_id
                self._next_id += 1

                bx, by, bw, bh = det["bbox"]
                det_cx = bx + bw / 2.0
                det_cy = by + bh / 2.0
                new_track = TrackedFace(new_id, det["bbox"], (det_cx, det_cy))
                new_track.update(
                    det["bbox"],
                    (det_cx, det_cy),
                    det["expression"],
                    det["confidence"],
                )
                self.tracks[new_id] = new_track

                assigned_results.append({
                    "track_id": new_id,
                    "face_id": new_id,
                    "bbox": det["bbox"],
                    "expression": det["expression"],
                    "confidence": det["confidence"],
                    "status": det.get("status", "face_detected"),
                    "blendshapes_summary": det.get("blendshapes_summary", {}),
                })

        # Age and purge tracks that left the frame
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        # Sort by track ID for deterministic output
        assigned_results.sort(key=lambda x: x["track_id"])
        return assigned_results


def classify_expression_from_blendshapes(
    blendshapes: Dict[str, float],
) -> Tuple[str, float, str, Dict[str, float]]:
    """Analyzes facial blendshapes (FACS Action Units) to classify expression into 9 categories or 'uncertain'.
    
    Returns:
        (expression, confidence, status, summary_dict)
    """
    b = blendshapes

    # Action Unit Components
    smile = (b.get("mouthSmileLeft", 0.0) + b.get("mouthSmileRight", 0.0)) / 2.0
    cheek_squint = (b.get("cheekSquintLeft", 0.0) + b.get("cheekSquintRight", 0.0)) / 2.0
    jaw_open = b.get("jawOpen", 0.0)
    brow_inner_up = b.get("browInnerUp", 0.0)
    brow_outer_up = (b.get("browOuterUpLeft", 0.0) + b.get("browOuterUpRight", 0.0)) / 2.0
    eye_wide = (b.get("eyeWideLeft", 0.0) + b.get("eyeWideRight", 0.0)) / 2.0
    eye_squint = (b.get("eyeSquintLeft", 0.0) + b.get("eyeSquintRight", 0.0)) / 2.0
    mouth_frown = (b.get("mouthFrownLeft", 0.0) + b.get("mouthFrownRight", 0.0)) / 2.0
    mouth_shrug = b.get("mouthShrugLower", 0.0)
    brow_down = (b.get("browDownLeft", 0.0) + b.get("browDownRight", 0.0)) / 2.0
    nose_sneer = (b.get("noseSneerLeft", 0.0) + b.get("noseSneerRight", 0.0)) / 2.0
    mouth_press = (b.get("mouthPressLeft", 0.0) + b.get("mouthPressRight", 0.0)) / 2.0
    mouth_stretch = (b.get("mouthStretchLeft", 0.0) + b.get("mouthStretchRight", 0.0)) / 2.0
    mouth_upper_up = (b.get("mouthUpperUpLeft", 0.0) + b.get("mouthUpperUpRight", 0.0)) / 2.0
    mouth_pucker = b.get("mouthPucker", 0.0)

    # 1. Happy (AU6 + AU12)
    happy_score = 0.70 * smile + 0.30 * cheek_squint

    # 2. Surprised (AU1 + AU2 + AU5 + AU26)
    surprised_score = 0.40 * jaw_open + 0.30 * brow_inner_up + 0.30 * eye_wide

    # 3. Sad (AU1 + AU4 + AU15)
    sad_score = 0.55 * mouth_frown + 0.30 * brow_inner_up + 0.15 * mouth_shrug
    if smile > 0.15:
        sad_score *= max(0.0, 1.0 - (smile - 0.15) * 3.0)

    # 4. Angry (AU4 + AU7 + AU23/24)
    angry_score = 0.50 * brow_down + 0.30 * mouth_press + 0.20 * nose_sneer
    if smile > 0.15:
        angry_score *= max(0.0, 1.0 - (smile - 0.15) * 3.0)

    # 5. Fearful (AU1 + AU2 + AU5 + AU20)
    fearful_score = 0.45 * eye_wide + 0.35 * brow_inner_up + 0.20 * mouth_stretch

    # 6. Disgusted (AU9 + AU10)
    disgusted_score = 0.55 * nose_sneer + 0.45 * mouth_upper_up

    # 7. Confused (Asymmetric brow raise or brow furrow + brow inner up)
    brow_diff = abs(b.get("browDownLeft", 0.0) - b.get("browDownRight", 0.0))
    outer_diff = abs(b.get("browOuterUpLeft", 0.0) - b.get("browOuterUpRight", 0.0))
    brow_combo = min(brow_inner_up, brow_down)
    confused_score = 0.60 * max(brow_diff, outer_diff) + 0.25 * brow_combo + 0.15 * mouth_pucker

    # 8. Focused (Mild brow furrow, eye squint, low mouth motion)
    focused_score = 0.55 * eye_squint + 0.45 * min(brow_down, 0.30)
    if (
        brow_down > 0.35
        or mouth_press > 0.25
        or nose_sneer > 0.25
        or smile > 0.15
        or jaw_open > 0.20
        or mouth_frown > 0.20
    ):
        focused_score *= 0.30

    scores = {
        "happy": happy_score,
        "surprised": surprised_score,
        "sad": sad_score,
        "angry": angry_score,
        "fearful": fearful_score,
        "disgusted": disgusted_score,
        "confused": confused_score,
        "focused": focused_score,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_expr, top_score = sorted_scores[0]
    second_expr, second_score = sorted_scores[1]

    summary = {
        "happy": round(happy_score, 3),
        "surprised": round(surprised_score, 3),
        "sad": round(sad_score, 3),
        "angry": round(angry_score, 3),
        "fearful": round(fearful_score, 3),
        "disgusted": round(disgusted_score, 3),
        "confused": round(confused_score, 3),
        "focused": round(focused_score, 3),
    }

    # Relaxed / Neutral when all active emotional signals are low (< 0.20)
    if top_score < 0.20:
        conf = round(min(0.90, 0.65 + (0.20 - top_score) * 1.2), 2)
        return "neutral", conf, "face_detected", summary

    # Uncertain when score is weak or ambiguous between top two categories
    if top_score < 0.32 or ((top_score - second_score) < 0.05 and top_score < 0.50):
        conf = round(max(0.30, min(0.60, top_score)), 2)
        return "uncertain", conf, "uncertain_expression", summary

    # Standard calibrated confidence (0.50 - 0.95)
    calibrated_conf = round(min(0.95, 0.40 + top_score * 0.90), 2)
    return top_expr, calibrated_conf, "face_detected", summary


class FacialAnalysisService:
    """Multi-person facial expression analyzer with real-time tracking and MediaPipe inference."""

    def __init__(
        self,
        is_active: bool = True,
        model_path: Optional[str] = None,
        max_faces: int = 10,
    ):
        self.is_active: bool = is_active
        self.model_path: Path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.max_faces: int = max_faces
        self.tracker: FaceTracker = FaceTracker()

        self._detector: Optional[Any] = None
        self._init_attempted: bool = False
        self._detector_error: Optional[str] = None

        self._last_expression: Optional[str] = None
        self._last_confidence: float = 0.0

    @property
    def status(self) -> str:
        """Returns the current operational status of the facial module."""
        if not self.is_active:
            return "module_disabled"
        return "ready"

    def set_active(self, active: bool) -> None:
        """Enables or disables facial expression analysis."""
        self.is_active = active
        if not active:
            self.tracker.reset()

    def reset_tracker(self) -> None:
        """Resets tracking history and IDs."""
        self.tracker.reset()

    def _ensure_detector(self) -> Optional[Any]:
        """Initializes the MediaPipe FaceLandmarker once and caches it."""
        if self._detector is not None:
            return self._detector

        if self._init_attempted and self._detector_error is not None:
            return None

        self._init_attempted = True
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not self.model_path.exists():
                self._detector_error = f"Model file not found: {self.model_path}"
                return None

            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=self.max_faces,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
            return self._detector
        except Exception as exc:
            self._detector_error = str(exc)
            return None

    def detect_face_contour_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detection using OpenCV skin segmentation when MediaPipe model is unavailable."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        min_face_area = int(0.02 * w * h)
        max_face_area = int(0.70 * w * h)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_face_area or area > max_face_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(bw, 1)

            if 0.8 <= aspect <= 2.5:
                # Basic gradient analysis in lower region for rough smile
                roi = frame[by : by + bh, bx : bx + bw]
                rh, rw = roi.shape[:2]
                expr = "neutral"
                conf = 0.50
                if rh >= 20 and rw >= 20:
                    mouth = roi[int(rh * 0.65) :, int(rw * 0.20) : int(rw * 0.80)]
                    gray_m = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                    grad_std = float(np.std(cv2.Sobel(gray_m, cv2.CV_64F, 0, 1, ksize=3)))
                    if grad_std > 35.0:
                        expr = "happy"
                        conf = min(0.75, 0.55 + (grad_std - 35.0) / 100.0)
                    elif grad_std > 25.0:
                        expr = "surprised"
                        conf = 0.60

                detected.append({
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "expression": expr,
                    "confidence": round(conf, 2),
                    "status": "face_detected_degraded",
                    "blendshapes_summary": {},
                })

        # Sort largest first
        detected.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        return detected[: self.max_faces]

    def analyze_face(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyzes all faces in frame and returns multi-person detections with tracking IDs.
        
        Contract:
            - success: bool
            - face_detected: bool
            - num_faces: int
            - faces: List[Dict] with track_id, face_id, bbox, expression, confidence, status, blendshapes_summary
            - expression: Optional[str] (primary face)
            - confidence: float (primary face)
            - bbox: Optional[List[int]] (primary face)
            - status: str
            - disclaimer: str
            - supported_expressions: List[str]
        """
        if not self.is_active:
            return {
                "success": False,
                "face_detected": False,
                "num_faces": 0,
                "faces": [],
                "expression": None,
                "confidence": 0.0,
                "bbox": None,
                "status": "module_disabled",
                "disclaimer": FACIAL_ANALYSIS_DISCLAIMER,
                "supported_expressions": SUPPORTED_EXPRESSIONS,
            }

        if frame is None or frame.size == 0:
            return {
                "success": False,
                "face_detected": False,
                "num_faces": 0,
                "faces": [],
                "expression": None,
                "confidence": 0.0,
                "bbox": None,
                "status": "invalid_frame",
                "disclaimer": FACIAL_ANALYSIS_DISCLAIMER,
                "supported_expressions": SUPPORTED_EXPRESSIONS,
            }

        h, w = frame.shape[:2]
        detector = self._ensure_detector()

        raw_detections: List[Dict[str, Any]] = []

        if detector is not None:
            try:
                import mediapipe as mp

                # MediaPipe requires RGB format
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)

                if res.face_landmarks:
                    for i, landmarks in enumerate(res.face_landmarks):
                        # Calculate pixel bounding box from normalized landmarks
                        xs = [lm.x * w for lm in landmarks]
                        ys = [lm.y * h for lm in landmarks]
                        min_x = max(0.0, min(xs))
                        max_x = min(float(w), max(xs))
                        min_y = max(0.0, min(ys))
                        max_y = min(float(h), max(ys))

                        bw = max_x - min_x
                        bh = max_y - min_y

                        # Add comfortable padding around the landmarks
                        pad_x = bw * 0.12
                        pad_y = bh * 0.12
                        bx = int(max(0, min_x - pad_x))
                        by = int(max(0, min_y - pad_y))
                        bw = int(min(w - bx, bw + 2 * pad_x))
                        bh = int(min(h - by, bh + 2 * pad_y))

                        # Extract blendshapes
                        blend_dict: Dict[str, float] = {}
                        if i < len(res.face_blendshapes):
                            for cat in res.face_blendshapes[i]:
                                blend_dict[cat.category_name] = float(cat.score)

                        expr, conf, status, summary = classify_expression_from_blendshapes(
                            blend_dict
                        )

                        # Flag partial or low confidence face detections
                        if (bw * bh) < (0.015 * w * h):
                            status = "partial_face"
                            if conf < 0.40:
                                expr = "uncertain"

                        raw_detections.append({
                            "bbox": [bx, by, bw, bh],
                            "expression": expr,
                            "confidence": conf,
                            "status": status,
                            "blendshapes_summary": summary,
                        })
            except Exception:
                raw_detections = self.detect_face_contour_fallback(frame)
        else:
            raw_detections = self.detect_face_contour_fallback(frame)

        # Update tracker with detections
        tracked_faces = self.tracker.update(raw_detections, (h, w))

        if not tracked_faces:
            return {
                "success": False,
                "face_detected": False,
                "num_faces": 0,
                "faces": [],
                "expression": None,
                "confidence": 0.0,
                "bbox": None,
                "status": "no_face_detected",
                "disclaimer": FACIAL_ANALYSIS_DISCLAIMER,
                "supported_expressions": SUPPORTED_EXPRESSIONS,
            }

        # Select primary face (largest area in frame)
        primary_face = max(
            tracked_faces, key=lambda f: f["bbox"][2] * f["bbox"][3]
        )
        self._last_expression = primary_face["expression"]
        self._last_confidence = primary_face["confidence"]

        overall_status = "face_detected" if len(tracked_faces) == 1 else "faces_detected"

        return {
            "success": True,
            "face_detected": True,
            "num_faces": len(tracked_faces),
            "faces": tracked_faces,
            "expression": primary_face["expression"],
            "confidence": primary_face["confidence"],
            "bbox": primary_face["bbox"],
            "status": overall_status,
            "disclaimer": FACIAL_ANALYSIS_DISCLAIMER,
            "supported_expressions": SUPPORTED_EXPRESSIONS,
        }


# Global Singleton Service
facial_service = FacialAnalysisService()
