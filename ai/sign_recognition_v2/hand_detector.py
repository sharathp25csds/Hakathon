"""Multi-Hand Detection and 21-Landmark Estimation Engine for SignBridge AI v2.
Preserves separate Left and Right hand landmarks using MediaPipe Tasks Vision API.
Explicitly flags degraded/fallback states when neural models are unavailable.
Never silently outputs fabricated points as real anatomical landmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ai.sign_recognition_v2.config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    TASK_MODEL_PATH,
)

# Standard Hand Skeleton Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm base
]


@dataclass
class HandData:
    """Individual hand landmark data, handedness, bounding box, and 3D orientation."""
    landmarks: List[Tuple[float, float, float]]  # 21 points (x, y, z)
    handedness: str  # "Left", "Right", or "Unknown"
    confidence: float
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    palm_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # 3D palm normal vector


@dataclass
class HandDetectionResultV2:
    """Multi-hand detection container preserving all detected hands and spatial relationships."""
    detected: bool = False
    num_hands: int = 0
    hands: List[HandData] = field(default_factory=list)
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    inter_hand_distance: Optional[float] = None  # Normalized distance between wrists
    relative_hand_vector: Optional[Tuple[float, float, float]] = None  # Vector from Left to Right wrist
    backend: str = "none"
    is_degraded: bool = False  # True if fallback / non-MediaPipe landmarks are used

    # Backward compatibility properties mapping to primary hand
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    handedness: str = "Unknown"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0


def compute_palm_normal(
    landmarks: List[Tuple[float, float, float]], handedness: str = "Right"
) -> Tuple[float, float, float]:
    """Computes normalized 3D palm normal vector (wrist, index MCP, pinky MCP plane)."""
    if len(landmarks) < 21:
        return (0.0, 0.0, 1.0)

    p0 = np.array(landmarks[0][:3], dtype=np.float32)   # Wrist
    p5 = np.array(landmarks[5][:3], dtype=np.float32)   # Index MCP
    p17 = np.array(landmarks[17][:3], dtype=np.float32) # Pinky MCP

    v1 = p5 - p0
    v2 = p17 - p0

    if handedness.lower() == "left":
        normal = np.cross(v2, v1)
    else:
        normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm > 1e-6:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return (float(normal[0]), float(normal[1]), float(normal[2]))


class CVHandDetectorFallbackV2:
    """Degraded OpenCV skin-segmentation fallback.
    
    EXPLICITLY FLAGS is_degraded=True to inform classifiers that landmarks
    are interpolated geometric contour approximations, not anatomical MediaPipe landmarks.
    """

    def __init__(self, min_area: int = 4000):
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> HandDetectionResultV2:
        """Analyzes frame contours; returns is_degraded=True."""
        if frame is None or frame.size == 0:
            return HandDetectionResultV2(detected=False, backend="opencv_degraded_fallback", is_degraded=True)

        h, w = frame.shape[:2]

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandDetectionResultV2(detected=False, backend="opencv_degraded_fallback", is_degraded=True)

        candidate = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(candidate)
        if area < self.min_area:
            return HandDetectionResultV2(detected=False, backend="opencv_degraded_fallback", is_degraded=True)

        bx, by, bw, bh = cv2.boundingRect(candidate)
        aspect = bw / float(bh)
        if aspect < 0.3 or aspect > 2.5:
            return HandDetectionResultV2(detected=False, backend="opencv_degraded_fallback", is_degraded=True)

        moments = cv2.moments(candidate)
        if moments["m00"] == 0:
            return HandDetectionResultV2(detected=False, backend="opencv_degraded_fallback", is_degraded=True)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        hull = cv2.convexHull(candidate, returnPoints=False)
        fingertips = []
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(candidate, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(candidate[s][0])
                        if start[1] < cy and d > 1200:
                            fingertips.append(start)
            except Exception:
                pass

        fingertips = sorted(fingertips, key=lambda p: p[0])[:5]

        wrist_norm = (cx / w, (by + bh) / h, 0.0)
        landmarks: List[Tuple[float, float, float]] = [wrist_norm]
        finger_bases_x = np.linspace(bx / w, (bx + bw) / w, 6)[1:]
        mcp_norm_y = cy / h

        for idx, base_x in enumerate(finger_bases_x[:5]):
            if idx < len(fingertips):
                tip_x, tip_y = fingertips[idx][0] / w, fingertips[idx][1] / h
            else:
                tip_x, tip_y = base_x, mcp_norm_y + 0.02  # Curled down near palm MCP

            for step in range(1, 5):
                alpha = step / 4.0
                jx = (1 - alpha) * base_x + alpha * tip_x
                jy = (1 - alpha) * mcp_norm_y + alpha * tip_y
                landmarks.append((float(jx), float(jy), 0.0))

        confidence = min(0.85, max(0.50, float(area / (w * h * 0.35))))
        hand = HandData(
            landmarks=landmarks[:21],
            handedness="Right",
            confidence=round(confidence, 2),
            bbox=(bx, by, bw, bh),
            palm_normal=(0.0, 0.0, 1.0),
        )

        return HandDetectionResultV2(
            detected=True,
            num_hands=1,
            hands=[hand],
            right_hand=hand,
            backend="opencv_degraded_fallback",
            is_degraded=True,
            landmarks=hand.landmarks,
            handedness=hand.handedness,
            bbox=hand.bbox,
            confidence=hand.confidence,
        )


class HandDetectorV2:
    """Primary Multi-Hand Detection interface for SignBridge AI v2."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or TASK_MODEL_PATH
        self.mediapipe_landmarker = None
        self.fallback_detector = CVHandDetectorFallbackV2()
        self.backend = "opencv_degraded_fallback"
        self._initialize_mediapipe()

    def _initialize_mediapipe(self) -> bool:
        """Verifies and initializes MediaPipe Tasks Vision API with multi-hand support."""
        if not self.model_path.exists():
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,  # Support both hands for ISL
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.mediapipe_landmarker = HandLandmarker.create_from_options(options)
            self.backend = "mediapipe_tasks"
            return True
        except Exception:
            self.backend = "opencv_degraded_fallback"
            return False

    def detect(self, frame: np.ndarray) -> HandDetectionResultV2:
        """Detects up to two hands, preserving landmarks, handedness, and spatial vectors."""
        if frame is None or frame.size == 0:
            return HandDetectionResultV2(detected=False, backend=self.backend)

        # 1. MediaPipe Tasks Multi-Hand Detection
        if self.mediapipe_landmarker is not None:
            try:
                import mediapipe as mp

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.mediapipe_landmarker.detect(mp_image)

                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    h, w = frame.shape[:2]
                    hands: List[HandData] = []
                    left_hand: Optional[HandData] = None
                    right_hand: Optional[HandData] = None

                    for idx, hand_lms in enumerate(result.hand_landmarks):
                        lms = [(float(lm.x), float(lm.y), float(lm.z)) for lm in hand_lms]
                        
                        handedness = "Unknown"
                        score = 0.95
                        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                            handedness = result.handedness[idx][0].category_name
                            score = float(result.handedness[idx][0].score)

                        xs = [int(lm[0] * w) for lm in lms]
                        ys = [int(lm[1] * h) for lm in lms]
                        bx, bw = min(xs), max(xs) - min(xs)
                        by, bh = min(ys), max(ys) - min(ys)
                        bbox = (max(0, bx), max(0, by), max(1, bw), max(1, bh))

                        palm_norm = compute_palm_normal(lms, handedness)
                        hand_obj = HandData(
                            landmarks=lms,
                            handedness=handedness,
                            confidence=round(score, 2),
                            bbox=bbox,
                            palm_normal=palm_norm,
                        )
                        hands.append(hand_obj)

                        if handedness.lower() == "left" and left_hand is None:
                            left_hand = hand_obj
                        elif handedness.lower() == "right" and right_hand is None:
                            right_hand = hand_obj

                    # Compute inter-hand spatial relationship if 2 hands present
                    inter_dist = None
                    rel_vec = None
                    if len(hands) >= 2:
                        w0 = np.array(hands[0].landmarks[0][:3])
                        w1 = np.array(hands[1].landmarks[0][:3])
                        inter_dist = float(np.linalg.norm(w0 - w1))
                        rel_vec = tuple(float(x) for x in (w1 - w0))

                    primary = hands[0]
                    return HandDetectionResultV2(
                        detected=True,
                        num_hands=len(hands),
                        hands=hands,
                        left_hand=left_hand,
                        right_hand=right_hand,
                        inter_hand_distance=inter_dist,
                        relative_hand_vector=rel_vec,
                        backend="mediapipe_tasks",
                        is_degraded=False,
                        # Backward compatibility
                        landmarks=primary.landmarks,
                        handedness=primary.handedness,
                        bbox=primary.bbox,
                        confidence=primary.confidence,
                    )
                else:
                    # MediaPipe inference succeeded with zero hands detected
                    return HandDetectionResultV2(
                        detected=False,
                        num_hands=0,
                        hands=[],
                        backend="mediapipe_tasks",
                        is_degraded=False,
                    )
            except Exception:
                pass

        # 2. Degraded OpenCV Fallback (invoked only when MediaPipe is unavailable or inference fails)
        return self.fallback_detector.detect(frame)

    @staticmethod
    def draw_landmarks(
        image: np.ndarray,
        result: HandDetectionResultV2,
        draw_bbox: bool = True,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Visualizes all detected hands, skeleton joints, handedness, and diagnostic HUD."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if not result.detected or not result.hands:
            if draw_hud:
                status_text = "No Hand Detected" if not result.is_degraded else "No Hand Detected [Degraded Fallback]"
                cv2.putText(
                    annotated,
                    status_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 240),
                    2,
                    cv2.LINE_AA,
                )
            return annotated

        for hand_idx, hand in enumerate(result.hands):
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]

            # Skeleton connections (Cyan for Right hand, Orange for Left hand)
            conn_color = (0, 220, 255) if hand.handedness == "Right" else (255, 140, 0)
            for s_idx, e_idx in HAND_CONNECTIONS:
                if s_idx < len(pts) and e_idx < len(pts):
                    cv2.line(annotated, pts[s_idx], pts[e_idx], conn_color, 2, cv2.LINE_AA)

            # Keypoint circles
            for idx, (px, py) in enumerate(pts):
                is_tip = idx in (4, 8, 12, 16, 20)
                color = (0, 255, 120) if is_tip else (255, 200, 0)
                radius = 6 if is_tip else 4
                cv2.circle(annotated, (px, py), radius, color, -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Bounding box & label
            if draw_bbox and hand.bbox != (0, 0, 0, 0):
                bx, by, bw, bh = hand.bbox
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), conn_color, 2, cv2.LINE_AA)
                badge_label = f"{hand.handedness} ({int(hand.confidence * 100)}%)"
                cv2.putText(
                    annotated,
                    badge_label,
                    (bx, max(20, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    conn_color,
                    2,
                    cv2.LINE_AA,
                )

        # Connect both wrists with a dashed line if 2 hands present
        if len(result.hands) >= 2:
            p_w0 = (int(result.hands[0].landmarks[0][0] * w), int(result.hands[0].landmarks[0][1] * h))
            p_w1 = (int(result.hands[1].landmarks[0][0] * w), int(result.hands[1].landmarks[0][1] * h))
            cv2.line(annotated, p_w0, p_w1, (255, 0, 255), 2, cv2.LINE_AA)

        if draw_hud:
            hands_str = ", ".join([h.handedness for h in result.hands])
            degraded_tag = " [DEGRADED]" if result.is_degraded else ""
            hud_text = f"Hands: {result.num_hands} ({hands_str}){degraded_tag}"
            cv2.putText(
                annotated,
                hud_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated
