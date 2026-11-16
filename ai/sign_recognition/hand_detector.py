"""Hand Detection and Landmark Extraction Engine for SignBridge AI.
Supports MediaPipe Tasks Vision API with robust OpenCV Computer Vision fallback.
Safely handles headless environments, missing cameras, and variable lighting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.sign_recognition.config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    TASK_MODEL_PATH,
)

# Standard Hand Skeleton Connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]


@dataclass
class HandDetectionResult:
    """Standardized result container for hand detection across all backends."""
    detected: bool = False
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)  # 21 points (x, y, z)
    handedness: str = "Unknown"  # "Right", "Left", or "Unknown"
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, width, height) in pixel coords
    confidence: float = 0.0
    num_hands: int = 0
    backend: str = "none"


class CVHandDetectorFallback:
    """Robust OpenCV skin-segmentation and convexity-defect hand detector fallback.
    
    Operates without external pre-trained model files, providing offline/WSL resilience.
    Extracts palm center, wrist, bounding box, and estimates landmark positions.
    """

    def __init__(self):
        self.min_area = 5000

    def detect(self, frame: np.ndarray) -> HandDetectionResult:
        """Detects hand using color segmentation and contour analysis."""
        h, w = frame.shape[:2]
        
        # Convert to HSV and YCrCb for robust skin color thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Skin range in YCrCb is typically very consistent across skin tones
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandDetectionResult(detected=False, backend="opencv_fallback")

        # Pick largest contour meeting minimum area threshold
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)
        if area < self.min_area:
            return HandDetectionResult(detected=False, backend="opencv_fallback")

        # Bounding box
        bx, by, bw, bh = cv2.boundingRect(hand_contour)
        
        # Moments for center of mass
        moments = cv2.moments(hand_contour)
        if moments["m00"] == 0:
            return HandDetectionResult(detected=False, backend="opencv_fallback")

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Convex hull and defects to locate fingertips
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        fingertips = []
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(hand_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand_contour[s][0])
                        # Filter points located above palm center (extended fingers)
                        if start[1] < cy and d > 1000:
                            fingertips.append(start)
            except Exception:
                pass

        # Sort fingertips from left to right (X coordinate)
        fingertips = sorted(fingertips, key=lambda p: p[0])[:5]

        # Synthesize normalized 21 landmark estimates based on palm geometry
        wrist_x, wrist_y = cx / w, (by + bh) / h
        landmarks: List[Tuple[float, float, float]] = [(wrist_x, wrist_y, 0.0)]

        # Interpolate standard finger chains (Thumb, Index, Middle, Ring, Pinky)
        finger_bases_x = np.linspace(bx / w, (bx + bw) / w, 6)[1:]
        mcp_y = cy / h

        for idx, base_x in enumerate(finger_bases_x[:5]):
            # Use detected fingertip if available, else estimate from bounding box
            if idx < len(fingertips):
                tip_x, tip_y = fingertips[idx][0] / w, fingertips[idx][1] / h
            else:
                tip_x, tip_y = base_x, (by / h) + 0.05

            # 4 joints per finger (CMC/MCP, PIP, DIP, TIP)
            for step in range(1, 5):
                alpha = step / 4.0
                jx = (1 - alpha) * base_x + alpha * tip_x
                jy = (1 - alpha) * mcp_y + alpha * tip_y
                landmarks.append((float(jx), float(jy), 0.0))

        confidence = min(0.90, max(0.55, float(area / (w * h * 0.4))))

        return HandDetectionResult(
            detected=True,
            landmarks=landmarks[:21],
            handedness="Right",
            bbox=(bx, by, bw, bh),
            confidence=confidence,
            num_hands=1,
            backend="opencv_fallback",
        )


class HandDetector:
    """Primary Hand Detection interface for SignBridge AI.
    
    Seamlessly manages MediaPipe Tasks HandLandmarker when task asset is available,
    falling back to CVHandDetectorFallback for maximum hackathon reliability.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or TASK_MODEL_PATH
        self.mediapipe_landmarker = None
        self.fallback_detector = CVHandDetectorFallback()
        self.backend = "opencv_fallback"
        self._initialize_mediapipe()

    def _initialize_mediapipe(self) -> bool:
        """Attempts to initialize MediaPipe Tasks HandLandmarker if model file exists."""
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
                num_hands=2,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.mediapipe_landmarker = HandLandmarker.create_from_options(options)
            self.backend = "mediapipe_tasks"
            return True
        except Exception as e:
            # Gracefully keep fallback active
            self.backend = "opencv_fallback"
            return False

    def detect(self, frame: np.ndarray) -> HandDetectionResult:
        """Processes a BGR image frame and extracts 21 hand landmarks.
        
        Args:
            frame: BGR image frame from webcam or API request.
            
        Returns:
            HandDetectionResult object with landmarks and detection metadata.
        """
        if frame is None or frame.size == 0:
            return HandDetectionResult(detected=False, backend=self.backend)

        # 1. MediaPipe Tasks processing if available
        if self.mediapipe_landmarker is not None:
            try:
                import mediapipe as mp

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.mediapipe_landmarker.detect(mp_image)

                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    primary_hand = result.hand_landmarks[0]
                    landmarks = [(float(lm.x), float(lm.y), float(lm.z)) for lm in primary_hand]
                    
                    # Handedness
                    handedness = "Unknown"
                    if result.handedness and len(result.handedness) > 0:
                        handedness = result.handedness[0][0].category_name

                    # Bounding box in pixel coords
                    h, w = frame.shape[:2]
                    xs = [int(lm.x * w) for lm in primary_hand]
                    ys = [int(lm.y * h) for lm in primary_hand]
                    min_x, max_x = max(0, min(xs)), min(w, max(xs))
                    min_y, max_y = max(0, min(ys)), min(h, max(ys))
                    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                    score = 0.95
                    if result.handedness and len(result.handedness) > 0:
                        score = float(result.handedness[0][0].score)

                    return HandDetectionResult(
                        detected=True,
                        landmarks=landmarks,
                        handedness=handedness,
                        bbox=bbox,
                        confidence=score,
                        num_hands=len(result.hand_landmarks),
                        backend="mediapipe_tasks",
                    )
            except Exception:
                pass  # Fall through to fallback detector

        # 2. OpenCV Fallback detector
        return self.fallback_detector.detect(frame)

    @staticmethod
    def draw_landmarks(
        image: np.ndarray,
        result: HandDetectionResult,
        draw_bbox: bool = True,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Renders hand landmarks, connections, and status HUD onto the image frame."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if not result.detected or not result.landmarks:
            if draw_hud:
                cv2.putText(
                    annotated,
                    "No Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            return annotated

        # Convert normalized coordinates to pixel coordinates
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in result.landmarks]

        # Draw skeleton connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(annotated, pts[start_idx], pts[end_idx], (0, 255, 200), 2, cv2.LINE_AA)

        # Draw landmark points
        for idx, (px, py) in enumerate(pts):
            # Fingertip landmarks highlighted in bright green
            is_tip = idx in (4, 8, 12, 16, 20)
            color = (0, 255, 0) if is_tip else (255, 128, 0)
            radius = 6 if is_tip else 4
            cv2.circle(annotated, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding box
        if draw_bbox and result.bbox != (0, 0, 0, 0):
            bx, by, bw, bh = result.bbox
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2, cv2.LINE_AA)
            label = f"{result.handedness} ({int(result.confidence * 100)}%)"
            cv2.putText(
                annotated,
                label,
                (bx, max(20, by - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw status HUD
        if draw_hud:
            status_text = f"Hand: {result.handedness} | Conf: {result.confidence:.2f} [{result.backend}]"
            cv2.putText(
                annotated,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated
