"""Temporal and Trajectory Recognition Engine for Dynamic ISL Letters (v2).
Tracks movement trajectories over consecutive frames to recognize dynamic letters
such as 'J' (hook stroke) and 'Z' (zig-zag stroke).
Verifies both trajectory path and handshape anatomy to eliminate false positives.
"""

from collections import deque
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ai.sign_recognition_v2.config import TRAJECTORY_BUFFER_SIZE
from ai.sign_recognition_v2.landmark_extractor import INDEX_TIP, PINKY_TIP, WRIST


class TemporalRecognizerV2:
    """Analyzes landmark trajectories over a sliding temporal window for dynamic letters."""

    def __init__(self, buffer_size: int = TRAJECTORY_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.index_trajectory: deque = deque(maxlen=buffer_size)
        self.pinky_trajectory: deque = deque(maxlen=buffer_size)
        self.timestamps: deque = deque(maxlen=buffer_size)
        self.frame_counter: int = 0
        self.last_trigger_time: float = 0.0
        self.last_trigger_frame: int = -100
        self.trigger_cooldown_sec: float = 1.5
        self.trigger_cooldown_frames: int = 20

    def reset(self) -> None:
        """Clears accumulated trajectory history."""
        self.index_trajectory.clear()
        self.pinky_trajectory.clear()
        self.timestamps.clear()

    def add_frame(
        self,
        normalized_landmarks: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """Records normalized fingertip positions for the current frame.
        
        Args:
            normalized_landmarks: Array of shape (21, 3).
            timestamp: Optional monotonic timestamp (seconds); defaults to time.monotonic().
        """
        now = time.monotonic() if timestamp is None else timestamp
        self.frame_counter += 1

        idx_pt = normalized_landmarks[INDEX_TIP, :2].copy()
        pnk_pt = normalized_landmarks[PINKY_TIP, :2].copy()

        self.index_trajectory.append(idx_pt)
        self.pinky_trajectory.append(pnk_pt)
        self.timestamps.append(now)

    def evaluate_dynamic_gesture(
        self,
        finger_states: Optional[Dict[str, bool]] = None,
        current_time: Optional[float] = None,
        language: str = "ISL",
    ) -> Tuple[Optional[str], float, Dict]:
        """Evaluates whether accumulated trajectory and current handshape match a dynamic letter stroke.
        
        Args:
            finger_states: Optional dictionary with boolean states for thumb, index, middle, ring, pinky.
            current_time: Optional timestamp for testing/simulation.
            language: "ISL" or "ASL" to apply language-specific handshape constraints.
            
        Returns:
            Tuple: (letter_or_None, confidence, metadata)
        """
        now = time.monotonic() if current_time is None else current_time

        # Check cooldown (both time-based and frame-based)
        if (now - self.last_trigger_time < self.trigger_cooldown_sec) or (
            self.frame_counter - self.last_trigger_frame < self.trigger_cooldown_frames
        ):
            return None, 0.0, {"status": "cooldown"}

        if len(self.index_trajectory) < 8:
            return None, 0.0, {"status": "insufficient_trajectory"}

        pts_index = np.array(self.index_trajectory)
        pts_pinky = np.array(self.pinky_trajectory)

        # 1. Evaluate 'Z' stroke:
        # Handshape prerequisite: Index finger must be extended.
        index_extended = True
        if finger_states is not None:
            if str(language).upper() == "ASL":
                index_extended = finger_states.get("index", False) and not finger_states.get("pinky", False)
            else:
                index_extended = finger_states.get("index", False)

        if index_extended:
            z_match, z_conf = self._check_z_stroke(pts_index)
            if z_match:
                self.last_trigger_time = now
                self.last_trigger_frame = self.frame_counter
                self.reset()
                return "Z", z_conf, {
                    "stroke": "zigzag",
                    "finger": "index",
                    "language": str(language).upper(),
                    "status": "experimental_dynamic",
                }

        # 2. Evaluate 'J' stroke:
        # In ASL, J strictly requires pinky finger extended (starting from 'I' handshape).
        # In ISL, ISLRTC permits index hook or pinky hook.
        j_handshape = True
        if finger_states is not None:
            if str(language).upper() == "ASL":
                j_handshape = finger_states.get("pinky", False) and not finger_states.get("index", False)
            else:
                j_handshape = finger_states.get("pinky", False) or finger_states.get("index", False)

        if j_handshape:
            # Check pinky trajectory first, then index trajectory (if allowed by language)
            j_match_p, j_conf_p = self._check_j_stroke(pts_pinky)
            j_match_i, j_conf_i = (False, 0.0) if str(language).upper() == "ASL" else self._check_j_stroke(pts_index)

            if j_match_p or j_match_i:
                conf = max(j_conf_p, j_conf_i)
                active_finger = "pinky" if j_conf_p >= j_conf_i else "index"
                self.last_trigger_time = now
                self.last_trigger_frame = self.frame_counter
                self.reset()
                return "J", conf, {
                    "stroke": "hook",
                    "finger": active_finger,
                    "language": str(language).upper(),
                    "status": "experimental_dynamic",
                }

        return None, 0.0, {"status": "no_dynamic_match"}

    def _check_z_stroke(self, pts: np.ndarray) -> Tuple[bool, float]:
        """Detects 3-segment zig-zag pattern in index fingertip trajectory.
        
        A valid 'Z' pattern consists of:
        - Segment 1: Rightward movement (+X)
        - Segment 2: Diagonal down-left movement (-X, +Y in camera screen coords)
        - Segment 3: Rightward movement (+X)
        """
        n = len(pts)
        if n < 8:
            return False, 0.0

        total_span_x = np.ptp(pts[:, 0])
        total_span_y = np.ptp(pts[:, 1])

        # Must span meaningful horizontal and vertical range
        if total_span_x < 0.20 or total_span_y < 0.15:
            return False, 0.0

        # Split trajectory into 3 approximately equal segments
        p1 = pts[: n // 3]
        p2 = pts[n // 3 : 2 * n // 3]
        p3 = pts[2 * n // 3 :]

        dx1 = p1[-1, 0] - p1[0, 0]
        dx2 = p2[-1, 0] - p2[0, 0]
        dy2 = p2[-1, 1] - p2[0, 1]
        dx3 = p3[-1, 0] - p3[0, 0]

        if dx1 > 0.04 and dx2 < -0.04 and dy2 > 0.04 and dx3 > 0.04:
            conf = min(0.95, 0.70 + float(total_span_x * 0.5))
            return True, round(conf, 2)

        return False, 0.0

    def _check_j_stroke(self, pts: np.ndarray) -> Tuple[bool, float]:
        """Detects downward hook curve in fingertip trajectory.
        
        A valid 'J' pattern consists of:
        - Downward stroke (+Y) followed by an upward hook curve (-Y) at the end.
        """
        n = len(pts)
        if n < 8:
            return False, 0.0

        total_span_y = np.ptp(pts[:, 1])
        if total_span_y < 0.18:
            return False, 0.0

        max_y_idx = int(np.argmax(pts[:, 1]))

        # Lowest point of hook (max Y) should occur in the second half of the stroke
        if max_y_idx > (n * 0.40) and max_y_idx < (n - 1):
            hook_dy = pts[-1, 1] - pts[max_y_idx, 1]
            if hook_dy < -0.025:  # Curving back upward
                conf = min(0.94, 0.72 + float(total_span_y * 0.4))
                return True, round(conf, 2)

        return False, 0.0
