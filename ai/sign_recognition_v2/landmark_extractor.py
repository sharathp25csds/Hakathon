"""Multi-Hand Landmark Preprocessing and Geometric Feature Extraction Engine (v2).
Computes translation-, scale-, rotation-, and orientation-normalized representations for both
single-hand and two-hand Indian Sign Language (ISL) gestures.
Rejects non-finite values and enforces feature version consistency.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ai.sign_recognition_v2.config import (
    FEATURE_VERSION,
    HAND_FEATURE_DIM,
    INTER_HAND_FEATURE_DIM,
    LEGACY_FEATURE_DIM,
    NUM_LANDMARKS,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition_v2.hand_detector import HandData, HandDetectionResultV2

# Canonical MediaPipe Joint Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class LandmarkExtractorV2:
    """Extracts scale-, rotation-, and orientation-invariant feature vectors from 1 or 2 hands."""

    def __init__(
        self,
        enable_rotation_normalization: bool = True,
        feature_version: str = FEATURE_VERSION,
        dominant_hand_preference: str = "auto",  # "auto", "right", or "left"
    ):
        self.enable_rotation_normalization = enable_rotation_normalization
        self.feature_version = feature_version
        self.dominant_hand_preference = dominant_hand_preference
        self.total_dim = TOTAL_FEATURE_DIM

    def normalize_single_hand(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Translates landmarks to wrist origin, scales by palm span, and rotates to vertical.
        
        Args:
            landmarks: Array of shape (21, 3).
            
        Returns:
            Tuple: (normalized_landmarks_array of shape (21, 3), palm_scale)
        """
        pts = np.array(landmarks, dtype=np.float32)

        if np.isnan(pts).any() or np.isinf(pts).any():
            raise ValueError("Landmarks contain non-finite values (NaN or Inf)")

        if pts.ndim != 2 or pts.shape[0] != NUM_LANDMARKS or pts.shape[1] < 3:
            raise ValueError(f"Expected landmarks array of shape (21, 3), got {pts.shape}")

        pts_3d = pts[:, :3].copy()

        # Translation: Move wrist to origin (0, 0, 0)
        wrist = pts_3d[WRIST].copy()
        translated = pts_3d - wrist

        # Scale Normalization: Middle MCP distance
        palm_scale = float(np.linalg.norm(translated[MIDDLE_MCP, :2]))
        if palm_scale < 1e-5:
            palm_scale = float(np.max(np.linalg.norm(translated[:, :2], axis=1)))
        if palm_scale < 1e-5:
            palm_scale = 1.0

        scaled = translated / palm_scale

        # Rotation Normalization (with minimum span guard to eliminate fist jitter)
        if self.enable_rotation_normalization:
            mid_vec = scaled[MIDDLE_MCP, :2]
            norm_mid = np.linalg.norm(mid_vec)
            # Only rotate if the palm axis is distinct (avoids noise on tight fists)
            if norm_mid >= 0.15:
                current_angle = np.arctan2(mid_vec[1], mid_vec[0])
                target_angle = -np.pi / 2.0  # Vertical upright
                rot_angle = target_angle - current_angle

                cos_a = np.cos(rot_angle)
                sin_a = np.sin(rot_angle)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                scaled[:, :2] = scaled[:, :2] @ rot_mat.T

        return scaled, palm_scale

    def normalize_landmarks(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> np.ndarray:
        """Backward compatible helper returning normalized (21, 3) landmarks."""
        norm_pts, _ = self.normalize_single_hand(landmarks)
        return norm_pts

    def compute_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """Calculates extension status (True/False) for each of the 5 fingers."""
        pts = landmarks[:, :3]
        wrist = pts[WRIST]

        def is_extended(tip_idx: int, pip_idx: int) -> bool:
            dist_tip = float(np.linalg.norm(pts[tip_idx, :2] - wrist[:2]))
            dist_pip = float(np.linalg.norm(pts[pip_idx, :2] - wrist[:2]))
            return dist_tip > (dist_pip * 1.10)

        thumb_tip_dist = float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[PINKY_MCP, :2]))
        thumb_ip_dist = float(np.linalg.norm(pts[THUMB_IP, :2] - pts[PINKY_MCP, :2]))
        thumb_extended = thumb_tip_dist > (thumb_ip_dist * 1.06)

        return {
            "thumb": thumb_extended,
            "index": is_extended(INDEX_TIP, INDEX_PIP),
            "middle": is_extended(MIDDLE_TIP, MIDDLE_PIP),
            "ring": is_extended(RING_TIP, RING_PIP),
            "pinky": is_extended(PINKY_TIP, PINKY_PIP),
        }

    def extract_single_hand_vector(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray], palm_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> np.ndarray:
        """Extracts 76-dimensional feature vector for an individual hand."""
        norm_pts, _ = self.normalize_single_hand(landmarks)
        coords_flat = norm_pts.flatten()  # 63 features

        norm_vec = np.array(palm_normal, dtype=np.float32)  # 3 features

        states = self.compute_finger_states(norm_pts)
        finger_vec = np.array(
            [1.0 if states[f] else 0.0 for f in ["thumb", "index", "middle", "ring", "pinky"]],
            dtype=np.float32,
        )  # 5 features

        distances = np.array(
            [
                float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[INDEX_TIP])),
                float(np.linalg.norm(norm_pts[INDEX_TIP] - norm_pts[MIDDLE_TIP])),
                float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[WRIST])),
                float(np.linalg.norm(norm_pts[MIDDLE_TIP] - norm_pts[WRIST])),
                float(np.linalg.norm(norm_pts[PINKY_TIP] - norm_pts[WRIST])),
            ],
            dtype=np.float32,
        )  # 5 features

        # Total = 63 + 3 + 5 + 5 = 76
        return np.concatenate([coords_flat, norm_vec, finger_vec, distances])

    def extract_multi_hand_features(self, detection: HandDetectionResultV2) -> np.ndarray:
        """Extracts complete 162-dimensional feature vector from HandDetectionResultV2.
        
        Handles both single-hand and dual-hand signs gracefully.
        Preserves dominant and non-dominant roles based on configuration.
        """
        if not detection.detected or not detection.hands:
            return np.zeros(self.total_dim, dtype=np.float32)

        # Determine dominant vs non-dominant hand order
        hands = detection.hands
        primary_hand: HandData = hands[0]
        secondary_hand: Optional[HandData] = hands[1] if len(hands) >= 2 else None

        if self.dominant_hand_preference == "left":
            # If user prefers Left hand dominance, place Left as primary
            if detection.left_hand:
                primary_hand = detection.left_hand
                secondary_hand = detection.right_hand
        elif self.dominant_hand_preference == "right":
            if detection.right_hand:
                primary_hand = detection.right_hand
                secondary_hand = detection.left_hand

        # 1. Primary Hand Features (76 features)
        primary_vec = self.extract_single_hand_vector(
            primary_hand.landmarks, primary_hand.palm_normal
        )

        # 2. Secondary Hand Features (76 features)
        if secondary_hand is not None:
            secondary_vec = self.extract_single_hand_vector(
                secondary_hand.landmarks, secondary_hand.palm_normal
            )
        else:
            secondary_vec = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)

        # 3. Inter-Hand Spatial Features (10 features)
        inter_hand_vec = np.zeros(INTER_HAND_FEATURE_DIM, dtype=np.float32)
        if secondary_hand is not None:
            p_norm, p_scale = self.normalize_single_hand(primary_hand.landmarks)
            s_raw = np.array(secondary_hand.landmarks, dtype=np.float32)

            # Wrist-to-wrist vector scaled by dominant palm scale
            w_diff = (s_raw[WRIST, :3] - np.array(primary_hand.landmarks[WRIST][:3])) / p_scale
            w_dist = float(np.linalg.norm(w_diff))

            # Dominant index tip to secondary fingertips (thumb, index, middle, ring, pinky)
            # Crucial for ISLRTC vowels A, E, I, O, U!
            p_idx_tip = np.array(primary_hand.landmarks[INDEX_TIP][:3])
            contact_dists = [
                float(np.linalg.norm(p_idx_tip - s_raw[tip_idx, :3]) / p_scale)
                for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            ]

            inter_hand_vec[0] = 1.0  # has_second_hand flag
            inter_hand_vec[1] = w_dist
            inter_hand_vec[2:5] = w_diff[:3]
            inter_hand_vec[5:10] = contact_dists

        return np.concatenate([primary_vec, secondary_vec, inter_hand_vec])

    def extract_features(
        self, input_data: Union[HandDetectionResultV2, List, np.ndarray]
    ) -> np.ndarray:
        """Unified feature extractor supporting HandDetectionResultV2 and raw landmark arrays."""
        if isinstance(input_data, HandDetectionResultV2):
            return self.extract_multi_hand_features(input_data)

        # Raw array passed (e.g. single-hand landmarks array of shape (21, 3))
        pts = np.array(input_data, dtype=np.float32)
        if pts.shape == (21, 3):
            # Format into primary hand + zero-padded second hand = 162 features
            primary_vec = self.extract_single_hand_vector(pts)
            secondary_vec = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
            inter_vec = np.zeros(INTER_HAND_FEATURE_DIM, dtype=np.float32)
            return np.concatenate([primary_vec, secondary_vec, inter_vec])

        raise ValueError(f"Unsupported input data shape for feature extraction: {pts.shape}")

    def extract_legacy_features(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> np.ndarray:
        """Extracts legacy 73-dimensional single-hand vector for backward compatibility."""
        norm_pts, _ = self.normalize_single_hand(landmarks)
        coords_flat = norm_pts.flatten()

        states = self.compute_finger_states(norm_pts)
        finger_vec = np.array(
            [1.0 if states[f] else 0.0 for f in ["thumb", "index", "middle", "ring", "pinky"]],
            dtype=np.float32,
        )

        distances = np.array(
            [
                float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[INDEX_TIP])),
                float(np.linalg.norm(norm_pts[INDEX_TIP] - norm_pts[MIDDLE_TIP])),
                float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[WRIST])),
                float(np.linalg.norm(norm_pts[MIDDLE_TIP] - norm_pts[WRIST])),
                float(np.linalg.norm(norm_pts[PINKY_TIP] - norm_pts[WRIST])),
            ],
            dtype=np.float32,
        )

        return np.concatenate([coords_flat, finger_vec, distances])
