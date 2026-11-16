"""Landmark Preprocessing and Feature Extraction for SignBridge AI.
Translates, scales, and computes geometric features from 21 hand landmarks.
Provides scale, position, and hand-size invariant representations.
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from ai.sign_recognition.config import (
    NUM_LANDMARKS,
    NUM_COORDINATE_FEATURES,
    TOTAL_FEATURE_DIM,
)

# Standard MediaPipe 21 Landmark Indices
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


class LandmarkExtractor:
    """Preprocesses raw 21 hand landmarks and extracts robust geometric features."""

    def __init__(self):
        self.num_landmarks = NUM_LANDMARKS
        self.total_dim = TOTAL_FEATURE_DIM

    def normalize_landmarks(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> np.ndarray:
        """Translates landmarks to wrist origin (landmark 0) and normalizes by palm scale.
        
        Args:
            landmarks: Array of shape (21, 3) representing (x, y, z) coordinates.
            
        Returns:
            Normalized array of shape (21, 3).
        """
        pts = np.array(landmarks, dtype=np.float32)
        if pts.shape[0] != self.num_landmarks or pts.shape[1] < 3:
            raise ValueError(f"Expected landmarks with shape (21, 3), got {pts.shape}")

        # Translation: Move wrist to origin (0, 0, 0)
        wrist = pts[WRIST, :3].copy()
        translated = pts[:, :3] - wrist

        # Scale normalization: Compute scale from wrist to middle finger MCP
        scale = np.linalg.norm(translated[MIDDLE_MCP, :2])
        if scale < 1e-6:
            # Fallback to maximum coordinate distance
            scale = np.max(np.linalg.norm(translated[:, :2], axis=1))
        if scale < 1e-6:
            scale = 1.0

        normalized = translated / scale
        return normalized

    def compute_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """Determines if each of the 5 fingers is extended or curled.
        
        Works with both normalized and raw landmarks.
        
        Args:
            landmarks: Array of shape (21, 3).
            
        Returns:
            Dictionary with boolean extension status for thumb, index, middle, ring, pinky.
        """
        pts = landmarks[:, :3]

        # In image coordinates, Y increases downward.
        # Finger is extended if tip Y is significantly above (less than) PIP Y
        # For thumb, compare tip X/Y with MCP/IP depending on hand orientation.
        wrist = pts[WRIST]
        
        # Distance from wrist to tip vs wrist to PIP
        def is_extended_by_distance(tip_idx: int, pip_idx: int) -> bool:
            dist_tip = np.linalg.norm(pts[tip_idx, :2] - wrist[:2])
            dist_pip = np.linalg.norm(pts[pip_idx, :2] - wrist[:2])
            return bool(dist_tip > (dist_pip * 1.15))

        # Thumb extension: tip distance to pinky MCP compared to IP distance
        thumb_dist_tip = np.linalg.norm(pts[THUMB_TIP, :2] - pts[PINKY_MCP, :2])
        thumb_dist_ip = np.linalg.norm(pts[THUMB_IP, :2] - pts[PINKY_MCP, :2])
        thumb_extended = bool(thumb_dist_tip > (thumb_dist_ip * 1.10))

        index_extended = is_extended_by_distance(INDEX_TIP, INDEX_PIP)
        middle_extended = is_extended_by_distance(MIDDLE_TIP, MIDDLE_PIP)
        ring_extended = is_extended_by_distance(RING_TIP, RING_PIP)
        pinky_extended = is_extended_by_distance(PINKY_TIP, PINKY_PIP)

        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }

    def extract_features(
        self, landmarks: Union[List[Tuple[float, float, float]], np.ndarray]
    ) -> np.ndarray:
        """Extracts complete 1D feature vector for classification.
        
        Combines:
        1. 63 normalized (x, y, z) coordinates.
        2. 5 binary finger extension states.
        3. 5 key Euclidean distance metrics.
        
        Returns:
            1D NumPy array of shape (73,).
        """
        norm_pts = self.normalize_landmarks(landmarks)
        coords_flat = norm_pts.flatten()  # 63 features

        finger_states = self.compute_finger_states(norm_pts)
        finger_vec = np.array(
            [
                1.0 if finger_states["thumb"] else 0.0,
                1.0 if finger_states["index"] else 0.0,
                1.0 if finger_states["middle"] else 0.0,
                1.0 if finger_states["ring"] else 0.0,
                1.0 if finger_states["pinky"] else 0.0,
            ],
            dtype=np.float32,
        )  # 5 features

        # Key distance measurements
        thumb_to_index = float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[INDEX_TIP]))
        index_to_middle = float(np.linalg.norm(norm_pts[INDEX_TIP] - norm_pts[MIDDLE_TIP]))
        thumb_to_wrist = float(np.linalg.norm(norm_pts[THUMB_TIP] - norm_pts[WRIST]))
        middle_to_wrist = float(np.linalg.norm(norm_pts[MIDDLE_TIP] - norm_pts[WRIST]))
        pinky_to_wrist = float(np.linalg.norm(norm_pts[PINKY_TIP] - norm_pts[WRIST]))

        dist_vec = np.array(
            [thumb_to_index, index_to_middle, thumb_to_wrist, middle_to_wrist, pinky_to_wrist],
            dtype=np.float32,
        )  # 5 features

        feature_vector = np.concatenate([coords_flat, finger_vec, dist_vec])
        return feature_vector
