"""SignBridge AI - American Sign Language (ASL) Geometric Rule Classifier (v2).
Deterministic geometric rule engine based on standard anatomical ASL fingerspelling configurations.

ASL fingerspelling is one-handed. This classifier evaluates single-hand landmark geometry,
relative thumb placement (A vs S vs E vs T vs M vs N), finger crossing (R),
horizontal orientation (G, H), downward orientation (P, Q), and finger loops (D, F, O).
Dynamic letters (J, Z) are arbitrated via TemporalRecognizerV2.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ai.sign_recognition_v2.config import (
    ASL_COMPLIANCE_DISCLAIMER,
    ASL_ALPHABET_METADATA,
)
from ai.sign_recognition_v2.hand_detector import HandDetectionResultV2
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


def align_hand_x(pts: np.ndarray) -> np.ndarray:
    """Mirrors X coordinates if the hand is left-oriented, ensuring handedness invariance.
    
    After alignment, the thumb is always on the -X (radial) side and the pinky is on the +X (ulnar) side.
    """
    aligned = pts.copy()
    if aligned[THUMB_CMC, 0] > aligned[PINKY_MCP, 0]:
        aligned[:, 0] = -aligned[:, 0]
    return aligned


class ASLRuleBasedClassifierV2:
    """Deterministic geometric rule engine for American Sign Language (ASL) alphabet fingerspelling."""

    def __init__(self, extractor: LandmarkExtractorV2):
        self.extractor = extractor

    def classify_detection(
        self, detection: HandDetectionResultV2
    ) -> Tuple[Optional[str], float, Dict]:
        """Classifies hand detection result using ASL geometric rules."""
        if not detection.detected or not detection.hands:
            return None, 0.0, {"error": "no_hands"}

        # Reject degraded / approximate fallback detections
        if getattr(detection, "is_degraded", False):
            return None, 0.0, {
                "source": "none",
                "status": "degraded_detection_rejected",
                "error": "degraded_detection_rejected",
                "explanation": "Approximate fallback landmarks cannot be used for ASL recognition.",
            }

        primary_hand = detection.hands[0]
        pts = np.array(primary_hand.landmarks, dtype=np.float32)

        metadata = {
            "num_hands": len(detection.hands),
            "handedness": primary_hand.handedness,
            "disclaimer": ASL_COMPLIANCE_DISCLAIMER,
        }

        # ASL alphabet is strictly one-handed. If two hands are detected, evaluate the primary hand
        # while noting two-hand presence in metadata
        letter, conf, geom_meta = self.classify_geometry(pts)
        metadata.update(geom_meta)
        return letter, conf, metadata

    def classify_geometry(
        self, landmarks: np.ndarray
    ) -> Tuple[Optional[str], float, Dict]:
        """Evaluates normalized landmark geometry against defined ASL alphabet handshapes."""
        if float(np.ptp(landmarks)) < 1e-3:
            return None, 0.0, {"error": "degenerate_hand_geometry"}

        norm_pts, p_scale = self.extractor.normalize_single_hand(landmarks)
        p_scale = max(float(p_scale), 1e-4)
        finger_states = self.extractor.compute_finger_states(norm_pts)

        # Handedness-aligned coordinates for reliable thumb placement checks
        aligned = align_hand_x(norm_pts)

        st = finger_states
        thumb_ext = st["thumb"]
        idx_ext = st["index"]
        mid_ext = st["middle"]
        ring_ext = st["ring"]
        pnk_ext = st["pinky"]

        # Landmark tip vectors
        t_tip = norm_pts[THUMB_TIP]
        i_tip = norm_pts[INDEX_TIP]
        m_tip = norm_pts[MIDDLE_TIP]
        r_tip = norm_pts[RING_TIP]
        p_tip = norm_pts[PINKY_TIP]
        wrist = norm_pts[WRIST]

        # Key inter-finger Euclidean distances
        t_i_dist = float(np.linalg.norm(t_tip - i_tip))
        i_m_dist = float(np.linalg.norm(i_tip - m_tip))
        m_r_dist = float(np.linalg.norm(m_tip - r_tip))
        r_p_dist = float(np.linalg.norm(r_tip - p_tip))
        t_m_dist = float(np.linalg.norm(t_tip - m_tip))
        t_p_dist = float(np.linalg.norm(t_tip - p_tip))
        t_r_dist = float(np.linalg.norm(t_tip - r_tip))

        # Camera-frame and unrotated orientation vectors
        raw_palm_vec = landmarks[MIDDLE_MCP, :2] - landmarks[WRIST, :2]
        raw_idx_vec = landmarks[INDEX_TIP, :2] - landmarks[INDEX_MCP, :2]
        raw_mid_vec = landmarks[MIDDLE_TIP, :2] - landmarks[MIDDLE_MCP, :2]
        raw_thumb_vec = landmarks[THUMB_TIP, :2] - landmarks[THUMB_MCP, :2]

        # Normalized orientation vectors
        vec_i = i_tip[:2] - norm_pts[INDEX_MCP, :2]
        vec_m = m_tip[:2] - norm_pts[MIDDLE_MCP, :2]
        vec_t = t_tip[:2] - norm_pts[THUMB_MCP, :2]

        # Downward check: in camera coordinates (+Y is down), hand or fingers point downward
        is_pointing_down = bool(
            (raw_palm_vec[1] > 0.05 and raw_idx_vec[1] > -0.05)
            or (raw_idx_vec[1] > 0.12)
            or (norm_pts[INDEX_TIP, 1] > norm_pts[INDEX_MCP, 1] + 0.15)
        )

        # Horizontal check: fingers point sideways
        is_pointing_horiz = bool(
            (abs(raw_palm_vec[0]) > abs(raw_palm_vec[1]) * 1.15)
            or (abs(raw_idx_vec[0]) > abs(raw_idx_vec[1]) * 1.15 and not is_pointing_down)
            or (abs(vec_i[0]) > abs(vec_i[1]) * 1.25 and not is_pointing_down)
        )

        # Upright check: fingers point upward
        is_pointing_up_i = bool(
            (raw_idx_vec[1] < -0.05 or vec_i[1] < -0.10) and not (is_pointing_down or is_pointing_horiz)
        )

        metadata = {
            "finger_states": finger_states,
            "thumb_index_dist": round(t_i_dist, 3),
            "index_middle_dist": round(i_m_dist, 3),
            "is_horizontal": is_pointing_horiz,
            "is_downward": is_pointing_down,
        }

        # =========================================================================
        # 1. DOWNWARD ORIENTATION FAMILY ('P', 'Q')
        # =========================================================================
        # 'P': Like 'K' handshape, but pointing downward
        if is_pointing_down and not (ring_ext or pnk_ext):
            d_thumb_mid_pip = float(np.linalg.norm(t_tip - norm_pts[MIDDLE_PIP]))
            if (t_m_dist < 0.45 or d_thumb_mid_pip < 0.45) and (idx_ext or raw_idx_vec[1] > 0.10):
                return "P", 0.94, metadata

        # 'Q': Like 'G' handshape, but pointing downward
        if is_pointing_down and not (mid_ext or ring_ext or pnk_ext):
            if t_i_dist < 0.50 and (idx_ext or raw_idx_vec[1] > 0.10):
                return "Q", 0.94, metadata

        # =========================================================================
        # 2. HORIZONTAL ORIENTATION FAMILY ('G', 'H')
        # =========================================================================
        # 'H': Index and middle fingers extended horizontally together
        if is_pointing_horiz and not (ring_ext or pnk_ext):
            if (idx_ext and mid_ext) or (abs(raw_mid_vec[0]) > abs(raw_mid_vec[1])):
                if i_m_dist < 0.35:
                    return "H", 0.94, metadata

        # 'G': Index finger and thumb extended horizontally, others closed
        if is_pointing_horiz and not (mid_ext or ring_ext or pnk_ext):
            if t_i_dist < 0.85:
                return "G", 0.94, metadata

        # =========================================================================
        # 3. VERTICAL TWO-FINGER UPRIGHT FAMILY ('U', 'V', 'R', 'K')
        # =========================================================================
        if idx_ext and mid_ext and not (ring_ext or pnk_ext) and not (is_pointing_down or is_pointing_horiz):
            # 'R': Index and middle fingers crossed over each other
            # In aligned coordinates, index crosses past middle to the ulnar (+X) side
            is_crossed = (aligned[INDEX_TIP, 0] > aligned[MIDDLE_TIP, 0] + 0.02)
            if is_crossed and i_m_dist < 0.35:
                return "R", 0.94, metadata

            # 'K': Index upright, middle forward or spread, thumb at middle knuckle
            d_thumb_mid_pip = float(np.linalg.norm(t_tip - norm_pts[MIDDLE_PIP]))
            if d_thumb_mid_pip < 0.40 and i_m_dist >= 0.20:
                return "K", 0.93, metadata

            # 'V': Index and middle upright spread apart in 'V'
            if i_m_dist >= 0.25:
                return "V", 0.95, metadata

            # 'U': Index and middle upright touching parallel together
            if i_m_dist < 0.25 and aligned[INDEX_TIP, 0] <= aligned[MIDDLE_TIP, 0] + 0.02:
                return "U", 0.94, metadata

        # =========================================================================
        # 4. THREE & FOUR FINGER EXTENDED FAMILY ('B', 'W', 'F')
        # =========================================================================
        # 'W': Index, middle, and ring extended upright spread apart
        if idx_ext and mid_ext and ring_ext and not pnk_ext:
            if i_m_dist >= 0.20 and m_r_dist >= 0.20:
                return "W", 0.94, metadata

        # 'B': Four fingers extended upright touching, thumb folded across palm
        if idx_ext and mid_ext and ring_ext and pnk_ext:
            if i_m_dist < 0.35 and m_r_dist < 0.35 and r_p_dist < 0.35:
                return "B", 0.94, metadata

        # 'F': Thumb and index touch forming circle; middle, ring, pinky extended upright
        if mid_ext and ring_ext and pnk_ext and not idx_ext and t_i_dist < 0.35:
            return "F", 0.94, metadata

        # =========================================================================
        # 5. SINGLE FINGER EXTENDED FAMILY ('D', 'L', 'Y', 'I', 'X')
        # =========================================================================
        # 'Y': Thumb and pinky extended wide ("shaka" shape)
        if thumb_ext and pnk_ext and not (idx_ext or mid_ext or ring_ext):
            if t_p_dist > 1.10:
                return "Y", 0.95, metadata

        # 'I': Pinky upright alone, others curled (static letter I)
        if pnk_ext and not (idx_ext or mid_ext or ring_ext or thumb_ext):
            return "I", 0.94, metadata

        # 'D': Index extended straight up; thumb and middle/ring tips touch in loop
        if idx_ext and not (mid_ext or ring_ext or pnk_ext):
            if (t_m_dist < 0.35 or t_r_dist < 0.35) and not (is_pointing_down or is_pointing_horiz):
                return "D", 0.93, metadata

        # 'L': Index upright, thumb ~90 deg horizontal, others curled
        if idx_ext and not (mid_ext or ring_ext or pnk_ext):
            if (thumb_ext or t_i_dist > 0.70) and t_m_dist > 0.40 and not (is_pointing_down or is_pointing_horiz):
                return "L", 0.95, metadata

        # 'X': Index hooked/bent at knuckle sticking up, others curled in fist
        idx_hooked = bool(
            norm_pts[INDEX_PIP, 1] < norm_pts[INDEX_MCP, 1] - 0.35
            and norm_pts[INDEX_TIP, 1] > norm_pts[INDEX_PIP, 1] + 0.10
            and norm_pts[MIDDLE_PIP, 1] >= norm_pts[MIDDLE_MCP, 1] - 0.20
        )
        if idx_hooked and not (mid_ext or ring_ext or pnk_ext):
            return "X", 0.92, metadata

        # =========================================================================
        # 6. CURVED HANDSHAPES ('C', 'O')
        # =========================================================================
        # 'O': All fingertips touch thumb forming a closed circle
        if (
            t_i_dist < 0.35
            and t_m_dist < 0.40
            and not (idx_ext or mid_ext or ring_ext or pnk_ext)
            and norm_pts[INDEX_PIP, 1] < norm_pts[INDEX_MCP, 1] - 0.20
        ):
            return "O", 0.92, metadata

        # 'C': Curved fingers and thumb forming open 'C' shape
        if (
            0.35 <= t_i_dist <= 0.90
            and not (idx_ext or mid_ext or ring_ext or pnk_ext)
            and t_p_dist < 1.0
            and norm_pts[INDEX_PIP, 1] < norm_pts[INDEX_MCP, 1] - 0.20
            and norm_pts[MIDDLE_PIP, 1] < norm_pts[MIDDLE_MCP, 1] - 0.20
        ):
            return "C", 0.91, metadata

        # =========================================================================
        # 7. FIST LETTERS FAMILY ('A', 'S', 'E', 'T', 'M', 'N')
        # =========================================================================
        # All 4 fingers are curled down
        if not (idx_ext or mid_ext or ring_ext or pnk_ext):
            t_x = aligned[THUMB_TIP, 0]
            t_y = aligned[THUMB_TIP, 1]
            idx_mcp_x = aligned[INDEX_MCP, 0]
            mid_mcp_x = aligned[MIDDLE_MCP, 0]
            ring_mcp_x = aligned[RING_MCP, 0]
            idx_mcp_y = aligned[INDEX_MCP, 1]
            mid_pip_pt = norm_pts[MIDDLE_PIP]

            # 'A': Thumb extended upright alongside index radial edge
            if t_x <= idx_mcp_x + 0.05 and t_y < aligned[THUMB_MCP, 1]:
                return "A", 0.93, metadata

            # 'E': All 4 fingertips curled tight down resting on the thumb beneath them
            if t_y > idx_mcp_y + 0.25 and t_x > idx_mcp_x + 0.05:
                return "E", 0.92, metadata

            # 'T': Thumb tucked under index (emerging between Index and Middle knuckles)
            if idx_mcp_x + 0.05 < t_x <= (mid_mcp_x + idx_mcp_x) / 2.0 + 0.02:
                return "T", 0.92, metadata

            # 'S': Thumb wrapped horizontally across front of all curled fingers (resting on Middle PIP)
            if (mid_mcp_x + idx_mcp_x) / 2.0 + 0.02 < t_x <= (ring_mcp_x + mid_mcp_x) / 2.0 - 0.03:
                return "S", 0.93, metadata

            # 'N': Thumb tucked under first two fingers (emerging between Middle and Ring knuckles)
            if (ring_mcp_x + mid_mcp_x) / 2.0 - 0.03 < t_x <= (ring_mcp_x + mid_mcp_x) / 2.0 + 0.05:
                return "N", 0.92, metadata

            # 'M': Thumb tucked under first three fingers (emerging between Ring and Pinky knuckles)
            if t_x > (ring_mcp_x + mid_mcp_x) / 2.0 + 0.05:
                return "M", 0.92, metadata

            # Default fist fallback
            return "S", 0.90, metadata

        # Default fallback for unrecognized pose
        return None, 0.20, metadata
