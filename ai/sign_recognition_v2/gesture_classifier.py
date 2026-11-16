"""ISL Alphabet & Gesture Classification Engine for SignBridge AI v2.
Integrates:
1. Dynamic stroke trajectory recognizer for motion letters (J, Z).
2. Deterministic geometric rule engine grounded in ISLRTC single- and two-hand configurations.
3. Scikit-Learn Random Forest multi-hand classifier (162 features).
4. Strict safety policies: conflict resolution (UNKNOWN on clash), mode isolation,
   and refusal to silently load legacy gesture models for ISL alphabet recognition.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from ai.sign_recognition_v2.asl_rules import ASLRuleBasedClassifierV2
from ai.sign_recognition_v2.config import (
    ASL_ALPHABET,
    ASL_ALPHABET_METADATA,
    ASL_COMPLIANCE_DISCLAIMER,
    CLASSIFIER_MODEL_PATH,
    DEFAULT_RECOGNITION_MODE,
    DEFAULT_SIGN_LANGUAGE,
    FEATURE_VERSION,
    ISL_ALPHABET,
    ISL_ALPHABET_METADATA,
    ISL_CLASSIFIER_MODEL_PATH,
    ISL_COMPLIANCE_DISCLAIMER,
    LEGACY_GESTURES_METADATA,
    MIN_RECOGNITION_CONFIDENCE,
    RecognitionMode,
    SignLanguage,
    SUPPORTED_ALPHABET,
    SUPPORTED_ASL_ALPHABET,
    SUPPORTED_GESTURES,
    TOTAL_FEATURE_DIM,
    UNKNOWN_GESTURE_THRESHOLD,
)
from ai.sign_recognition_v2.hand_detector import HandData, HandDetectionResultV2
from ai.sign_recognition_v2.landmark_extractor import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    LandmarkExtractorV2,
)
from ai.sign_recognition_v2.temporal_recognizer import TemporalRecognizerV2


class ISLRuleBasedClassifierV2:
    """Deterministic geometric rule engine based on verified ISLRTC alphabet handshapes."""

    def __init__(self, extractor: LandmarkExtractorV2, mode: RecognitionMode = RecognitionMode.ISL_ALPHABET):
        self.extractor = extractor
        self.mode = mode

    def classify_detection(
        self, detection: HandDetectionResultV2
    ) -> Tuple[Optional[str], float, Dict]:
        """Classifies a multi-hand detection result using ISLRTC geometry."""
        if not detection.detected or not detection.hands:
            return None, 0.0, {"error": "no_hands"}

        # Reject degraded / approximate fallback detections (e.g. OpenCV contour fallback)
        if getattr(detection, "is_degraded", False):
            return None, 0.0, {
                "source": "none",
                "status": "degraded_detection_rejected",
                "error": "degraded_detection_rejected",
                "explanation": (
                    "Approximate fallback landmarks cannot be used for ISL recognition. "
                    "Genuine anatomical MediaPipe hand landmarks are required."
                ),
            }

        primary_hand = detection.hands[0]
        secondary_hand = detection.hands[1] if len(detection.hands) >= 2 else None

        primary_pts = np.array(primary_hand.landmarks, dtype=np.float32)
        secondary_pts = (
            np.array(secondary_hand.landmarks, dtype=np.float32)
            if secondary_hand is not None
            else None
        )

        return self.classify_geometry(primary_pts, secondary_pts)

    def classify_geometry(
        self,
        primary_landmarks: np.ndarray,
        secondary_landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float, Dict]:
        """Evaluates normalized landmark geometry against defined ISL alphabet handshapes."""
        # Reject degenerate / collapsed landmark arrays (e.g. all zeros or flat dummy inputs)
        if float(np.ptp(primary_landmarks)) < 1e-3:
            return None, 0.0, {"error": "degenerate_hand_geometry"}

        norm_primary, p_scale = self.extractor.normalize_single_hand(primary_landmarks)
        p_scale = max(float(p_scale), 1e-4)
        states_p = self.extractor.compute_finger_states(norm_primary)

        tp = states_p["thumb"]
        ip = states_p["index"]
        mp = states_p["middle"]
        rp = states_p["ring"]
        pp = states_p["pinky"]

        # Primary hand normalized metrics
        t_i_dist = float(np.linalg.norm(norm_primary[THUMB_TIP] - norm_primary[INDEX_TIP]))
        i_m_dist = float(np.linalg.norm(norm_primary[INDEX_TIP] - norm_primary[MIDDLE_TIP]))
        m_r_dist = float(np.linalg.norm(norm_primary[MIDDLE_TIP] - norm_primary[RING_TIP]))
        t_p_dist = float(np.linalg.norm(norm_primary[THUMB_TIP] - norm_primary[PINKY_TIP]))
        t_wrist_y = float(norm_primary[THUMB_TIP][1] - norm_primary[WRIST][1])

        metadata = {
            "finger_states_primary": states_p,
            "thumb_index_dist": round(t_i_dist, 3),
            "index_middle_dist": round(i_m_dist, 3),
            "has_second_hand": secondary_landmarks is not None,
        }

        # -------------------------------------------------------------
        # ISL ALPHABET MODE (Strictly ISLRTC Alphabet A - Z)
        # -------------------------------------------------------------
        if self.mode == RecognitionMode.ISL_ALPHABET:
            # 1. TWO-HANDED ISLRTC RULES (When secondary hand is present)
            if secondary_landmarks is not None:
                norm_sec, s_scale = self.extractor.normalize_single_hand(secondary_landmarks)
                s_scale = max(float(s_scale), 1e-4)
                states_s = self.extractor.compute_finger_states(norm_sec)
                metadata["finger_states_secondary"] = states_s

                # Distances from primary index tip to secondary fingertips
                p_idx_tip = primary_landmarks[INDEX_TIP, :3]
                s_tips = [
                    float(np.linalg.norm(p_idx_tip - secondary_landmarks[tip_idx, :3]) / p_scale)
                    for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
                ]
                d_to_thumb, d_to_index, d_to_middle, d_to_ring, d_to_pinky = s_tips

                metadata["vowel_contact_distances"] = {
                    "to_thumb": round(d_to_thumb, 3),
                    "to_index": round(d_to_index, 3),
                    "to_middle": round(d_to_middle, 3),
                    "to_ring": round(d_to_ring, 3),
                    "to_pinky": round(d_to_pinky, 3),
                }

                s_t_i_dist = float(np.linalg.norm(norm_sec[THUMB_TIP] - norm_sec[INDEX_TIP]))

                # 1A. SYMMETRIC TWO-HANDED RULES
                # ISLRTC 'B' & 'Q': Both hands forming index-thumb loops
                if t_i_dist < 0.28 and s_t_i_dist < 0.28:
                    wrist_dist = float(np.linalg.norm(primary_landmarks[WRIST] - secondary_landmarks[WRIST]) / p_scale)
                    d_ii = float(np.linalg.norm(primary_landmarks[INDEX_TIP] - secondary_landmarks[INDEX_TIP]) / p_scale)
                    d_it1 = float(np.linalg.norm(primary_landmarks[INDEX_TIP] - secondary_landmarks[THUMB_TIP]) / p_scale)
                    d_it2 = float(np.linalg.norm(secondary_landmarks[INDEX_TIP] - primary_landmarks[THUMB_TIP]) / p_scale)
                    min_it = min(d_it1, d_it2)
                    min_loop_contact = min(d_ii, min_it)

                    dy_wrists = float(abs(primary_landmarks[WRIST, 1] - secondary_landmarks[WRIST, 1]) / p_scale)

                    # B: Both hands side-by-side touching at index-thumb loops (glasses shape)
                    if wrist_dist < 2.0 and dy_wrists < 0.40:
                        return "B", 0.93, metadata

                    # Q: Dominant circle hooks into non-dominant circle / thumb
                    if min_loop_contact < 0.50:
                        return "Q", 0.93, metadata

                # ISLRTC 'G': Two closed fists stacked vertically
                fist_p = not (ip or mp or rp or pp)
                fist_s = not (states_s["index"] or states_s["middle"] or states_s["ring"] or states_s["pinky"])
                if fist_p and fist_s:
                    mcp_dist = float(np.linalg.norm(primary_landmarks[MIDDLE_MCP] - secondary_landmarks[MIDDLE_MCP]) / p_scale)
                    if mcp_dist < 3.5:
                        dy = float(abs(primary_landmarks[WRIST, 1] - secondary_landmarks[WRIST, 1]) / p_scale)
                        dx = float(abs(primary_landmarks[WRIST, 0] - secondary_landmarks[WRIST, 0]) / p_scale)
                        if dy > 0.25 and dy >= dx * 0.5:
                            return "G", 0.93, metadata

                # ISLRTC 'S': Two pinkies interlocked
                pinky_only_p = pp and not (ip or mp or rp)
                pinky_only_s = states_s["pinky"] and not (states_s["index"] or states_s["middle"] or states_s["ring"])
                if pinky_only_p and pinky_only_s:
                    d_pp_tip = float(np.linalg.norm(primary_landmarks[PINKY_TIP] - secondary_landmarks[PINKY_TIP]) / p_scale)
                    d_pp_pip = float(np.linalg.norm(primary_landmarks[PINKY_PIP] - secondary_landmarks[PINKY_PIP]) / p_scale)
                    if min(d_pp_tip, d_pp_pip) < 0.45:
                        return "S", 0.93, metadata

                # ISLRTC 'X': Two index fingers crossed into an 'X'
                idx_only_p = ip and not (mp or rp or pp)
                idx_only_s = states_s["index"] and not (states_s["middle"] or states_s["ring"] or states_s["pinky"])
                if idx_only_p and idx_only_s:
                    d_xx_tip = float(np.linalg.norm(primary_landmarks[INDEX_TIP] - secondary_landmarks[INDEX_TIP]) / p_scale)
                    d_xx_pip = float(np.linalg.norm(primary_landmarks[INDEX_PIP] - secondary_landmarks[INDEX_PIP]) / p_scale)
                    if min(d_xx_tip, d_xx_pip) < 0.45:
                        return "X", 0.94, metadata

                # ISLRTC 'W': Two hands with spread fingers interlocking
                spread_p = ip and mp and rp
                spread_s = states_s["index"] and states_s["middle"] and states_s["ring"]
                if spread_p and spread_s:
                    wrist_dist = float(np.linalg.norm(primary_landmarks[WRIST] - secondary_landmarks[WRIST]) / p_scale)
                    d_mm = float(np.linalg.norm(primary_landmarks[MIDDLE_TIP] - secondary_landmarks[MIDDLE_TIP]) / p_scale)
                    d_ii = float(np.linalg.norm(primary_landmarks[INDEX_TIP] - secondary_landmarks[INDEX_TIP]) / p_scale)
                    if wrist_dist < 2.0 and min(d_mm, d_ii) < 0.60:
                        return "W", 0.94, metadata

                # 1B. ASYMMETRIC TWO-HANDED RULES (Evaluated across both hand roles for dominant-hand invariance)
                hand_pairs = [
                    (primary_landmarks, secondary_landmarks, norm_primary, norm_sec, states_p, states_s, p_scale),
                    (secondary_landmarks, primary_landmarks, norm_sec, norm_primary, states_s, states_p, s_scale),
                ]

                for dom, non, norm_d, norm_n, st_d, st_n, scale in hand_pairs:
                    dom_idx_tip = dom[INDEX_TIP, :3]
                    non_tips = [
                        float(np.linalg.norm(dom_idx_tip - non[tip_idx, :3]) / scale)
                        for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
                    ]
                    d_th, d_in, d_mi, d_ri, d_pi = non_tips
                    min_tip_dist = min(non_tips)

                    # ISLRTC Vowels: Dominant index tip contacts non-dominant spread fingertips
                    if st_d["index"] and not (st_d["middle"] or st_d["ring"] or st_d["pinky"]):
                        touch_threshold = 0.40
                        if d_th < touch_threshold and d_th < min(d_in, d_mi):
                            return "A", 0.94, metadata
                        if d_in < touch_threshold and d_in < min(d_th, d_mi):
                            return "E", 0.94, metadata
                        if d_mi < touch_threshold and d_mi < min(d_in, d_ri):
                            return "I", 0.93, metadata
                        if d_ri < touch_threshold and d_ri < min(d_mi, d_pi):
                            return "O", 0.92, metadata
                        if d_pi < touch_threshold and d_pi < min(d_ri, d_mi):
                            return "U", 0.93, metadata

                    # ISLRTC 'D': Non-dominant semicircle base; dominant index touches both tips
                    if st_d["index"] and not (st_d["middle"] or st_d["ring"]):
                        d_ti_non = float(np.linalg.norm(norm_n[THUMB_TIP] - norm_n[INDEX_TIP]))
                        if d_ti_non > 0.28:
                            d_base_thumb = float(np.linalg.norm(dom_idx_tip - non[THUMB_TIP, :3]) / scale)
                            d_base_index = float(np.linalg.norm(dom_idx_tip - non[INDEX_TIP, :3]) / scale)
                            if d_base_thumb < 0.45 and d_base_index < 0.45:
                                return "D", 0.92, metadata

                    # ISLRTC 'K' & 'P': Non-dominant upright index
                    if st_n["index"] and not (st_n["middle"] or st_n["ring"] or st_n["pinky"]):
                        d_t_i_dom = float(np.linalg.norm(norm_d[THUMB_TIP] - norm_d[INDEX_TIP]))
                        # 'P': Dominant index & thumb form downward loop touching non-dominant index
                        if d_t_i_dom < 0.28 and not (st_d["middle"] or st_d["ring"] or st_d["pinky"]):
                            d_contact_p = min(
                                float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_TIP]) / scale),
                                float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_PIP]) / scale),
                            )
                            if d_contact_p < 0.45:
                                return "P", 0.93, metadata
                        # 'K': Dominant hooked index contacts side/knuckle of non-dominant upright index
                        if not (st_d["middle"] or st_d["ring"] or st_d["pinky"]) and d_t_i_dom >= 0.22:
                            d_pip = float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_PIP]) / scale)
                            d_mcp = float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_MCP]) / scale)
                            if min(d_pip, d_mcp) < 0.45:
                                return "K", 0.92, metadata

                    # ISLRTC Palm Letters Family ('H', 'M', 'N', 'R', 'T')
                    # Non-dominant hand is flat open palm
                    if st_n["index"] and st_n["middle"] and st_n["ring"] and st_n["pinky"]:
                        d_palm_wrist = float(np.linalg.norm(dom[INDEX_TIP] - non[WRIST]) / scale)
                        d_palm_mcp = float(np.linalg.norm(dom[INDEX_TIP] - non[MIDDLE_MCP]) / scale)
                        d_palm_center = min(d_palm_wrist, d_palm_mcp)

                        # Ensure contact is on the palm/base, not touching fingertips (which are vowels)
                        if min_tip_dist > 0.30:
                            # 'H': Flat open palm brushing across non-dominant palm
                            if st_d["index"] and st_d["middle"] and st_d["ring"] and st_d["pinky"]:
                                if d_palm_center < 0.85:
                                    return "H", 0.92, metadata

                            # 'M': Three fingers (index, middle, ring) resting on palm
                            elif st_d["index"] and st_d["middle"] and st_d["ring"] and not st_d["pinky"]:
                                d_mid_palm = min(
                                    float(np.linalg.norm(dom[MIDDLE_TIP] - non[WRIST]) / scale),
                                    float(np.linalg.norm(dom[MIDDLE_TIP] - non[MIDDLE_MCP]) / scale),
                                )
                                if d_mid_palm < 0.85 or d_palm_center < 0.85:
                                    return "M", 0.94, metadata

                            # 'N': Two fingers (index, middle) resting on palm
                            elif st_d["index"] and st_d["middle"] and not (st_d["ring"] or st_d["pinky"]):
                                if d_palm_center < 0.85:
                                    return "N", 0.94, metadata

                            # 'T' vs 'R': One finger (index) on palm
                            elif st_d["index"] and not (st_d["middle"] or st_d["ring"] or st_d["pinky"]):
                                d_edge = min(
                                    float(np.linalg.norm(dom[INDEX_TIP] - non[PINKY_MCP]) / scale),
                                    float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_MCP]) / scale),
                                )
                                # 'T': Index touches side edge of palm (ulnar/radial boundary)
                                if d_edge < 0.48 and d_edge < d_palm_mcp:
                                    return "T", 0.92, metadata
                                # 'R': Curved index resting on palm center
                                elif d_palm_center < 0.85:
                                    return "R", 0.93, metadata

                    # ISLRTC 'F': Dominant index & middle fingers cross over non-dominant index & middle
                    if st_d["index"] and st_d["middle"] and not (st_d["ring"] or st_d["pinky"]):
                        if st_n["index"] and st_n["middle"] and not st_n["ring"]:
                            d_im = float(np.linalg.norm(dom[INDEX_TIP] - non[INDEX_TIP]) / scale)
                            if d_im < 0.45:
                                return "F", 0.92, metadata

                    # ISLRTC 'Y' (two-handed): Dominant index placed between non-dominant thumb and index V-cleft
                    if st_d["index"] and not (st_d["middle"] or st_d["ring"] or st_d["pinky"]):
                        d_ti_non = float(np.linalg.norm(norm_n[THUMB_TIP] - norm_n[INDEX_TIP]))
                        if st_n["thumb"] and st_n["index"] and d_ti_non > 0.35:
                            cleft_pt = (non[INDEX_MCP] + non[THUMB_MCP]) / 2.0
                            d_cleft = float(np.linalg.norm(dom[INDEX_TIP] - cleft_pt) / scale)
                            if d_cleft < 0.45:
                                return "Y", 0.93, metadata

            # 2. SINGLE-HANDED ISLRTC RULES (and fallback single-hand configurations)
            # 'L': Index upright, thumb ~90 deg horizontal, others curled
            if tp and ip and not mp and not rp and not pp:
                if t_i_dist > 0.38:
                    return "L", 0.95, metadata

            # 'V': Index and middle upright spread apart, others curled
            if ip and mp and not rp and not pp and not tp:
                if i_m_dist >= 0.14:
                    return "V", 0.95, metadata
                else:
                    return "U", 0.92, metadata  # Index & middle upright touching parallel

            # 'I': Pinky upright alone, others curled
            if pp and not ip and not mp and not rp and not tp:
                return "I", 0.94, metadata

            # 'Y': Thumb and pinky extended wide, others curled
            if tp and pp and not ip and not mp and not rp:
                if t_p_dist > 0.48:
                    return "Y", 0.95, metadata

            # 'C': Curved fingers and thumb forming open 'C' shape
            if not (ip and mp and rp and pp) and 0.22 < t_i_dist < 0.50:
                # Check for curved arc
                if norm_primary[INDEX_TIP, 0] > norm_primary[INDEX_PIP, 0] * 0.7:
                    return "C", 0.90, metadata

            # 'O': Fingertips touching thumb forming a circle
            if t_i_dist < 0.22 and not (ip and mp and rp and pp):
                t_mid = float(np.linalg.norm(norm_primary[THUMB_TIP] - norm_primary[MIDDLE_TIP]))
                if t_mid < 0.26:
                    return "O", 0.91, metadata

            # 'W': Index, middle, and ring extended upright spread apart
            if ip and mp and rp and not pp and not tp:
                return "W", 0.93, metadata

            # 'D' (single hand variant): Index upright, thumb touches middle tip
            if ip and not mp and not rp and not pp:
                t_mid = float(np.linalg.norm(norm_primary[THUMB_TIP] - norm_primary[MIDDLE_TIP]))
                if t_mid < 0.32:
                    return "D", 0.91, metadata

            # 'B' (single hand variant): All 4 fingers upright together, thumb curled
            if ip and mp and rp and pp and not tp:
                if i_m_dist < 0.15 and m_r_dist < 0.15:
                    return "B", 0.91, metadata

            # 'F' (single hand variant): Thumb & index touch in loop, middle/ring/pinky upright
            if mp and rp and pp and t_i_dist < 0.22:
                return "F", 0.91, metadata

            # 'A' (single hand fist with thumb upright alongside)
            if not ip and not mp and not rp and not pp:
                if tp and t_wrist_y < 0:
                    return "A", 0.90, metadata

        # -------------------------------------------------------------
        # LEGACY GESTURES MODE (Isolated demo gestures)
        # -------------------------------------------------------------
        elif self.mode == RecognitionMode.LEGACY_GESTURES:
            if tp and ip and mp and rp and pp:
                if t_i_dist > 0.38:
                    return "HELLO", 0.94, metadata
                return "STOP", 0.92, metadata

            if tp and not ip and not mp and not rp and not pp:
                if t_wrist_y < 0:
                    return "YES", 0.93, metadata
                return "HELP", 0.85, metadata

            if not rp and not pp and t_i_dist < 0.28:
                return "NO", 0.91, metadata

            if tp and ip and not mp and not rp and pp:
                return "I_LOVE_YOU", 0.96, metadata

        # Ambiguous / Unrecognized hand pose
        return None, 0.20, metadata


class ISLMLClassifierV2:
    """Scikit-Learn Random Forest model for data-driven ISL alphabet classification.
    
    Enforces strict 8-point model and metadata validation in ISL Alphabet mode.
    Safely rejects incompatible, unverified, or unprovenanced models.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        mode: RecognitionMode = RecognitionMode.ISL_ALPHABET,
    ):
        self.mode = mode
        if model_path is not None:
            self.model_path = Path(model_path)
        elif mode == RecognitionMode.ISL_ALPHABET:
            # In ISL mode: strictly point to ISL classifier; NEVER silently fall back to legacy
            self.model_path = ISL_CLASSIFIER_MODEL_PATH
        else:
            self.model_path = CLASSIFIER_MODEL_PATH

        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.metadata: Optional[Dict] = None
        self._load_model()

    def _load_model(self) -> bool:
        self.model = None
        self.is_loaded = False
        self.load_error = None
        self.metadata = None

        if not self.model_path.exists():
            self.load_error = f"Model file not found at {self.model_path}"
            return False

        # In legacy mode, load legacy model if available
        if self.mode == RecognitionMode.LEGACY_GESTURES:
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                return True
            except Exception as e:
                self.load_error = f"Failed to load legacy model: {e}"
                return False

        # --- STRICT ISL ALPHABET MODEL VALIDATION (8 Requirements) ---
        # 1. Model file can be loaded
        try:
            loaded_model = joblib.load(self.model_path)
        except Exception as e:
            self.load_error = f"Failed to deserialize model file from {self.model_path}: {e}"
            return False

        # 2. n_features_in_ equals TOTAL_FEATURE_DIM (162)
        n_features = getattr(loaded_model, "n_features_in_", None)
        if n_features != TOTAL_FEATURE_DIM:
            self.load_error = (
                f"Model feature dimension mismatch: model has n_features_in_={n_features}, "
                f"expected TOTAL_FEATURE_DIM={TOTAL_FEATURE_DIM} (162)."
            )
            return False

        # 3. Model classes are valid members of SUPPORTED_ALPHABET
        model_classes = getattr(loaded_model, "classes_", None)
        if model_classes is None or len(model_classes) == 0:
            self.load_error = "Model has no classes_ attribute or classes list is empty."
            return False

        model_classes_set = set(str(c) for c in model_classes)
        invalid_classes = [c for c in model_classes_set if c not in SUPPORTED_ALPHABET]
        if invalid_classes:
            self.load_error = (
                f"Model contains invalid or non-ISL class labels: {invalid_classes}. "
                f"All model classes must be valid members of SUPPORTED_ALPHABET."
            )
            return False

        # 4. Required metadata JSON exists
        meta_path = self.model_path.with_suffix(".json")
        if not meta_path.exists():
            self.load_error = (
                f"Required metadata JSON sidecar not found at {meta_path}."
            )
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
        except Exception as e:
            self.load_error = f"Failed to parse metadata JSON at {meta_path}: {e}"
            return False

        # 5. Metadata feature dimension matches 162
        meta_dim = meta_json.get("feature_dimension")
        if meta_dim != TOTAL_FEATURE_DIM:
            self.load_error = (
                f"Metadata feature_dimension mismatch: found {meta_dim}, "
                f"expected {TOTAL_FEATURE_DIM} (162)."
            )
            return False

        # 6. Metadata feature version matches FEATURE_VERSION
        meta_version = meta_json.get("feature_version")
        if meta_version != FEATURE_VERSION:
            self.load_error = (
                f"Metadata feature_version mismatch: found '{meta_version}', "
                f"expected '{FEATURE_VERSION}'."
            )
            return False

        # 7. Metadata class labels match the model's classes
        meta_classes = meta_json.get("classes") or meta_json.get("trained_class_labels")
        if not meta_classes:
            self.load_error = "Metadata is missing 'classes' or 'trained_class_labels' list."
            return False

        meta_classes_set = set(str(c) for c in meta_classes)
        if meta_classes_set != model_classes_set:
            self.load_error = (
                f"Metadata classes {sorted(list(meta_classes_set))} do not match "
                f"model classes {sorted(list(model_classes_set))}."
            )
            return False

        # 8. Metadata identifies the training dataset and its provenance
        has_dataset_ident = bool(meta_json.get("dataset_path") or meta_json.get("dataset_sha256"))
        has_provenance = bool(
            meta_json.get("source_type_counts")
            or meta_json.get("sources")
            or ("is_benchmark_mode" in meta_json)
        )
        if not (has_dataset_ident and has_provenance):
            self.load_error = (
                "Metadata does not adequately identify training dataset and provenance "
                "(requires dataset_path/dataset_sha256 and source provenance)."
            )
            return False

        # All 8 validation checks passed!
        self.model = loaded_model
        self.is_loaded = True
        self.metadata = meta_json
        return True

    def predict(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        if not self.is_loaded or self.model is None:
            return None, 0.0, {
                "error": "model_not_loaded",
                "reason": self.load_error or "Model validation failed or model not found",
            }

        features = feature_vector.reshape(1, -1)
        try:
            pred_class = str(self.model.predict(features)[0])
            confidence = 0.80
            proba_dict = {}

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                confidence = float(np.max(probs))
                proba_dict = {str(c): float(p) for c, p in zip(classes, probs)}

            return pred_class, round(confidence, 2), {"probabilities": proba_dict}
        except Exception as e:
            return None, 0.0, {"error": str(e)}


class GestureClassifierV2:
    """Master ISL Alphabet & Gesture Classifier for SignBridge AI v2.
    
    Integrates:
    1. Dynamic Stroke Trajectory Recognizer (J, Z)
    2. Deterministic ISL Geometric Rule Engine
    3. Scikit-Learn Trained Random Forest Classifier
    4. Conflict Arbiter (returns UNKNOWN on conflicting high-confidence predictions)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        mode: RecognitionMode = DEFAULT_RECOGNITION_MODE,
        language: Union[SignLanguage, str] = DEFAULT_SIGN_LANGUAGE,
    ):
        self.mode = mode
        if isinstance(language, SignLanguage):
            self.language = language
        else:
            val = str(language).upper().split(".")[-1]
            self.language = SignLanguage(val)

        self.extractor = LandmarkExtractorV2()
        self.rule_classifier = ISLRuleBasedClassifierV2(self.extractor, mode=mode)
        self.asl_rule_classifier = ASLRuleBasedClassifierV2(self.extractor)
        self.ml_classifier = ISLMLClassifierV2(model_path=model_path, mode=mode)
        self.temporal_recognizer = TemporalRecognizerV2()
        self.min_confidence = MIN_RECOGNITION_CONFIDENCE
        self.unknown_threshold = UNKNOWN_GESTURE_THRESHOLD

    def set_language(self, language: Union[SignLanguage, str]) -> None:
        """Switches active sign language recognition mode between ISL and ASL."""
        if isinstance(language, SignLanguage):
            self.language = language
        else:
            val = str(language).upper().split(".")[-1]
            self.language = SignLanguage(val)
        self.temporal_recognizer.reset()

    def classify_detection(
        self, detection: HandDetectionResultV2
    ) -> Tuple[Optional[str], float, bool, Dict]:
        """Classifies a multi-hand detection result into an ISL/ASL letter or supported gesture."""
        disclaimer = (
            ASL_COMPLIANCE_DISCLAIMER
            if self.language == SignLanguage.ASL
            else ISL_COMPLIANCE_DISCLAIMER
        )

        if not detection.detected or not detection.hands:
            return None, 0.0, False, {
                "source": "none",
                "status": "no_hand_detected",
                "language": self.language.value,
                "disclaimer": disclaimer,
            }

        # Reject degraded / approximate fallback detections (e.g. OpenCV contour fallback)
        if getattr(detection, "is_degraded", False):
            return None, 0.0, False, {
                "source": "none",
                "status": "degraded_detection_rejected",
                "explanation": (
                    f"Approximate fallback landmarks cannot be used for {self.language.value} recognition. "
                    "Genuine anatomical MediaPipe hand landmarks are required."
                ),
                "language": self.language.value,
                "disclaimer": disclaimer,
            }

        primary_hand = detection.hands[0]
        pts = np.array(primary_hand.landmarks, dtype=np.float32)

        # Extract 162-dimensional feature vector across available hands
        feature_vec = self.extractor.extract_features(detection)
        norm_pts, _ = self.extractor.normalize_single_hand(pts)
        finger_states = self.extractor.compute_finger_states(norm_pts)

        # 1. Temporal / Dynamic Letter Stroke Check (J, Z)
        if self.mode == RecognitionMode.ISL_ALPHABET or self.language == SignLanguage.ASL:
            self.temporal_recognizer.add_frame(norm_pts)
            dyn_label, dyn_conf, dyn_meta = self.temporal_recognizer.evaluate_dynamic_gesture(
                finger_states=finger_states, language=self.language.value
            )
            if dyn_label is not None and dyn_conf >= self.min_confidence:
                return dyn_label, dyn_conf, True, {
                    "source": "temporal_trajectory",
                    "status": "experimental_dynamic",
                    "stroke_meta": dyn_meta,
                    "language": self.language.value,
                    "disclaimer": disclaimer,
                }

        # 2. Geometric Rule Engine Evaluation based on active language
        if self.language == SignLanguage.ASL:
            rule_label, rule_conf, rule_meta = self.asl_rule_classifier.classify_detection(detection)
            rule_meta["language"] = "ASL"
            rule_meta["disclaimer"] = disclaimer
            is_rec = bool(rule_label is not None and rule_conf >= self.min_confidence)
            return rule_label, rule_conf, is_rec, {
                "source": "asl_rule_engine",
                "status": "recognized" if is_rec else "unknown_gesture",
                "metadata": rule_meta,
                "language": "ASL",
                "disclaimer": disclaimer,
            }
        else:
            # ISL Mode
            rule_label, rule_conf, rule_meta = self.rule_classifier.classify_detection(detection)
            rule_meta["language"] = "ISL"

            # 3. ML Model Evaluation
            ml_label, ml_conf, ml_meta = None, 0.0, {}
            if self.ml_classifier.is_loaded:
                ml_label, ml_conf, ml_meta = self.ml_classifier.predict(feature_vec)

            return self._arbitrate_predictions(rule_label, rule_conf, rule_meta, ml_label, ml_conf, ml_meta)

    def classify_landmarks(
        self, landmarks: np.ndarray
    ) -> Tuple[Optional[str], float, bool, Dict]:
        """Backward-compatible entry point for array of landmarks (21, 3)."""
        disclaimer = (
            ASL_COMPLIANCE_DISCLAIMER
            if self.language == SignLanguage.ASL
            else ISL_COMPLIANCE_DISCLAIMER
        )

        pts = np.array(landmarks, dtype=np.float32)
        norm_pts, _ = self.extractor.normalize_single_hand(pts)
        finger_states = self.extractor.compute_finger_states(norm_pts)
        feature_vec = self.extractor.extract_features(pts)

        # 1. Temporal / Dynamic Letter Stroke Check (J, Z)
        if self.mode == RecognitionMode.ISL_ALPHABET or self.language == SignLanguage.ASL:
            self.temporal_recognizer.add_frame(norm_pts)
            dyn_label, dyn_conf, dyn_meta = self.temporal_recognizer.evaluate_dynamic_gesture(
                finger_states=finger_states, language=self.language.value
            )
            if dyn_label is not None and dyn_conf >= self.min_confidence:
                return dyn_label, dyn_conf, True, {
                    "source": "temporal_trajectory",
                    "status": "experimental_dynamic",
                    "stroke_meta": dyn_meta,
                    "language": self.language.value,
                    "disclaimer": disclaimer,
                }

        # 2. Geometric Rule Engine Evaluation
        if self.language == SignLanguage.ASL:
            rule_label, rule_conf, rule_meta = self.asl_rule_classifier.classify_geometry(pts)
            rule_meta["language"] = "ASL"
            rule_meta["disclaimer"] = disclaimer
            is_rec = bool(rule_label is not None and rule_conf >= self.min_confidence)
            return rule_label, rule_conf, is_rec, {
                "source": "asl_rule_engine",
                "status": "recognized" if is_rec else "unknown_gesture",
                "metadata": rule_meta,
                "language": "ASL",
                "disclaimer": disclaimer,
            }
        else:
            rule_label, rule_conf, rule_meta = self.rule_classifier.classify_geometry(pts)
            rule_meta["language"] = "ISL"

            ml_label, ml_conf, ml_meta = None, 0.0, {}
            if self.ml_classifier.is_loaded:
                ml_label, ml_conf, ml_meta = self.ml_classifier.predict(feature_vec)

            return self._arbitrate_predictions(rule_label, rule_conf, rule_meta, ml_label, ml_conf, ml_meta)

    def _arbitrate_predictions(
        self,
        rule_label: Optional[str],
        rule_conf: float,
        rule_meta: Dict,
        ml_label: Optional[str],
        ml_conf: float,
        ml_meta: Dict,
    ) -> Tuple[Optional[str], float, bool, Dict]:
        """Arbitrates between Geometric Rule and ML Model predictions.
        
        Strict Safety Rule:
        If both Rule and ML make strong predictions that CONFLICT with each other,
        the arbiter rejects both and safely returns UNKNOWN with status 'conflicting_predictions'.
        """
        chosen_label = None
        chosen_conf = 0.0
        source = "none"
        status = "recognized"

        # Check for direct consensus
        if rule_label is not None and ml_label is not None:
            if rule_label == ml_label:
                chosen_label = rule_label
                chosen_conf = min(0.99, max(rule_conf, ml_conf) + 0.05)
                source = "hybrid_consensus"
                status = "consensus"
            else:
                # CONFLICT CHECK:
                # If both have high confidence and disagree, do not arbitrarily pick one!
                if rule_conf >= self.min_confidence and ml_conf >= self.min_confidence:
                    return None, 0.30, False, {
                        "source": "conflict_arbiter",
                        "status": "conflicting_predictions",
                        "rule_prediction": rule_label,
                        "rule_confidence": rule_conf,
                        "ml_prediction": ml_label,
                        "ml_confidence": ml_conf,
                        "disclaimer": ISL_COMPLIANCE_DISCLAIMER,
                    }
                elif rule_conf >= self.min_confidence:
                    chosen_label = rule_label
                    chosen_conf = rule_conf
                    source = "rule_engine"
                elif ml_conf >= self.min_confidence:
                    chosen_label = ml_label
                    chosen_conf = ml_conf
                    source = "ml_model"
                else:
                    chosen_label = None
                    chosen_conf = 0.25
                    source = "unrecognized"
        elif rule_label is not None and rule_conf >= self.min_confidence:
            chosen_label = rule_label
            chosen_conf = rule_conf
            source = "rule_engine"
        elif ml_label is not None and ml_conf >= self.min_confidence:
            chosen_label = ml_label
            chosen_conf = ml_conf
            source = "ml_model"
        else:
            chosen_label = None
            chosen_conf = max(rule_conf, ml_conf)
            source = "unrecognized"

        # Vocabulary verification against current mode
        if self.mode == RecognitionMode.ISL_ALPHABET:
            is_valid_vocab = chosen_label in SUPPORTED_ALPHABET or chosen_label in ISL_ALPHABET
            if chosen_label:
                meta = ISL_ALPHABET_METADATA.get(chosen_label, {})
                if meta.get("status") == "unsupported":
                    return None, 0.25, False, {
                        "source": source,
                        "status": "unsupported_letter",
                        "letter": chosen_label,
                        "disclaimer": ISL_COMPLIANCE_DISCLAIMER,
                    }
        else:
            is_valid_vocab = chosen_label in SUPPORTED_GESTURES

        is_recognized = (
            chosen_label is not None
            and is_valid_vocab
            and chosen_conf >= self.min_confidence
        )

        final_label = chosen_label if is_recognized else None
        final_conf = chosen_conf if is_recognized else min(chosen_conf, 0.35)

        metadata = {
            "source": source,
            "status": status if is_recognized else "unrecognized",
            "rule_prediction": rule_label,
            "rule_confidence": rule_conf,
            "ml_prediction": ml_label,
            "ml_confidence": ml_conf,
            "mode": str(self.mode),
            "language": self.language.value,
            "disclaimer": ISL_COMPLIANCE_DISCLAIMER,
            "raw_metrics": {**rule_meta, **ml_meta},
        }

        return final_label, round(final_conf, 2), is_recognized, metadata
