"""Configuration, Vocabulary Metadata, and ISLRTC Architectural Settings for SignBridge AI v2.
Defines the official Indian Sign Language (ISL) Alphabet mapping, multi-hand feature dimensions,
temporal thresholds, buffers, and model/dataset paths.
"""

from enum import Enum
from pathlib import Path
import sys

# Base Module Paths (Self-Contained)
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = MODULE_DIR / "models"
DATA_DIR = MODULE_DIR / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Isolated Model & Dataset Paths
TASK_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
ISL_CLASSIFIER_MODEL_PATH = MODELS_DIR / "isl_alphabet_classifier.joblib"
ISL_DATASET_PATH = DATA_DIR / "isl_alphabet.csv"

# Legacy paths preserved for backward compatibility
CLASSIFIER_MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
DATASET_PATH = DATA_DIR / "gestures.csv"

# Current Multi-Hand Feature Version
FEATURE_VERSION = "v2.3_isl_multihand"

# MediaPipe Standard Landmark Count
NUM_LANDMARKS = 21

# Recognition Mode & Supported Languages
class RecognitionMode(str, Enum):
    ISL_ALPHABET = "isl_alphabet"
    LEGACY_GESTURES = "legacy_gestures"

DEFAULT_RECOGNITION_MODE = RecognitionMode.ISL_ALPHABET

class SignLanguage(str, Enum):
    ISL = "ISL"
    ASL = "ASL"

DEFAULT_SIGN_LANGUAGE = SignLanguage.ISL

# Clear Categorization & Transparency Disclaimers
ISL_COMPLIANCE_DISCLAIMER = (
    "DISCLAIMER: This system is an Indian Sign Language (ISL) alphabet fingerspelling prototype "
    "grounded in the official ISLRTC curriculum. It recognizes discrete letter handshapes and combines them "
    "into words and sentences. It does NOT claim full conversational ISL grammar, syntax, or complete translation."
)

ASL_COMPLIANCE_DISCLAIMER = (
    "DISCLAIMER: This system is an American Sign Language (ASL) alphabet fingerspelling prototype. "
    "It recognizes discrete letter handshapes and combines them into words and sentences. "
    "It does NOT claim full conversational ASL grammar, syntax, or complete translation."
)

# Full ISL Alphabet Definition (A - Z)
ISL_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Full ASL Alphabet Definition (A - Z)
ASL_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Official ISLRTC Alphabet Reference Mapping
# Based on the official ISLRTC (Indian Sign Language Research and Training Centre) fingerspelling chart.
# Every letter has an explicit hand requirement, motion requirement, and verified/experimental status.
ISL_ALPHABET_METADATA = {
    "A": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, pointing to non-dominant thumb tip",
        "non_dominant_hand": "Open flat hand with thumb spread",
        "relative_position": "Dominant index tip contacts non-dominant thumb tip",
        "status": "experimental",  # Modeled, awaiting live signer demonstration
    },
    "B": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Thumb and fingers curved into loops adjacent to non-dominant hand (glasses shape)",
        "non_dominant_hand": "Thumb and index curved into loop touching dominant hand loops",
        "relative_position": "Two hands touching at index-thumb loops in front of chest",
        "status": "experimental",
    },
    "C": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Fingers and thumb curved into an open 'C' shape facing sideways",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright",
        "status": "experimental",
    },
    "D": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended upright, touching non-dominant index/thumb tips",
        "non_dominant_hand": "Thumb and index forming semicircle base of 'D'",
        "relative_position": "Dominant vertical index closes non-dominant semicircle into 'D'",
        "status": "experimental",
    },
    "E": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, pointing to non-dominant index fingertip",
        "non_dominant_hand": "Open flat hand with fingers spread",
        "relative_position": "Dominant index tip contacts non-dominant index fingertip",
        "status": "experimental",
    },
    "F": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended forward together",
        "non_dominant_hand": "Index and middle fingers extended forward; dominant fingers rest across them",
        "relative_position": "Dominant fingers cross perpendicularly over non-dominant fingers",
        "status": "experimental",
    },
    "G": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Closed fist resting on top of non-dominant fist",
        "non_dominant_hand": "Closed fist held horizontal base",
        "relative_position": "Two fists stacked vertically",
        "status": "experimental",
    },
    "H": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Flat open palm brushing forward across non-dominant palm",
        "non_dominant_hand": "Flat open palm facing upward/forward",
        "relative_position": "Dominant palm strokes across non-dominant palm",
        "status": "experimental",
    },
    "I": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, pointing to non-dominant middle fingertip",
        "non_dominant_hand": "Open flat hand with fingers spread",
        "relative_position": "Dominant index tip contacts non-dominant middle fingertip",
        "status": "experimental",
    },
    "J": {
        "hands_required": 1,
        "is_dynamic": True,
        "dominant_hand": "Index or pinky extended, tracing a downward curve and hook in the air",
        "non_dominant_hand": None,
        "relative_position": "Continuous trajectory stroke in camera view",
        "status": "experimental_dynamic",
    },
    "K": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger bent into hook, resting at the non-dominant index knuckle",
        "non_dominant_hand": "Index finger pointing upright",
        "relative_position": "Dominant hooked index contacts side of non-dominant upright index",
        "status": "experimental",
    },
    "L": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index finger pointing straight up, thumb extended horizontally at 90 degrees forming 'L'",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "M": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Three fingers (index, middle, ring) resting on non-dominant flat palm",
        "non_dominant_hand": "Flat open palm facing upward",
        "relative_position": "Dominant three fingertips placed on non-dominant palm",
        "status": "experimental",
    },
    "N": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Two fingers (index, middle) resting on non-dominant flat palm",
        "non_dominant_hand": "Flat open palm facing upward",
        "relative_position": "Dominant two fingertips placed on non-dominant palm",
        "status": "experimental",
    },
    "O": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, pointing to non-dominant ring fingertip",
        "non_dominant_hand": "Open flat hand with fingers spread",
        "relative_position": "Dominant index tip contacts non-dominant ring fingertip",
        "status": "experimental",
    },
    "P": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index and thumb forming loop, pointing downward against non-dominant index",
        "non_dominant_hand": "Index finger pointing upright",
        "relative_position": "Downward dominant loop contacts non-dominant upright index",
        "status": "experimental",
    },
    "Q": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Thumb and index forming circle hooking onto non-dominant thumb",
        "non_dominant_hand": "Thumb and index forming circle",
        "relative_position": "Dominant circle hooks into non-dominant circle",
        "status": "experimental",
    },
    "R": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger curved into small hook resting on non-dominant palm",
        "non_dominant_hand": "Flat open palm facing upward",
        "relative_position": "Dominant curved finger contacts non-dominant palm",
        "status": "experimental",
    },
    "S": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Pinky finger hooked around non-dominant pinky finger",
        "non_dominant_hand": "Fist with pinky extended, hooking dominant pinky",
        "relative_position": "Two pinkies interlocked",
        "status": "experimental",
    },
    "T": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended horizontally, touching non-dominant palm edge",
        "non_dominant_hand": "Flat open hand vertical, thumb side facing dominant hand",
        "relative_position": "Dominant index tip touches side edge of non-dominant hand",
        "status": "experimental",
    },
    "U": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, pointing to non-dominant pinky fingertip",
        "non_dominant_hand": "Open flat hand with fingers spread",
        "relative_position": "Dominant index tip contacts non-dominant pinky fingertip",
        "status": "experimental",
    },
    "V": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended upright spread apart in a 'V' shape",
        "non_dominant_hand": None,
        "relative_position": "Single hand upright facing forward",
        "status": "experimental",
    },
    "W": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Fingers spread, interlocking with non-dominant spread fingers",
        "non_dominant_hand": "Fingers spread, palms facing each other",
        "relative_position": "Both hands interlocked at fingers",
        "status": "experimental",
    },
    "X": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, crossing over non-dominant index finger",
        "non_dominant_hand": "Index finger extended upright",
        "relative_position": "Two index fingers crossed into an 'X'",
        "status": "experimental",
    },
    "Y": {
        "hands_required": 2,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended, placed between non-dominant thumb and index V-cleft",
        "non_dominant_hand": "Open hand with thumb and index spread wide forming a 'V'",
        "relative_position": "Dominant index rests in the cleft between non-dominant thumb and index",
        "status": "experimental",
    },
    "Z": {
        "hands_required": 1,
        "is_dynamic": True,
        "dominant_hand": "Index finger extended, tracing a zig-zag 'Z' shape in the air",
        "non_dominant_hand": None,
        "relative_position": "Continuous trajectory stroke in camera view",
        "status": "experimental_dynamic",
    },
}

# Official ASL Alphabet Reference Mapping (American Sign Language A - Z)
ASL_ALPHABET_METADATA = {
    "A": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Closed fist with thumb extended upright alongside the index finger",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "B": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Four fingers extended vertically and touching; thumb folded across palm",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "C": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Fingers and thumb curved to form an open 'C' shape facing sideways",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright",
        "status": "experimental",
    },
    "D": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended straight up; thumb and other fingertips touch in a loop",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "E": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "All four fingers curled down tight; thumb tucked across horizontally under fingertips",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "F": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Thumb and index tips touching in a circle; middle, ring, pinky extended upright",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "G": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index finger and thumb extended horizontally pointing sideways; others curled into fist",
        "non_dominant_hand": None,
        "relative_position": "Single hand held horizontally pointing across chest",
        "status": "experimental",
    },
    "H": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended horizontally together; others curled into fist",
        "non_dominant_hand": None,
        "relative_position": "Single hand held horizontally pointing across chest",
        "status": "experimental",
    },
    "I": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Pinky finger extended upright alone; other fingers and thumb curled into fist",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "J": {
        "hands_required": 1,
        "is_dynamic": True,
        "dominant_hand": "Pinky finger extended, tracing a downward hook curve ('J') in the air",
        "non_dominant_hand": None,
        "relative_position": "Continuous trajectory stroke in camera view",
        "status": "experimental_dynamic",
    },
    "K": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index upright, middle forward/angled, thumb resting at middle finger base/joint",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "L": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index finger extended upright, thumb extended horizontally at 90 degrees forming 'L'",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "M": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Closed fist with thumb tucked under the first three fingers (emerging between ring & pinky)",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "N": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Closed fist with thumb tucked under the first two fingers (emerging between middle & ring)",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "O": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "All fingertips and thumb curved touching together to form an 'O' circle",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "P": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index pointing forward, middle pointing downward, thumb between them (downward K)",
        "non_dominant_hand": None,
        "relative_position": "Single hand angled downward",
        "status": "experimental",
    },
    "Q": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and thumb extended pointing downward (downward G); others curled in fist",
        "non_dominant_hand": None,
        "relative_position": "Single hand angled downward",
        "status": "experimental",
    },
    "R": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended upright and crossed over each other; others curled",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "S": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Closed fist with thumb wrapped horizontally across the front of all curled fingers",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "T": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Closed fist with thumb tucked under the index finger (emerging between index & middle)",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "U": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended upright touching parallel together; others curled",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "V": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index and middle fingers extended upright spread apart in a 'V' shape; others curled",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "W": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index, middle, and ring fingers extended upright spread in a 'W'; pinky curled held by thumb",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "X": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Index finger hooked/bent at middle knuckle; others curled into fist",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "Y": {
        "hands_required": 1,
        "is_dynamic": False,
        "dominant_hand": "Thumb and pinky extended wide; index, middle, ring curled into fist (shaka shape)",
        "non_dominant_hand": None,
        "relative_position": "Single hand held upright facing forward",
        "status": "experimental",
    },
    "Z": {
        "hands_required": 1,
        "is_dynamic": True,
        "dominant_hand": "Index finger extended, tracing a zig-zag 'Z' shape in the air",
        "non_dominant_hand": None,
        "relative_position": "Continuous trajectory stroke in camera view",
        "status": "experimental_dynamic",
    },
}

# Currently Active Validated/Experimental Subsets
SUPPORTED_ALPHABET = [
    letter for letter, meta in ISL_ALPHABET_METADATA.items()
    if meta["status"] in ("experimental", "experimental_dynamic", "verified")
]

SUPPORTED_ISL_ALPHABET = SUPPORTED_ALPHABET

SUPPORTED_ASL_ALPHABET = [
    letter for letter, meta in ASL_ALPHABET_METADATA.items()
    if meta["status"] in ("supported", "experimental", "experimental_dynamic")
]

# Legacy demonstration gestures (Strictly isolated from Alphabet mode)
LEGACY_GESTURES_METADATA = {
    "HELLO": {"description": "Open palm waving upward"},
    "YES": {"description": "Thumbs-up affirmative"},
    "NO": {"description": "Index and middle pinch to thumb"},
    "HELP": {"description": "Thumb upright from fist"},
    "THANK YOU": {"description": "Flat hand forward"},
    "STOP": {"description": "Open flat palm forward"},
    "PEACE": {"description": "Two fingers in V shape"},
    "OK": {"description": "Thumb and index tips touching"},
    "I_LOVE_YOU": {"description": "Thumb, index, pinky extended"},
}

SUPPORTED_GESTURES = list(LEGACY_GESTURES_METADATA.keys())

# Recognition Thresholds & Hyperparameters
MIN_DETECTION_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50
MIN_RECOGNITION_CONFIDENCE = 0.65
UNKNOWN_GESTURE_THRESHOLD = 0.50

# Continuous Letter & Sentence Formation Parameters
LETTER_STABILITY_FRAMES = 4    # Consecutive identical frames required to confirm letter
LETTER_DEBOUNCE_SEC = 1.0      # Minimum seconds before same letter can re-append without hand release
WORD_PAUSE_SEC = 2.5           # Optional pause duration to trigger word separation
AUTO_SPACE_ON_PAUSE = False    # Disabled by default; requires explicit user/config enablement
TRAJECTORY_BUFFER_SIZE = 25    # Frames tracked for dynamic letters (J, Z)

# Backward compatibility aliases
SMOOTHING_WINDOW_SIZE = LETTER_STABILITY_FRAMES + 1
REQUIRED_STABLE_FRAMES = LETTER_STABILITY_FRAMES
DEBOUNCE_COOLDOWN_SEC = LETTER_DEBOUNCE_SEC

# Camera Defaults
DEFAULT_CAMERA_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Multi-Hand Feature Vector Dimensions (Version 2.3)
# Per Hand (Primary & Secondary):
# - 63 coordinates: 21 landmarks * (x, y, z)
# - 3 palm normal vector: (nx, ny, nz)
# - 5 finger states: Thumb, Index, Middle, Ring, Pinky (1.0 or 0.0)
# - 5 key intra-hand distances: (T-I, I-M, T-W, M-W, P-W)
# Total per hand = 76
HAND_FEATURE_DIM = 76

# Inter-Hand Spatial Features (When 2 hands detected, else zero-padded):
# - 1 presence flag: has_second_hand (1.0 or 0.0)
# - 1 wrist-to-wrist distance
# - 3 relative wrist vector: (dx, dy, dz)
# - 5 dominant index tip to non-dominant fingertip distances (T, I, M, R, P)
INTER_HAND_FEATURE_DIM = 10

# Total Multi-Hand Dimension: Primary (76) + Secondary (76) + Inter-Hand (10) = 162
TOTAL_FEATURE_DIM = (HAND_FEATURE_DIM * 2) + INTER_HAND_FEATURE_DIM  # 162

# Legacy Single-Hand Dimension (for backward compatibility tests)
LEGACY_FEATURE_DIM = 73
