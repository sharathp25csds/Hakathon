"""Centralized configuration for SignBridge AI Hand Recognition Module.
Configurable thresholds, model paths, vocabulary, and system parameters.
"""

import sys
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model & Data Paths
TASK_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
DATASET_PATH = DATA_DIR / "gestures.csv"

# Supported Sign Vocabulary
SUPPORTED_GESTURES = [
    "HELLO",        # Open palm, all fingers extended upward
    "YES",          # Thumbs-up or closed fist nodding
    "NO",           # Index and middle finger pinch to thumb
    "HELP",         # Thumb upward from flat hand/fist
    "THANK YOU",    # Flat hand forward from chin
    "STOP",         # Open palm facing forward, fingers extended
    "PEACE",        # Index & middle fingers extended in V shape
    "OK",           # Thumb & index forming a circle, other fingers extended
    "I_LOVE_YOU",   # Thumb, index, and pinky extended (ASL ILY)
]

# Gesture Descriptions for UI & Users
GESTURE_DESCRIPTIONS = {
    "HELLO": "Open palm with all fingers spread and pointing upward",
    "YES": "Fist with thumb pointing upward (Thumbs-up / affirmative)",
    "NO": "Index and middle fingers pinched together with thumb",
    "HELP": "Thumb pointed straight up with hand upright",
    "THANK YOU": "Flat open hand facing forward",
    "STOP": "Open palm facing forward with fingers held upright",
    "PEACE": "Index and middle fingers extended in a V shape",
    "OK": "Thumb and index tips touching to form a circle",
    "I_LOVE_YOU": "Thumb, index, and pinky extended (ASL ILY sign)",
}

# Recognition Thresholds & Hyperparameters
MIN_DETECTION_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50
MIN_RECOGNITION_CONFIDENCE = 0.65
UNKNOWN_GESTURE_THRESHOLD = 0.50

# Temporal Smoothing & Stability
SMOOTHING_WINDOW_SIZE = 5      # Consecutive frames for majority vote
REQUIRED_STABLE_FRAMES = 3     # Minimum identical votes to confirm a gesture
DEBOUNCE_COOLDOWN_SEC = 1.5    # Minimum seconds before re-appending the same gesture to a sentence

# Camera & Performance Defaults
DEFAULT_CAMERA_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Feature Extraction Dimensions
NUM_LANDMARKS = 21
NUM_COORDINATE_FEATURES = 63   # 21 * 3 (x, y, z)
NUM_FINGER_STATE_FEATURES = 5  # Thumb, Index, Middle, Ring, Pinky extended flags
NUM_DISTANCE_FEATURES = 5      # Fingertip-to-palm/wrist Euclidean distances
TOTAL_FEATURE_DIM = NUM_COORDINATE_FEATURES + NUM_FINGER_STATE_FEATURES + NUM_DISTANCE_FEATURES  # 73
