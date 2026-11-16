"""SignBridge AI - ISL Alphabet & Continuous Sign Recognition Module (v2).
Independent assistive communication vision engine for deaf and hard-of-hearing users.

Provides:
- 21-landmark translation, scale, and rotation normalized hand pose estimation
- Multi-hand feature extraction (162 dims: primary 76, secondary 76, inter-hand 10)
- Real-time Indian Sign Language (ISL) alphabet recognition (A-Z)
- Continuous letter stabilization, duplicate suppression, and debouncing
- Real-time word assembly and sentence building
- Dynamic stroke trajectory tracking for motion letters (J, Z)
- Multi-tier Text-to-Speech (TTS) synthesis
"""

__version__ = "2.3.0"

from ai.sign_recognition_v2.config import (
    FEATURE_VERSION,
    ISL_ALPHABET,
    ISL_ALPHABET_METADATA,
    ISL_COMPLIANCE_DISCLAIMER,
    RecognitionMode,
    SUPPORTED_ALPHABET,
    SUPPORTED_GESTURES,
    TOTAL_FEATURE_DIM,
)
from ai.sign_recognition_v2.gesture_classifier import GestureClassifierV2
from ai.sign_recognition_v2.hand_detector import HandData, HandDetectionResultV2, HandDetectorV2
from ai.sign_recognition_v2.landmark_extractor import LandmarkExtractorV2
from ai.sign_recognition_v2.letter_buffer import LetterBufferV2
from ai.sign_recognition_v2.prediction import SignRecognitionServiceV2
from ai.sign_recognition_v2.sentence_builder import SentenceBuilderV2
from ai.sign_recognition_v2.speech import TextToSpeechEngineV2, tts_engine_v2
from ai.sign_recognition_v2.temporal_recognizer import TemporalRecognizerV2
from ai.sign_recognition_v2.word_builder import WordBuilderV2

__all__ = [
    "SignRecognitionServiceV2",
    "HandDetectorV2",
    "HandData",
    "HandDetectionResultV2",
    "LandmarkExtractorV2",
    "GestureClassifierV2",
    "LetterBufferV2",
    "WordBuilderV2",
    "SentenceBuilderV2",
    "TemporalRecognizerV2",
    "TextToSpeechEngineV2",
    "tts_engine_v2",
    "RecognitionMode",
    "FEATURE_VERSION",
    "TOTAL_FEATURE_DIM",
    "ISL_ALPHABET",
    "ISL_ALPHABET_METADATA",
    "SUPPORTED_ALPHABET",
    "SUPPORTED_GESTURES",
    "ISL_COMPLIANCE_DISCLAIMER",
]
