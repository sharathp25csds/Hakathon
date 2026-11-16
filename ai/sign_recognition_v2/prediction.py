"""Continuous Letter Recognition, Word & Sentence Formation, and Master Service for SignBridge AI v2.
Integrates multi-hand detection, ISLRTC geometric & ML classification,
letter stabilization, duplicate suppression, word assembly, sentence building,
real latency benchmarking, and speech synthesis.
"""

import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from ai.sign_recognition_v2.config import (
    ASL_ALPHABET_METADATA,
    ASL_COMPLIANCE_DISCLAIMER,
    DEFAULT_RECOGNITION_MODE,
    DEFAULT_SIGN_LANGUAGE,
    ISL_ALPHABET_METADATA,
    ISL_COMPLIANCE_DISCLAIMER,
    RecognitionMode,
    SignLanguage,
    SUPPORTED_ASL_ALPHABET,
    SUPPORTED_ISL_ALPHABET,
)
from ai.sign_recognition_v2.gesture_classifier import GestureClassifierV2
from ai.sign_recognition_v2.hand_detector import HandDetectionResultV2, HandDetectorV2
from ai.sign_recognition_v2.letter_buffer import LetterBufferV2
from ai.sign_recognition_v2.sentence_builder import SentenceBuilderV2
from ai.sign_recognition_v2.speech import tts_engine_v2


# Backward compatibility aliases
SentenceBufferV2 = SentenceBuilderV2
TemporalSmootherV2 = LetterBufferV2


class SignRecognitionServiceV2:
    """Master high-level service providing real-time ISL/ASL alphabet recognition and sentence building."""

    def __init__(
        self,
        detector: Optional[HandDetectorV2] = None,
        classifier: Optional[GestureClassifierV2] = None,
        mode: RecognitionMode = DEFAULT_RECOGNITION_MODE,
        language: Union[SignLanguage, str] = DEFAULT_SIGN_LANGUAGE,
    ):
        self.mode = mode
        self.detector = detector or HandDetectorV2()
        self.classifier = classifier or GestureClassifierV2(mode=mode, language=language)
        if classifier is not None and language != DEFAULT_SIGN_LANGUAGE:
            self.classifier.set_language(language)
        self.letter_buffer = LetterBufferV2()
        self.sentence_builder = SentenceBuilderV2()
        self.tts = tts_engine_v2

        # Backward compatibility properties
        self.smoother = self.letter_buffer
        self.sentence_buffer = self.sentence_builder

        self.last_detected_letter: Optional[str] = None
        self.last_confidence: float = 0.0
        self.current_status: str = "NO HAND DETECTED"

    @property
    def current_language(self) -> str:
        """Returns the currently active sign language ('ISL' or 'ASL')."""
        return self.classifier.language.value

    def set_language(self, language: Union[SignLanguage, str]) -> None:
        """Sets the active sign language recognition mode."""
        self.classifier.set_language(language)
        self.letter_buffer.reset()

    @property
    def current_letter(self) -> Optional[str]:
        """Returns the currently detected stable letter."""
        return self.last_detected_letter

    @property
    def confidence(self) -> float:
        """Returns the confidence score of the current letter prediction."""
        return self.last_confidence

    @property
    def current_word(self) -> str:
        """Returns the word currently being spelled."""
        return self.sentence_builder.current_word

    @property
    def completed_sentence(self) -> str:
        """Returns the full accumulated sentence."""
        return self.sentence_builder.full_sentence

    @property
    def recognition_status(self) -> str:
        """Returns the descriptive recognition status."""
        return self.current_status

    def insert_space(self) -> Optional[str]:
        """Completes the current word and separates it in the sentence."""
        return self.sentence_builder.insert_space()

    def backspace(self) -> Optional[str]:
        """Removes the last letter or reopens the last completed word."""
        return self.sentence_builder.backspace()

    def clear_sentence(self) -> None:
        """Clears both the current word and the completed sentence."""
        self.sentence_builder.clear_sentence()
        self.letter_buffer.reset()

    def get_supported_gestures(
        self, language: Optional[Union[SignLanguage, str]] = None
    ) -> Dict[str, Any]:
        """Returns metadata, supported alphabet, counts, and compliance disclaimers for the active or requested language."""
        lang_str = (
            str(language if language is not None else self.current_language)
            .upper()
            .split(".")[-1]
        )
        if lang_str == "ASL":
            supported = list(SUPPORTED_ASL_ALPHABET)
            metadata = dict(ASL_ALPHABET_METADATA)
            disclaimer = ASL_COMPLIANCE_DISCLAIMER
            target_lang = "ASL"
        else:
            supported = list(SUPPORTED_ISL_ALPHABET)
            metadata = dict(ISL_ALPHABET_METADATA)
            disclaimer = ISL_COMPLIANCE_DISCLAIMER
            target_lang = "ISL"

        counts_by_status = {
            "supported": 0,
            "experimental": 0,
            "experimental_dynamic": 0,
            "unsupported": 0,
            "total": len(metadata),
        }
        for meta in metadata.values():
            st = meta.get("status", "unsupported")
            counts_by_status[st] = counts_by_status.get(st, 0) + 1

        return {
            "language": target_lang,
            "supported_alphabet": supported,
            "metadata": metadata,
            "count": len(supported),
            "disclaimer": disclaimer,
            "counts_by_status": counts_by_status,
            "gestures": supported,
        }

    def speak_sentence(self, browser_fallback: bool = False) -> bool:
        """Speaks the completed sentence aloud via the TTS engine contract."""
        text = self.completed_sentence
        if text:
            return self.tts.speak(text, browser_fallback=browser_fallback)
        return False

    def process_frame(
        self, frame: np.ndarray, annotate: bool = True
    ) -> Dict[str, Any]:
        """Processes a single BGR video frame and produces the standardized API response.
        
        Measures actual processing time using time.perf_counter().
        """
        t_start = time.perf_counter()

        # 1. Check for empty or invalid frame
        if frame is None or frame.size == 0:
            self.current_status = "CAMERA UNAVAILABLE"
            return {
                "success": False,
                "recognized_text": None,
                "confidence": 0.0,
                "hand_detected": False,
                "gesture_recognized": False,
                "status": "invalid_frame",
                "language": self.current_language,
                "current_word": self.current_word,
                "sentence": self.completed_sentence,
                "latency_ms": 0.0,
            }

        # 2. Multi-Hand Detection
        detection: HandDetectionResultV2 = self.detector.detect(frame)

        if not detection.detected or not detection.hands:
            self.letter_buffer.record_no_hand()
            self.last_detected_letter = None
            self.last_confidence = 0.0
            self.current_status = "NO HAND DETECTED"

            annotated_frame = (
                self.detector.draw_landmarks(frame, detection) if annotate else None
            )
            elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "success": True,
                "recognized_text": None,
                "confidence": 0.0,
                "hand_detected": False,
                "num_hands": 0,
                "gesture_recognized": False,
                "status": "no_hand_detected",
                "language": self.current_language,
                "system_status": self.current_status,
                "current_letter": self.current_letter,
                "current_word": self.current_word,
                "sentence": self.completed_sentence,
                "latency_ms": elapsed_ms,
                "annotated_frame": annotated_frame,
            }

        # 3. Gesture & ISL Alphabet Classification (Multi-Hand)
        raw_label, raw_conf, is_recognized, meta = self.classifier.classify_detection(detection)

        # 4. Continuous Letter Buffer & Stability Evaluation
        display_letter, smoothed_conf, is_newly_accepted, buffer_status = (
            self.letter_buffer.process_prediction(
                raw_label if is_recognized else None, raw_conf
            )
        )

        self.last_detected_letter = display_letter
        self.last_confidence = smoothed_conf
        self.current_status = buffer_status

        # 5. Continuous Word Formation (Append newly accepted letter)
        if is_newly_accepted and display_letter:
            self.sentence_builder.add_letter(display_letter)

        # 6. Format API Status
        if buffer_status == "LETTER ACCEPTED":
            api_status = "letter_accepted"
        elif buffer_status == "STABLE LETTER":
            api_status = "recognized"
        elif buffer_status == "RECOGNIZING":
            api_status = "stabilizing"
        elif buffer_status == "LOW CONFIDENCE":
            api_status = "low_confidence"
        else:
            api_status = "unknown_gesture"

        # 7. Annotate Visual Output
        annotated_frame = None
        if annotate:
            annotated_frame = self.detector.draw_landmarks(frame, detection)
            h, w = annotated_frame.shape[:2]

            # Display active sign banner
            if display_letter:
                banner = f"[{self.current_language}] Letter: {display_letter} ({int(smoothed_conf * 100)}%) - {buffer_status}"
                cv2.putText(
                    annotated_frame,
                    banner,
                    (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated_frame,
                    f"[{self.current_language}] {buffer_status}",
                    (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display Word & Sentence Builder HUD at bottom
            word_hud = f"Word: '{self.current_word}' | Sentence: '{self.completed_sentence}'"
            cv2.putText(
                annotated_frame,
                word_hud,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "success": True,
            "recognized_text": display_letter,
            "confidence": round(smoothed_conf, 2),
            "language": self.current_language,
            "hand_detected": True,
            "num_hands": len(detection.hands),
            "gesture_recognized": bool(display_letter is not None),
            "status": api_status,
            "system_status": self.current_status,
            "current_letter": self.current_letter,
            "current_word": self.current_word,
            "sentence": self.completed_sentence,
            "handedness": detection.handedness,
            "bbox": list(detection.bbox),
            "landmarks": detection.landmarks,
            "latency_ms": elapsed_ms,
            "annotated_frame": annotated_frame,
            "metadata": meta,
        }
