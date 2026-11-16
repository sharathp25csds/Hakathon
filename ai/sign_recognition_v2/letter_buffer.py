"""Letter Buffer and Continuous Prediction Stabilization Engine (v2).
Implements rolling-window stability, duplicate letter suppression, genuine hand-release detection,
and supports intentional repeated letters (e.g., 'LL' in 'HELLO') without runaway duplication on hold.
"""

from collections import Counter, deque
import time
from typing import Optional, Tuple

from ai.sign_recognition_v2.config import (
    AUTO_SPACE_ON_PAUSE,
    LETTER_DEBOUNCE_SEC,
    LETTER_STABILITY_FRAMES,
    MIN_RECOGNITION_CONFIDENCE,
    WORD_PAUSE_SEC,
)


class LetterBufferV2:
    """Stabilizes continuous frame-by-frame letter predictions into distinct accepted characters."""

    def __init__(
        self,
        stability_threshold: int = LETTER_STABILITY_FRAMES,
        debounce_sec: float = LETTER_DEBOUNCE_SEC,
        min_confidence: float = MIN_RECOGNITION_CONFIDENCE,
        auto_space_on_pause: bool = AUTO_SPACE_ON_PAUSE,
        pause_duration_sec: float = WORD_PAUSE_SEC,
    ):
        self.stability_threshold = stability_threshold
        self.debounce_sec = debounce_sec
        self.min_confidence = min_confidence
        self.auto_space_on_pause = auto_space_on_pause
        self.pause_duration_sec = pause_duration_sec

        self._history: deque = deque(maxlen=stability_threshold + 2)
        self._confidences: deque = deque(maxlen=stability_threshold + 2)

        self.last_accepted_letter: Optional[str] = None
        self.last_accepted_time: float = 0.0
        self.hand_was_released: bool = True
        self.hand_present: bool = False
        self.current_status: str = "NO HAND DETECTED"

    def reset(self) -> None:
        """Resets stabilization history and state."""
        self._history.clear()
        self._confidences.clear()
        self.last_accepted_letter = None
        self.last_accepted_time = 0.0
        self.hand_was_released = True
        self.hand_present = False
        self.current_status = "NO HAND DETECTED"

    def record_no_hand(self) -> None:
        """Informs the buffer that no hand was detected in the current frame.
        
        This is the ONLY event that marks a genuine hand release, allowing
        subsequent identical letters (e.g. repeated 'L' in 'HELLO') to be accepted.
        """
        self._history.clear()
        self._confidences.clear()
        self.hand_was_released = True
        self.hand_present = False
        self.current_status = "NO HAND DETECTED"

    def process_prediction(
        self,
        letter: Optional[str],
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[str], float, bool, str]:
        """Evaluates a raw frame prediction against stability and debouncing rules.
        
        Args:
            letter: Predicted letter string (e.g. 'A') or None.
            confidence: Float confidence score (0.0 to 1.0).
            timestamp: Optional monotonic timestamp (seconds); defaults to time.monotonic().
            
        Returns:
            Tuple of:
            - display_letter: Currently detected stable letter or None
            - smoothed_confidence: Average confidence across the window
            - is_newly_accepted: True only on the exact frame a letter is locked in
            - status: Detailed descriptive system status
        """
        now = time.monotonic() if timestamp is None else timestamp
        self.hand_present = True

        # 1. Low confidence or unknown gesture handling
        # CRITICAL SAFETY INVARIANT: An unknown gesture or low confidence while hand is present
        # does NOT count as a hand release! Doing so would cause continuous holds with
        # brief noise to rapidly append runaway duplicates.
        if letter is None or letter.strip().upper() == "UNKNOWN":
            self._history.clear()
            self._confidences.clear()
            self.current_status = "UNKNOWN GESTURE"
            return None, round(confidence, 2), False, self.current_status

        if confidence < self.min_confidence:
            self._history.clear()
            self._confidences.clear()
            self.current_status = "LOW CONFIDENCE"
            return None, round(confidence, 2), False, self.current_status

        clean_letter = letter.strip().upper()
        self._history.append(clean_letter)
        self._confidences.append(confidence)

        # 2. Window accumulation check
        if len(self._history) < self.stability_threshold:
            self.current_status = "RECOGNIZING"
            return clean_letter, round(confidence, 2), False, self.current_status

        # 3. Majority vote across stability window
        counts = Counter(self._history)
        most_common, count = counts.most_common(1)[0]
        avg_conf = float(sum(self._confidences) / len(self._confidences))

        if count < self.stability_threshold or most_common != clean_letter:
            self.current_status = "RECOGNIZING"
            return clean_letter, round(avg_conf, 2), False, self.current_status

        self.current_status = "STABLE LETTER"

        # 4. Duplicate Suppression & Repeated Letter Rule:
        # A letter is accepted ONLY IF:
        # (a) It is a DIFFERENT letter from the last accepted letter (e.g. 'H' -> 'E'), OR
        # (b) It is the SAME letter, but the signer genuinely released the hand between signs
        #     (e.g., spelling 'HELLO' by signing 'L', dropping hand, and signing 'L' again).
        can_accept = False
        if most_common != self.last_accepted_letter:
            can_accept = True
        elif self.hand_was_released:
            can_accept = True

        if can_accept:
            self.last_accepted_letter = most_common
            self.last_accepted_time = now
            self.hand_was_released = False
            self.current_status = "LETTER ACCEPTED"
            return most_common, round(avg_conf, 2), True, self.current_status

        # Continuous gesture hold: retain visual feedback, but DO NOT append again!
        return most_common, round(avg_conf, 2), False, self.current_status
