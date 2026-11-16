"""Text-to-Speech (TTS) Engine and Audio Synthesis Contract for SignBridge AI v2.
Provides clean status reporting, system binary probing, and zero-crash degradation.
"""

import logging
import shutil
import subprocess
from typing import Dict, Optional

logger = logging.getLogger("SignBridgeAI_v2.Speech")


class TextToSpeechEngineV2:
    """Manages audio speech output with strict execution contracts and browser fallback."""

    def __init__(self):
        self.available_backend: Optional[str] = None
        self._detect_backend()

    def _detect_backend(self) -> None:
        """Probes for available system speech synthesis binaries."""
        for binary in ["spd-say", "espeak", "say"]:
            if shutil.which(binary) is not None:
                self.available_backend = binary
                return
        self.available_backend = None

    @property
    def has_native_tts(self) -> bool:
        """Returns True if a native CLI speech synthesizer is installed."""
        return self.available_backend is not None

    def speak(self, text: str, browser_fallback: bool = False) -> bool:
        """Speaks the text through available system TTS.
        
        Contract:
        - Returns True ONLY if native CLI TTS successfully launched, OR if
          browser_fallback is explicitly True (indicating browser audio handles it).
        - Returns False if text is empty, execution failed, or native TTS is missing
          and no browser fallback is requested.
        
        Args:
            text: Sentence or token to speak.
            browser_fallback: Whether client browser TTS is connected and handling speech.
            
        Returns:
            bool: True if speech was executed or delegated to active browser; False otherwise.
        """
        if not text or not text.strip():
            return False

        clean_text = text.strip()

        # 1. Attempt native system CLI execution if available
        if self.available_backend in ("spd-say", "espeak", "say"):
            try:
                subprocess.Popen(
                    [self.available_backend, clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as e:
                logger.warning(f"Native TTS execution failed for '{clean_text}': {e}")
                return False

        # 2. If native TTS is not available on host/container
        if browser_fallback:
            # Client browser window.speechSynthesis will handle audio
            return True

        # Native binary is missing and browser client is not active
        return False

    def get_speech_payload(self, text: str) -> Dict[str, str]:
        """Formats a standardized payload for frontend Web Speech API integration."""
        return {
            "text": text.strip() if text else "",
            "native_tts_available": str(self.has_native_tts),
            "backend": self.available_backend or "web_speech_api",
        }


# Global Singleton for v2
tts_engine_v2 = TextToSpeechEngineV2()
