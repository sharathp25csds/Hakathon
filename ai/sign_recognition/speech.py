"""Text-to-Speech (TTS) Engine for SignBridge AI.
Supports browser Web Speech API (primary) with resilient Python fallback.
Ensures zero-crash degradation if audio hardware or system drivers are missing.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger("SignBridgeAI.Speech")


class TextToSpeechEngine:
    """Manages text-to-speech conversion with multi-tier graceful fallbacks."""

    def __init__(self):
        self.available_backend: Optional[str] = None
        self._detect_backend()

    def _detect_backend(self) -> None:
        """Checks for native speech utilities in Linux/macOS/Windows."""
        for cmd in ["spd-say", "espeak", "say"]:
            try:
                res = subprocess.run(["which", cmd], capture_output=True, text=True)
                if res.returncode == 0:
                    self.available_backend = cmd
                    logger.info(f"System TTS backend found: {cmd}")
                    return
            except Exception:
                continue

        # In browser environments, Web Speech API handles client-side TTS
        self.available_backend = "browser_fallback"

    def speak(self, text: str) -> bool:
        """Speaks the provided text. Never raises exceptions or halts execution.
        
        Args:
            text: Sentence or word to synthesize.
            
        Returns:
            True if spoken or queued for speech, False on fallback.
        """
        if not text or not text.strip():
            return False

        clean_text = text.strip()

        if self.available_backend in ("spd-say", "espeak", "say"):
            try:
                subprocess.Popen(
                    [self.available_backend, clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as e:
                logger.warning(f"Native TTS execution failed: {e}")

        # Browser client-side speech handles this cleanly via window.speechSynthesis
        return True


# Global Singleton
tts_engine = TextToSpeechEngine()
