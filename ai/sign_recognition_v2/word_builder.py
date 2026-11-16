"""Word Formation and Letter Assembly Engine for SignBridge AI v2.
Assembles recognized letters into words with backspace, space, and clear operations.
"""

from typing import List, Optional


class WordBuilderV2:
    """Manages character buffering and word-level composition from fingerspelled letters."""

    def __init__(self):
        self._letters: List[str] = []

    def add_letter(self, letter: str) -> str:
        """Appends an accepted letter to the active word buffer."""
        if letter and len(letter.strip()) == 1:
            clean = letter.strip().upper()
            self._letters.append(clean)
        return self.current_word

    def backspace(self) -> Optional[str]:
        """Removes and returns the most recently added letter from the word."""
        if self._letters:
            return self._letters.pop()
        return None

    def clear_word(self) -> None:
        """Clears the active word buffer."""
        self._letters.clear()

    def complete_word(self) -> Optional[str]:
        """Completes and resets the active word buffer, returning the finished word."""
        if not self._letters:
            return None
        completed = "".join(self._letters)
        self._letters.clear()
        return completed

    @property
    def current_word(self) -> str:
        """Returns the active word as a string."""
        return "".join(self._letters)

    @property
    def length(self) -> int:
        return len(self._letters)

    def __len__(self) -> int:
        return len(self._letters)

    def __repr__(self) -> str:
        return f"<WordBuilderV2: '{self.current_word}'>"
