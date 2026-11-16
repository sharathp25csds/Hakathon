"""Sentence Assembly and Continuous Phrase Formation for SignBridge AI v2.
Combines fingerspelled letters and completed words into formatted sentences.
Supports space insertion, backspacing, clearing, and text-to-speech integration.
"""

from typing import List, Optional
from ai.sign_recognition_v2.word_builder import WordBuilderV2


class SentenceBuilderV2:
    """Orchestrates word building and sentence formation for fingerspelled communication."""

    def __init__(self):
        self.word_builder = WordBuilderV2()
        self._completed_words: List[str] = []

    def add_letter(self, letter: str) -> str:
        """Appends a recognized letter to the current active word."""
        self.word_builder.add_letter(letter)
        return self.full_sentence

    def insert_space(self) -> Optional[str]:
        """Completes the current word and appends it to the completed sentence list."""
        finished_word = self.word_builder.complete_word()
        if finished_word:
            self._completed_words.append(finished_word)
        return finished_word

    def backspace(self) -> Optional[str]:
        """Removes the last letter, or if the current word is empty, the last completed word."""
        if len(self.word_builder) > 0:
            return self.word_builder.backspace()
        elif self._completed_words:
            # Re-open the last completed word back into the active word builder
            last_word = self._completed_words.pop()
            for char in last_word[:-1]:
                self.word_builder.add_letter(char)
            return last_word[-1]
        return None

    def clear_sentence(self) -> None:
        """Clears both the active word and all completed words."""
        self.word_builder.clear_word()
        self._completed_words.clear()

    @property
    def current_word(self) -> str:
        """Returns the word currently being spelled."""
        return self.word_builder.current_word

    @property
    def completed_words(self) -> List[str]:
        """Returns the list of previously completed words."""
        return list(self._completed_words)

    @property
    def full_sentence(self) -> str:
        """Returns the full sentence including completed words and active word."""
        words = list(self._completed_words)
        if self.current_word:
            words.append(self.current_word)
        return " ".join(words)

    def confirm_sentence(self) -> str:
        """Finalizes any active word and returns the final sentence string."""
        self.insert_space()
        return " ".join(self._completed_words)

    def __repr__(self) -> str:
        return f"<SentenceBuilderV2: '{self.full_sentence}'>"
