"""Text utilities for CJK character handling, punctuation, and character counting."""

from typing import Iterable

# ── Character sets ───────────────────────────────────────────────

_PUNCT_CHARS = '，。、！？；：\u201c\u201d\u2018\u2019《》【】（）,.!?;:\'"()[]{}·…—~ '
_SENTENCE_END_CHARS = '。！？!?'
_COMMA_CHARS = '，,;；：:'

PUNCTUATION = frozenset(_PUNCT_CHARS)
SENTENCE_END = frozenset(_SENTENCE_END_CHARS)
COMMA_LIKE = frozenset(_COMMA_CHARS)
TRAILING_STRIP = frozenset('，。、,.')

# CJK Unicode ranges
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
)


# ── Core predicates ──────────────────────────────────────────────

def is_cjk(char: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def is_punctuation(char: str) -> bool:
    """Check if character is a punctuation mark."""
    return char in PUNCTUATION


def char_weight(char: str) -> float:
    """Return character weight for counting purposes.

    - CJK char = 1.0
    - Non-CJK, non-punct char = 0.5
    - Punctuation/space = 0.0
    """
    if is_cjk(char):
        return 1.0
    if char in PUNCTUATION or char == ' ':
        return 0.0
    return 0.5


# ── Text measurement ─────────────────────────────────────────────

def text_cjk_count(text: str) -> float:
    """Count CJK chars + non-CJK word units in a text string."""
    return sum(char_weight(ch) for ch in text)


def word_cjk_len(word_text: str) -> float:
    """Count meaningful characters in a word."""
    return sum(char_weight(ch) for ch in word_text)


def words_cjk_len(words: Iterable[dict]) -> float:
    """Sum CJK count for a list of word dicts."""
    return sum(word_cjk_len(w.get("text", "")) for w in words)


# ── Text predicates ──────────────────────────────────────────────

def ends_with_any(text: str, char_set: set[str] | frozenset[str]) -> bool:
    """Check if text ends with a character from char_set (skipping trailing punctuation)."""
    for ch in reversed(text):
        if ch in char_set:
            return True
        if ch not in PUNCTUATION:
            break
    return False


def text_of_words(words: Iterable[dict]) -> str:
    """Concatenate word texts."""
    return "".join(w.get("text", "") for w in words)
