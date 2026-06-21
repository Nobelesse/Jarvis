# core/wake_word.py

from __future__ import annotations

import re
from difflib import SequenceMatcher

from config import (
    WAKE_WINDOW_SECONDS,
    WAKE_WORD,
)
from core.speech_to_text import listen_for_text


WAKE_WORD_ALIASES = {
    "jarvis",
    "darvis",
    "dervis",
    "jervis",
    "jarves",
    "jarvez",
    "garvis",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_wake_word_candidates(text: str) -> list[str]:
    words = normalize_text(text).split()
    candidates: list[str] = []

    for index, word in enumerate(words):
        candidates.append(word)

        if index + 1 < len(words):
            combined = word + words[index + 1]

            if len(combined) <= 10:
                candidates.append(combined)

    return candidates


def is_safe_fuzzy_variant(candidate: str, wake_word: str) -> bool:
    if not 5 <= len(candidate) <= 7:
        return False

    if candidate[0] not in {"j", "d", "g"}:
        return False

    if candidate[-2:] != wake_word[-2:]:
        return False

    similarity = SequenceMatcher(
        None,
        candidate,
        wake_word,
    ).ratio()

    return similarity >= 0.83


def find_wake_word_match(text: str) -> str | None:
    normalized_wake_word = normalize_text(WAKE_WORD).replace(" ", "")

    aliases = {
        normalize_text(alias).replace(" ", "")
        for alias in WAKE_WORD_ALIASES
    }
    aliases.add(normalized_wake_word)

    for candidate in get_wake_word_candidates(text):
        if candidate in aliases:
            return candidate

        if is_safe_fuzzy_variant(
            candidate,
            normalized_wake_word,
        ):
            return candidate

    return None


def contains_wake_word(text: str) -> bool:
    return find_wake_word_match(text) is not None


def wait_for_wake_word() -> None:
    while True:
        print(f"\n[Sleeping... Say '{WAKE_WORD}' to activate Jarvis.]")

        heard_text = listen_for_text(
            WAKE_WINDOW_SECONDS,
            wake_word_mode=True,
        )

        heard_text = normalize_text(heard_text)

        if not heard_text:
            continue

        print(f"[Heard: {heard_text}]")

        matched_word = find_wake_word_match(heard_text)

        if matched_word:
            print(f"[Wake word detected: {matched_word}]")
            return