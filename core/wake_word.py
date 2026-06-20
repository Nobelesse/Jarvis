import re

from config import WAKE_WINDOW_SECONDS, WAKE_WORD
from core.speech_to_text import listen_for_text


def contains_wake_word(text: str) -> bool:
    """
    Checks whether spoken text contains the wake word.
    """

    pattern = rf"\b{re.escape(WAKE_WORD)}\b"

    return bool(
        re.search(
            pattern,
            text.lower()
        )
    )


def wait_for_wake_word() -> None:
    """
    Keeps listening silently until the user says 'Jarvis'.
    """

    print(f"\n[Sleeping... Say '{WAKE_WORD}' to activate Jarvis.]")

    while True:
        spoken_text = listen_for_text(WAKE_WINDOW_SECONDS)

        if spoken_text:
            print(f"[Heard: {spoken_text}]")

        if contains_wake_word(spoken_text):
            return