# core/speech_to_text.py

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from config import (
    COMMAND_WINDOW_SECONDS,
    MODELS_DIR,
    WHISPER_MODEL,
)
from core.audio_recorder import record_wav


LOGGER = logging.getLogger(__name__)

WAKE_WORD_PROMPT = "Jarvis."
WAKE_WORD_HOTWORDS = "Jarvis"


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    print("[Loading local speech-recognition model...]")

    return WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS_DIR),
    )


def resolve_listen_seconds(
    seconds: float | None = None,
    timeout: float | None = None,
    phrase_time_limit: float | None = None,
) -> float:
    values = (
        seconds,
        timeout,
        phrase_time_limit,
        COMMAND_WINDOW_SECONDS,
    )

    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return float(value)

    return float(COMMAND_WINDOW_SECONDS)


def transcribe_wav(
    audio_path: Path,
    *,
    wake_word_mode: bool = False,
) -> str:
    audio_path = Path(audio_path)

    try:
        model = get_whisper_model()

        transcription_options = {
            "language": "en",
            "beam_size": 3,
            "best_of": 3,
            "temperature": 0.0,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }

        if wake_word_mode:
            transcription_options.update(
                {
                    "beam_size": 5,
                    "best_of": 5,
                    "initial_prompt": WAKE_WORD_PROMPT,
                    "hotwords": WAKE_WORD_HOTWORDS,
                }
            )

        segments, _ = model.transcribe(
            str(audio_path),
            **transcription_options,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        )

        return text.strip()

    except Exception as error:
        LOGGER.exception("Speech transcription failed: %s", error)
        return ""

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError as error:
            LOGGER.warning(
                "Could not remove temporary audio file %s: %s",
                audio_path,
                error,
            )


def listen_for_text(
    seconds: float | None = None,
    *,
    timeout: float | None = None,
    phrase_time_limit: float | None = None,
    wake_word_mode: bool = False,
) -> str:
    listen_seconds = resolve_listen_seconds(
        seconds=seconds,
        timeout=timeout,
        phrase_time_limit=phrase_time_limit,
    )

    audio_path = record_wav(listen_seconds)

    return transcribe_wav(
        audio_path,
        wake_word_mode=wake_word_mode,
    )