import logging
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from config import MODELS_DIR, WHISPER_MODEL
from core.audio_recorder import record_wav


logging.getLogger("faster_whisper").setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """
    Loads the local Whisper model once and reuses it.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Loading local speech-recognition model...]")

    return WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS_DIR)
    )


def transcribe_wav(audio_path: Path) -> str:
    """
    Converts a WAV file into English text.
    """

    try:
        model = get_whisper_model()

        segments, _ = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return text

    except Exception as error:
        print(f"[Speech recognition error: {error}]")
        return ""

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass


def listen_for_text(seconds: float) -> str:
    """
    Records speech for a fixed duration and returns recognized text.
    """

    audio_path = record_wav(seconds)
    return transcribe_wav(audio_path)