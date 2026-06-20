from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from config import AUDIO_DIR, CHANNELS, SAMPLE_RATE


def record_wav(seconds: float) -> Path:
    """
    Records microphone audio and saves it as a temporary WAV file.
    """

    if seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    frames = int(seconds * SAMPLE_RATE)

    try:
        print(f"[Listening for {seconds:.1f} seconds...]")

        audio = sd.rec(
            frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32"
        )

        sd.wait()

    except Exception as error:
        raise RuntimeError(
            "Jarvis could not access the microphone. "
            "Check your microphone connection and Windows microphone permissions."
        ) from error

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audio_path = AUDIO_DIR / f"recording_{timestamp}.wav"

    write(audio_path, SAMPLE_RATE, audio_int16)

    return audio_path