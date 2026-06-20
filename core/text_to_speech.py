import threading

import pyttsx3

from config import TTS_RATE, TTS_VOLUME


_engine = None
_engine_lock = threading.RLock()


def get_engine():
    """
    Creates the offline Windows SAPI voice engine once.
    """

    global _engine

    with _engine_lock:
        if _engine is None:
            _engine = pyttsx3.init("sapi5")
            _engine.setProperty("rate", TTS_RATE)
            _engine.setProperty("volume", TTS_VOLUME)

        return _engine


def speak(text: str) -> None:
    """
    Speaks text using the offline Windows voice engine.
    """

    message = str(text).strip()

    if not message:
        return

    print(f"Jarvis: {message}")

    with _engine_lock:
        engine = get_engine()
        engine.stop()
        engine.say(message)
        engine.runAndWait()


def stop_speaking() -> None:
    """
    Stops the current speech output.
    """

    with _engine_lock:
        if _engine is not None:
            _engine.stop()