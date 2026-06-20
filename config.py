from pathlib import Path
import os

from dotenv import load_dotenv


# --------------------------------------------------
# Base folders
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load secrets from .env
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# Phase 1: Audio recording settings
# --------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1


# --------------------------------------------------
# Phase 1: Offline speech-to-text settings
# --------------------------------------------------

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "base"
).strip()


# --------------------------------------------------
# Phase 1: Text-to-speech settings
# --------------------------------------------------

TTS_RATE = 185
TTS_VOLUME = 1.0


# --------------------------------------------------
# Phase 2: OpenRouter AI settings
# --------------------------------------------------

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
).strip()

OPENROUTER_API_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)