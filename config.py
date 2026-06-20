from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# Phase 1: Offline speech-to-text settings
# --------------------------------------------------

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "base"
).strip()


# --------------------------------------------------
# Phase 2: OpenRouter online AI settings
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