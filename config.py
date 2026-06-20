# config.py

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)


def _read_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default

    if minimum is not None:
        value = max(value, minimum)

    return value


def _read_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()

    return value in {"1", "true", "yes", "on"}


# --------------------------------------------------
# Phase 1 and Phase 2 voice settings
# --------------------------------------------------

WAKE_WORD = os.getenv("WAKE_WORD", "jarvis").strip().lower() or "jarvis"

SAMPLE_RATE = _read_int("SAMPLE_RATE", 16000, minimum=8000)
CHANNELS = _read_int("CHANNELS", 1, minimum=1)

WAKE_WINDOW_SECONDS = _read_float(
    "WAKE_WINDOW_SECONDS",
    3.5,
    minimum=0.5,
)

COMMAND_WINDOW_SECONDS = _read_float(
    "COMMAND_WINDOW_SECONDS",
    12.0,
    minimum=2.0,
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en").strip() or "base.en"

TTS_RATE = _read_int("TTS_RATE", 185, minimum=80)
TTS_VOLUME = _read_float("TTS_VOLUME", 1.0, minimum=0.0, maximum=1.0)


# --------------------------------------------------
# Jarvis personality
# --------------------------------------------------

SYSTEM_PROMPT = (
    os.getenv("JARVIS_SYSTEM_PROMPT")
    or os.getenv("SYSTEM_PROMPT")
    or (
        "You are Jarvis, a helpful personal laptop assistant for Boss. "
        "Answer clearly, naturally, and briefly unless a detailed answer is requested. "
        "Never claim that you completed an action unless it truly happened."
    )
)


# --------------------------------------------------
# Phase 3: OpenRouter online AI settings
# --------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "",
).strip()

OPENROUTER_API_URL = os.getenv(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
).strip()

OPENROUTER_TIMEOUT_SECONDS = _read_int(
    "OPENROUTER_TIMEOUT_SECONDS",
    45,
    minimum=5,
)


# --------------------------------------------------
# Phase 4: Ollama offline AI settings
# --------------------------------------------------

AI_PREFERRED_BACKEND = os.getenv(
    "AI_PREFERRED_BACKEND",
    "auto",
).strip().lower()

if AI_PREFERRED_BACKEND not in {"auto", "openrouter", "ollama"}:
    AI_PREFERRED_BACKEND = "auto"

OLLAMA_ENABLED = _read_bool("OLLAMA_ENABLED", True)

OLLAMA_BASE_URL = (
    os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    )
    .strip()
    .rstrip("/")
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
).strip() or "llama3.2:3b"

OLLAMA_TIMEOUT_SECONDS = _read_int(
    "OLLAMA_TIMEOUT_SECONDS",
    120,
    minimum=15,
)

AI_NETWORK_CONNECT_TIMEOUT_SECONDS = _read_int(
    "AI_NETWORK_CONNECT_TIMEOUT_SECONDS",
    5,
    minimum=1,
)

AI_FAILURE_COOLDOWN_SECONDS = _read_int(
    "AI_FAILURE_COOLDOWN_SECONDS",
    300,
    minimum=30,
)

AI_TEMPERATURE = _read_float(
    "AI_TEMPERATURE",
    0.4,
    minimum=0.0,
    maximum=2.0,
)