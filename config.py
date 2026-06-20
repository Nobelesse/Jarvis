import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    log_level: str
    gemini_api_key: str | None


def load_settings() -> Settings:
    return Settings(
        app_name="Jarvis",
        environment=os.getenv("JARVIS_ENV", "development"),
        log_level=os.getenv("JARVIS_LOG_LEVEL", "INFO"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
    )


settings = load_settings()