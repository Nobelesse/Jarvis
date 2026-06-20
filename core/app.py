from config import settings
from core.logger import configure_logging


def run_jarvis() -> None:
    logger = configure_logging()

    logger.info(
        "Jarvis Phase 0 started | environment=%s",
        settings.environment,
    )

    print("\nJarvis Phase 0 is ready.")
    print("Project foundation, settings, and logging are working.")
    print("Jarvis is not listening yet.\n")