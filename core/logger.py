import logging
from logging.handlers import RotatingFileHandler

from config import PROJECT_ROOT, settings


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(settings.app_name)

    if logger.handlers:
        return logger

    log_directory = PROJECT_ROOT / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_directory / "jarvis.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger