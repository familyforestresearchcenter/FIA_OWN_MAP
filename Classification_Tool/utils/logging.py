from __future__ import annotations
import logging
from datetime import datetime

LOGGER_NAME = "classification"

def get_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
