"""
config/logger.py
----------------
Single-call logger factory used across every module.
Usage:
    from config.logger import get_logger
    log = get_logger(__name__)
"""

import logging
import sys
from config.settings import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:                     # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger