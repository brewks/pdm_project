"""
Logging configuration for the GA Predictive Maintenance System.

This module sets up centralized logging for all components of the PdM system,
replacing scattered print statements with structured logging.

Author: Ndubuisi Chibuogwu
Date: Dec 2024 - July 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configure and return a logger instance with file and console handlers.

    Args:
        name: Name of the logger (typically __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL from settings.
        log_to_file: If True, log messages to file
        log_to_console: If True, log messages to console

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("System initialized")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level or LOG_LEVEL))

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # File handler
    if log_to_file:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
   
    # Get an existing logger or create a new one with default configuration.   
    logger = logging.getLogger(name)  # name: Name of the logger
    if not logger.hasHandlers():
        return setup_logger(name)
    return logger  # logging.Logger: Logger instance
