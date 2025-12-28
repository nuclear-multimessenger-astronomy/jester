"""
Logging configuration for jesterTOV package.

This module provides a centralized logging configuration to help users
understand what's happening during inference and TOV solving operations.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "jester",
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger for jester with consistent formatting.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is "jester".
    level : int, optional
        Logging level (e.g., logging.DEBUG, logging.INFO). Default is INFO.
    fmt : str, optional
        Custom format string. If None, uses default format.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from jesterTOV.logging_config import setup_logger
    >>> logger = setup_logger()
    >>> logger.info("Starting inference...")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        if fmt is None:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def get_logger(name: str = "jester") -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This function retrieves an existing logger or creates a new one if
    it doesn't exist. Use this in modules instead of setup_logger().

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is "jester".

    Returns
    -------
    logging.Logger
        Logger instance.

    Examples
    --------
    >>> from jesterTOV.logging_config import get_logger
    >>> logger = get_logger("jester")
    >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)

    # If this is a child logger and parent exists, return it
    if "." in name and not logger.handlers:
        parent_name = name.split(".")[0]
        parent_logger = logging.getLogger(parent_name)
        if parent_logger.handlers:
            return logger

    # Otherwise, set up if needed
    if not logger.handlers:
        setup_logger(name)

    return logger


def set_log_level(level: int, name: str = "jester") -> None:
    """
    Change the logging level for jester loggers.

    Parameters
    ----------
    level : int
        New logging level (e.g., logging.DEBUG, logging.WARNING).
    name : str, optional
        Name of the logger to modify. Default is "jester".

    Examples
    --------
    >>> from jesterTOV.logging_config import set_log_level
    >>> import logging
    >>> set_log_level(logging.DEBUG)  # Enable debug messages
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# Initialize the main jester logger on import
_main_logger = setup_logger("jester", level=logging.INFO)
