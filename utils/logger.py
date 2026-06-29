"""
Logging Utilities

Provides centralized logging configuration and helper functions.

Author: AI Computer Vision Team
"""

import logging
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
