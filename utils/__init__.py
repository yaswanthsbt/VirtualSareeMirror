"""
Utilities Package

Provides common utility functions and helpers for the Virtual Saree Mirror application.

Modules:
    logger: Logging utilities
    config: Configuration management
"""

from .config import ConfigManager
from .logger import get_logger

__all__ = ["get_logger", "ConfigManager"]
