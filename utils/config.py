"""
Configuration Manager

Centralized configuration management with YAML support.

Author: AI Computer Vision Team
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager for YAML configuration files.

    This class manages application configuration loaded from YAML files.
    It provides type-safe access to configuration values with defaults.

    Example:
        >>> config = ConfigManager()
        >>> fps = config.get("camera.fps", default=30)
        >>> camera_config = config.get("camera")
    """

    _instance: Optional["ConfigManager"] = None
    _config: dict = {}

    def __new__(cls, config_path: str = "config/default.yaml"):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path: str) -> None:
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            self._config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("camera.fps", default=30)
            30
            >>> config.get("camera")  # Returns entire camera config dict
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_all(self) -> dict:
        """Get entire configuration.

        Returns:
            Full configuration dictionary
        """
        return self._config.copy()

    def reload(self, config_path: str = "config/default.yaml") -> None:
        """Reload configuration from file.

        Args:
            config_path: Path to configuration file
        """
        self._initialize(config_path)
        logger.info("Configuration reloaded")
