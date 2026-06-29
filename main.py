"""
Virtual Saree Mirror - Main Application Entry Point

This is the entry point for the Virtual Saree Mirror application.
It initializes all modules, loads configuration, and starts the GUI.

Author: AI Computer Vision Team
Version: 1.0.0
"""

import sys
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


class ApplicationConfig:
    """Application configuration manager."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: dict = {}
        self.load_config()

    def load_config(self) -> None:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration: {e}")
            raise

    def get(self, key: str, default: Optional[any] = None) -> any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default


def setup_logging() -> None:
    """Setup logging configuration from YAML."""
    log_config_path = Path("config/logging.yaml")

    if log_config_path.exists():
        with open(log_config_path, "r") as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    logging.info("Logging system initialized")


def create_directories() -> None:
    """Create required directories if they don't exist."""
    directories = [
        "logs",
        "models",
        "sarees",
        "captures",
        "temp",
        "assets",
        "database",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.debug(f"Directory ensured: {directory}")


def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load environment variables
        load_dotenv()
        logging.info("Environment variables loaded")

        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Virtual Saree Mirror - Application Starting")
        logger.info("=" * 60)

        # Create required directories
        create_directories()
        logger.info("Required directories created")

        # Load configuration
        config = ApplicationConfig("config/default.yaml")
        app_config = config.get("app", {})
        logger.info(f"Application: {app_config.get('name', 'Virtual Saree Mirror')}")
        logger.info(f"Version: {app_config.get('version', '1.0.0')}")

        # TODO: Initialize modules in following phases
        # Phase 2: Camera Manager
        # Phase 3: Pose Detection
        # Phase 4: Segmentation
        # Phase 5: Saree Loader
        # Phase 6: Draping Engine
        # Phase 7: Motion Tracking
        # Phase 8: Renderer
        # Phase 9: UI

        logger.info("Application initialization complete (Phase 1)")
        logger.info("All modules will be added in subsequent phases")

        # TODO: Start GUI in Phase 9
        logger.warning("GUI implementation pending (Phase 9)")

        return 0

    except Exception as e:
        logging.error(f"Fatal error during initialization: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
