"""
Logger Configuration for AI Trading Machine
==========================================

Centralized logging configuration with proper formatting,
file rotation, and different log levels for development and production.

Author: AI Trading Machine
Licensed by SJ Trading
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional


class Logger:
    """
    Wrapper class for logging functionality.
    Provides a consistent interface for logging across the application.
    """

    def __init__(self, name: str, level: Optional[str] = None):
        """Initialize logger with given name and level."""
        self.logger = setup_logger(name, level)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting and handlers.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output:
        if log_file is None:
            # Default log file location
            log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, "ai_trading_machine_{datetime.now().strftime('%Y%m%d')}.log"
            )

        # Rotating file handler (10MB max, keep 5 files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_trading_logger() -> logging.Logger:
    """Get the main trading logger."""
    return setup_logger("ai_trading_machine.trading")


def get_backtest_logger() -> logging.Logger:
    """Get the backtesting logger."""
    return setup_logger("ai_trading_machine.backtest")


def get_data_logger() -> logging.Logger:
    """Get the data ingestion logger."""
    return setup_logger("ai_trading_machine.data")


def get_strategy_logger() -> logging.Logger:
    """Get the strategy logger."""
    return setup_logger("ai_trading_machine.strategy")


# Configure root logger for the package
def configure_package_logging():
    """Configure logging for the entire AI Trading Machine package."""
    # Set up root logger
    root_logger = setup_logger("ai_trading_machine")

    # Suppress verbose logging from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    return root_logger


# Auto-configure when module is imported
configure_package_logging()
