#!/usr/bin/env python3
"""
Enhanced Error Logging System
=============================
Provides concise, focused error tracking and logging to improve GA log usability.
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class LogLevel(Enum):
    CONCISE = "concise"
    VERBOSE = "verbose"
    DEBUG = "debug"


class EnhancedLogger:
    """Enhanced logger with concise error tracking"""

    def __init__(self, name: str, log_level: LogLevel = LogLevel.CONCISE):
        self.name = name
        self.log_level = log_level
        self.logger = logging.getLogger(name)
        self.setup_logging()

    def setup_logging(self):
        """Setup enhanced logging configuration"""
        # Create logs directory
        log_dir = Path("logs/enhanced")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure formatter based on log level
        if self.log_level == LogLevel.CONCISE:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s() - %(message)s"
            )

        # File handler for all logs
        file_handler = logging.FileHandler(
            log_dir / "{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler with concise format
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(levelname)s | %(name)s:%(lineno)d | %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # Error file handler for errors only
        error_handler = logging.FileHandler(
            log_dir / "{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)

    def log_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """Log error with concise context"""
        error_info = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "file": self._get_caller_info(),
            "context": context or {},
        }

        if exception:
            error_info["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }

            if self.log_level != LogLevel.CONCISE:
                error_info["traceback"] = traceback.format_exc()
            else:
                # Get only the relevant traceback line
                tb = traceback.extract_tb(exception.__traceback__)
                if tb:
                    last_frame = tb[-1]
                    error_info["location"] = (
                        "{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
                    )

        # Log concise error
        if self.log_level == LogLevel.CONCISE:
            log_message = "❌ {message}"
            if exception:
                log_message += " | {type(exception).__name__}: {str(exception)}"
            if "location" in error_info:
                log_message += " | {error_info['location']}"
        else:
            log_message = "❌ {message}"
            if exception:
                log_message += "\nException: {exception}"
                log_message += "\nTraceback:\n{traceback.format_exc()}"

        self.logger.error(log_message)

        # Save detailed error info to file
        self._save_error_details(error_info)

    def log_function_entry(self, func_name: str, args: Optional[dict] = None):
        """Log function entry with minimal noise"""
        if self.log_level != LogLevel.CONCISE:
            message = "🔄 Entering {func_name}"
            if args:
                message += " | Args: {args}"
            self.logger.debug(message)

    def log_function_exit(self, func_name: str, result: Optional[Any] = None):
        """Log function exit with minimal noise"""
        if self.log_level != LogLevel.CONCISE:
            message = "✅ Exiting {func_name}"
            if result is not None:
                message += " | Result type: {type(result).__name__}"
            self.logger.debug(message)

    def log_progress(self, message: str, current: int, total: int):
        """Log progress with clean formatting"""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_msg = "📊 {message} | {current}/{total} ({percentage:.1f}%)"
        self.logger.info(progress_msg)

    def log_performance(
        self, operation: str, duration: float, details: Optional[dict] = None
    ):
        """Log performance metrics concisely"""
        perf_msg = "⏱️ {operation} | {duration:.3f}s"
        if details:
            perf_msg += " | {details}"
        self.logger.info(perf_msg)

    def _get_caller_info(self) -> dict[str, Any]:
        """Get concise caller information"""
        frame = sys._getframe(2)  # Go back 2 frames to get actual caller
        return {
            "file": os.path.basename(frame.f_code.co_filename),
            "function": frame.f_code.co_name,
            "line": frame.f_lineno,
        }

    def _save_error_details(self, error_info: dict[str, Any]):
        """Save detailed error info for debugging"""
        error_dir = Path("logs/errors")
        error_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%")[:-3]
        error_file = error_dir / f"error_{timestamp}.json"

        import json

        with open(error_file, "w") as f:
            json.dump(error_info, f, indent=2, default=str)

    def info(self, message: str):
        """Log info message"""
        self.logger.info("ℹ️ {message}")

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning("⚠️ {message}")

    def debug(self, message: str):
        """Log debug message"""
        if self.log_level != LogLevel.CONCISE:
            self.logger.debug("🔍 {message}")

    def success(self, message: str):
        """Log success message"""
        self.logger.info("✅ {message}")


# Factory function to create enhanced loggers
def get_enhanced_logger(name: str, log_level: str = "concise") -> EnhancedLogger:
    """Get an enhanced logger instance"""
    try:
        level = LogLevel(log_level.lower())
    except ValueError:
        level = LogLevel.CONCISE

    return EnhancedLogger(name, level)


# Context manager for function logging
class LoggedFunction:
    """Context manager for automatic function entry/exit logging"""

    def __init__(
        self, logger: EnhancedLogger, func_name: str, args: Optional[dict] = None
    ):
        self.logger = logger
        self.func_name = func_name
        self.args = args
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log_function_entry(self.func_name, self.args)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            self.logger.log_error("Function {self.func_name} failed", exc_val)
        else:
            self.logger.log_function_exit(self.func_name)
            self.logger.log_performance(self.func_name, duration)


# Decorator for automatic function logging
def logged_function(logger_name: str = None, log_level: str = "concise"):
    """Decorator to automatically log function entry, exit, and errors"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_enhanced_logger(logger_name or func.__module__, log_level)

            with LoggedFunction(
                logger,
                func.__name__,
                {"args_count": len(args), "kwargs": list(kwargs.keys())},
            ):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.log_error("Function {func.__name__} failed", e)
                    raise

        return wrapper

    return decorator


# Error aggregation for GA logs
class ErrorAggregator:
    """Aggregates and summarizes errors for GA log analysis"""

    def __init__(self):
        self.errors = []
        self.error_patterns = {}

    def add_error(self, error_info: dict[str, Any]):
        """Add an error to the aggregator"""
        self.errors.append(error_info)

        # Group by error type and message pattern
        error_type = error_info.get("exception", {}).get("type", "Unknown")
        error_pattern = self._extract_pattern(error_info.get("message", ""))

        key = "{error_type}:{error_pattern}"
        if key not in self.error_patterns:
            self.error_patterns[key] = {
                "count": 0,
                "first_seen": error_info.get("timestamp"),
                "last_seen": error_info.get("timestamp"),
                "examples": [],
            }

        self.error_patterns[key]["count"] += 1
        self.error_patterns[key]["last_seen"] = error_info.get("timestamp")

        if len(self.error_patterns[key]["examples"]) < 3:
            self.error_patterns[key]["examples"].append(error_info)

    def _extract_pattern(self, message: str) -> str:
        """Extract error pattern from message"""
        # Simple pattern extraction - could be enhanced
        words = message.split()[:5]  # First 5 words
        return " ".join(words)

    def get_summary(self) -> dict[str, Any]:
        """Get error summary for GA logs"""
        return {
            "total_errors": len(self.errors),
            "unique_patterns": len(self.error_patterns),
            "top_errors": sorted(
                self.error_patterns.items(), key=lambda x: x[1]["count"], reverse=True
            )[:10],
            "summary_timestamp": datetime.now().isoformat(),
        }


# Global error aggregator instance
error_aggregator = ErrorAggregator()


def main():
    """Demo of enhanced logging system"""
    # Test different log levels
    concise_logger = get_enhanced_logger("demo_concise", "concise")
    verbose_logger = get_enhanced_logger("demo_verbose", "verbose")

    # Test various logging scenarios
    concise_logger.info("Starting enhanced logging demo")

    try:
        # Simulate an error
        raise ValueError("This is a test error for demonstration")
    except Exception as e:
        concise_logger.log_error("Demo error occurred", e, {"demo": True})

    concise_logger.success("Enhanced logging demo completed")

    # Test function logging
    @logged_function("demo_function", "concise")
    def test_function(x, y):
        return x + y

    result = test_function(1, 2)
    concise_logger.info("Function result: {result}")


if __name__ == "__main__":
    main()
