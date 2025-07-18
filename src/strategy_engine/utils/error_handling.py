#!/usr/bin/env python3
"""
Standardized Error Handling Framework
====================================

Provides consistent error handling patterns across the AI Trading Machine.
"""

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingSystemError(Exception):
    """Base exception for trading system errors"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: str = "unknown",
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.component = component
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class CriticalSystemError(TradingSystemError):
    """Critical system error that requires immediate attention"""

    def __init__(
        self,
        message: str,
        component: str = "unknown",
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, ErrorSeverity.CRITICAL, component, context)


class DataValidationError(TradingSystemError):
    """Data validation error"""

    def __init__(
        self,
        message: str,
        data_source: str = "unknown",
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message, ErrorSeverity.HIGH, "data_validation_{data_source}", context
        )


class ConfigurationError(TradingSystemError):
    """Configuration error"""

    def __init__(
        self,
        message: str,
        config_component: str = "unknown",
        context: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message, ErrorSeverity.HIGH, "config_{config_component}", context
        )


def fail_fast_on_critical(func: Callable) -> Callable:
    """Decorator that ensures critical errors cause immediate system failure"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CriticalSystemError as e:
            logger = logging.getLogger(func.__module__)
            logger.critical("CRITICAL ERROR in {func.__name__}: {e}")
            logger.critical("Component: {e.component}")
            logger.critical("Context: {e.context}")
            logger.critical("Traceback: {traceback.format_exc()}")

            # Trigger alerts and shutdown
            _trigger_critical_alert(e, func.__name__)
            sys.exit(1)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error("Unexpected error in {func.__name__}: {e}")
            logger.error("Traceback: {traceback.format_exc()}")
            raise TradingSystemError("Unexpected error in {func.__name__}: {e}")

    return wrapper


def handle_with_recovery(
    fallback_value: Any = None, recovery_func: Optional[Callable] = None
):
    """Decorator for graceful error handling with recovery"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CriticalSystemError:
                # Critical errors should not be recovered from
                raise
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.warning("Error in {func.__name__}: {e}")
                logger.warning("Attempting recovery...")

                if recovery_func:
                    try:
                        return recovery_func(*args, **kwargs)
                    except Exception as recovery_error:
                        logger.error("Recovery failed: {recovery_error}")

                if fallback_value is not None:
                    logger.warning("Using fallback value: {fallback_value}")
                    return fallback_value

                # If no recovery possible, convert to system error
                raise TradingSystemError("Error in {func.__name__}: {e}")

        return wrapper

    return decorator


def validate_and_handle(
    validation_func: Callable[[Any], bool], error_message: str = "Validation failed"
):
    """Decorator for input validation with proper error handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            try:
                if not validation_func(*args, **kwargs):
                    raise DataValidationError(
                        "{error_message} in {func.__name__}",
                        context={"args": args, "kwargs": kwargs},
                    )
            except Exception as e:
                raise DataValidationError(
                    "Validation error in {func.__name__}: {e}",
                    context={"args": args, "kwargs": kwargs},
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _trigger_critical_alert(error: CriticalSystemError, function_name: str) -> None:
    """Trigger critical system alert"""
    try:
        # Save critical error for monitoring system
        alert_data = {
            "timestamp": error.timestamp,
            "error_type": "CRITICAL_SYSTEM_ERROR",
            "component": error.component,
            "function": function_name,
            "message": str(error),
            "context": error.context,
        }

        # Write to critical alerts file
        os.makedirs("logs/critical_alerts", exist_ok=True)
        alert_file = "logs/critical_alerts/critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(alert_file, "w") as f:
            json.dump(alert_data, f, indent=2)

        print("🚨 CRITICAL ALERT TRIGGERED: {alert_file}")

    except Exception as alert_error:
        print("Failed to trigger critical alert: {alert_error}")


# Utility functions for common error handling patterns


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """Safely load JSON file with proper error handling"""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("JSON file not found: {file_path}")
        return default
    except json.JSONDecodeError as e:
        raise DataValidationError("Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise TradingSystemError("Failed to load JSON {file_path}: {e}")


def safe_file_write(file_path: str, content: str, critical: bool = False) -> bool:
    """Safely write file with proper error handling"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        if critical:
            raise CriticalSystemError("Failed to write critical file {file_path}: {e}")
        else:
            logger.error("Failed to write file {file_path}: {e}")
            return False


def require_config(config_key: str, config_dict: dict[str, Any]) -> Any:
    """Require configuration key with proper error handling"""
    if config_key not in config_dict:
        raise ConfigurationError("Required configuration missing: {config_key}")
    return config_dict[config_key]
