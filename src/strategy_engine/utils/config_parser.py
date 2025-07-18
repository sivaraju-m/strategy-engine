"""
Configuration Parser for AI Trading Machine
===========================================

Centralized configuration management with support for:
- JSON configuration files
- Environment variable overrides
- Secrets management integration
- Configuration validation

Author: AI Trading Machine
Licensed by SJ Trading
"""

import json
import os
from pathlib import Path
from typing import Any, Optional


class ConfigParser:
    """
    Configuration parser with environment variable support and validation.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration parser.

        Args:
            config_file: Path to JSON configuration file
        """
        self.config = {}
        self.config_file = config_file

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(file_path) as f:
                self.config = json.load(f)
        except Exception as e:
            raise ValueError("Failed to load config from {file_path}: {e}")

    def get(self, key: str, default: Any = None, env_var: Optional[str] = None) -> Any:
        """
        Get configuration value with environment variable override.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            env_var: Environment variable name to check first

        Returns:
            Configuration value
        """
        # Check environment variable first
        if env_var and os.getenv(env_var):
            return os.getenv(env_var)

        # Check environment variable with key name
        env_key = key.upper().replace(".", "_")
        if os.getenv(env_key):
            return os.getenv(env_key)

        # Navigate nested dictionary using dot notation
        value = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split(".")
        current = self.config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def load_config(config_file: Optional[str] = None) -> ConfigParser:
    """
    Load configuration from file or create default configuration.

    Args:
        config_file: Path to configuration file

    Returns:
        ConfigParser instance
    """
    if not config_file:
        # Look for default config files
        project_root = Path(__file__).parent.parent.parent.parent
        possible_configs = [
            project_root / "configs" / "config.json",
            project_root / "config.json",
            project_root / "configs" / "nifty50.json",
        ]

        for config_path in possible_configs:
            if config_path.exists():
                config_file = str(config_path)
                break

    return ConfigParser(config_file)


def load_nifty50_config() -> list[str]:
    """
    Load NIFTY 50 stock symbols from configuration.

    Returns:
        List of NIFTY 50 stock symbols
    """
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        nifty50_file = project_root / "configs" / "nifty50.json"

        if nifty50_file.exists():
            with open(nifty50_file) as f:
                data = json.load(f)
                return data.get("symbols", [])
        else:
            # Default NIFTY 50 symbols if file not found
            return get_default_nifty50_symbols()

    except Exception as e:
        print("Error loading NIFTY 50 config: {e}")
        return get_default_nifty50_symbols()


def get_default_nifty50_symbols() -> list[str]:
    """
    Get default NIFTY 50 symbols.

    Returns:
        List of NIFTY 50 stock symbols
    """
    return [
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "BHARTIARTL",
        "ICICIBANK",
        "INFOSYS",
        "SBIN",
        "LICI",
        "ITC",
        "HINDUNILVR",
        "LT",
        "HCLTECH",
        "MARUTI",
        "SUNPHARMA",
        "BAJFINANCE",
        "ONGC",
        "COALINDIA",
        "NTPC",
        "ASIANPAINT",
        "M&M",
        "NESTLEIND",
        "WIPRO",
        "ULTRACEMCO",
        "ADANIENT",
        "JSWSTEEL",
        "POWERGRID",
        "AXISBANK",
        "BAJAJFINSV",
        "KOTAKBANK",
        "TITAN",
        "INDUSINDBK",
        "TECHM",
        "GRASIM",
        "HDFCLIFE",
        "ADANIPORTS",
        "TATACONSUM",
        "HINDALCO",
        "TATAMOTORS",
        "CIPLA",
        "SBILIFE",
        "BAJAJ-AUTO",
        "BPCL",
        "EICHERMOT",
        "APOLLOHOSP",
        "HEROMOTOCO",
        "DRREDDY",
        "BRITANNIA",
        "TATASTEEL",
        "DIVISLAB",
        "UPL",
    ]


def get_trading_config() -> dict[str, Any]:
    """
    Get default trading configuration.

    Returns:
        Dictionary with trading configuration
    """
    return {
        "risk_management": {
            "max_position_size": 0.05,  # 5% of portfolio per position
            "max_daily_loss": 0.02,  # 2% daily loss limit
            "max_drawdown": 0.10,  # 10% max drawdown
            "stop_loss": 0.05,  # 5% stop loss
            "take_profit": 0.15,  # 15% take profit
        },
        "data": {
            "default_exchange": "NSE",
            "data_interval": "day",
            "lookback_days": 30,
        },
        "strategies": {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "momentum": {"lookback_period": 10, "threshold": 0.02},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        },
        "kite": {
            "environment": "production",
            "default_product": "MIS",
            "default_order_type": "MARKET",
        },
    }


def validate_kite_config() -> bool:
    """
    Validate KiteConnect configuration.

    Returns:
        True if configuration is valid, False otherwise
    """
    required_env_vars = ["KITE_API_KEY", "KITE_API_SECRET"]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("Missing required environment variables: {missing_vars}")
        return False

    return True


def create_sample_config() -> dict[str, Any]:
    """
    Create a sample configuration file structure.

    Returns:
        Dictionary with sample configuration
    """
    return {
        "project": {
            "name": "AI Trading Machine",
            "version": "1.0.0",
            "environment": "development",
        },
        "logging": {"level": "INFO", "file_output": True, "console_output": True},
        "data": {
            "source": "yfinance",
            "symbols": get_default_nifty50_symbols()[:10],  # First 10 for sample
            "interval": "1d",
            "period": "1y",
        },
        "trading": get_trading_config(),
        "kite": {
            "api_key": "your_api_key_here",
            "api_secret": "your_api_secret_here",
            "environment": "sandbox",
        },
    }


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully")

    # Test NIFTY 50 loading
    symbols = load_nifty50_config()
    print("Loaded {len(symbols)} NIFTY 50 symbols")

    # Test trading config
    trading_config = get_trading_config()
    print("Trading config: {trading_config}")

    # Validate Kite config
    is_valid = validate_kite_config()
    print("Kite config valid: {is_valid}")
