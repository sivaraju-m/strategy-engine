"""
Technical indicators module for AI Trading Machine.

This module provides functions to calculate various technical indicators
used in trading strategy development and feature engineering.
"""

import numpy as np
import pandas as pd


def add_moving_averages(
    df: pd.DataFrame, price_col: str = "close", windows: list[int] = None
) -> pd.DataFrame:
    """
    Add Simple Moving Averages (SMA) for different window sizes.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        windows: List of window sizes

    Returns:
        DataFrame with additional SMA columns
    """
    if windows is None:
        windows = [5, 10, 20, 50, 200]

    result = df.copy()

    for window in windows:
        result["sma_{window}"] = result[price_col].rolling(window=window).mean()

    return result


def add_exponential_moving_averages(
    df: pd.DataFrame, price_col: str = "close", windows: list[int] = None
) -> pd.DataFrame:
    """
    Add Exponential Moving Averages (EMA) for different window sizes.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        windows: List of window sizes

    Returns:
        DataFrame with additional EMA columns
    """
    if windows is None:
        windows = [5, 10, 20, 50, 200]

    result = df.copy()

    for window in windows:
        result["ema_{window}"] = result[price_col].ewm(span=window, adjust=False).mean()

    return result


def add_rsi(
    df: pd.DataFrame, price_col: str = "close", windows: list[int] = None
) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) for different window sizes.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        windows: List of window sizes

    Returns:
        DataFrame with additional RSI columns
    """
    if windows is None:
        windows = [6, 14, 21]

    result = df.copy()

    for window in windows:
        # Calculate price changes
        delta = result[price_col].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result["rsi_{window}"] = 100 - (100 / (1 + rs))

    return result


def add_macd(
    df: pd.DataFrame,
    price_col: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Add Moving Average Convergence Divergence (MACD) indicator.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line

    Returns:
        DataFrame with additional MACD columns
    """
    result = df.copy()

    # Calculate fast and slow EMAs
    fast_ema = result[price_col].ewm(span=fast_period, adjust=False).mean()
    slow_ema = result[price_col].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    result["macd_line"] = fast_ema - slow_ema

    # Calculate signal line
    result["macd_signal"] = (
        result["macd_line"].ewm(span=signal_period, adjust=False).mean()
    )

    # Calculate histogram
    result["macd_histogram"] = result["macd_line"] - result["macd_signal"]

    return result


def add_bollinger_bands(
    df: pd.DataFrame, price_col: str = "close", window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Add Bollinger Bands indicator.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        window: Window size for moving average
        num_std: Number of standard deviations

    Returns:
        DataFrame with additional Bollinger Bands columns
    """
    result = df.copy()

    # Calculate middle band (SMA)
    result["bb_middle"] = result[price_col].rolling(window=window).mean()

    # Calculate standard deviation
    result["bb_std"] = result[price_col].rolling(window=window).std()

    # Calculate upper and lower bands
    result["bb_upper"] = result["bb_middle"] + (result["bb_std"] * num_std)
    result["bb_lower"] = result["bb_middle"] - (result["bb_std"] * num_std)

    # Calculate bandwidth and %B
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
    result["bb_pct_b"] = (result[price_col] - result["bb_lower"]) / (
        result["bb_upper"] - result["bb_lower"]
    )

    return result


def add_stochastic_oscillator(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Add Stochastic Oscillator indicator.

    Args:
        df: DataFrame with price data
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        k_period: Period for %K line
        d_period: Period for %D line

    Returns:
        DataFrame with additional Stochastic Oscillator columns
    """
    result = df.copy()

    # Calculate %K
    lowest_low = result[low_col].rolling(window=k_period).min()
    highest_high = result[high_col].rolling(window=k_period).max()

    result["stoch_k"] = 100 * (
        (result[close_col] - lowest_low) / (highest_high - lowest_low)
    )

    # Calculate %D (SMA of %K)
    result["stoch_d"] = result["stoch_k"].rolling(window=d_period).mean()

    return result


def add_atr(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:
    """
    Add Average True Range (ATR) indicator.

    Args:
        df: DataFrame with price data
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
        window: Period for ATR calculation

    Returns:
        DataFrame with additional ATR columns
    """
    result = df.copy()

    # Calculate True Range
    result["tr1"] = abs(result[high_col] - result[low_col])
    result["tr2"] = abs(result[high_col] - result[close_col].shift(1))
    result["tr3"] = abs(result[low_col] - result[close_col].shift(1))

    result["true_range"] = result[["tr1", "tr2", "tr3"]].max(axis=1)

    # Calculate ATR
    result["atr"] = result["true_range"].rolling(window=window).mean()

    # Drop intermediate columns
    result = result.drop(columns=["tr1", "tr2", "tr3", "true_range"])

    return result


def add_momentum_indicators(
    df: pd.DataFrame, price_col: str = "close", periods: list[int] = None
) -> pd.DataFrame:
    """
    Add momentum indicators for different periods.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        periods: List of periods for momentum calculation

    Returns:
        DataFrame with additional momentum columns
    """
    if periods is None:
        periods = [1, 5, 10, 20, 50]

    result = df.copy()

    for period in periods:
        # Simple momentum (current price / past price)
        result["momentum_{period}"] = result[price_col] / result[price_col].shift(
            period
        )

        # Percentage change
        result["pct_change_{period}"] = result[price_col].pct_change(periods=period)

        # Rate of change
        result["roc_{period}"] = (
            (result[price_col] - result[price_col].shift(period))
            / result[price_col].shift(period)
            * 100
        )

    return result


def add_volatility_indicators(
    df: pd.DataFrame, price_col: str = "close", windows: list[int] = None
) -> pd.DataFrame:
    """
    Add volatility indicators for different window sizes.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        windows: List of window sizes

    Returns:
        DataFrame with additional volatility columns
    """
    if windows is None:
        windows = [5, 10, 20, 50]

    result = df.copy()

    # Calculate returns
    result["daily_return"] = result[price_col].pct_change()

    for window in windows:
        # Standard deviation of returns
        result["volatility_{window}"] = (
            result["daily_return"].rolling(window=window).std()
        )

        # Normalized volatility (annualized)
        result["annualized_vol_{window}"] = result["volatility_{window}"] * np.sqrt(252)

    # Drop intermediate columns
    result = result.drop(columns=["daily_return"])

    return result


def add_all_indicators(
    df: pd.DataFrame, ohlcv_cols: dict[str, str] = None
) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.

    Args:
        df: DataFrame with price data
        ohlcv_cols: Dictionary mapping OHLCV column names
            (e.g., {'open': 'open_price', 'high': 'high_price', ...})

    Returns:
        DataFrame with all technical indicators
    """
    if ohlcv_cols is None:
        ohlcv_cols = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

    # Check if all required columns exist
    for col in ohlcv_cols.values():
        if col not in df.columns:
            raise ValueError("Required column '{col}' not found in DataFrame")

    result = df.copy()

    # Add moving averages
    result = add_moving_averages(result, price_col=ohlcv_cols["close"])
    result = add_exponential_moving_averages(result, price_col=ohlcv_cols["close"])

    # Add oscillators
    result = add_rsi(result, price_col=ohlcv_cols["close"])
    result = add_macd(result, price_col=ohlcv_cols["close"])
    result = add_stochastic_oscillator(
        result,
        high_col=ohlcv_cols["high"],
        low_col=ohlcv_cols["low"],
        close_col=ohlcv_cols["close"],
    )

    # Add volatility indicators
    result = add_bollinger_bands(result, price_col=ohlcv_cols["close"])
    result = add_atr(
        result,
        high_col=ohlcv_cols["high"],
        low_col=ohlcv_cols["low"],
        close_col=ohlcv_cols["close"],
    )
    result = add_volatility_indicators(result, price_col=ohlcv_cols["close"])

    # Add momentum indicators
    result = add_momentum_indicators(result, price_col=ohlcv_cols["close"])

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate technical indicators for trading data"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file with OHLCV data"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output CSV file with indicators"
    )
    parser.add_argument(
        "--open-col", default="open", help="Column name for open prices"
    )
    parser.add_argument(
        "--high-col", default="high", help="Column name for high prices"
    )
    parser.add_argument("--low-col", default="low", help="Column name for low prices")
    parser.add_argument(
        "--close-col", default="close", help="Column name for close prices"
    )
    parser.add_argument("--volume-col", default="volume", help="Column name for volume")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Map column names
    ohlcv_cols = {
        "open": args.open_col,
        "high": args.high_col,
        "low": args.low_col,
        "close": args.close_col,
        "volume": args.volume_col,
    }

    # Add indicators
    result_df = add_all_indicators(df, ohlcv_cols)

    # Save result
    result_df.to_csv(args.output, index=False)
    print("Added technical indicators and saved to {args.output}")
