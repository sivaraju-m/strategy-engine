"""
================================================================================
Simple Trading Strategies Implementation
================================================================================

This module provides simple and robust trading strategies for live trading. It
features:

- Includes SMA, RSI, and Momentum strategies
- Reliable and easy to use
- Real-time signal generation and backtesting

================================================================================
"""

"""
Simple Trading Strategies for Live Trading
=========================================

This module contains simple, robust trading strategies that can be used
for live trading and testing. These strategies are designed to be reliable
and easy to understand.

Author: AI Trading Machine
Licensed by SJ Trading
"""

import logging
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy:
    """
    Simple Moving Average Crossover Strategy.

    This strategy generates buy signals when the short-term moving average
    crosses above the long-term moving average, and sell signals when it
    crosses below.
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        Initialize the strategy.

        Args:
            short_window: Period for short-term moving average
            long_window: Period for long-term moving average
        """
        self.short_window = short_window
        self.long_window = long_window
        self.name = "SMA_{short_window}_{long_window}"

    def generate_signals(self, data) -> Any:
        """
        Generate trading signals based on moving average crossover.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals and confidence scores
        """
        if not PANDAS_AVAILABLE:
            logger.error("Pandas not available for strategy calculation")
            return None

        try:
            # Ensure we have enough data
            if len(data) < self.long_window:
                logger.warning("Insufficient data: {len(data)} < {self.long_window}")
                return pd.DataFrame()

            # Calculate moving averages
            data = data.copy()
            data["sma_short"] = data["close"].rolling(window=self.short_window).mean()
            data["sma_long"] = data["close"].rolling(window=self.long_window).mean()

            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals["signal"] = 0.0
            signals["confidence"] = 0.5

            # Buy signal: short MA crosses above long MA
            buy_condition = (data["sma_short"] > data["sma_long"]) & (
                data["sma_short"].shift(1) <= data["sma_long"].shift(1)
            )

            # Sell signal: short MA crosses below long MA
            sell_condition = (data["sma_short"] < data["sma_long"]) & (
                data["sma_short"].shift(1) >= data["sma_long"].shift(1)
            )

            # Set signals
            signals.loc[buy_condition, "signal"] = 1.0
            signals.loc[buy_condition, "confidence"] = 0.7

            signals.loc[sell_condition, "signal"] = -1.0
            signals.loc[sell_condition, "confidence"] = 0.7

            # Hold signal when short MA is above long MA
            hold_condition = data["sma_short"] > data["sma_long"]
            signals.loc[hold_condition, "signal"] = 0.5
            signals.loc[hold_condition, "confidence"] = 0.6

            return signals

        except Exception as e:
            logger.error("Strategy calculation failed: {e}")
            return pd.DataFrame() if PANDAS_AVAILABLE else None


class RSIStrategy:
    """
    Simple RSI (Relative Strength Index) Strategy.

    This strategy uses RSI to identify overbought and oversold conditions.
    """

    def __init__(
        self, rsi_period: int = 14, overbought: float = 70, oversold: float = 30
    ):
        """
        Initialize the RSI strategy.

        Args:
            rsi_period: Period for RSI calculation
            overbought: RSI level for overbought condition
            oversold: RSI level for oversold condition
        """
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.name = "RSI_{rsi_period}_{oversold}_{overbought}"

    def calculate_rsi(self, prices, period: int = 14):
        """Calculate RSI indicator."""
        if not PANDAS_AVAILABLE:
            return None

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data) -> Any:
        """
        Generate trading signals based on RSI.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals and confidence scores
        """
        if not PANDAS_AVAILABLE:
            logger.error("Pandas not available for strategy calculation")
            return None

        try:
            # Ensure we have enough data
            if len(data) < self.rsi_period * 2:
                logger.warning("Insufficient data: {len(data)} < {self.rsi_period * 2}")
                return pd.DataFrame()

            # Calculate RSI
            data = data.copy()
            data["rsi"] = self.calculate_rsi(data["close"], self.rsi_period)

            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals["signal"] = 0.0
            signals["confidence"] = 0.5

            # Buy signal: RSI below oversold threshold
            buy_condition = data["rsi"] < self.oversold

            # Sell signal: RSI above overbought threshold
            sell_condition = data["rsi"] > self.overbought

            # Set signals with confidence based on how extreme the RSI is
            signals.loc[buy_condition, "signal"] = 1.0
            signals.loc[buy_condition, "confidence"] = (
                0.6 + (self.oversold - data.loc[buy_condition, "rsi"]) / 100
            )

            signals.loc[sell_condition, "signal"] = -1.0
            signals.loc[sell_condition, "confidence"] = (
                0.6 + (data.loc[sell_condition, "rsi"] - self.overbought) / 100
            )

            # Neutral zone
            neutral_condition = (data["rsi"] >= self.oversold) & (
                data["rsi"] <= self.overbought
            )
            signals.loc[neutral_condition, "signal"] = 0.0
            signals.loc[neutral_condition, "confidence"] = 0.4

            # Cap confidence at 0.9
            signals["confidence"] = signals["confidence"].clip(upper=0.9)

            return signals

        except Exception as e:
            logger.error("RSI strategy calculation failed: {e}")
            return pd.DataFrame() if PANDAS_AVAILABLE else None


class MomentumStrategy:
    """
    Simple Momentum Strategy.

    This strategy identifies momentum based on price changes over a period.
    """

    def __init__(self, lookback_period: int = 10, threshold: float = 0.02):
        """
        Initialize the momentum strategy.

        Args:
            lookback_period: Period to look back for momentum calculation
            threshold: Minimum price change percentage to generate signal
        """
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.name = "Momentum_{lookback_period}_{threshold}"

    def generate_signals(self, data) -> Any:
        """
        Generate trading signals based on momentum.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with signals and confidence scores
        """
        if not PANDAS_AVAILABLE:
            logger.error("Pandas not available for strategy calculation")
            return None

        try:
            # Ensure we have enough data
            if len(data) < self.lookback_period + 5:
                logger.warning(
                    "Insufficient data: {len(data)} < {self.lookback_period + 5}"
                )
                return pd.DataFrame()

            # Calculate momentum
            data = data.copy()
            data["momentum"] = (
                data["close"] - data["close"].shift(self.lookback_period)
            ) / data["close"].shift(self.lookback_period)

            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals["signal"] = 0.0
            signals["confidence"] = 0.5

            # Buy signal: positive momentum above threshold
            buy_condition = data["momentum"] > self.threshold

            # Sell signal: negative momentum below negative threshold
            sell_condition = data["momentum"] < -self.threshold

            # Set signals with confidence based on momentum strength
            signals.loc[buy_condition, "signal"] = 1.0
            signals.loc[buy_condition, "confidence"] = 0.5 + (
                data.loc[buy_condition, "momentum"] * 2
            ).clip(upper=0.4)

            signals.loc[sell_condition, "signal"] = -1.0
            signals.loc[sell_condition, "confidence"] = 0.5 + (
                -data.loc[sell_condition, "momentum"] * 2
            ).clip(upper=0.4)

            # Weak momentum
            weak_momentum = data["momentum"].abs() <= self.threshold
            signals.loc[weak_momentum, "signal"] = 0.0
            signals.loc[weak_momentum, "confidence"] = 0.3

            return signals

        except Exception as e:
            logger.error("Momentum strategy calculation failed: {e}")
            return pd.DataFrame() if PANDAS_AVAILABLE else None


# Strategy factory for easy access
AVAILABLE_STRATEGIES = {
    "sma": SimpleMovingAverageStrategy,
    "rsi": RSIStrategy,
    "momentum": MomentumStrategy,
    "adaptive_rsi": RSIStrategy,  # Alias for compatibility
}


def create_strategy(strategy_name: str, **kwargs) -> Any:
    """
    Create a strategy instance by name.

    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance
    """
    strategy_name = strategy_name.lower()

    if strategy_name not in AVAILABLE_STRATEGIES:
        logger.error("Unknown strategy: {strategy_name}")
        logger.info("Available strategies: {list(AVAILABLE_STRATEGIES.keys())}")
        return None

    strategy_class = AVAILABLE_STRATEGIES[strategy_name]

    try:
        return strategy_class(**kwargs)
    except Exception as e:
        logger.error("Failed to create strategy {strategy_name}: {e}")
        return None


def get_strategy_list() -> list:
    """Get list of available strategies."""
    return list(AVAILABLE_STRATEGIES.keys())


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Simple Trading Strategies - Test Mode")
    print("=" * 50)

    # Test strategy creation
    for strategy_name in get_strategy_list():
        print("\n📊 Testing {strategy_name} strategy...")
        strategy = create_strategy(strategy_name)
        if strategy:
            print("✅ {strategy.name} created successfully")
        else:
            print("❌ Failed to create {strategy_name}")

    # Test with mock data if pandas is available
    if PANDAS_AVAILABLE:
        print("\n🧪 Testing with mock data...")

        # Create mock price data
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        mock_data = pd.DataFrame(
            {
                "open": np.random.normal(100, 2, 50),
                "high": np.random.normal(102, 2, 50),
                "low": np.random.normal(98, 2, 50),
                "close": np.random.normal(100, 2, 50),
                "volume": np.random.randint(100000, 1000000, 50),
            },
            index=dates,
        )

        # Test SMA strategy
        sma_strategy = create_strategy("sma")
        if sma_strategy:
            signals = sma_strategy.generate_signals(mock_data)
            if signals is not None and not signals.empty:
                print("✅ SMA strategy generated {len(signals)} signals")
                print(
                    "📊 Signal distribution: {signals['signal'].value_counts().to_dict()}"
                )
            else:
                print("⚠️  SMA strategy returned empty signals")

        print("\n✅ Strategy testing completed")
    else:
        print("\n⚠️  Pandas not available - skipping data tests")

    print("=" * 50)
