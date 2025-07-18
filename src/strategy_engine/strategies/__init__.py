"""
Strategy module for the strategy engine

This module provides various trading strategies.
"""

from strategy_engine.strategies.base_strategy import (
    BaseStrategy,
    StrategyType,
    StrategyMetadata,
    SignalType,
    TradingSignal,
    SEBILimits
)

# For backwards compatibility and easier imports
strategy_base = BaseStrategy

try:
    from .momentum import MomentumStrategy
    from .rsi_strategy import RSIStrategy
except ImportError:
    # Fallback for when modules are not available
    RSIStrategy = None
    MomentumStrategy = None


def run_rsi(df, **kwargs):
    """Run RSI strategy on DataFrame with parameters"""
    try:
        # Extract RSI parameters with defaults
        rsi_period = kwargs.get("period", kwargs.get("rsi_period", 14))
        oversold = kwargs.get("oversold", 30)
        overbought = kwargs.get("overbought", 70)

        # Create strategy with parameters
        strategy = RSIStrategy(
            rsi_period=rsi_period, oversold=oversold, overbought=overbought
        )

        # Convert DataFrame to dict format expected by RSIStrategy
        # yfinance_loader converts columns to lowercase
        data = {
            "symbol": "TEST",  # Add required symbol field
            "close": df["close"].tolist(),
            "high": df["high"].tolist(),
            "low": df["low"].tolist(),
            "volume": df["volume"].tolist(),
            "timestamp": df.index.tolist(),
            "current_price": float(
                df["close"].iloc[-1]
            ),  # Add required current_price field
        }

        # Calculate indicators and generate signals
        data_with_indicators = strategy.calculate_indicators(data)
        signals = strategy.generate_signals(data_with_indicators)

        # Convert to simple signal format for backtesting
        signal_values = []
        for signal in signals:
            if signal.signal.name == "BUY":
                signal_values.append(1)
            elif signal.signal.name == "SELL":
                signal_values.append(-1)
            else:
                signal_values.append(0)

        # Pad with zeros if needed
        while len(signal_values) < len(df):
            signal_values.append(0)

        return signal_values, 0.8  # signals, confidence

    except Exception as e:
        print("Error in RSI strategy: {e}")
        return [0] * len(df), 0.0


def run_momentum(df, **kwargs):
    """Run Momentum strategy on DataFrame with parameters"""
    try:
        # yfinance_loader already provides lowercase columns
        strategy = MomentumStrategy(**kwargs)
        result = strategy.run(df)

        # Convert to signal array format
        signal = result.get("signal", 0)
        confidence = result.get("confidence", 0.0)
        signals = [signal] * len(df)  # Simple: same signal for all periods

        return signals, confidence

    except Exception as e:
        print("Error in momentum strategy: {e}")
        return [0] * len(df), 0.0


registry = {"rsi": run_rsi, "momentum": run_momentum}
