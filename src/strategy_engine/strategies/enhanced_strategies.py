#!/usr/bin/env python3
"""
Enhanced Trading Strategies
Advanced technical analysis and signal generation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    signal: SignalType
    confidence: float
    price: float
    timestamp: Any
    reason: str
    indicators: dict[str, float] = None


class EnhancedRSIStrategy:
    """
    Enhanced RSI Strategy with multiple timeframes and adaptive thresholds
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        adaptive_thresholds: bool = True,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.adaptive_thresholds = adaptive_thresholds

    def calculate_rsi(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate RSI with improved accuracy"""
        if period is None:
            period = self.period

        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use Wilder's smoothing (exponential moving average)
        alpha = 1.0 / period
        avg_gains = np.zeros(len(gains))
        avg_losses = np.zeros(len(losses))

        # Initial average
        avg_gains[period - 1] = np.mean(gains[:period])
        avg_losses[period - 1] = np.mean(losses[:period])

        # Calculate smoothed averages
        for i in range(period, len(gains)):
            avg_gains[i] = alpha * gains[i] + (1 - alpha) * avg_gains[i - 1]
            avg_losses[i] = alpha * losses[i] + (1 - alpha) * avg_losses[i - 1]

        # Calculate RSI
        rs = np.divide(
            avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0
        )
        rsi = 100 - (100 / (1 + rs))

        # Pad with 50 for initial values
        rsi_full = np.full(len(prices), 50.0)
        rsi_full[period:] = rsi[period - 1 :]

        return rsi_full

    def get_adaptive_thresholds(
        self, rsi_values: np.ndarray, lookback: int = 100
    ) -> tuple[float, float]:
        """Calculate adaptive RSI thresholds based on historical distribution"""
        if len(rsi_values) < lookback:
            return self.oversold, self.overbought

        recent_rsi = rsi_values[-lookback:]
        p25 = np.percentile(recent_rsi, 25)
        p75 = np.percentile(recent_rsi, 75)

        # Adaptive thresholds based on recent distribution
        adaptive_oversold = max(20, min(35, p25))
        adaptive_overbought = min(80, max(65, p75))

        return adaptive_oversold, adaptive_overbought

    def generate_signals(self, df: pd.DataFrame) -> list[TradingSignal]:
        """Generate enhanced RSI signals"""
        prices = df["close"].values
        rsi = self.calculate_rsi(prices)

        # Multi-timeframe RSI
        rsi_short = self.calculate_rsi(prices, period=7)
        rsi_long = self.calculate_rsi(prices, period=21)

        signals = []

        for i in range(len(df)):
            current_rsi = rsi[i]
            current_rsi_short = rsi_short[i]
            current_rsi_long = rsi_long[i]
            price = prices[i]
            timestamp = df.index[i]

            # Get thresholds
            if self.adaptive_thresholds and i >= 100:
                oversold_thresh, overbought_thresh = self.get_adaptive_thresholds(
                    rsi[: i + 1]
                )
            else:
                oversold_thresh, overbought_thresh = self.oversold, self.overbought

            # Signal generation logic
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Neutral RSI"

            # Strong buy signals
            if (
                current_rsi < oversold_thresh
                and current_rsi_short < oversold_thresh
                and current_rsi > current_rsi_short
            ):  # RSI starting to turn up
                signal_type = SignalType.BUY
                confidence = min(
                    0.9, 0.6 + (oversold_thresh - current_rsi) / oversold_thresh * 0.3
                )
                reason = "Oversold RSI({current_rsi:.1f}) with upturn"

            # Moderate buy signals
            elif current_rsi < oversold_thresh:
                signal_type = SignalType.BUY
                confidence = (
                    0.6 + (oversold_thresh - current_rsi) / oversold_thresh * 0.2
                )
                reason = "Oversold RSI({current_rsi:.1f})"

            # Strong sell signals
            elif (
                current_rsi > overbought_thresh
                and current_rsi_short > overbought_thresh
                and current_rsi < current_rsi_short
            ):  # RSI starting to turn down
                signal_type = SignalType.SELL
                confidence = min(
                    0.9,
                    0.6
                    + (current_rsi - overbought_thresh)
                    / (100 - overbought_thresh)
                    * 0.3,
                )
                reason = "Overbought RSI({current_rsi:.1f}) with downturn"

            # Moderate sell signals
            elif current_rsi > overbought_thresh:
                signal_type = SignalType.SELL
                confidence = (
                    0.6
                    + (current_rsi - overbought_thresh)
                    / (100 - overbought_thresh)
                    * 0.2
                )
                reason = "Overbought RSI({current_rsi:.1f})"

            indicators = {
                "rsi": current_rsi,
                "rsi_short": current_rsi_short,
                "rsi_long": current_rsi_long,
                "oversold_thresh": oversold_thresh,
                "overbought_thresh": overbought_thresh,
            }

            signals.append(
                TradingSignal(
                    signal=signal_type,
                    confidence=confidence,
                    price=price,
                    timestamp=timestamp,
                    reason=reason,
                    indicators=indicators,
                )
            )

        return signals


class EnhancedMomentumStrategy:
    """
    Enhanced Momentum Strategy with multiple timeframes and trend confirmation
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        momentum_threshold: float = 0.02,
        trend_confirmation: bool = True,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_threshold = momentum_threshold
        self.trend_confirmation = trend_confirmation

    def calculate_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate momentum over a given window"""
        momentum = np.zeros(len(prices))
        for i in range(window, len(prices)):
            momentum[i] = (prices[i] - prices[i - window]) / prices[i - window]
        return momentum

    def calculate_moving_average(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate simple moving average"""
        ma = np.zeros(len(prices))
        for i in range(window - 1, len(prices)):
            ma[i] = np.mean(prices[i - window + 1 : i + 1])
        return ma

    def calculate_volatility(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility (standard deviation of returns)"""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.zeros(len(prices))

        for i in range(window, len(prices)):
            volatility[i] = np.std(returns[i - window : i])

        return volatility

    def generate_signals(self, df: pd.DataFrame) -> list[TradingSignal]:
        """Generate enhanced momentum signals"""
        prices = df["close"].values
        volumes = df["volume"].values

        # Calculate indicators
        short_momentum = self.calculate_momentum(prices, self.short_window)
        long_momentum = self.calculate_momentum(prices, self.long_window)
        ma_short = self.calculate_moving_average(prices, self.short_window)
        ma_long = self.calculate_moving_average(prices, self.long_window)
        volatility = self.calculate_volatility(prices)

        # Volume moving average for confirmation
        volume_ma = self.calculate_moving_average(volumes, 20)

        signals = []

        for i in range(len(df)):
            price = prices[i]
            timestamp = df.index[i]

            current_short_momentum = short_momentum[i]
            current_long_momentum = long_momentum[i]
            current_volatility = volatility[i]

            # Trend confirmation
            trend_up = ma_short[i] > ma_long[i] if i >= self.long_window else False
            trend_down = ma_short[i] < ma_long[i] if i >= self.long_window else False

            # Volume confirmation
            volume_confirmation = volumes[i] > volume_ma[i] * 1.2 if i >= 20 else False

            # Signal generation
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Neutral momentum"

            # Strong buy signals (positive momentum + trend + volume)
            if (
                current_short_momentum > self.momentum_threshold
                and current_long_momentum > 0
                and (not self.trend_confirmation or trend_up)
                and volume_confirmation
            ):
                signal_type = SignalType.BUY
                confidence = min(
                    0.9,
                    0.6
                    + min(current_short_momentum / (self.momentum_threshold * 2), 0.3),
                )
                reason = (
                    "Strong momentum({current_short_momentum:.3f}) with trend & volume"
                )

            # Moderate buy signals
            elif current_short_momentum > self.momentum_threshold:
                signal_type = SignalType.BUY
                confidence = 0.6 + min(
                    current_short_momentum / (self.momentum_threshold * 2), 0.2
                )
                reason = "Positive momentum({current_short_momentum:.3f})"

            # Strong sell signals (negative momentum + trend + volume)
            elif (
                current_short_momentum < -self.momentum_threshold
                and current_long_momentum < 0
                and (not self.trend_confirmation or trend_down)
                and volume_confirmation
            ):
                signal_type = SignalType.SELL
                confidence = min(
                    0.9,
                    0.6
                    + min(
                        abs(current_short_momentum) / (self.momentum_threshold * 2), 0.3
                    ),
                )
                reason = "Strong negative momentum({current_short_momentum:.3f}) with trend & volume"

            # Moderate sell signals
            elif current_short_momentum < -self.momentum_threshold:
                signal_type = SignalType.SELL
                confidence = 0.6 + min(
                    abs(current_short_momentum) / (self.momentum_threshold * 2), 0.2
                )
                reason = "Negative momentum({current_short_momentum:.3f})"

            indicators = {
                "short_momentum": current_short_momentum,
                "long_momentum": current_long_momentum,
                "ma_short": ma_short[i],
                "ma_long": ma_long[i],
                "volatility": current_volatility,
                "volume_ratio": volumes[i] / volume_ma[i] if volume_ma[i] > 0 else 1,
                "trend_up": trend_up,
                "trend_down": trend_down,
            }

            signals.append(
                TradingSignal(
                    signal=signal_type,
                    confidence=confidence,
                    price=price,
                    timestamp=timestamp,
                    reason=reason,
                    indicators=indicators,
                )
            )

        return signals


class MACDStrategy:
    """
    MACD (Moving Average Convergence Divergence) Strategy
    """

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros(len(prices))
        alpha = 2.0 / (period + 1)

        # Initialize with simple moving average
        ema[period - 1] = np.mean(prices[:period])

        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_macd(
        self, prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD line, signal line, and histogram"""
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)

        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signals(self, df: pd.DataFrame) -> list[TradingSignal]:
        """Generate MACD signals"""
        prices = df["close"].values
        macd_line, signal_line, histogram = self.calculate_macd(prices)

        signals = []

        for i in range(1, len(df)):  # Start from 1 for comparison with previous
            price = prices[i]
            timestamp = df.index[i]

            current_macd = macd_line[i]
            current_signal = signal_line[i]
            current_histogram = histogram[i]

            prev_macd = macd_line[i - 1]
            prev_signal = signal_line[i - 1]
            prev_histogram = histogram[i - 1]

            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Neutral MACD"

            # MACD crossover signals
            if prev_macd <= prev_signal and current_macd > current_signal:
                # Bullish crossover
                signal_type = SignalType.BUY
                confidence = 0.7 + min(
                    abs(current_macd - current_signal) / abs(current_macd) * 0.2, 0.2
                )
                reason = "MACD bullish crossover"

            elif prev_macd >= prev_signal and current_macd < current_signal:
                # Bearish crossover
                signal_type = SignalType.SELL
                confidence = 0.7 + min(
                    abs(current_macd - current_signal) / abs(current_macd) * 0.2, 0.2
                )
                reason = "MACD bearish crossover"

            # Histogram momentum signals
            elif current_histogram > 0 and prev_histogram <= 0:
                signal_type = SignalType.BUY
                confidence = 0.6
                reason = "MACD histogram turned positive"

            elif current_histogram < 0 and prev_histogram >= 0:
                signal_type = SignalType.SELL
                confidence = 0.6
                reason = "MACD histogram turned negative"

            indicators = {
                "macd_line": current_macd,
                "signal_line": current_signal,
                "histogram": current_histogram,
                "crossover_strength": abs(current_macd - current_signal),
            }

            signals.append(
                TradingSignal(
                    signal=signal_type,
                    confidence=confidence,
                    price=price,
                    timestamp=timestamp,
                    reason=reason,
                    indicators=indicators,
                )
            )

        # Add initial neutral signal
        if len(signals) < len(df):
            signals.insert(
                0,
                TradingSignal(
                    signal=SignalType.HOLD,
                    confidence=0.5,
                    price=prices[0],
                    timestamp=df.index[0],
                    reason="Initial state",
                    indicators={"macd_line": 0, "signal_line": 0, "histogram": 0},
                ),
            )

        return signals


# Wrapper functions for backward compatibility
def enhanced_rsi_signals(df: pd.DataFrame, **kwargs) -> tuple[list[int], float]:
    """Wrapper function that returns simple signal format"""
    strategy = EnhancedRSIStrategy(**kwargs)
    signals = strategy.generate_signals(df)

    signal_values = [s.signal.value for s in signals]
    avg_confidence = sum(s.confidence for s in signals) / len(signals)

    return signal_values, avg_confidence


def enhanced_momentum_signals(df: pd.DataFrame, **kwargs) -> tuple[list[int], float]:
    """Wrapper function that returns simple signal format"""
    strategy = EnhancedMomentumStrategy(**kwargs)
    signals = strategy.generate_signals(df)

    signal_values = [s.signal.value for s in signals]
    avg_confidence = sum(s.confidence for s in signals) / len(signals)

    return signal_values, avg_confidence


def macd_signals(df: pd.DataFrame, **kwargs) -> tuple[list[int], float]:
    """Wrapper function that returns simple signal format"""
    strategy = MACDStrategy(**kwargs)
    signals = strategy.generate_signals(df)

    signal_values = [s.signal.value for s in signals]
    avg_confidence = sum(s.confidence for s in signals) / len(signals)

    return signal_values, avg_confidence
