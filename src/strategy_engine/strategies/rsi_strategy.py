"""
================================================================================
RSI Strategy Implementation
================================================================================

This module implements a mean reversion strategy based on Relative Strength
Index (RSI). It features:

- Buy signals when RSI < oversold threshold
- Sell signals when RSI > overbought threshold
- SEBI compliance and risk management

================================================================================
"""

import logging
from datetime import datetime
from typing import Any

from .base_strategy import BaseStrategy, SignalType, TradingSignal

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy with SEBI compliance

    Strategy Logic:
    - Buy when RSI < oversold_threshold (default 30)
    - Sell when RSI > overbought_threshold (default 70)
    - Include margin calculations, position limits, and risk-adjusted returns
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        **kwargs,
    ):
        super().__init__(strategy_name="RSI_Strategy", **kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

        logger.info(
            "🎯 RSI Strategy initialized: period={rsi_period}, "
            "oversold={oversold}, overbought={overbought}"
        )

    def calculate_rsi(self, prices: list[float], period: int = None) -> float:
        """
        Calculate RSI (Relative Strength Index)

        Args:
            prices: List of closing prices
            period: RSI period (defaults to self.rsi_period)

        Returns:
            RSI value (0-100)
        """
        if period is None:
            period = self.rsi_period

        if len(prices) < period + 1:
            return 50.0  # Neutral RSI when insufficient data

        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            changes.append(prices[i] - prices[i - 1])

        if len(changes) < period:
            return 50.0

        # Separate gains and losses
        gains = [max(change, 0) for change in changes[-period:]]
        losses = [-min(change, 0) for change in changes[-period:]]

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Calculate RSI
        if avg_loss == 0:
            return 100.0  # No losses, RSI = 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate RSI and other technical indicators

        Args:
            data: Market data dictionary with 'close' prices

        Returns:
            Dictionary with additional indicator data
        """
        if "close" not in data:
            logger.error("❌ Missing 'close' prices in data")
            return data

        close_prices = data["close"]
        if (
            not isinstance(close_prices, list)
            or len(close_prices) < self.rsi_period + 1
        ):
            logger.warning(
                "⚠️ Insufficient data for RSI calculation: {len(close_prices) if isinstance(close_prices, list) else 'invalid'}"
            )
            return data

        # Calculate RSI
        rsi = self.calculate_rsi(close_prices)

        # Calculate additional indicators
        sma_20 = self._calculate_sma(close_prices, 20)
        volatility = self._calculate_volatility(close_prices)

        # Add indicators to data
        result = data.copy()
        result.update(
            {
                "rsi": rsi,
                "sma_20": sma_20,
                "volatility": volatility,
                "rsi_signal": self._get_rsi_signal(rsi),
            }
        )

        logger.debug(
            "📊 Indicators calculated: RSI={rsi:.2f}, SMA20={sma_20:.2f}, Vol={volatility:.4f}"
        )
        return result

    def _calculate_sma(self, prices: list[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return sum(prices[-period:]) / period

    def _calculate_volatility(self, prices: list[float]) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance**0.5

    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal based on thresholds"""
        if rsi <= self.oversold:
            return "BUY"
        elif rsi >= self.overbought:
            return "SELL"
        else:
            return "HOLD"

    def generate_signals(self, data: dict[str, Any]) -> list[TradingSignal]:
        """
        Generate trading signals based on RSI analysis

        Args:
            data: Market data dictionary with symbol, close prices, etc.

        Returns:
            List of trading signals with confidence scores
        """
        signals = []

        # Validate required data
        if not all(key in data for key in ["symbol", "close", "current_price"]):
            logger.error(
                "❌ Missing required data fields: symbol, close, current_price"
            )
            return signals

        symbol = data["symbol"]
        current_price = data["current_price"]
        close_prices = data["close"]

        # Calculate indicators
        indicators = self.calculate_indicators(data)
        rsi = indicators.get("rsi", 50.0)
        volatility = indicators.get("volatility", 0.0)

        # Generate signal based on RSI
        signal_type = None
        confidence = 0.0
        reasoning = ""

        if rsi <= self.oversold:
            signal_type = SignalType.BUY
            confidence = min(
                1.0, (self.oversold - rsi) / self.oversold
            )  # Higher confidence for lower RSI
            reasoning = "RSI oversold at {rsi:.2f} (threshold: {self.oversold})"
        elif rsi >= self.overbought:
            signal_type = SignalType.SELL
            confidence = min(1.0, (rsi - self.overbought) / (100 - self.overbought))
            reasoning = "RSI overbought at {rsi:.2f} (threshold: {self.overbought})"
        else:
            signal_type = SignalType.HOLD
            confidence = 0.1  # Low confidence for hold signals
            reasoning = (
                "RSI neutral at {rsi:.2f} (between {self.oversold}-{self.overbought})"
            )

        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(close_prices, signal_type)

        # Calculate position size (only for BUY/SELL signals)
        position_size = 0.0
        if signal_type in [SignalType.BUY, SignalType.SELL] and confidence > 0.3:
            portfolio_value = data.get(
                "portfolio_value", 1000000
            )  # Default 10L portfolio
            position_size = self.calculate_position_size(
                current_price, portfolio_value, volatility, confidence
            )

        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now(),
            strategy_name=self.strategy_name,
            risk_metrics=risk_metrics,
            position_size=position_size,
            reasoning=reasoning,
        )

        signals.append(signal)

        logger.info(
            "🎯 {symbol}: {signal_type.value} signal with {confidence:.2f} confidence "
            "(RSI: {rsi:.2f}, Position: {position_size} shares)"
        )

        return signals
