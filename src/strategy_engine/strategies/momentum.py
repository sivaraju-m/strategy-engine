"""
================================================================================
Momentum Strategy Implementation
================================================================================

This module generates trading signals based on price momentum. It features:

- Momentum indicators from historical price data
- Real-time signal generation and backtesting
- Designed for integration into larger trading systems

================================================================================
"""

from typing import Union

import pandas as pd


class MomentumStrategy:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period

    def run(self, df: pd.DataFrame) -> dict[str, Union[float, int]]:
        """
        Calculate momentum based signals
        Args:
            df: DataFrame with OHLCV data
        Returns:
            dict with signal, confidence and returns
        """
        try:
            if df.empty or len(df) < self.lookback_period:
                return {"signal": 0, "confidence": 0.0, "returns": 0.0}

            if "close" not in df.columns:
                raise ValueError("DataFrame must contain 'close' column")

            # Calculate momentum indicators
            df = df.copy()
            returns = df["close"].pct_change(self.lookback_period)
            returns = returns.fillna(0)

            latest_return = float(returns.iloc[-1])
            signal = 1 if latest_return > 0 else -1
            confidence = abs(latest_return)

            return {
                "signal": signal,
                "confidence": round(confidence, 4),
                "returns": round(latest_return, 4),
            }

        except Exception as e:
            print("Error in momentum strategy: {str(e)}")
            return {"signal": 0, "confidence": 0.0, "returns": 0.0}
