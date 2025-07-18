import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Any, Dict, List

from strategy_engine.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, SEBILimits

class TestStrategy(BaseStrategy):
    """Test implementation of BaseStrategy for unit testing"""
    
    def __init__(self, strategy_name="Test Strategy", **kwargs):
        super().__init__(strategy_name=strategy_name, **kwargs)
    
    def generate_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Implementation of the abstract method for testing purposes"""
        signals = []
        
        if "symbol" not in data or "price" not in data:
            return signals
            
        # Create a simple signal based on provided data
        signal = TradingSignal(
            symbol=data["symbol"],
            signal=SignalType.BUY if data.get("buy_signal", True) else SignalType.SELL,
            confidence=data.get("confidence", 0.8),
            price=data["price"],
            timestamp=data.get("timestamp", datetime.now()),
            strategy_name=self.strategy_name,
            risk_metrics={"stop_loss": self.stop_loss_pct, "take_profit": self.take_profit_pct},
            position_size=data.get("position_size", 1.0),
            reasoning="Test signal for unit testing"
        )
        signals.append(signal)
        
        return signals
        
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the required abstract method for calculating indicators"""
        # Simple implementation for testing
        if not data or "dataframe" not in data:
            return data
            
        df = data["dataframe"]
        if isinstance(df, pd.DataFrame) and "close" in df.columns:
            # Add a simple moving average indicator
            if len(df) > 5:
                df["sma_5"] = df["close"].rolling(window=5).mean()
            
            # Add RSI if data is provided
            if "rsi" not in df.columns and len(df) > 14:
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["rsi"] = 100 - (100 / (1 + rs))
                
            data["dataframe"] = df
            
        return data


def test_strategy_base():
    """Test the base strategy functionality"""
    # Initialize the test strategy
    strategy = TestStrategy(
        strategy_name="UnitTestStrategy",
        max_position_pct=1.5,
        stop_loss_pct=3.0,
        take_profit_pct=7.0
    )
    
    # Verify initialization parameters
    assert strategy.strategy_name == "UnitTestStrategy"
    assert strategy.max_position_pct == 1.5
    assert strategy.stop_loss_pct == 3.0
    assert strategy.take_profit_pct == 7.0
    assert strategy.is_active == True
    assert isinstance(strategy.sebi_limits, SEBILimits)
    
    # Test signal generation
    test_data = {
        "symbol": "AAPL",
        "price": 150.0,
        "buy_signal": True,
        "confidence": 0.9,
        "position_size": 1.5
    }
    
    signals = strategy.generate_signals(test_data)
    
    # Verify generated signals
    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "AAPL"
    assert signal.signal == SignalType.BUY
    assert signal.confidence == 0.9
    assert signal.price == 150.0
    assert signal.strategy_name == "UnitTestStrategy"
    assert signal.position_size == 1.5
    
    # Test indicator calculation
    dates = [datetime.now()] * 20
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0,
              102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0]
    
    # Create a DataFrame with OHLC data
    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [10000] * len(prices)
    })
    
    indicator_data = {
        "symbol": "AAPL",
        "dataframe": df
    }
    
    result = strategy.calculate_indicators(indicator_data)
    
    # Verify indicators were added
    assert "dataframe" in result
    result_df = result["dataframe"]
    assert "sma_5" in result_df.columns
    assert "rsi" in result_df.columns
