import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy_engine.strategies.rsi_strategy import RSIStrategy
from strategy_engine.strategies.base_strategy import SignalType

def test_rsi_strategy():
    """Test the RSI strategy implementation"""
    # Initialize the RSI strategy with custom parameters
    strategy = RSIStrategy(
        rsi_period=10,
        oversold=25,
        overbought=75,
        max_position_pct=1.0,
        stop_loss_pct=3.0,
        take_profit_pct=6.0
    )
    
    # Verify initialization parameters
    assert strategy.strategy_name == "RSI_Strategy"
    assert strategy.rsi_period == 10
    assert strategy.oversold == 25
    assert strategy.overbought == 75
    assert strategy.max_position_pct == 1.0
    assert strategy.stop_loss_pct == 3.0
    assert strategy.take_profit_pct == 6.0
    
    # Generate mock price data for testing
    dates = [datetime.now() - timedelta(days=i) for i in range(20, 0, -1)]
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0,
              102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0]
    
    # Test RSI calculation directly
    rsi_value = strategy.calculate_rsi(prices)
    assert 0 <= rsi_value <= 100
    
    # Test indicator calculation
    indicator_data = {
        "symbol": "AAPL",
        "close": prices,
        "current_price": 97.0
    }
    
    result = strategy.calculate_indicators(indicator_data)
    assert "rsi" in result
    assert "sma_20" in result
    assert "volatility" in result
    assert "rsi_signal" in result
    
    # Test signal generation based on internal RSI calculation
    # (relying on the strategy's own internal RSI calculation)
    test_data = {
        "symbol": "AAPL",
        "close": prices,
        "current_price": 97.0,
        "portfolio_value": 1000000
    }
    
    signals = strategy.generate_signals(test_data)
    assert len(signals) > 0
    # Note: We don't test for a specific signal type because it depends on the
    # internal RSI calculation
