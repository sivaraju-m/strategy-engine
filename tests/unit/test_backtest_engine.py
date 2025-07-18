import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from strategy_engine.backtest.engine import calculate_performance_metrics, BacktestResult

def test_calculate_performance_metrics():
    """Test the performance metrics calculation function"""
    # Create sample returns, prices, and trades for testing
    # Simple upward trend with 10% return
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01, -0.01, 0.02, 0.01, 0.005, 0.015])
    prices = np.array([100.0, 101.0, 103.0, 102.5, 104.0, 105.0, 104.0, 106.0, 107.0, 107.5, 109.0])
    
    # Sample trades
    trades = [
        {
            "symbol": "AAPL",
            "entry_price": 100.0,
            "entry_date": datetime.now() - timedelta(days=10),
            "exit_price": 103.0,
            "exit_date": datetime.now() - timedelta(days=8),
            "profit_pct": 3.0,
            "profit_amount": 3.0,
            "position_size": 1.0,
            "position_type": "LONG",
            "pnl": 3.0  # Adding pnl field as expected by the implementation
        },
        {
            "symbol": "AAPL",
            "entry_price": 104.0,
            "entry_date": datetime.now() - timedelta(days=6),
            "exit_price": 102.0,
            "exit_date": datetime.now() - timedelta(days=5),
            "profit_pct": -1.92,
            "profit_amount": -2.0,
            "position_size": 1.0,
            "position_type": "LONG",
            "pnl": -2.0  # Adding pnl field as expected by the implementation
        },
        {
            "symbol": "AAPL",
            "entry_price": 104.0,
            "entry_date": datetime.now() - timedelta(days=4),
            "exit_price": 109.0,
            "exit_date": datetime.now(),
            "profit_pct": 4.81,
            "profit_amount": 5.0,
            "position_size": 1.0,
            "position_type": "LONG",
            "pnl": 5.0  # Adding pnl field as expected by the implementation
        }
    ]
    
    # Calculate metrics
    result = calculate_performance_metrics(returns, prices, trades)
    
    # Validate result is a BacktestResult
    assert isinstance(result, BacktestResult)
    
    # Check core metrics are calculated and have reasonable values
    assert hasattr(result, 'total_return')
    assert hasattr(result, 'annualized_return')
    assert hasattr(result, 'sharpe_ratio')
    assert hasattr(result, 'max_drawdown')
    assert hasattr(result, 'win_rate')
    
    # Check trade analysis
    assert result.total_trades == 3
    assert result.profitable_trades == 2
    assert result.losing_trades == 1
    assert result.win_rate == pytest.approx(2/3, 0.01)  # Should be about 66.7%
