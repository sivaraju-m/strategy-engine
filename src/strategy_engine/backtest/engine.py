from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BacktestResult:
    """Comprehensive backtest results container"""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    portfolio_value: list[float]
    drawdown_series: list[float]
    trade_log: list[dict[str, Any]]


def calculate_performance_metrics(
    returns: np.ndarray, prices: np.ndarray, trades: list[dict[str, Any]]
) -> BacktestResult:
    """
    Calculate comprehensive performance metrics for backtesting

    Args:
        returns: Array of strategy returns
        prices: Array of asset prices
        trades: List of trade dictionaries with entry/exit info

    Returns:
        BacktestResult with all performance metrics
    """
    if len(returns) == 0:
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            total_trades=0,
            profitable_trades=0,
            losing_trades=0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_duration=0.0,
            portfolio_value=[],
            drawdown_series=[],
            trade_log=[],
        )

    # Portfolio value calculation
    portfolio_value = np.cumprod(1 + returns)

    # Basic metrics
    total_return = float(portfolio_value[-1] - 1)
    annualized_return = float((portfolio_value[-1] ** (252 / len(returns))) - 1)
    volatility = float(np.std(returns) * np.sqrt(252))

    # Sharpe ratio (assuming 5% risk-free rate)
    risk_free_rate = 0.05
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = (
        float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
        if np.std(excess_returns) > 0
        else 0.0
    )

    # Drawdown calculation
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown_series = (portfolio_value - running_max) / running_max
    max_drawdown = float(np.min(drawdown_series))

    # Calmar ratio
    calmar_ratio = (
        float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    )

    # Trade analysis
    if trades:
        profitable_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in trades if t.get("pnl", 0) <= 0])
        total_trades = len(trades)

        win_rate = float(profitable_trades / total_trades) if total_trades > 0 else 0.0

        winning_trades_pnl = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
        losing_trades_pnl = [t["pnl"] for t in trades if t.get("pnl", 0) <= 0]

        avg_win = float(np.mean(winning_trades_pnl)) if winning_trades_pnl else 0.0
        avg_loss = float(np.mean(losing_trades_pnl)) if losing_trades_pnl else 0.0

        largest_win = float(max(winning_trades_pnl)) if winning_trades_pnl else 0.0
        largest_loss = float(min(losing_trades_pnl)) if losing_trades_pnl else 0.0

        profit_factor = (
            float(abs(sum(winning_trades_pnl) / sum(losing_trades_pnl)))
            if sum(losing_trades_pnl) != 0
            else float("in")
        )

        # Average trade duration (assuming each trade has duration in days)
        durations = [t.get("duration", 1) for t in trades]
        avg_trade_duration = float(np.mean(durations)) if durations else 0.0
    else:
        profitable_trades = losing_trades = total_trades = 0
        win_rate = avg_win = avg_loss = largest_win = largest_loss = (
            avg_trade_duration
        ) = 0.0
        profit_factor = 0.0

    return BacktestResult(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        total_trades=total_trades,
        profitable_trades=profitable_trades,
        losing_trades=losing_trades,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_trade_duration=avg_trade_duration,
        portfolio_value=portfolio_value.tolist(),
        drawdown_series=drawdown_series.tolist(),
        trade_log=trades,
    )


def run_enhanced_backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 100000,
    commission_rate: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """
    Enhanced backtest with realistic transaction costs and position tracking

    Args:
        prices: Array of asset prices
        signals: Array of trading signals (-1, 0, 1)
        initial_capital: Starting portfolio value
        commission_rate: Commission as percentage of trade value
        slippage: Slippage as percentage of trade value

    Returns:
        Dictionary with comprehensive backtest results
    """
    if len(prices) < 2 or len(signals) < 2:
        return {"error": "Insufficient data for backtesting"}

    # Ensure signals and prices are aligned
    min_length = min(len(prices), len(signals))
    prices = prices[:min_length]
    signals = signals[:min_length]

    # Initialize tracking variables
    portfolio_value = initial_capital
    position = 0  # Number of shares held
    cash = initial_capital
    trades = []
    portfolio_values = []
    returns = []

    current_trade = None

    for i in range(1, len(prices)):
        current_price = prices[i]
        current_signal = signals[i]
        prev_signal = signals[i - 1] if i > 0 else 0

        # Track portfolio value
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)

        # Calculate returns
        if i > 0:
            ret = (
                (portfolio_values[i - 1] / portfolio_values[i - 2] - 1) if i > 1 else 0
            )
            returns.append(ret)

        # Signal change detection
        if current_signal != prev_signal:
            # Close existing position
            if position != 0:
                trade_value = abs(position) * current_price
                total_cost = trade_value * (commission_rate + slippage)

                if position > 0:  # Closing long position
                    cash += trade_value - total_cost
                else:  # Closing short position
                    cash -= trade_value + total_cost

                # Complete the trade log
                if current_trade:
                    pnl = cash - current_trade["entry_cash"]
                    current_trade.update(
                        {
                            "exit_price": current_price,
                            "exit_date": i,
                            "pnl": pnl,
                            "duration": i - current_trade["entry_date"],
                        }
                    )
                    trades.append(current_trade)

                position = 0

            # Open new position
            if current_signal != 0:
                available_cash = cash * 0.95  # Use 95% of cash for position
                shares_to_buy = int(available_cash / current_price)

                if shares_to_buy > 0:
                    position = (
                        shares_to_buy * current_signal
                    )  # Positive for long, negative for short
                    trade_value = abs(position) * current_price
                    total_cost = trade_value * (commission_rate + slippage)

                    if current_signal > 0:  # Long position
                        cash -= trade_value + total_cost
                    else:  # Short position
                        cash += trade_value - total_cost

                    # Start new trade log
                    current_trade = {
                        "entry_date": i,
                        "entry_price": current_price,
                        "entry_cash": cash,
                        "position_size": position,
                        "signal": current_signal,
                    }

    # Close final position if any
    if position != 0 and len(prices) > 0:
        final_price = prices[-1]
        trade_value = abs(position) * final_price
        total_cost = trade_value * (commission_rate + slippage)

        if position > 0:
            cash += trade_value - total_cost
        else:
            cash -= trade_value + total_cost

        if current_trade:
            pnl = cash - current_trade["entry_cash"]
            current_trade.update(
                {
                    "exit_price": final_price,
                    "exit_date": len(prices) - 1,
                    "pnl": pnl,
                    "duration": len(prices) - 1 - current_trade["entry_date"],
                }
            )
            trades.append(current_trade)

    # Calculate final metrics
    returns_array = np.array(returns) if returns else np.array([0])
    result = calculate_performance_metrics(returns_array, prices, trades)

    return {
        "performance_metrics": result,
        "final_portfolio_value": cash,
        "total_return_pct": ((cash / initial_capital) - 1) * 100,
        "transaction_costs": sum(
            [abs(t.get("pnl", 0)) * (commission_rate + slippage) for t in trades]
        ),
        "number_of_trades": len(trades),
    }
