import argparse
import sys
from datetime import datetime, timezone

from trading_data_pipeline.ingest.yfinance_loader import load_yfinance_data
from strategy_engine.strategies import registry
from strategy_engine.utils.data_cleaner import clean_ohlcv_data
from src.bootstrap import *  # Ensures ai_trading_machine is importable


def log_backtest_result(result: dict) -> None:
    """
    Log the backtest results to the console
    Args:
        result: Dictionary containing backtest results
    """
    print("\n📊 Backtest Results:")
    for key, value in result.items():
        print("{key}: {value}")


def run_backtest(
    ticker: str, strategy_name: str, start_date: str, end_date: str
) -> None:
    """
    Run backtest for given ticker and strategy
    Args:
        ticker: Stock ticker symbol
        strategy_name: Strategy name to test
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    try:
        if strategy_name not in registry:
            raise ValueError(
                "Strategy '{strategy_name}' not found. Available strategies: {list(registry.keys())}"
            )

        raw_df = load_yfinance_data(ticker, start_date, end_date)
        df = clean_ohlcv_data(raw_df)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        strategy_fn = registry[strategy_name]
        strategy_result = strategy_fn(df)

        result = {}
        if isinstance(strategy_result, tuple):
            result = {"signal": strategy_result[0], "confidence": strategy_result[1]}
        elif isinstance(strategy_result, dict):
            result = strategy_result

        log_backtest_result(
            {
                "ticker": ticker,
                "strategy": strategy_name,
                "start_date": start_date,
                "end_date": end_date,
                "cagr": result.get("cagr", 0.0),
                "sharpe": result.get("sharpe", 0.0),
                **result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        print("❌ Error running backtest: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--strategy", required=True, help="Strategy name to test")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    run_backtest(args.ticker, args.strategy, args.start_date, args.end_date)
