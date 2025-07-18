#!/usr/bin/env python3
"""
Backtesting runner for strategy engine.
"""
import argparse
import sys
from strategy_engine.backtest.engine import run_backtest
from shared_services.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run backtesting for a strategy")
    parser.add_argument("--config", required=True, help="Path to strategy config file")
    parser.add_argument("--universe", required=True, help="Universe to backtest on")
    parser.add_argument("--start-date", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for backtest (YYYY-MM-DD)")
    args = parser.parse_args()
    
    try:
        run_backtest(args.config, args.universe, args.start_date, args.end_date)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
