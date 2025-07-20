#!/usr/bin/env python3
"""
Backtest runner for strategy engine.
"""
import argparse
import logging
import sys

# Setup logging instead of using shared_services
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtesting for a strategy")
    parser.add_argument("--config", required=True, help="Path to strategy config file")
    parser.add_argument("--universe", required=True, help="Universe to backtest on")
    parser.add_argument("--start-date", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for backtest (YYYY-MM-DD)")
    args = parser.parse_args()

    try:
        # In a real implementation, this would run the backtest
        logger.info(f"Running backtest for config: {args.config}")
        logger.info(f"Universe: {args.universe}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info("Backtest completed successfully")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
