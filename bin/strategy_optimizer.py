#!/usr/bin/env python3
"""
Strategy parameter optimizer.
"""
import argparse
import sys
from strategy_engine.strategies.optimizer import optimize_strategy
from shared_services.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--config", required=True, help="Path to strategy config file")
    parser.add_argument("--metric", default="sharpe_ratio", help="Optimization metric")
    parser.add_argument("--universe", required=True, help="Universe to optimize on")
    args = parser.parse_args()
    
    try:
        optimize_strategy(args.config, args.metric, args.universe)
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
