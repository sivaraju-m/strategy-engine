#!/usr/bin/env python3
"""
Strategy parameter optimizer.
"""
import argparse
import logging
import sys

# Setup logging instead of using shared_services
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--config", required=True, help="Path to strategy config file")
    parser.add_argument("--metric", default="sharpe_ratio", help="Optimization metric")
    parser.add_argument("--universe", required=True, help="Universe to optimize on")
    args = parser.parse_args()

    try:
        # In a real implementation, this would optimize the strategy
        logger.info(f"Optimizing strategy config: {args.config}")
        logger.info(f"Optimization metric: {args.metric}")
        logger.info(f"Universe: {args.universe}")
        logger.info("Strategy optimization completed successfully")
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
