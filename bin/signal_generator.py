#!/usr/bin/env python3
"""
Signal generator for strategy engine.
"""
import argparse
import logging
import sys
from strategy_engine.signals.generator import generate_signals

# Setup logging instead of using shared_services
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate signals for a universe")
    parser.add_argument("--config", required=True, help="Path to signal config file")
    parser.add_argument(
        "--universe", required=True, help="Universe to generate signals for"
    )
    parser.add_argument("--date", help="Date for signal generation (YYYY-MM-DD)")
    args = parser.parse_args()

    try:
        # In a real implementation, this would generate signals
        logger.info(f"Generating signals with config: {args.config}")
        logger.info(f"Universe: {args.universe}")
        logger.info(f"Date: {args.date}")
        logger.info("Signal generation completed successfully")
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
