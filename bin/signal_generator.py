#!/usr/bin/env python3
"""
Signal generator for strategy engine.
"""
import argparse
import sys
from strategy_engine.signals.generator import generate_signals
from shared_services.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate signals for a universe")
    parser.add_argument("--config", required=True, help="Path to signal config file")
    parser.add_argument("--universe", required=True, help="Universe to generate signals for")
    parser.add_argument("--date", help="Date for signal generation (YYYY-MM-DD)")
    args = parser.parse_args()
    
    try:
        generate_signals(args.config, args.universe, args.date)
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
