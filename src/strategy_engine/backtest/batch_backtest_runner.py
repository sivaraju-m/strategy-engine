"""Batch backtest runner for processing multiple tickers sequentially."""

import json
import logging
import sys
from pathlib import Path

from strategy_engine.backtest.backtest_runner import run_backtest


class BatchBacktestRunner:
    """Handles batch processing of backtests with progress tracking and error handling."""

    def __init__(self, config_path: str, strategy: str, start_date: str, end_date: str):
        """Initialize batch runner with parameters."""
        self.config_path = Path(config_path)
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.failed: list[tuple[str, str]] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging with file and console output."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / "batch_backtest_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

    def load_tickers(self) -> list[str]:
        """Load and validate tickers from config file."""
        try:
            with open(self.config_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "tickers" in data:
                return data["tickers"]

            raise ValueError("Config must be list or dict with 'tickers' key")
        except Exception as e:
            logging.error("❌ Failed to load tickers: {e}")
            raise

    def run(self) -> dict[str, bool]:
        """Execute backtests for all tickers with progress tracking."""
        tickers = self.load_tickers()
        results = {}
        total = len(tickers)

        logging.info("\n🚀 Starting batch backtest")
        logging.info("Strategy: {self.strategy}")
        logging.info("Period: {self.start_date} to {self.end_date}")
        logging.info("Tickers: {total}\n")

        for idx, ticker in enumerate(tickers, 1):
            try:
                logging.info("[{idx}/{total}] Processing {ticker}...")
                run_backtest(ticker, self.strategy, self.start_date, self.end_date)
                results[ticker] = True
                logging.info("✅ {ticker} completed successfully\n")
            except Exception as e:
                error_msg = str(e)
                self.failed.append((ticker, error_msg))
                results[ticker] = False
                logging.error("❌ Failed: {ticker} - {error_msg}\n")

        self._print_summary(total)
        return results

    def _print_summary(self, total: int) -> None:
        """Print execution summary with success rate."""
        success_count = total - len(self.failed)
        success_rate = (success_count / total) * 100

        logging.info("\n=== 📊 Batch Summary ===")
        logging.info("Total processed: {total}")
        logging.info("Succeeded: {success_count}")
        logging.info("Failed: {len(self.failed)}")
        logging.info("Success rate: {success_rate:.1f}%")

        if self.failed:
            logging.info("\n⚠️ Failed tickers:")
            for ticker, error in self.failed:
                logging.info("- {ticker}: {error}")


def main():
    """CLI entry point for batch backtest runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run backtests for multiple tickers")
    parser.add_argument("--config", required=True, help="Path to tickers config")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    try:
        runner = BatchBacktestRunner(
            args.config, args.strategy, args.start_date, args.end_date
        )
        runner.run()
    except KeyboardInterrupt:
        logging.info("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error("\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
