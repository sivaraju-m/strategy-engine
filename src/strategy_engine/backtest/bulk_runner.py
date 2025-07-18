"""Bulk backtest runner for multiple tickers with error handling and resumability."""

import json
import logging
import sys
from pathlib import Path

from .backtest_runner import run_backtest


class BulkBacktestRunner:
    """Runs backtests for multiple tickers with error handling and progress tracking."""

    def __init__(self, config_path: str, skip_failed: bool = True):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.config_dir = self.project_root / "configs"

        # Create necessary directories
        self.logs_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

        # Resolve config path
        self.config_path = (
            Path(config_path)
            if Path(config_path).is_absolute()
            else self.project_root / config_path.lstrip("/")
        )

        if not self.config_path.exists():
            raise FileNotFoundError(
                "Config file not found: {self.config_path}\n"
                "Please put config files in {self.config_dir}\n"
                "Example usage: --config configs/nifty50.json"
            )

        self.skip_failed = skip_failed
        self.failed_tickers: set[str] = set()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging with proper format."""
        log_file = (
            self.logs_dir
            / 'backtest_bulk_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        logging.info("Log file: {log_file}")
        logging.info("Config file: {self.config_path}")

    def load_tickers(self) -> list[str]:
        """Load tickers from JSON config file."""
        try:
            with open(self.config_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get("tickers", [])
        except Exception as e:
            logging.error("❌ Failed to load tickers: {e}")
            raise

    def verify_results(self, results: dict[str, bool]) -> None:
        """Verify and validate backtest results."""
        logging.info("\n=== 🔍 Result Verification ===")

        # Verify all tickers were processed
        expected_tickers = set(self.load_tickers())
        processed_tickers = set(results.keys())
        missing = expected_tickers - processed_tickers

        if missing:
            logging.warning("⚠️ Missing results for tickers: {missing}")

        # Check success rate
        success_rate = (sum(results.values()) / len(results)) * 100
        logging.info("Success rate: {success_rate:.2f}%")

        if success_rate < 80:
            logging.warning("⚠️ Success rate below 80% - please investigate failures")

    def run(self, strategy: str, start_date: str, end_date: str) -> dict[str, bool]:
        """Run backtests for all tickers."""
        tickers = self.load_tickers()
        results = {}
        total = len(tickers)

        logging.info("\n📊 Starting bulk backtest for {total} tickers")

        for idx, ticker in enumerate(tickers, 1):
            if ticker in self.failed_tickers and self.skip_failed:
                logging.info("⏭️  Skipping previously failed ticker: {ticker}")
                continue

            try:
                logging.info("\n🚀 [{idx}/{total}] Processing {ticker}...")
                run_backtest(ticker, strategy, start_date, end_date)
                results[ticker] = True
                logging.info("✅ {ticker} completed successfully")
            except Exception as e:
                logging.error("❌ Failed to process {ticker}: {e}")
                self.failed_tickers.add(ticker)
                results[ticker] = False

        self._print_summary(results)
        self.verify_results(results)  # Add verification step
        return results

    def _print_summary(self, results: dict[str, bool]) -> None:
        """Print final execution summary."""
        succeeded = sum(results.values())
        failed = len(results) - succeeded

        logging.info("\n=== 📈 Backtest Summary ===")
        logging.info("Total: {len(results)} tickers")
        logging.info("✅ Succeeded: {succeeded}")
        logging.info("❌ Failed: {failed}")
        if failed > 0:
            logging.info("\nFailed tickers:")
            for ticker in [t for t, s in results.items() if not s]:
                logging.info("- {ticker}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk backtest runner for multiple tickers"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to tickers config JSON (relative to project root)",
    )
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--skip-failed", action="store_true", help="Skip failed tickers"
    )
    args = parser.parse_args()

    try:
        runner = BulkBacktestRunner(args.config, args.skip_failed)
        runner.run(args.strategy, args.start_date, args.end_date)
    except KeyboardInterrupt:
        logging.info("\n⚠️ Process interrupted by user")
    except Exception as e:
        logging.error("\n❌ Fatal error: {e}")
        sys.exit(1)
