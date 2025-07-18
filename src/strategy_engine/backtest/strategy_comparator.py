"""
Enhanced Strategy Comparison and Backtesting Framework
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy_engine.backtest.engine import run_enhanced_backtest
from trading_data_pipeline.ingest.yfinance_loader import load_yfinance_data
from strategy_engine.strategies.strategy_registry import registry
from strategy_engine.utils.data_cleaner import clean_ohlcv_data


class StrategyComparator:
    """Compare multiple strategies on multiple assets with comprehensive analysis"""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.results = {}
        self.logger = logging.getLogger(__name__)

    def run_strategy_comparison(
        self,
        tickers: list[str],
        strategies: list[str],
        start_date: str,
        end_date: str,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run multiple strategies on multiple tickers and compare performance

        Args:
            tickers: List of ticker symbols
            strategies: List of strategy names from registry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_results: Whether to save results to file

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            "metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "tickers": tickers,
                "strategies": strategies,
                "run_timestamp": datetime.now().isoformat(),
            },
            "individual_results": {},
            "strategy_summary": {},
            "ticker_summary": {},
            "best_performers": {},
        }

        total_combinations = len(tickers) * len(strategies)
        current_combination = 0

        # Run each strategy on each ticker
        for ticker in tickers:
            comparison_results["individual_results"][ticker] = {}

            try:
                # Load and clean data
                raw_df = load_yfinance_data(ticker, start_date, end_date)
                df = clean_ohlcv_data(raw_df)
                df.fillna(method="ffill", inplace=True)
                df.fillna(method="bfill", inplace=True)

                if len(df) < 50:  # Skip if insufficient data
                    self.logger.warning("Insufficient data for {ticker}, skipping...")
                    continue

                prices = df["Close"].values

                for strategy_name in strategies:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100

                    self.logger.info(
                        "[{current_combination}/{total_combinations}] "
                        "Testing {strategy_name} on {ticker} ({progress:.1f}%)"
                    )

                    try:
                        # Generate signals
                        strategy_fn = registry[strategy_name]
                        strategy_result = strategy_fn(df)

                        if isinstance(strategy_result, tuple):
                            signals = strategy_result[0]
                        elif isinstance(strategy_result, dict):
                            signals = strategy_result.get("signals", np.zeros(len(df)))
                        else:
                            signals = strategy_result

                        # Ensure signals are numeric array
                        if isinstance(signals, pd.Series):
                            signals = signals.values
                        signals = np.array(signals, dtype=float)

                        # Run backtest
                        backtest_result = run_enhanced_backtest(
                            prices,
                            signals,
                            self.initial_capital,
                            self.commission_rate,
                            self.slippage,
                        )

                        comparison_results["individual_results"][ticker][
                            strategy_name
                        ] = backtest_result

                    except Exception as e:
                        self.logger.error(
                            "Error testing {strategy_name} on {ticker}: {str(e)}"
                        )
                        comparison_results["individual_results"][ticker][
                            strategy_name
                        ] = {"error": str(e)}

            except Exception as e:
                self.logger.error("Error loading data for {ticker}: {str(e)}")
                continue

        # Generate summary statistics
        self._generate_summary_statistics(comparison_results)

        if save_results:
            self._save_results(comparison_results)

        return comparison_results

    def _generate_summary_statistics(self, results: dict[str, Any]) -> None:
        """Generate strategy and ticker summary statistics"""

        # Strategy summary
        strategy_metrics = {}
        for strategy in results["metadata"]["strategies"]:
            strategy_results = []

            for ticker in results["individual_results"]:
                if strategy in results["individual_results"][ticker]:
                    result = results["individual_results"][ticker][strategy]
                    if "error" not in result and "performance_metrics" in result:
                        metrics = result["performance_metrics"]
                        strategy_results.append(
                            {
                                "total_return": metrics.total_return,
                                "sharpe_ratio": metrics.sharpe_ratio,
                                "max_drawdown": metrics.max_drawdown,
                                "win_rate": metrics.win_rate,
                                "profit_factor": metrics.profit_factor,
                                "total_trades": metrics.total_trades,
                            }
                        )

            if strategy_results:
                strategy_metrics[strategy] = {
                    "avg_total_return": np.mean(
                        [r["total_return"] for r in strategy_results]
                    ),
                    "avg_sharpe_ratio": np.mean(
                        [r["sharpe_ratio"] for r in strategy_results]
                    ),
                    "avg_max_drawdown": np.mean(
                        [r["max_drawdown"] for r in strategy_results]
                    ),
                    "avg_win_rate": np.mean([r["win_rate"] for r in strategy_results]),
                    "avg_profit_factor": np.mean(
                        [
                            r["profit_factor"]
                            for r in strategy_results
                            if r["profit_factor"] != float("in")
                        ]
                    ),
                    "total_tests": len(strategy_results),
                    "profitable_tests": len(
                        [r for r in strategy_results if r["total_return"] > 0]
                    ),
                }

        results["strategy_summary"] = strategy_metrics

        # Find best performers
        best_performers = {
            "best_total_return": {"strategy": "", "ticker": "", "value": -float("in")},
            "best_sharpe_ratio": {"strategy": "", "ticker": "", "value": -float("in")},
            "lowest_drawdown": {"strategy": "", "ticker": "", "value": float("in")},
            "highest_win_rate": {"strategy": "", "ticker": "", "value": -float("in")},
        }

        for ticker in results["individual_results"]:
            for strategy in results["individual_results"][ticker]:
                result = results["individual_results"][ticker][strategy]
                if "error" not in result and "performance_metrics" in result:
                    metrics = result["performance_metrics"]

                    if (
                        metrics.total_return
                        > best_performers["best_total_return"]["value"]
                    ):
                        best_performers["best_total_return"].update(
                            {
                                "strategy": strategy,
                                "ticker": ticker,
                                "value": metrics.total_return,
                            }
                        )

                    if (
                        metrics.sharpe_ratio
                        > best_performers["best_sharpe_ratio"]["value"]
                    ):
                        best_performers["best_sharpe_ratio"].update(
                            {
                                "strategy": strategy,
                                "ticker": ticker,
                                "value": metrics.sharpe_ratio,
                            }
                        )

                    if (
                        metrics.max_drawdown
                        < best_performers["lowest_drawdown"]["value"]
                    ):
                        best_performers["lowest_drawdown"].update(
                            {
                                "strategy": strategy,
                                "ticker": ticker,
                                "value": metrics.max_drawdown,
                            }
                        )

                    if metrics.win_rate > best_performers["highest_win_rate"]["value"]:
                        best_performers["highest_win_rate"].update(
                            {
                                "strategy": strategy,
                                "ticker": ticker,
                                "value": metrics.win_rate,
                            }
                        )

        results["best_performers"] = best_performers

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "strategy_comparison_{timestamp}.json"
        filepath = Path("logs") / "strategy_comparisons" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to JSON serializable
        json_results = self._make_json_serializable(results)

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        self.logger.info("Results saved to {filepath}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif hasattr(obj, "__dict__"):  # Handle dataclass objects
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def generate_performance_report(self, results: dict[str, Any]) -> str:
        """Generate a formatted performance report"""

        report = []
        report.append("=" * 80)
        report.append("🚀 STRATEGY COMPARISON REPORT")
        report.append("=" * 80)
        report.append(
            "Period: {results['metadata']['start_date']} to {results['metadata']['end_date']}"
        )
        report.append(
            "Strategies Tested: {', '.join(results['metadata']['strategies'])}"
        )
        report.append("Tickers Analyzed: {len(results['metadata']['tickers'])}")
        report.append("")

        # Strategy Summary
        report.append("📊 STRATEGY PERFORMANCE SUMMARY")
        report.append("-" * 50)

        if results["strategy_summary"]:
            strategy_df = pd.DataFrame(results["strategy_summary"]).T
            strategy_df = strategy_df.round(4)
            report.append(strategy_df.to_string())

        report.append("")

        # Best Performers
        report.append("🏆 BEST PERFORMERS")
        report.append("-" * 30)

        for metric, data in results["best_performers"].items():
            if data["strategy"]:
                report.append(
                    "{metric.replace('_', ' ').title()}: "
                    "{data['strategy']} on {data['ticker']} ({data['value']:.4f})"
                )

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def quick_strategy_test(
    strategy_name: str, ticker: str = "^NSEI", period_days: int = 365
) -> dict[str, Any]:
    """
    Quick test of a single strategy on a single ticker

    Args:
        strategy_name: Name of strategy from registry
        ticker: Ticker symbol (default: NIFTY)
        period_days: Number of days to test (default: 1 year)

    Returns:
        Backtest results
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")

    comparator = StrategyComparator()
    results = comparator.run_strategy_comparison(
        tickers=[ticker],
        strategies=[strategy_name],
        start_date=start_date,
        end_date=end_date,
        save_results=False,
    )

    if (
        ticker in results["individual_results"]
        and strategy_name in results["individual_results"][ticker]
    ):
        return results["individual_results"][ticker][strategy_name]
    else:
        return {"error": "Strategy test failed"}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test all strategies on NIFTY50 sample
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    available_strategies = list(registry.keys())

    comparator = StrategyComparator()
    results = comparator.run_strategy_comparison(
        tickers=sample_tickers,
        strategies=available_strategies,
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    print(comparator.generate_performance_report(results))
