"""
Multi-asset strategy validation module.
Tests strategies across different market caps, sectors, and volatility regimes.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns

from src.ai_trading_machine.utils.bq_logger import log_backtest_result


class MultiAssetValidator:
    """
    Validates trading strategies across different asset classes and categories.

    Attributes:
        client: BigQuery client for data retrieval and logging
        project_id: GCP project ID
        dataset: BigQuery dataset for trading data
    """

    def __init__(
        self,
        bq_client: Optional[bigquery.Client] = None,
        project_id: str = "ai-trading-gcp-459813",
        dataset: str = "trading_data",
    ):
        """
        Initialize the multi-asset validator.

        Args:
            bq_client: BigQuery client for data retrieval
            project_id: GCP project ID
            dataset: BigQuery dataset for trading data
        """
        self.client = bq_client or bigquery.Client()
        self.project_id = project_id
        self.dataset = dataset

    def get_symbols_by_market_cap(
        self, cap_segment: str = "large_cap", limit: int = 100
    ) -> List[str]:
        """
        Get list of symbols by market capitalization segment.

        Args:
            cap_segment: Market cap segment ('large_cap', 'mid_cap', 'small_cap')
            limit: Maximum number of symbols to return

        Returns:
            List of symbol strings

        Raises:
            ValueError: If cap_segment is invalid
        """
        valid_segments = ["large_cap", "mid_cap", "small_cap"]
        if cap_segment not in valid_segments:
            raise ValueError(
                f"Invalid market cap segment. Must be one of {valid_segments}"
            )

        query = f"""
        SELECT DISTINCT symbol
        FROM `{self.project_id}.{self.dataset}.{cap_segment}_price_data`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
        LIMIT {limit}
        """

        query_job = self.client.query(query)
        results = query_job.result()

        return [row.symbol for row in results]

    def get_symbols_by_sector(self, sector: str, limit: int = 100) -> List[str]:
        """
        Get list of symbols by sector.

        Args:
            sector: Sector name (e.g., 'IT', 'Finance', 'Energy')
            limit: Maximum number of symbols to return

        Returns:
            List of symbol strings
        """
        query = f"""
        SELECT DISTINCT symbol
        FROM `{self.project_id}.{self.dataset}.symbol_metadata`
        WHERE sector = '{sector}'
        LIMIT {limit}
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()
            return [row.symbol for row in results]
        except Exception as e:
            print(f"Error querying symbols by sector: {e}")
            return []

    def get_symbols_by_volatility(
        self, volatility_level: str = "high", lookback_days: int = 90, limit: int = 100
    ) -> List[str]:
        """
        Get list of symbols by historical volatility level.

        Args:
            volatility_level: Volatility level ('high', 'medium', 'low')
            lookback_days: Number of days to calculate volatility
            limit: Maximum number of symbols to return

        Returns:
            List of symbol strings

        Raises:
            ValueError: If volatility_level is invalid
        """
        valid_levels = ["high", "medium", "low"]
        if volatility_level not in valid_levels:
            raise ValueError(f"Invalid volatility level. Must be one of {valid_levels}")

        # Define percentile thresholds for each level
        percentiles = {
            "high": 0.67,  # Top third
            "medium": 0.33,  # Middle third
            "low": 0.0,  # Bottom third
        }

        # Get volatility for all symbols
        query = f"""
        WITH symbol_volatility AS (
            SELECT 
                symbol,
                STDDEV(close / LAG(close) OVER (PARTITION BY symbol ORDER BY date) - 1) * SQRT(252) AS volatility
            FROM `{self.project_id}.{self.dataset}.large_cap_price_data`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
            GROUP BY symbol
        ),
        volatility_percentiles AS (
            SELECT
                symbol,
                volatility,
                PERCENT_RANK() OVER (ORDER BY volatility) AS percentile
            FROM symbol_volatility
            WHERE volatility IS NOT NULL
        )
        SELECT symbol
        FROM volatility_percentiles
        WHERE percentile >= {percentiles[volatility_level]}
        ORDER BY volatility DESC
        LIMIT {limit}
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()
            return [row.symbol for row in results]
        except Exception as e:
            print(f"Error querying symbols by volatility: {e}")
            return []

    def validate_across_market_caps(
        self,
        strategy_class: Any,
        strategy_params: Dict[str, Any],
        start_date: str,
        end_date: str,
        cap_segments: Optional[List[str]] = None,
        symbols_per_segment: int = 20,
        tag: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate strategy across different market cap segments.

        Args:
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for the strategy
            start_date: Start date for validation (YYYY-MM-DD)
            end_date: End date for validation (YYYY-MM-DD)
            cap_segments: List of market cap segments to test
            symbols_per_segment: Number of symbols to test per segment
            tag: Optional tag for logging results

        Returns:
            Dictionary of results by market cap segment
        """
        cap_segments = cap_segments or ["large_cap", "mid_cap", "small_cap"]
        results = {}

        for segment in cap_segments:
            # Get symbols for this segment
            symbols = self.get_symbols_by_market_cap(segment, symbols_per_segment)

            if not symbols:
                print(f"No symbols found for segment: {segment}")
                continue

            # Collect results for each symbol
            segment_results = {
                "symbols": [],
                "sharpe_ratios": [],
                "total_returns": [],
                "max_drawdowns": [],
                "win_rates": [],
            }

            # Test each symbol
            for symbol in symbols:
                # Get historical data
                query = f"""
                SELECT date, open, high, low, close, volume
                FROM `{self.project_id}.{self.dataset}.{segment}_price_data`
                WHERE symbol = '{symbol}'
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                """

                try:
                    df = self.client.query(query).to_dataframe()

                    if df.empty:
                        print(f"No data found for {symbol}")
                        continue

                    # Initialize strategy
                    strategy = strategy_class(strategy_params)

                    # Run backtest
                    signals = []
                    returns = []

                    for i in range(len(df) - 1):
                        try:
                            result = strategy.run(df.iloc[: i + 1])

                            if result and "signal" in result:
                                signal = result["signal"]
                                signals.append(signal)

                                # Calculate return based on signal
                                next_return = df["close"].pct_change().iloc[i + 1]
                                trade_return = (
                                    next_return * signal
                                    if not np.isnan(next_return)
                                    else 0
                                )
                                returns.append(trade_return)
                        except Exception as e:
                            print(f"Error in strategy execution: {e}")
                            returns.append(0)

                    # Calculate performance metrics
                    if returns:
                        sharpe_ratio = (
                            np.mean(returns) / np.std(returns) * np.sqrt(252)
                            if np.std(returns) > 0
                            else 0
                        )
                        total_return = np.sum(returns)
                        max_drawdown = self._calculate_max_drawdown(returns)
                        win_rate = sum(1 for r in returns if r > 0) / len(returns)

                        segment_results["symbols"].append(symbol)
                        segment_results["sharpe_ratios"].append(sharpe_ratio)
                        segment_results["total_returns"].append(total_return)
                        segment_results["max_drawdowns"].append(max_drawdown)
                        segment_results["win_rates"].append(win_rate)

                        # Log to BigQuery
                        log_data = {
                            "strategy_id": strategy_class.__name__,
                            "params": str(strategy_params),
                            "market_cap": segment,
                            "start_date": df["date"].min(),
                            "end_date": df["date"].max(),
                            "symbol": symbol,
                            "sharpe_ratio": sharpe_ratio,
                            "total_return": total_return,
                            "max_drawdown": max_drawdown,
                            "win_rate": win_rate,
                            "tag": tag or f"market_cap_{segment}",
                            "timestamp": datetime.now(),
                        }

                        log_backtest_result(**log_data)

                except Exception as e:
                    print(f"Error processing symbol {symbol}: {e}")
                    continue

            # Calculate averages
            results[segment] = {
                "avg_sharpe_ratio": (
                    np.mean(segment_results["sharpe_ratios"])
                    if segment_results["sharpe_ratios"]
                    else 0
                ),
                "avg_total_return": (
                    np.mean(segment_results["total_returns"])
                    if segment_results["total_returns"]
                    else 0
                ),
                "avg_max_drawdown": (
                    np.mean(segment_results["max_drawdowns"])
                    if segment_results["max_drawdowns"]
                    else 0
                ),
                "avg_win_rate": (
                    np.mean(segment_results["win_rates"])
                    if segment_results["win_rates"]
                    else 0
                ),
                "symbol_count": len(segment_results["symbols"]),
                "raw_results": segment_results,
            }

        return results

    def validate_across_sectors(
        self,
        strategy_class: Any,
        strategy_params: Dict[str, Any],
        start_date: str,
        end_date: str,
        sectors: Optional[List[str]] = None,
        symbols_per_sector: int = 10,
        tag: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate strategy across different market sectors.

        Args:
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for the strategy
            start_date: Start date for validation (YYYY-MM-DD)
            end_date: End date for validation (YYYY-MM-DD)
            sectors: List of sectors to test
            symbols_per_sector: Number of symbols to test per sector
            tag: Optional tag for logging results

        Returns:
            Dictionary of results by sector
        """
        sectors = sectors or ["IT", "Finance", "Energy", "Consumer", "Healthcare"]
        results = {}

        for sector in sectors:
            # Get symbols for this sector
            symbols = self.get_symbols_by_sector(sector, symbols_per_sector)

            if not symbols:
                print(f"No symbols found for sector: {sector}")
                continue

            # Collect results for each symbol
            sector_results = {
                "symbols": [],
                "sharpe_ratios": [],
                "total_returns": [],
                "max_drawdowns": [],
                "win_rates": [],
            }

            # Test each symbol - using large_cap data for simplicity
            for symbol in symbols:
                # Get historical data
                query = f"""
                SELECT date, open, high, low, close, volume
                FROM `{self.project_id}.{self.dataset}.large_cap_price_data`
                WHERE symbol = '{symbol}'
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                """

                try:
                    df = self.client.query(query).to_dataframe()

                    if df.empty:
                        print(f"No data found for {symbol}")
                        continue

                    # Initialize strategy
                    strategy = strategy_class(strategy_params)

                    # Run backtest
                    signals = []
                    returns = []

                    for i in range(len(df) - 1):
                        try:
                            result = strategy.run(df.iloc[: i + 1])

                            if result and "signal" in result:
                                signal = result["signal"]
                                signals.append(signal)

                                # Calculate return based on signal
                                next_return = df["close"].pct_change().iloc[i + 1]
                                trade_return = (
                                    next_return * signal
                                    if not np.isnan(next_return)
                                    else 0
                                )
                                returns.append(trade_return)
                        except Exception as e:
                            print(f"Error in strategy execution: {e}")
                            returns.append(0)

                    # Calculate performance metrics
                    if returns:
                        sharpe_ratio = (
                            np.mean(returns) / np.std(returns) * np.sqrt(252)
                            if np.std(returns) > 0
                            else 0
                        )
                        total_return = np.sum(returns)
                        max_drawdown = self._calculate_max_drawdown(returns)
                        win_rate = sum(1 for r in returns if r > 0) / len(returns)

                        sector_results["symbols"].append(symbol)
                        sector_results["sharpe_ratios"].append(sharpe_ratio)
                        sector_results["total_returns"].append(total_return)
                        sector_results["max_drawdowns"].append(max_drawdown)
                        sector_results["win_rates"].append(win_rate)

                        # Log to BigQuery
                        log_data = {
                            "strategy_id": strategy_class.__name__,
                            "params": str(strategy_params),
                            "sector": sector,
                            "start_date": df["date"].min(),
                            "end_date": df["date"].max(),
                            "symbol": symbol,
                            "sharpe_ratio": sharpe_ratio,
                            "total_return": total_return,
                            "max_drawdown": max_drawdown,
                            "win_rate": win_rate,
                            "tag": tag or f"sector_{sector}",
                            "timestamp": datetime.now(),
                        }

                        log_backtest_result(**log_data)

                except Exception as e:
                    print(f"Error processing symbol {symbol}: {e}")
                    continue

            # Calculate averages
            results[sector] = {
                "avg_sharpe_ratio": (
                    np.mean(sector_results["sharpe_ratios"])
                    if sector_results["sharpe_ratios"]
                    else 0
                ),
                "avg_total_return": (
                    np.mean(sector_results["total_returns"])
                    if sector_results["total_returns"]
                    else 0
                ),
                "avg_max_drawdown": (
                    np.mean(sector_results["max_drawdowns"])
                    if sector_results["max_drawdowns"]
                    else 0
                ),
                "avg_win_rate": (
                    np.mean(sector_results["win_rates"])
                    if sector_results["win_rates"]
                    else 0
                ),
                "symbol_count": len(sector_results["symbols"]),
                "raw_results": sector_results,
            }

        return results

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: List of return values

        Returns:
            Maximum drawdown as a positive value
        """
        # Convert returns to cumulative equity curve
        cumulative = np.cumprod(1 + np.array(returns))

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Return maximum drawdown (as a positive value)
        return abs(min(drawdown)) if len(drawdown) > 0 else 0

    def plot_sector_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "avg_sharpe_ratio",
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot comparison of strategy performance across sectors.

        Args:
            results: Results dictionary from validate_across_sectors
            metric: Metric to plot ('avg_sharpe_ratio', 'avg_total_return', etc.)
            title: Plot title (or use default)
            output_path: Path to save plot (or display inline)
        """
        try:
            # Extract data for plotting
            sectors = list(results.keys())
            values = [results[sector][metric] for sector in sectors]

            # Create plot
            plt.figure(figsize=(10, 6))

            # Set color palette based on values
            colors = sns.color_palette("RdYlGn", len(sectors))
            colors = [colors[i] for i in np.argsort(np.argsort(values))]

            bars = plt.bar(sectors, values, color=colors)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{values[i]:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Set title and labels
            metric_labels = {
                "avg_sharpe_ratio": "Average Sharpe Ratio",
                "avg_total_return": "Average Total Return",
                "avg_max_drawdown": "Average Max Drawdown",
                "avg_win_rate": "Average Win Rate",
            }

            plt.title(title or f"{metric_labels.get(metric, metric)} by Sector")
            plt.ylabel(metric_labels.get(metric, metric))
            plt.xlabel("Sector")
            plt.xticks(rotation=45)

            # Add number of symbols as text
            for i, sector in enumerate(sectors):
                plt.text(
                    i,
                    0.01,
                    f"n={results[sector]['symbol_count']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="darkgray",
                )

            plt.tight_layout()

            # Save or display
            if output_path:
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()

        except ImportError:
            print(
                "Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn"
            )
