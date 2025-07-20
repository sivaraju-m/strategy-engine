"""
Advanced backtesting analytics module.
Implements Monte Carlo simulation, stress testing, and statistical significance testing.
"""

from typing import Dict, List, Callable, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.cloud import bigquery
import matplotlib.pyplot as plt

from src.ai_trading_machine.utils.bq_logger import log_backtest_result


class MonteCarloSimulator:
    """
    Implements Monte Carlo simulation for strategy performance evaluation.

    Attributes:
        n_simulations: Number of simulations to run
        bq_client: BigQuery client for result logging
    """

    def __init__(
        self, n_simulations: int = 1000, bq_client: Optional[bigquery.Client] = None
    ):
        """
        Initialize the Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run
            bq_client: BigQuery client for result logging
        """
        self.n_simulations = n_simulations
        self.bq_client = bq_client

    def simulate_price_paths(
        self, df: pd.DataFrame, n_simulations: Optional[int] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate simulated price paths based on historical data.

        Args:
            df: Historical price data with OHLCV columns
            n_simulations: Number of simulations (overrides instance attribute)

        Returns:
            Dictionary of simulated price dataframes
        """
        n_simulations = n_simulations or self.n_simulations

        # Calculate returns and volatility
        returns = df["close"].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()

        # Generate price paths
        simulated_paths = {}

        for i in range(n_simulations):
            # Generate random returns based on historical distribution
            sim_returns = np.random.normal(mean_return, std_return, len(df))

            # Create price series starting from the first historical price
            start_price = df["close"].iloc[0]
            sim_prices = [start_price]

            for ret in sim_returns:
                sim_prices.append(sim_prices[-1] * (1 + ret))

            # Create simulated dataframe
            sim_df = df.copy()
            sim_df["close"] = sim_prices[: len(df)]

            # Adjust other price columns (open, high, low) based on close price changes
            for col in ["open", "high", "low"]:
                if col in sim_df.columns:
                    ratio = df[col] / df["close"]
                    sim_df[col] = sim_df["close"] * ratio

            simulated_paths[i] = sim_df

        return simulated_paths

    def simulate_strategy_performance(
        self,
        strategy_func: Callable,
        df: pd.DataFrame,
        params: Dict[str, Any],
        n_simulations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simulate strategy performance across multiple price paths.

        Args:
            strategy_func: Function that generates strategy signals
            df: Historical price data
            params: Strategy parameters
            n_simulations: Number of simulations (overrides instance attribute)

        Returns:
            Dictionary of performance metrics
        """
        n_simulations = n_simulations or self.n_simulations

        # Generate simulated price paths
        simulated_paths = self.simulate_price_paths(df, n_simulations)

        # Apply strategy to each path and calculate performance
        performance_results = []

        for i, sim_df in simulated_paths.items():
            # Generate signals using the strategy function
            signals = strategy_func(sim_df, params)

            # Create returns series
            sim_df = sim_df.copy()
            sim_df["signal"] = signals
            sim_df["return"] = sim_df["close"].pct_change()
            sim_df["strategy_return"] = sim_df["signal"].shift(1) * sim_df["return"]

            # Calculate performance metrics
            returns = sim_df["strategy_return"].dropna()

            if len(returns) > 0:
                total_return = (1 + returns).prod() - 1
                sharpe_ratio = (
                    np.sqrt(252) * returns.mean() / returns.std()
                    if returns.std() > 0
                    else 0
                )
                max_drawdown = (sim_df["close"] / sim_df["close"].cummax() - 1).min()

                performance_results.append(
                    {
                        "simulation_id": i,
                        "total_return": total_return,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                    }
                )

        # Aggregate results
        if not performance_results:
            return {"error": "No valid simulation results"}

        df_results = pd.DataFrame(performance_results)

        aggregated_results = {
            "mean_return": df_results["total_return"].mean(),
            "median_return": df_results["total_return"].median(),
            "return_std": df_results["total_return"].std(),
            "return_5th_percentile": df_results["total_return"].quantile(0.05),
            "return_95th_percentile": df_results["total_return"].quantile(0.95),
            "mean_sharpe": df_results["sharpe_ratio"].mean(),
            "median_sharpe": df_results["sharpe_ratio"].median(),
            "sharpe_std": df_results["sharpe_ratio"].std(),
            "sharpe_5th_percentile": df_results["sharpe_ratio"].quantile(0.05),
            "sharpe_95th_percentile": df_results["sharpe_ratio"].quantile(0.95),
            "mean_drawdown": df_results["max_drawdown"].mean(),
            "worst_drawdown": df_results["max_drawdown"].min(),
            "simulation_count": n_simulations,
            "statistically_significant": df_results["sharpe_ratio"].quantile(0.05) > 0,
        }

        # Log results to BigQuery if client is available
        if self.bq_client:
            try:
                log_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_type": "monte_carlo",
                    "parameters": json.dumps(params),
                    "results": json.dumps(aggregated_results),
                    "simulation_count": n_simulations,
                }

                log_backtest_result(
                    client=self.bq_client,
                    data=log_data,
                    table_id="advanced_analytics_results",
                )
            except Exception as e:
                print(f"Error logging Monte Carlo results to BigQuery: {e}")

        return aggregated_results


class StressTester:
    """
    Implements stress testing for strategy robustness evaluation.

    Attributes:
        bq_client: BigQuery client for result logging
    """

    def __init__(self, bq_client: Optional[bigquery.Client] = None):
        """
        Initialize the stress tester.

        Args:
            bq_client: BigQuery client for result logging
        """
        self.bq_client = bq_client

        # Define standard stress scenarios
        self.scenarios = {
            "baseline": {
                "description": "Original data without modification",
                "apply": lambda df: df.copy(),
            },
            "market_crash": {
                "description": "Simulated market crash (20% drop over 10 days)",
                "apply": self._apply_market_crash,
            },
            "high_volatility": {
                "description": "Period of high volatility (2x normal)",
                "apply": self._apply_high_volatility,
            },
            "low_liquidity": {
                "description": "Low liquidity period (90% volume reduction)",
                "apply": self._apply_low_liquidity,
            },
            "sudden_gap": {
                "description": "Sudden price gap (10% overnight)",
                "apply": self._apply_sudden_gap,
            },
            "trending_market": {
                "description": "Strong trending market (0.5% daily trend)",
                "apply": self._apply_trending_market,
            },
            "choppy_market": {
                "description": "Choppy market with frequent reversals",
                "apply": self._apply_choppy_market,
            },
        }

    def _apply_market_crash(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply market crash scenario"""
        result = df.copy()

        # Find a suitable location for the crash (after 30% of the data)
        crash_start = int(len(df) * 0.3)
        crash_duration = 10  # 10 days

        # Apply crash: 20% drop over 10 days
        daily_factor = (1 - 0.2) ** (1 / crash_duration)

        for i in range(crash_duration):
            if crash_start + i < len(result):
                idx = crash_start + i
                result.loc[result.index[idx], "close"] = (
                    result.iloc[idx - 1]["close"] * daily_factor
                )

                # Adjust other price columns
                for col in ["open", "high", "low"]:
                    if col in result.columns:
                        ratio = df.iloc[idx][col] / df.iloc[idx]["close"]
                        result.loc[result.index[idx], col] = (
                            result.iloc[idx]["close"] * ratio
                        )

        # Propagate the effect to the rest of the data
        final_crash_idx = crash_start + crash_duration - 1
        if final_crash_idx < len(result):
            crash_ratio = (
                result.iloc[final_crash_idx]["close"]
                / df.iloc[final_crash_idx]["close"]
            )

            for i in range(final_crash_idx + 1, len(result)):
                for col in ["open", "high", "low", "close"]:
                    if col in result.columns:
                        result.loc[result.index[i], col] = df.iloc[i][col] * crash_ratio

        return result

    def _apply_high_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply high volatility scenario"""
        result = df.copy()

        # Start after 30% of the data
        vol_start = int(len(df) * 0.3)
        vol_end = int(len(df) * 0.7)  # End after 70% of the data

        # Calculate returns
        returns = df["close"].pct_change().dropna()

        # Double volatility by amplifying returns
        for i in range(vol_start, min(vol_end, len(result))):
            if i > 0:
                # Calculate amplified return
                orig_return = (df.iloc[i]["close"] / df.iloc[i - 1]["close"]) - 1
                amplified_return = orig_return * 2  # Double volatility

                # Apply amplified return
                result.loc[result.index[i], "close"] = result.iloc[i - 1]["close"] * (
                    1 + amplified_return
                )

                # Adjust other price columns
                for col in ["open", "high", "low"]:
                    if col in result.columns:
                        ratio = df.iloc[i][col] / df.iloc[i]["close"]
                        result.loc[result.index[i], col] = (
                            result.iloc[i]["close"] * ratio
                        )

        return result

    def _apply_low_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply low liquidity scenario"""
        result = df.copy()

        # Reduce volume by 90%
        if "volume" in result.columns:
            result["volume"] = result["volume"] * 0.1

        # Increase high-low spread by 50%
        if "high" in result.columns and "low" in result.columns:
            for i in range(len(result)):
                mid = (result.iloc[i]["high"] + result.iloc[i]["low"]) / 2
                spread = result.iloc[i]["high"] - result.iloc[i]["low"]
                new_spread = spread * 1.5

                result.loc[result.index[i], "high"] = mid + new_spread / 2
                result.loc[result.index[i], "low"] = mid - new_spread / 2

        return result

    def _apply_sudden_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sudden price gap scenario"""
        result = df.copy()

        # Find a suitable location for the gap (around 40% of the data)
        gap_idx = int(len(df) * 0.4)

        if gap_idx < len(result):
            # 10% overnight gap
            gap_factor = 0.9  # 10% drop

            # Apply gap
            for i in range(gap_idx, len(result)):
                for col in ["open", "high", "low", "close"]:
                    if col in result.columns:
                        result.loc[result.index[i], col] = df.iloc[i][col] * gap_factor

        return result

    def _apply_trending_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trending market scenario"""
        result = df.copy()

        # Start after 20% of the data
        trend_start = int(len(df) * 0.2)

        # Apply consistent 0.5% daily uptrend
        daily_factor = 1.005  # 0.5% daily increase

        if "close" in result.columns:
            base_price = result.iloc[trend_start]["close"]

            for i in range(trend_start + 1, len(result)):
                # Calculate trend price
                days_from_start = i - trend_start
                trend_price = base_price * (daily_factor**days_from_start)

                # Apply trend while preserving some of the original pattern
                orig_price = df.iloc[i]["close"]
                orig_ratio = orig_price / df.iloc[trend_start]["close"]

                # Blend original pattern with trend (70% trend, 30% original)
                result.loc[result.index[i], "close"] = trend_price * 0.7 + (
                    base_price * orig_ratio * 0.3
                )

                # Adjust other price columns
                for col in ["open", "high", "low"]:
                    if col in result.columns:
                        ratio = df.iloc[i][col] / df.iloc[i]["close"]
                        result.loc[result.index[i], col] = (
                            result.iloc[i]["close"] * ratio
                        )

        return result

    def _apply_choppy_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply choppy market scenario with frequent reversals"""
        result = df.copy()

        # Start after 20% of the data
        choppy_start = int(len(df) * 0.2)

        if "close" in result.columns:
            # Generate oscillating pattern
            for i in range(choppy_start + 1, len(result)):
                if i % 5 < 3:  # 3 days up, 2 days down pattern
                    change = 0.01  # 1% up
                else:
                    change = -0.01  # 1% down

                # Apply change while preserving some original pattern
                if i > 0:
                    orig_change = (df.iloc[i]["close"] / df.iloc[i - 1]["close"]) - 1
                    # Blend: 70% choppy pattern, 30% original
                    blended_change = 0.7 * change + 0.3 * orig_change

                    result.loc[result.index[i], "close"] = result.iloc[i - 1][
                        "close"
                    ] * (1 + blended_change)

                    # Adjust other price columns
                    for col in ["open", "high", "low"]:
                        if col in result.columns:
                            ratio = df.iloc[i][col] / df.iloc[i]["close"]
                            result.loc[result.index[i], col] = (
                                result.iloc[i]["close"] * ratio
                            )

        return result

    def run_strategy_stress_test(
        self,
        strategy_func: Callable,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run stress tests for a strategy against multiple scenarios.

        Args:
            strategy_func: Function that generates strategy signals
            df: Historical price data
            params: Strategy parameters
            scenarios: List of scenario names to test (or None for all)

        Returns:
            Dictionary of stress test results by scenario
        """
        # Determine which scenarios to run
        if scenarios is None:
            scenarios_to_run = list(self.scenarios.keys())
        else:
            scenarios_to_run = [s for s in scenarios if s in self.scenarios]

        # Run strategy on each scenario
        results = {}

        # First run baseline for comparison
        baseline_df = self.scenarios["baseline"]["apply"](df)
        baseline_signals = strategy_func(baseline_df, params)

        # Calculate baseline performance
        baseline_df = baseline_df.copy()
        baseline_df["signal"] = baseline_signals
        baseline_df["return"] = baseline_df["close"].pct_change()
        baseline_df["strategy_return"] = (
            baseline_df["signal"].shift(1) * baseline_df["return"]
        )

        baseline_returns = baseline_df["strategy_return"].dropna()

        baseline_metrics = {
            "total_return": (
                (1 + baseline_returns).prod() - 1 if len(baseline_returns) > 0 else 0
            ),
            "sharpe_ratio": (
                np.sqrt(252) * baseline_returns.mean() / baseline_returns.std()
                if len(baseline_returns) > 0 and baseline_returns.std() > 0
                else 0
            ),
            "max_drawdown": (
                (baseline_df["close"] / baseline_df["close"].cummax() - 1).min()
                if len(baseline_df) > 0
                else 0
            ),
            "win_rate": (
                (baseline_returns > 0).mean() if len(baseline_returns) > 0 else 0
            ),
        }

        results["baseline"] = {
            "description": self.scenarios["baseline"]["description"],
            "metrics": baseline_metrics,
        }

        # Run each stress scenario
        for scenario in scenarios_to_run:
            if scenario == "baseline":
                continue

            scenario_df = self.scenarios[scenario]["apply"](df)
            scenario_signals = strategy_func(scenario_df, params)

            # Calculate scenario performance
            scenario_df = scenario_df.copy()
            scenario_df["signal"] = scenario_signals
            scenario_df["return"] = scenario_df["close"].pct_change()
            scenario_df["strategy_return"] = (
                scenario_df["signal"].shift(1) * scenario_df["return"]
            )

            scenario_returns = scenario_df["strategy_return"].dropna()

            scenario_metrics = {
                "total_return": (
                    (1 + scenario_returns).prod() - 1
                    if len(scenario_returns) > 0
                    else 0
                ),
                "sharpe_ratio": (
                    np.sqrt(252) * scenario_returns.mean() / scenario_returns.std()
                    if len(scenario_returns) > 0 and scenario_returns.std() > 0
                    else 0
                ),
                "max_drawdown": (
                    (scenario_df["close"] / scenario_df["close"].cummax() - 1).min()
                    if len(scenario_df) > 0
                    else 0
                ),
                "win_rate": (
                    (scenario_returns > 0).mean() if len(scenario_returns) > 0 else 0
                ),
            }

            # Calculate impact compared to baseline
            impact = {
                "return_impact": scenario_metrics["total_return"]
                - baseline_metrics["total_return"],
                "sharpe_impact": scenario_metrics["sharpe_ratio"]
                - baseline_metrics["sharpe_ratio"],
                "drawdown_impact": scenario_metrics["max_drawdown"]
                - baseline_metrics["max_drawdown"],
                "win_rate_impact": scenario_metrics["win_rate"]
                - baseline_metrics["win_rate"],
            }

            results[scenario] = {
                "description": self.scenarios[scenario]["description"],
                "metrics": scenario_metrics,
                "impact": impact,
            }

        # Log results to BigQuery if client is available
        if self.bq_client:
            try:
                log_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_type": "stress_test",
                    "parameters": json.dumps(params),
                    "results": json.dumps(results),
                    "scenario_count": len(scenarios_to_run),
                }

                log_backtest_result(
                    client=self.bq_client,
                    data=log_data,
                    table_id="advanced_analytics_results",
                )
            except Exception as e:
                print(f"Error logging stress test results to BigQuery: {e}")

        return results


class StatisticalSignificanceTester:
    """
    Implements statistical significance testing for strategy performance.
    """

    def __init__(self):
        """Initialize the statistical significance tester."""
        pass

    def t_test(
        self, strategy_returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Perform t-test on strategy returns.

        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with t-test results
        """
        from scipy import stats

        if len(strategy_returns) < 30:
            return {
                "error": "Insufficient data for t-test (minimum 30 observations required)"
            }

        # Test if mean return is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)

        results = {
            "mean_return": strategy_returns.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }

        # Compare to benchmark if provided
        if benchmark_returns is not None and len(benchmark_returns) == len(
            strategy_returns
        ):
            # Test if strategy returns are significantly different from benchmark
            t_stat_vs_bench, p_value_vs_bench = stats.ttest_ind(
                strategy_returns, benchmark_returns, equal_var=False
            )

            results.update(
                {
                    "benchmark_mean_return": benchmark_returns.mean(),
                    "vs_benchmark_t_statistic": t_stat_vs_bench,
                    "vs_benchmark_p_value": p_value_vs_bench,
                    "is_significant_vs_benchmark": p_value_vs_bench < 0.05,
                }
            )

        return results

    def bootstrap_test(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series = None,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for statistical significance.

        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Optional benchmark returns for comparison
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with bootstrap test results
        """
        if len(strategy_returns) < 30:
            return {
                "error": "Insufficient data for bootstrap test (minimum 30 observations required)"
            }

        # Bootstrap mean return
        mean_returns_bootstrap = []
        sharpe_ratios_bootstrap = []

        for _ in range(n_iterations):
            # Sample with replacement
            bootstrap_sample = strategy_returns.sample(
                len(strategy_returns), replace=True
            )

            # Calculate metrics
            mean_return = bootstrap_sample.mean()
            sharpe_ratio = (
                np.sqrt(252) * mean_return / bootstrap_sample.std()
                if bootstrap_sample.std() > 0
                else 0
            )

            mean_returns_bootstrap.append(mean_return)
            sharpe_ratios_bootstrap.append(sharpe_ratio)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        mean_return_lower = np.percentile(mean_returns_bootstrap, alpha / 2 * 100)
        mean_return_upper = np.percentile(mean_returns_bootstrap, (1 - alpha / 2) * 100)

        sharpe_lower = np.percentile(sharpe_ratios_bootstrap, alpha / 2 * 100)
        sharpe_upper = np.percentile(sharpe_ratios_bootstrap, (1 - alpha / 2) * 100)

        # Calculate p-values
        mean_return_p_value = np.mean([r <= 0 for r in mean_returns_bootstrap])
        sharpe_p_value = np.mean([s <= 0 for s in sharpe_ratios_bootstrap])

        results = {
            "mean_return": {
                "value": strategy_returns.mean(),
                "p_value": mean_return_p_value,
                "confidence_interval": [mean_return_lower, mean_return_upper],
                "is_significant": mean_return_p_value < 0.05,
            },
            "sharpe_ratio": {
                "value": (
                    np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
                    if strategy_returns.std() > 0
                    else 0
                ),
                "p_value": sharpe_p_value,
                "confidence_interval": [sharpe_lower, sharpe_upper],
                "is_significant": sharpe_p_value < 0.05,
            },
        }

        # Calculate alpha if benchmark returns are provided
        if benchmark_returns is not None and len(benchmark_returns) == len(
            strategy_returns
        ):
            import statsmodels.api as sm

            # Calculate alpha and beta using regression
            X = sm.add_constant(benchmark_returns)
            model = sm.OLS(strategy_returns, X).fit()

            alpha = model.params[0]
            beta = model.params[1]

            # Bootstrap alpha
            alphas_bootstrap = []

            for _ in range(n_iterations):
                # Generate indices with replacement
                indices = np.random.choice(
                    len(strategy_returns), len(strategy_returns), replace=True
                )

                # Create bootstrap samples
                strat_sample = strategy_returns.iloc[indices]
                bench_sample = benchmark_returns.iloc[indices]

                # Calculate alpha
                X_boot = sm.add_constant(bench_sample)
                model_boot = sm.OLS(strat_sample, X_boot).fit()

                alphas_bootstrap.append(model_boot.params[0])

            # Calculate confidence interval for alpha
            alpha_lower = np.percentile(alphas_bootstrap, alpha / 2 * 100)
            alpha_upper = np.percentile(alphas_bootstrap, (1 - alpha / 2) * 100)

            # Calculate p-value for alpha
            alpha_p_value = np.mean([a <= 0 for a in alphas_bootstrap])

            results["alpha"] = {
                "value": alpha,
                "p_value": alpha_p_value,
                "confidence_interval": [alpha_lower, alpha_upper],
                "is_significant": alpha_p_value < 0.05,
                "beta": beta,
            }

        return results
