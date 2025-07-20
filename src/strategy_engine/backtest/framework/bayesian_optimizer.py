"""
Bayesian optimization module for trading strategy parameter tuning.
Utilizes Gaussian Process Regression to efficiently search parameter spaces.
"""

from typing import Dict, List, Callable, Optional, Tuple, Union, Any, TypeVar
import pandas as pd
import numpy as np
import time
from datetime import datetime
from google.cloud import bigquery

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

from src.ai_trading_machine.utils.bq_logger import (
    log_backtest_result,
    ensure_bq_dataset_and_table,
)

# Type variables
StrategyType = TypeVar("StrategyType")


class BayesianStrategyOptimizer:
    """
    Bayesian optimization for trading strategy parameters using Gaussian Process Regression.

    Attributes:
        strategy_class: Trading strategy class to optimize
        param_space: List of parameter space definitions
        evaluation_func: Function to evaluate strategy performance
        client: BigQuery client for result logging
    """

    def __init__(
        self,
        strategy_class: Any,
        param_space: List,
        evaluation_func: Optional[Callable] = None,
        bq_client: Optional[bigquery.Client] = None,
        results_table: str = "strategy_parameter_results",
    ):
        """
        Initialize the Bayesian optimizer.

        Args:
            strategy_class: Trading strategy class to optimize
            param_space: List of skopt space definitions (Real, Integer, Categorical)
            evaluation_func: Custom evaluation function (or use default Sharpe)
            bq_client: BigQuery client for result logging
            results_table: Table name for parameter results

        Example:
            param_space = [
                Integer(10, 30, name='window'),
                Real(0.1, 0.5, name='threshold'),
                Categorical(['simple', 'exponential'], name='ma_type')
            ]
        """
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.evaluation_func = evaluation_func or self._default_evaluation_func
        self.client = bq_client or bigquery.Client()
        self.results_table = results_table
        self.param_names = [p.name for p in param_space]
        self.results_history = []
        self.optimization_data = None
        self.last_result = None

        # Ensure BigQuery table exists
        ensure_bq_dataset_and_table()

    def _default_evaluation_func(self, strategy: Any, df: pd.DataFrame) -> float:
        """
        Default evaluation function that optimizes for Sharpe ratio.

        Args:
            strategy: Instantiated strategy object
            df: OHLCV DataFrame

        Returns:
            Negative Sharpe ratio (for minimization)
        """
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
                        next_return * signal if not np.isnan(next_return) else 0
                    )
                    returns.append(trade_return)
            except Exception as e:
                print(f"Error in strategy execution: {e}")
                returns.append(0)

        # Calculate Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            return -sharpe_ratio  # Negative because we're minimizing
        else:
            return 0.0  # Neutral result if no valid returns

    @use_named_args
    def _objective_function(self, **params):
        """
        Objective function for Bayesian optimization.
        Decorated with use_named_args to handle parameter space.

        Args:
            **params: Strategy parameters from search space

        Returns:
            Optimization target value (lower is better)
        """
        # Create strategy instance with parameters
        strategy = self.strategy_class(params)

        # Apply evaluation function to get optimization target
        target_value = self.evaluation_func(strategy, self.optimization_data)

        # Store result for logging
        result = {
            "params": params,
            "target_value": target_value,
            "timestamp": datetime.now(),
        }
        self.results_history.append(result)

        # Log to BigQuery if we have a table defined
        if self.results_table:
            metrics = {
                "strategy_id": self.strategy_class.__name__,
                "params": str(params),
                "sharpe_ratio": -target_value,  # Convert back to positive
                "timestamp": datetime.now(),
                "symbol": (
                    self.optimization_data["symbol"].iloc[0]
                    if "symbol" in self.optimization_data.columns
                    else "unknown"
                ),
                "start_date": self.optimization_data["date"].min(),
                "end_date": self.optimization_data["date"].max(),
                "tag": "bayesian_opt",
            }

            log_backtest_result(**metrics)

        return target_value

    def optimize(
        self,
        df: pd.DataFrame,
        n_calls: int = 20,
        n_random_starts: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization of strategy parameters.

        Args:
            df: OHLCV DataFrame for optimization
            n_calls: Number of optimization iterations
            n_random_starts: Number of random points before optimization
            verbose: Whether to print progress

        Returns:
            Dictionary with optimal parameters and metrics

        Raises:
            ValueError: If input data is invalid
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        required_cols = ["date", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store data for optimization
        self.optimization_data = df

        # Reset results history
        self.results_history = []

        start_time = time.time()

        if verbose:
            print(
                f"Starting Bayesian optimization for {self.strategy_class.__name__}..."
            )
            print(f"Parameter space: {self.param_names}")
            print(f"Data period: {df['date'].min()} to {df['date'].max()}")
            print(f"Number of data points: {len(df)}")

        # Run optimization
        result = gp_minimize(
            self._objective_function,
            self.param_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            verbose=verbose,
            random_state=42,
        )

        elapsed_time = time.time() - start_time

        # Construct result dictionary
        optimal_params = {
            name: value for name, value in zip(self.param_names, result.x)
        }

        optimization_result = {
            "optimal_params": optimal_params,
            "sharpe_ratio": -result.fun,  # Convert back to positive
            "convergence": result.func_vals.tolist(),
            "n_iterations": n_calls,
            "elapsed_time": elapsed_time,
            "parameter_importance": self._calculate_parameter_importance(result),
            "all_results": self.results_history,
        }

        # Store the result for later reference
        self.last_result = result

        if verbose:
            print(f"Optimization complete in {elapsed_time:.2f} seconds")
            print(f"Optimal parameters: {optimal_params}")
            print(f"Best Sharpe ratio: {-result.fun:.4f}")

        return optimization_result

    def _calculate_parameter_importance(self, result) -> Dict[str, float]:
        """
        Calculate relative importance of each parameter.

        Args:
            result: Optimization result from gp_minimize

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        # Simple approach based on variance of parameter values
        # in the top-performing iterations
        top_indices = np.argsort(result.func_vals)[:10]  # Top 10 results

        importance = {}
        for i, param_name in enumerate(self.param_names):
            # Extract values for this parameter from top results
            values = [result.x_iters[idx][i] for idx in top_indices]

            # Calculate variance of parameter in top results
            if isinstance(values[0], (int, float)):
                importance[param_name] = np.std(values)
            else:
                # For categorical, use entropy
                unique_values = set(values)
                counts = [values.count(val) for val in unique_values]
                probs = [count / len(values) for count in counts]
                entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
                importance[param_name] = entropy

        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def plot_optimization_results(
        self, result: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None
    ) -> None:
        """
        Plot optimization results.

        Args:
            result: Optimization result dictionary (or use last result)
            output_dir: Directory to save plots (or display inline)
        """
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_convergence

            if result is None and hasattr(self, "last_result"):
                result = self.last_result

            if result is None:
                raise ValueError("No optimization result available")

            # Plot convergence
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = plot_convergence(result, ax=ax)
            ax.set_title(f"Convergence for {self.strategy_class.__name__}")
            if output_dir:
                plt.savefig(
                    f"{output_dir}/convergence_{self.strategy_class.__name__}.png"
                )
                plt.close()
            else:
                plt.show()

            # Plot parameter importance
            if "parameter_importance" in result:
                fig, ax = plt.subplots(figsize=(10, 6))
                importances = result["parameter_importance"]
                sorted_params = sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                )
                params = [p[0] for p in sorted_params]
                scores = [p[1] for p in sorted_params]

                ax.bar(params, scores)
                ax.set_title(f"Parameter Importance for {self.strategy_class.__name__}")
                ax.set_ylabel("Relative Importance")
                ax.set_xlabel("Parameter")
                plt.xticks(rotation=45)

                if output_dir:
                    plt.tight_layout()
                    plt.savefig(
                        f"{output_dir}/importance_{self.strategy_class.__name__}.png"
                    )
                    plt.close()
                else:
                    plt.tight_layout()
                    plt.show()

        except ImportError:
            print(
                "Plotting requires matplotlib and scikit-optimize. Install with: pip install matplotlib scikit-optimize"
            )

    def multi_objective_optimize(
        self,
        df: pd.DataFrame,
        objective_funcs: List[Callable],
        weights: List[float],
        n_calls: int = 20,
        n_random_starts: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform multi-objective Bayesian optimization of strategy parameters.

        Args:
            df: OHLCV DataFrame for optimization
            objective_funcs: List of objective functions to optimize
            weights: List of weights for each objective (must sum to 1)
            n_calls: Number of optimization iterations
            n_random_starts: Number of random points before optimization
            verbose: Whether to print progress

        Returns:
            Dictionary with optimal parameters and metrics

        Raises:
            ValueError: If input parameters are invalid
        """
        if len(objective_funcs) != len(weights):
            raise ValueError(
                "Number of objective functions must match number of weights"
            )

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")

        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        # Store original evaluation function
        original_eval_func = self.evaluation_func

        # Create combined objective function
        def combined_objective(strategy, data):
            results = []
            for i, func in enumerate(objective_funcs):
                # Each function should return a value to minimize
                value = func(strategy, data)
                results.append(value * weights[i])
            return sum(results)

        # Set combined objective function
        self.evaluation_func = combined_objective

        # Run optimization
        result = self.optimize(df, n_calls, n_random_starts, verbose)

        # Restore original evaluation function
        self.evaluation_func = original_eval_func

        return result
