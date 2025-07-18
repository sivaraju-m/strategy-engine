"""
Advanced backtesting analytics for robust strategy evaluation.
Includes Monte Carlo simulation, stress testing, and statistical significance testing.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from google.cloud import bigquery
import time
import logging
import json
from scipy import stats

from src.ai_trading_machine.utils.bq_logger import log_backtest_result, ensure_bq_dataset_and_table


class MonteCarloSimulator:
    """
    Monte Carlo simulation for testing strategy robustness.
    
    Attributes:
        n_simulations: Number of Monte Carlo simulations to run
        confidence_level: Confidence level for simulation results
        bq_client: BigQuery client for logging results
        results_table: BigQuery table for results
    """
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
        bq_client: Optional[bigquery.Client] = None,
        results_table: str = "monte_carlo_results"
    ):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for simulation results
            bq_client: BigQuery client for logging results
            results_table: BigQuery table for results
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.client = bq_client
        self.results_table = results_table
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure BQ table exists if client is provided
        if self.client:
            ensure_bq_dataset_and_table(self.client, results_table)
    
    def simulate_price_paths(
        self,
        df: pd.DataFrame,
        n_paths: Optional[int] = None,
        forward_periods: int = 252,
        volatility_window: int = 20
    ) -> np.ndarray:
        """
        Simulate multiple price paths using Geometric Brownian Motion.
        
        Args:
            df: Historical price data
            n_paths: Number of paths to simulate (defaults to self.n_simulations)
            forward_periods: Number of periods to project forward
            volatility_window: Window for volatility calculation
            
        Returns:
            3D array of simulated paths [n_paths, n_periods, 5 (OHLCV)]
        """
        if not n_paths:
            n_paths = self.n_simulations
            
        # Extract price data
        close_prices = df['close'].values
        
        # Calculate returns and volatility
        returns = np.diff(np.log(close_prices)) 
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Initialize simulation array [paths, periods, OHLCV]
        paths = np.zeros((n_paths, forward_periods, 5))
        
        # Last known price
        last_price = close_prices[-1]
        
        # Simulate paths
        for i in range(n_paths):
            # Generate random returns with drift and volatility
            random_returns = np.random.normal(mu, sigma, forward_periods)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(random_returns)
            
            # Generate price path
            price_path = last_price * np.exp(cumulative_returns)
            
            # Add some random high-low spread based on historical data
            avg_hl_ratio = np.mean(df['high'] / df['low'])
            hl_spread = avg_hl_ratio - 1
            
            # Generate OHLCV data
            for j in range(forward_periods):
                price = price_path[j]
                high_price = price * (1 + np.random.random() * hl_spread)
                low_price = price * (1 - np.random.random() * hl_spread)
                
                # Simple volume model based on price movement
                rel_volume = 1.0 + abs(random_returns[j]) / sigma
                avg_volume = np.mean(df['volume'].values)
                volume = avg_volume * rel_volume
                
                paths[i, j] = [price, high_price, low_price, price, volume]  # Open, High, Low, Close, Volume
        
        return paths
    
    def simulate_strategy_performance(
        self,
        strategy_func: Callable,
        df: pd.DataFrame,
        params: Dict[str, Any],
        n_simulations: Optional[int] = None,
        bootstrap_pct: float = 0.5
    ) -> Dict[str, Any]:
        """
        Simulate strategy performance across bootstrapped samples.
        
        Args:
            strategy_func: Strategy function that returns signals
            df: Historical price data
            params: Strategy parameters
            n_simulations: Number of simulations to run
            bootstrap_pct: Percentage of data to sample in each simulation
            
        Returns:
            Dict of simulation results
        """
        if not n_simulations:
            n_simulations = self.n_simulations
        
        # Initialize results
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []
        win_rates = []
        
        start_time = time.time()
        self.logger.info(f"Starting Monte Carlo simulation with {n_simulations} iterations")
        
        # Run simulations
        for i in range(n_simulations):
            # Bootstrap sample
            sample_size = int(len(df) * bootstrap_pct)
            sample_indices = np.random.choice(len(df), size=sample_size, replace=True)
            sample = df.iloc[sample_indices].sort_index()
            
            # Run strategy
            try:
                signals = strategy_func(sample, params)
                
                # Calculate returns based on signals
                sample['signal'] = signals
                sample['return'] = sample['close'].pct_change()
                sample['strategy_return'] = sample['signal'].shift(1) * sample['return']
                
                # Calculate metrics
                total_return = (1 + sample['strategy_return']).prod() - 1
                
                # Annualized Sharpe Ratio
                daily_returns = sample['strategy_return'].dropna()
                if len(daily_returns) > 0:
                    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
                else:
                    sharpe = 0
                    
                # Max Drawdown
                cum_returns = (1 + sample['strategy_return']).cumprod()
                running_max = cum_returns.cummax()
                drawdown = (cum_returns / running_max) - 1
                max_drawdown = drawdown.min()
                
                # Win Rate
                wins = (sample['strategy_return'] > 0).sum()
                total_trades = (sample['signal'] != 0).sum()
                win_rate = wins / total_trades if total_trades > 0 else 0
                
                # Store results
                sharpe_ratios.append(sharpe)
                total_returns.append(total_return)
                max_drawdowns.append(max_drawdown)
                win_rates.append(win_rate)
                
            except Exception as e:
                self.logger.warning(f"Error in simulation {i}: {e}")
                continue
                
        # Calculate statistics
        results = {
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'sharpe_5th_percentile': np.percentile(sharpe_ratios, 5),
            'sharpe_95th_percentile': np.percentile(sharpe_ratios, 95),
            
            'mean_return': np.mean(total_returns),
            'median_return': np.median(total_returns),
            'return_5th_percentile': np.percentile(total_returns, 5),
            'return_95th_percentile': np.percentile(total_returns, 95),
            
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.min(max_drawdowns),
            
            'mean_win_rate': np.mean(win_rates),
            'win_rate_5th_percentile': np.percentile(win_rates, 5),
            
            'n_simulations': n_simulations,
            'n_successful': len(sharpe_ratios),
            'runtime_seconds': time.time() - start_time
        }
        
        # Calculate p-value for hypothesis that returns > 0
        t_stat, p_value = stats.ttest_1samp(total_returns, 0)
        results['p_value'] = p_value
        results['t_statistic'] = t_stat
        results['statistically_significant'] = p_value < 0.05
        
        # Log results to BigQuery if client is available
        if self.client:
            log_data = {
                'strategy_name': params.get('strategy_name', 'unnamed_strategy'),
                'timestamp': datetime.now().isoformat(),
                'n_simulations': n_simulations,
                'mean_sharpe_ratio': results['mean_sharpe'],
                'mean_return': results['mean_return'],
                'mean_max_drawdown': results['mean_max_drawdown'],
                'p_value': results['p_value'],
                'statistically_significant': results['statistically_significant'],
                'simulation_params': json.dumps(params)
            }
            
            log_backtest_result(self.client, self.results_table, log_data)
        
        return results
    
    def plot_simulation_results(
        self,
        results: Dict[str, Any],
        metric: str = 'sharpe',
        file_path: Optional[str] = None
    ) -> None:
        """
        Plot histogram of simulation results.
        
        Args:
            results: Simulation results from simulate_strategy_performance
            metric: Metric to plot ('sharpe' or 'return')
            file_path: Optional path to save the plot
        """
        if metric == 'sharpe':
            data = results.get('sharpe_ratios', [])
            title = f"Distribution of Sharpe Ratios (Mean: {results['mean_sharpe']:.4f})"
            xlabel = "Sharpe Ratio"
        elif metric == 'return':
            data = results.get('total_returns', [])
            title = f"Distribution of Total Returns (Mean: {results['mean_return']:.4f})"
            xlabel = "Total Return"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, alpha=0.7, color='blue')
        plt.axvline(np.mean(data), color='red', linestyle='dashed', linewidth=2)
        plt.axvline(np.percentile(data, 5), color='green', linestyle='dashed', linewidth=2)
        plt.axvline(np.percentile(data, 95), color='green', linestyle='dashed', linewidth=2)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
        plt.show()


class StressTester:
    """
    Stress testing framework for strategies under extreme market conditions.
    
    Attributes:
        bq_client: BigQuery client for logging results
        results_table: BigQuery table for results
    """
    
    def __init__(
        self,
        bq_client: Optional[bigquery.Client] = None,
        results_table: str = "stress_test_results"
    ):
        """
        Initialize the stress tester.
        
        Args:
            bq_client: BigQuery client for logging results
            results_table: BigQuery table for results
        """
        self.client = bq_client
        self.results_table = results_table
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure BQ table exists if client is provided
        if self.client:
            ensure_bq_dataset_and_table(self.client, results_table)
    
    def apply_market_crash(
        self,
        df: pd.DataFrame,
        crash_percent: float = -0.15,
        recovery_days: int = 60,
        crash_day: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply a market crash scenario to the data.
        
        Args:
            df: Historical price data
            crash_percent: Percentage change during crash (negative)
            recovery_days: Days for partial recovery
            crash_day: Specific date for crash (or random if None)
            
        Returns:
            DataFrame with crash scenario applied
        """
        # Make a copy to avoid modifying original
        scenario_df = df.copy()
        
        # Sort by date
        scenario_df = scenario_df.sort_values('date')
        
        # Determine crash day index
        if crash_day:
            crash_idx = scenario_df[scenario_df['date'] == crash_day].index
            if len(crash_idx) == 0:
                crash_idx = int(len(scenario_df) * 0.7)  # Default to 70% through data
            else:
                crash_idx = crash_idx[0]
        else:
            # Random crash somewhere in middle 50% of the data
            min_idx = int(len(scenario_df) * 0.25)
            max_idx = int(len(scenario_df) * 0.75)
            crash_idx = np.random.randint(min_idx, max_idx)
        
        # Apply crash
        pre_crash_price = scenario_df.iloc[crash_idx]['close']
        crash_price = pre_crash_price * (1 + crash_percent)
        
        # Determine recovery price (60% recovery)
        recovery_percent = -0.6 * crash_percent
        recovery_price = crash_price * (1 + recovery_percent)
        
        # Apply crash and recovery
        for i in range(crash_idx + 1, min(crash_idx + recovery_days + 1, len(scenario_df))):
            # Linear model for simplicity
            progress = (i - crash_idx) / recovery_days
            progress = min(progress, 1.0)
            
            # Price interpolation
            if progress < 0.2:  # First 20% is crash
                crash_progress = progress / 0.2
                price = pre_crash_price * (1 - crash_progress * abs(crash_percent))
            else:  # Recovery phase
                recovery_progress = (progress - 0.2) / 0.8
                price = crash_price * (1 + recovery_progress * recovery_percent)
            
            # Apply to OHLC
            ratio = price / scenario_df.iloc[i]['close']
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['open', 'high', 'low', 'close'])] *= ratio
            
            # Increase volume during crash (typically 2-3x)
            volume_multiplier = 3.0 if progress < 0.2 else 2.0
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['volume'])] *= volume_multiplier
        
        # Add scenario metadata
        scenario_df['scenario'] = 'market_crash'
        scenario_df['crash_day'] = scenario_df.iloc[crash_idx]['date']
        scenario_df['crash_percent'] = crash_percent
        
        return scenario_df
    
    def apply_volatility_shock(
        self,
        df: pd.DataFrame,
        volatility_multiplier: float = 3.0,
        shock_days: int = 30,
        shock_day: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply a volatility shock scenario to the data.
        
        Args:
            df: Historical price data
            volatility_multiplier: Factor to increase volatility
            shock_days: Duration of volatility shock
            shock_day: Specific date for shock (or random if None)
            
        Returns:
            DataFrame with volatility shock applied
        """
        # Make a copy to avoid modifying original
        scenario_df = df.copy()
        
        # Sort by date
        scenario_df = scenario_df.sort_values('date')
        
        # Determine shock day index
        if shock_day:
            shock_idx = scenario_df[scenario_df['date'] == shock_day].index
            if len(shock_idx) == 0:
                shock_idx = int(len(scenario_df) * 0.7)  # Default to 70% through data
            else:
                shock_idx = shock_idx[0]
        else:
            # Random shock somewhere in middle 50% of the data
            min_idx = int(len(scenario_df) * 0.25)
            max_idx = int(len(scenario_df) * 0.75)
            shock_idx = np.random.randint(min_idx, max_idx)
        
        # Calculate historical volatility
        returns = scenario_df['close'].pct_change().dropna()
        historical_vol = returns.std()
        
        # Apply volatility shock
        for i in range(shock_idx + 1, min(shock_idx + shock_days + 1, len(scenario_df))):
            # Random return with amplified volatility
            shock_vol = historical_vol * volatility_multiplier
            random_return = np.random.normal(0, shock_vol)
            
            # Previous close
            prev_close = scenario_df.iloc[i-1]['close']
            
            # New close with shock
            new_close = prev_close * (1 + random_return)
            
            # Apply to OHLC with appropriate range
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['close'])] = new_close
            
            # Adjust high/low for increased volatility
            high_low_range = np.abs(random_return) * 1.5
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['high'])] = new_close * (1 + high_low_range/2)
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['low'])] = new_close * (1 - high_low_range/2)
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['open'])] = prev_close * (1 + random_return/2)
            
            # Increase volume during volatility shock
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['volume'])] *= (1.5 + np.abs(random_return) * 10)
        
        # Add scenario metadata
        scenario_df['scenario'] = 'volatility_shock'
        scenario_df['shock_day'] = scenario_df.iloc[shock_idx]['date']
        scenario_df['volatility_multiplier'] = volatility_multiplier
        
        return scenario_df
    
    def apply_liquidity_crisis(
        self,
        df: pd.DataFrame,
        volume_reduction: float = 0.8,
        spread_multiplier: float = 5.0,
        crisis_days: int = 45,
        crisis_day: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply a liquidity crisis scenario to the data.
        
        Args:
            df: Historical price data
            volume_reduction: Percentage reduction in volume
            spread_multiplier: Factor to increase bid-ask spread (modeled by high-low)
            crisis_days: Duration of liquidity crisis
            crisis_day: Specific date for crisis (or random if None)
            
        Returns:
            DataFrame with liquidity crisis applied
        """
        # Make a copy to avoid modifying original
        scenario_df = df.copy()
        
        # Sort by date
        scenario_df = scenario_df.sort_values('date')
        
        # Determine crisis day index
        if crisis_day:
            crisis_idx = scenario_df[scenario_df['date'] == crisis_day].index
            if len(crisis_idx) == 0:
                crisis_idx = int(len(scenario_df) * 0.7)  # Default to 70% through data
            else:
                crisis_idx = crisis_idx[0]
        else:
            # Random crisis somewhere in middle 50% of the data
            min_idx = int(len(scenario_df) * 0.25)
            max_idx = int(len(scenario_df) * 0.75)
            crisis_idx = np.random.randint(min_idx, max_idx)
        
        # Apply liquidity crisis
        for i in range(crisis_idx + 1, min(crisis_idx + crisis_days + 1, len(scenario_df))):
            # Reduce volume
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['volume'])] *= (1 - volume_reduction)
            
            # Increase high-low spread as proxy for bid-ask spread
            mid_price = scenario_df.iloc[i]['close']
            current_spread = scenario_df.iloc[i]['high'] - scenario_df.iloc[i]['low']
            new_spread = current_spread * spread_multiplier
            
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['high'])] = mid_price + new_spread/2
            scenario_df.iloc[i, scenario_df.columns.get_indexer(['low'])] = mid_price - new_spread/2
        
        # Add scenario metadata
        scenario_df['scenario'] = 'liquidity_crisis'
        scenario_df['crisis_day'] = scenario_df.iloc[crisis_idx]['date']
        scenario_df['volume_reduction'] = volume_reduction
        scenario_df['spread_multiplier'] = spread_multiplier
        
        return scenario_df
    
    def run_strategy_stress_test(
        self,
        strategy_func: Callable,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scenarios: List[str] = ['market_crash', 'volatility_shock', 'liquidity_crisis'],
        scenario_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run stress tests for a strategy under various extreme scenarios.
        
        Args:
            strategy_func: Strategy function that returns signals
            df: Historical price data
            params: Strategy parameters
            scenarios: List of scenarios to test
            scenario_params: Parameters for each scenario
            
        Returns:
            Dict of stress test results by scenario
        """
        results = {}
        
        # Initialize default scenario parameters
        default_scenario_params = {
            'market_crash': {'crash_percent': -0.15, 'recovery_days': 60},
            'volatility_shock': {'volatility_multiplier': 3.0, 'shock_days': 30},
            'liquidity_crisis': {'volume_reduction': 0.8, 'spread_multiplier': 5.0, 'crisis_days': 45}
        }
        
        # Use provided parameters or defaults
        if scenario_params:
            for scenario, params_dict in scenario_params.items():
                if scenario in default_scenario_params:
                    default_scenario_params[scenario].update(params_dict)
        
        # Run baseline strategy for comparison
        try:
            baseline_signals = strategy_func(df, params)
            baseline_metrics = self._calculate_metrics(df, baseline_signals)
            results['baseline'] = baseline_metrics
        except Exception as e:
            self.logger.error(f"Error in baseline strategy: {e}")
            return {}
        
        # Run stress tests for each scenario
        for scenario in scenarios:
            if scenario == 'market_crash':
                scenario_df = self.apply_market_crash(
                    df,
                    crash_percent=default_scenario_params['market_crash']['crash_percent'],
                    recovery_days=default_scenario_params['market_crash']['recovery_days']
                )
            elif scenario == 'volatility_shock':
                scenario_df = self.apply_volatility_shock(
                    df,
                    volatility_multiplier=default_scenario_params['volatility_shock']['volatility_multiplier'],
                    shock_days=default_scenario_params['volatility_shock']['shock_days']
                )
            elif scenario == 'liquidity_crisis':
                scenario_df = self.apply_liquidity_crisis(
                    df,
                    volume_reduction=default_scenario_params['liquidity_crisis']['volume_reduction'],
                    spread_multiplier=default_scenario_params['liquidity_crisis']['spread_multiplier'],
                    crisis_days=default_scenario_params['liquidity_crisis']['crisis_days']
                )
            else:
                self.logger.warning(f"Unknown scenario: {scenario}")
                continue
            
            try:
                # Run strategy on stressed data
                stress_signals = strategy_func(scenario_df, params)
                stress_metrics = self._calculate_metrics(scenario_df, stress_signals)
                
                # Calculate impact
                impact = {
                    'sharpe_impact': stress_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
                    'return_impact': stress_metrics['total_return'] - baseline_metrics['total_return'],
                    'drawdown_impact': stress_metrics['max_drawdown'] - baseline_metrics['max_drawdown'],
                    'win_rate_impact': stress_metrics['win_rate'] - baseline_metrics['win_rate']
                }
                
                # Store results
                results[scenario] = {
                    **stress_metrics,
                    'impact': impact,
                    'scenario_params': default_scenario_params[scenario]
                }
                
                # Log results
                self.logger.info(f"{scenario} stress test results:")
                self.logger.info(f"  Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.4f}, "
                               f"Stress Sharpe: {stress_metrics['sharpe_ratio']:.4f}, "
                               f"Impact: {impact['sharpe_impact']:.4f}")
                self.logger.info(f"  Baseline Return: {baseline_metrics['total_return']:.4f}, "
                               f"Stress Return: {stress_metrics['total_return']:.4f}, "
                               f"Impact: {impact['return_impact']:.4f}")
                
                # Log to BigQuery if client is available
                if self.client:
                    log_data = {
                        'strategy_name': params.get('strategy_name', 'unnamed_strategy'),
                        'timestamp': datetime.now().isoformat(),
                        'scenario': scenario,
                        'baseline_sharpe': baseline_metrics['sharpe_ratio'],
                        'stress_sharpe': stress_metrics['sharpe_ratio'],
                        'sharpe_impact': impact['sharpe_impact'],
                        'baseline_return': baseline_metrics['total_return'],
                        'stress_return': stress_metrics['total_return'],
                        'return_impact': impact['return_impact'],
                        'baseline_drawdown': baseline_metrics['max_drawdown'],
                        'stress_drawdown': stress_metrics['max_drawdown'],
                        'drawdown_impact': impact['drawdown_impact'],
                        'scenario_params': json.dumps(default_scenario_params[scenario]),
                        'strategy_params': json.dumps(params)
                    }
                    
                    log_backtest_result(self.client, self.results_table, log_data)
                
            except Exception as e:
                self.logger.error(f"Error in {scenario} stress test: {e}")
                results[scenario] = {'error': str(e)}
        
        return results
    
    def _calculate_metrics(self, df: pd.DataFrame, signals: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            df: Historical price data
            signals: Strategy signals
            
        Returns:
            Dict of performance metrics
        """
        # Prepare data
        data = df.copy()
        data['signal'] = signals
        data['return'] = data['close'].pct_change()
        data['strategy_return'] = data['signal'].shift(1) * data['return']
        
        # Calculate metrics
        total_return = (1 + data['strategy_return'].fillna(0)).prod() - 1
        
        # Annualized Sharpe Ratio
        daily_returns = data['strategy_return'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0
            
        # Max Drawdown
        cum_returns = (1 + data['strategy_return'].fillna(0)).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win Rate
        wins = (data['strategy_return'] > 0).sum()
        total_trades = (data['signal'] != 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': total_trades
        }


class StatisticalSignificanceTester:
    """
    Tests for statistical significance of strategy returns.
    
    Attributes:
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for testing
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        Initialize the statistical significance tester.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for testing
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def bootstrap_test(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run bootstrap test for statistical significance.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dict of test results
        """
        # Clean data
        strategy_returns = strategy_returns.dropna()
        
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.dropna()
            # Align dates
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_returns = strategy_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate actual metrics
        actual_mean = strategy_returns.mean()
        actual_sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        
        if benchmark_returns is not None:
            actual_alpha = strategy_returns.mean() - benchmark_returns.mean()
            actual_beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
        
        # Bootstrap sampling
        bootstrap_means = []
        bootstrap_sharpes = []
        bootstrap_alphas = []
        bootstrap_betas = []
        
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            sample_indices = np.random.choice(len(strategy_returns), size=len(strategy_returns), replace=True)
            strategy_sample = strategy_returns.iloc[sample_indices]
            
            # Calculate metrics
            bootstrap_means.append(strategy_sample.mean())
            bootstrap_sharpes.append(np.sqrt(252) * strategy_sample.mean() / strategy_sample.std() if strategy_sample.std() > 0 else 0)
            
            if benchmark_returns is not None:
                benchmark_sample = benchmark_returns.iloc[sample_indices]
                bootstrap_alphas.append(strategy_sample.mean() - benchmark_sample.mean())
                bootstrap_betas.append(np.cov(strategy_sample, benchmark_sample)[0, 1] / np.var(benchmark_sample) if np.var(benchmark_sample) > 0 else 0)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'mean_return': {
                'actual': actual_mean,
                'bootstrap_mean': np.mean(bootstrap_means),
                'ci_lower': np.percentile(bootstrap_means, lower_percentile),
                'ci_upper': np.percentile(bootstrap_means, upper_percentile),
                'p_value': np.mean(np.array(bootstrap_means) <= 0)
            },
            'sharpe_ratio': {
                'actual': actual_sharpe,
                'bootstrap_mean': np.mean(bootstrap_sharpes),
                'ci_lower': np.percentile(bootstrap_sharpes, lower_percentile),
                'ci_upper': np.percentile(bootstrap_sharpes, upper_percentile),
                'p_value': np.mean(np.array(bootstrap_sharpes) <= 0)
            }
        }
        
        if benchmark_returns is not None:
            results['alpha'] = {
                'actual': actual_alpha,
                'bootstrap_mean': np.mean(bootstrap_alphas),
                'ci_lower': np.percentile(bootstrap_alphas, lower_percentile),
                'ci_upper': np.percentile(bootstrap_alphas, upper_percentile),
                'p_value': np.mean(np.array(bootstrap_alphas) <= 0)
            }
            results['beta'] = {
                'actual': actual_beta,
                'bootstrap_mean': np.mean(bootstrap_betas),
                'ci_lower': np.percentile(bootstrap_betas, lower_percentile),
                'ci_upper': np.percentile(bootstrap_betas, upper_percentile)
            }
        
        # Add interpretations
        results['mean_return']['significant'] = results['mean_return']['p_value'] < (1 - self.confidence_level)
        results['sharpe_ratio']['significant'] = results['sharpe_ratio']['p_value'] < (1 - self.confidence_level)
        
        if benchmark_returns is not None:
            results['alpha']['significant'] = results['alpha']['p_value'] < (1 - self.confidence_level)
        
        # Log results
        self.logger.info("Statistical significance test results:")
        self.logger.info(f"  Mean Return: {actual_mean:.6f}, p-value: {results['mean_return']['p_value']:.4f}, "
                       f"Significant: {results['mean_return']['significant']}")
        self.logger.info(f"  Sharpe Ratio: {actual_sharpe:.4f}, p-value: {results['sharpe_ratio']['p_value']:.4f}, "
                       f"Significant: {results['sharpe_ratio']['significant']}")
        
        if benchmark_returns is not None:
            self.logger.info(f"  Alpha: {actual_alpha:.6f}, p-value: {results['alpha']['p_value']:.4f}, "
                           f"Significant: {results['alpha']['significant']}")
            self.logger.info(f"  Beta: {actual_beta:.4f}")
        
        return results
"""
