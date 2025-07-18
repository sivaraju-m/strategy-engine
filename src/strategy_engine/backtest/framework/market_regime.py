"""
Market regime-based backtesting pipeline.
Provides tools to classify market periods into different regimes and
run strategy backtests against specific market conditions.
"""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery

from src.ai_trading_machine.utils.bq_logger import log_backtest_result
from src.ai_trading_machine.utils.metrics import calculate_performance_metrics


class MarketRegimeClassifier:
    """
    Classifies market data into different regimes (bull, bear, sideways).
    
    Attributes:
        lookback_period: Period for trend calculation
        volatility_window: Window size for volatility calculation
        bull_threshold: Minimum return to classify as bull market
        bear_threshold: Maximum return to classify as bear market
        vol_threshold: Volatility threshold for classification
    """
    
    def __init__(
        self,
        lookback_period: int = 90,
        volatility_window: int = 20,
        bull_threshold: float = 0.1,
        bear_threshold: float = -0.1,
        vol_threshold: float = 0.15
    ):
        """
        Initialize the market regime classifier.
        
        Args:
            lookback_period: Period for trend calculation in days
            volatility_window: Window size for volatility calculation
            bull_threshold: Minimum return to classify as bull market
            bear_threshold: Maximum return to classify as bear market
            vol_threshold: Volatility threshold for classification
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.vol_threshold = vol_threshold
    
    def classify(self, df: pd.DataFrame, index_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Classify market data into regimes.
        
        Args:
            df: OHLCV DataFrame for a single symbol
            index_data: Optional market index data for classification
            
        Returns:
            DataFrame with regime column added
            
        Raises:
            ValueError: If input data is invalid
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
            
        # Use index data if provided, otherwise use the symbol data
        data = index_data if index_data is not None else df
        
        # Calculate returns
        data = data.sort_values('date')
        data['pct_change'] = data['close'].pct_change()
        data['rolling_return'] = data['close'].pct_change(self.lookback_period)
        data['volatility'] = data['pct_change'].rolling(self.volatility_window).std() * np.sqrt(252)
        
        # Classify regimes
        data['regime'] = 'sideways'  # Default
        data.loc[data['rolling_return'] >= self.bull_threshold, 'regime'] = 'bull'
        data.loc[data['rolling_return'] <= self.bear_threshold, 'regime'] = 'bear'
        data.loc[data['volatility'] >= self.vol_threshold, 'regime'] = 'volatile'
        
        # Merge regime information with original dataframe
        result = df.copy()
        result = pd.merge(
            result,
            data[['date', 'regime']],
            on='date',
            how='left'
        )
        
        # Forward fill any missing regimes
        result['regime'] = result['regime'].fillna(method='ffill').fillna('sideways')
        
        return result

    def get_regime_periods(self, df: pd.DataFrame) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """
        Extract date ranges for each regime type.
        
        Args:
            df: DataFrame with regime classification
            
        Returns:
            Dictionary of regime types with list of (start_date, end_date) tuples
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must have regime column")
            
        regimes = {}
        df = df.sort_values('date')
        
        # Group consecutive regime periods
        regime_changes = df['regime'] != df['regime'].shift(1)
        regime_start_indices = df.index[regime_changes].tolist()
        
        if len(regime_start_indices) == 0:
            return regimes
            
        for i in range(len(regime_start_indices)):
            start_idx = regime_start_indices[i]
            end_idx = regime_start_indices[i+1] - 1 if i < len(regime_start_indices) - 1 else df.index[-1]
            
            regime = df.loc[start_idx, 'regime']
            start_date = df.loc[start_idx, 'date']
            end_date = df.loc[end_idx, 'date']
            
            if regime not in regimes:
                regimes[regime] = []
                
            # Only include periods of sufficient length (at least 30 days)
            if (end_date - start_date).days >= 30:
                regimes[regime].append((start_date, end_date))
                
        return regimes


class RegimeBacktester:
    """
    Runs backtests on different market regimes.
    
    Attributes:
        classifier: MarketRegimeClassifier instance
        client: BigQuery client for result logging
    """
    
    def __init__(
        self, 
        classifier: Optional[MarketRegimeClassifier] = None,
        bq_client: Optional[bigquery.Client] = None
    ):
        """
        Initialize the regime backtester.
        
        Args:
            classifier: MarketRegimeClassifier instance (or uses default)
            bq_client: BigQuery client for result logging
        """
        self.classifier = classifier or MarketRegimeClassifier()
        self.client = bq_client or bigquery.Client()
    
    def run_regime_backtest(
        self,
        df: pd.DataFrame,
        strategy_class,
        strategy_params: Dict,
        regimes: Optional[List[str]] = None,
        index_data: Optional[pd.DataFrame] = None,
        tag: str = ""
    ) -> Dict[str, Dict]:
        """
        Run backtests on specific market regimes.
        
        Args:
            df: OHLCV DataFrame
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for the strategy
            regimes: List of regimes to test ('bull', 'bear', 'sideways', 'volatile')
            index_data: Optional market index data for classification
            tag: Optional tag for logging results
            
        Returns:
            Dictionary of regime results with performance metrics
            
        Raises:
            ValueError: If input data is invalid
        """
        # Default to all regimes if not specified
        regimes = regimes or ['bull', 'bear', 'sideways', 'volatile']
        
        # Classify market data
        classified_df = self.classifier.classify(df, index_data)
        
        results = {}
        strategy_id = strategy_class.__name__
        
        # Run backtest for each regime
        for regime in regimes:
            regime_data = classified_df[classified_df['regime'] == regime]
            
            if len(regime_data) < 20:  # Skip regimes with insufficient data
                continue
                
            # Initialize strategy
            strategy = strategy_class(strategy_params)
            
            # Run strategy on regime data
            signals = []
            returns = []
            
            for i in range(len(regime_data) - 1):
                row = regime_data.iloc[i:i+1]
                next_row = regime_data.iloc[i+1:i+2]
                
                if row.empty or next_row.empty:
                    continue
                    
                result = strategy.run(regime_data.iloc[:i+1])
                
                if result and 'signal' in result:
                    signal = result['signal']
                    signals.append(signal)
                    
                    # Calculate return based on signal
                    if signal != 0:
                        price_change = (next_row['close'].values[0] - row['close'].values[0]) / row['close'].values[0]
                        trade_return = price_change * signal  # Positive for long, negative for short
                        returns.append(trade_return)
                    else:
                        returns.append(0)
            
            # Calculate performance metrics
            if returns:
                metrics = calculate_performance_metrics(returns)
                metrics['regime'] = regime
                metrics['sample_size'] = len(returns)
                metrics['win_rate'] = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
                
                results[regime] = metrics
                
                # Log to BigQuery
                log_data = {
                    'strategy_id': strategy_id,
                    'params': str(strategy_params),
                    'regime': regime,
                    'start_date': regime_data['date'].min(),
                    'end_date': regime_data['date'].max(),
                    'symbol': df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown',
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_return': metrics.get('total_return', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'sample_size': metrics.get('sample_size', 0),
                    'tag': tag,
                    'timestamp': datetime.now()
                }
                
                log_backtest_result(**log_data)
        
        return results
    
    def run_multi_regime_comparison(
        self,
        symbols: List[str],
        strategy_class,
        strategy_params: Dict,
        start_date: str,
        end_date: str,
        regimes: Optional[List[str]] = None,
        market_index: str = 'NIFTY50',
        tag: str = ""
    ) -> Dict:
        """
        Run regime backtest comparison across multiple symbols.
        
        Args:
            symbols: List of symbol strings
            strategy_class: Strategy class to instantiate
            strategy_params: Parameters for the strategy
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            regimes: List of regimes to test
            market_index: Market index to use for regime classification
            tag: Optional tag for logging results
            
        Returns:
            Dictionary of aggregated results by regime
        """
        # Query to get market index data
        index_query = f"""
        SELECT date, open, high, low, close, volume
        FROM `ai-trading-gcp-459813.trading_data.large_cap_price_data`
        WHERE symbol = '{market_index}'
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """
        
        # Get index data
        index_df = self.client.query(index_query).to_dataframe()
        
        all_results = {}
        
        for symbol in symbols:
            # Query to get symbol data
            symbol_query = f"""
            SELECT date, open, high, low, close, volume, '{symbol}' as symbol
            FROM `ai-trading-gcp-459813.trading_data.large_cap_price_data`
            WHERE symbol = '{symbol}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """
            
            # Get symbol data
            symbol_df = self.client.query(symbol_query).to_dataframe()
            
            if symbol_df.empty:
                print(f"No data found for {symbol}")
                continue
                
            # Run regime backtest
            symbol_results = self.run_regime_backtest(
                symbol_df,
                strategy_class,
                strategy_params,
                regimes,
                index_df,
                tag=f"{tag}_{symbol}"
            )
            
            # Aggregate results
            for regime, metrics in symbol_results.items():
                if regime not in all_results:
                    all_results[regime] = {
                        'symbols': [],
                        'sharpe_ratios': [],
                        'total_returns': [],
                        'max_drawdowns': [],
                        'win_rates': []
                    }
                
                all_results[regime]['symbols'].append(symbol)
                all_results[regime]['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))
                all_results[regime]['total_returns'].append(metrics.get('total_return', 0))
                all_results[regime]['max_drawdowns'].append(metrics.get('max_drawdown', 0))
                all_results[regime]['win_rates'].append(metrics.get('win_rate', 0))
        
        # Calculate averages
        for regime, data in all_results.items():
            data['avg_sharpe_ratio'] = np.mean(data['sharpe_ratios']) if data['sharpe_ratios'] else 0
            data['avg_total_return'] = np.mean(data['total_returns']) if data['total_returns'] else 0
            data['avg_max_drawdown'] = np.mean(data['max_drawdowns']) if data['max_drawdowns'] else 0
            data['avg_win_rate'] = np.mean(data['win_rates']) if data['win_rates'] else 0
            data['symbol_count'] = len(data['symbols'])
        
        return all_results


def get_predefined_regime_periods() -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns predefined market regime periods based on historical analysis.
    
    Returns:
        Dictionary mapping regime types to list of (start_date, end_date) tuples
    """
    return {
        'bull': [
            ('2010-01-01', '2011-12-31'),
            ('2013-01-01', '2014-12-31'),
            ('2017-01-01', '2020-02-28'),
            ('2023-01-01', '2025-07-15')
        ],
        'bear': [
            ('2011-01-01', '2012-12-31'),
            ('2015-01-01', '2016-12-31'),
            ('2020-03-01', '2022-06-30')
        ],
        'sideways': [
            ('2012-01-01', '2013-12-31'),
            ('2016-01-01', '2017-12-31'),
            ('2022-07-01', '2023-12-31')
        ]
    }
