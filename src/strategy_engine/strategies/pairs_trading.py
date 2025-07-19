"""
================================================================================
Pairs Trading Strategy Implementation
================================================================================

This module implements a statistical arbitrage strategy for trading pairs of
correlated stocks. It features:

- Trades based on deviations of price ratios from their mean
- Real-time signal generation and backtesting
- Designed for integration into larger trading systems

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple
import logging
from scipy import stats

from .base_strategy import BaseStrategy, TradingSignal, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy that identifies and trades correlated stock pairs
    """

    def __init__(self, config: dict[str, Any]):
        strategy_name = config.get('strategy_name', 'pairs_trading')
        super().__init__(strategy_name)
        self.name = strategy_name
        self.lookback_period = config.get('lookback_period', 60)
        self.min_correlation = config.get('min_correlation', 0.8)
        self.zscore_threshold = config.get('zscore_threshold', 2.0)
        self.min_confidence = config.get('min_confidence', 0.7)
        
        # Common pairs in Indian markets
        self.known_pairs = [
            ('ICICIBANK.NS', 'HDFCBANK.NS'),
            ('TCS.NS', 'INFY.NS'),
            ('WIPRO.NS', 'HCLTECH.NS'),
            ('RELIANCE.NS', 'ONGC.NS'),
            ('SBIN.NS', 'KOTAKBANK.NS'),
            ('MARUTI.NS', 'TATAMOTORS.NS'),
            ('ADANIPORTS.NS', 'ADANIENT.NS'),
            ('HINDALCO.NS', 'JSWSTEEL.NS'),
            ('CIPLA.NS', 'DRREDDY.NS'),
            ('BAJFINANCE.NS', 'BAJAJFINSV.NS')
        ]

    def calculate_correlation(self, data: pd.DataFrame, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        if symbol1 not in data.columns or symbol2 not in data.columns:
            return 0.0
        
        returns1 = data[symbol1].pct_change().dropna()
        returns2 = data[symbol2].pct_change().dropna()
        
        # Align the series
        aligned_data = pd.concat([returns1, returns2], axis=1, join='inner')
        if len(aligned_data) < 20:  # Need minimum data
            return 0.0
        
        correlation = aligned_data.corr().iloc[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def calculate_zscore(self, data: pd.DataFrame, symbol1: str, symbol2: str) -> float:
        """Calculate z-score of price ratio"""
        if symbol1 not in data.columns or symbol2 not in data.columns:
            return 0.0
        
        prices1 = data[symbol1].tail(self.lookback_period)
        prices2 = data[symbol2].tail(self.lookback_period)
        
        # Calculate ratio
        ratio = prices1 / prices2
        ratio = ratio.dropna()
        
        if len(ratio) < 20:
            return 0.0
        
        current_ratio = ratio.iloc[-1]
        mean_ratio = ratio.mean()
        std_ratio = ratio.std()
        
        if std_ratio == 0:
            return 0.0
        
        zscore = (current_ratio - mean_ratio) / std_ratio
        return zscore

    def generate_signals(self, data: dict[str, Any]) -> list[TradingSignal]:
        """Generate pairs trading signals"""
        signals: list[TradingSignal] = []
        
        try:
            # Convert data to DataFrame if needed
            if 'prices' in data:
                df = pd.DataFrame(data['prices'])
            else:
                logger.warning("Expected 'prices' key in data dict")
                return signals
            
            # Check each known pair
            for symbol1, symbol2 in self.known_pairs:
                correlation = self.calculate_correlation(df, symbol1, symbol2)
                
                if abs(correlation) >= self.min_correlation:
                    zscore = self.calculate_zscore(df, symbol1, symbol2)
                    
                    if abs(zscore) >= self.zscore_threshold:
                        # Generate signals based on z-score
                        if zscore > self.zscore_threshold:
                            # Ratio is too high, short symbol1, long symbol2
                            self._create_pair_signals(
                                signals, df, symbol1, symbol2, zscore, 
                                'SELL', 'BUY', correlation
                            )
                        elif zscore < -self.zscore_threshold:
                            # Ratio is too low, long symbol1, short symbol2
                            self._create_pair_signals(
                                signals, df, symbol1, symbol2, zscore, 
                                'BUY', 'SELL', correlation
                            )
            
            logger.info(f"Generated {len(signals)} pairs trading signals")
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signals: {str(e)}")
        
        return signals

    def _create_pair_signals(self, signals: list[TradingSignal], df: pd.DataFrame, 
                           symbol1: str, symbol2: str, zscore: float,
                           signal1: str, signal2: str, correlation: float):
        """Create signals for a pair"""
        try:
            # Signal for symbol1
            if symbol1 in df.columns:
                price1 = float(df[symbol1].iloc[-1])
                confidence1 = min(0.9, 0.5 + (abs(zscore) * 0.1) + (abs(correlation) * 0.2))
                
                if confidence1 >= self.min_confidence:
                    signal_type1 = SignalType.BUY if signal1 == 'BUY' else SignalType.SELL
                    signal1_obj = TradingSignal(
                        symbol=symbol1,
                        signal=signal_type1,
                        confidence=confidence1,
                        price=price1,
                        timestamp=datetime.now(),
                        strategy_name=self.name,
                        risk_metrics={
                            'zscore': abs(zscore),
                            'correlation': abs(correlation),
                            'pair_score': abs(zscore) * abs(correlation)
                        },
                        position_size=0.5,  # Half position for each pair
                        reasoning=f"Pairs trade: {symbol1} vs {symbol2}, z-score: {zscore:.2f}, correlation: {correlation:.2f}"
                    )
                    signals.append(signal1_obj)
            
            # Signal for symbol2
            if symbol2 in df.columns:
                price2 = float(df[symbol2].iloc[-1])
                confidence2 = min(0.9, 0.5 + (abs(zscore) * 0.1) + (abs(correlation) * 0.2))
                
                if confidence2 >= self.min_confidence:
                    signal_type2 = SignalType.BUY if signal2 == 'BUY' else SignalType.SELL
                    signal2_obj = TradingSignal(
                        symbol=symbol2,
                        signal=signal_type2,
                        confidence=confidence2,
                        price=price2,
                        timestamp=datetime.now(),
                        strategy_name=self.name,
                        risk_metrics={
                            'zscore': abs(zscore),
                            'correlation': abs(correlation),
                            'pair_score': abs(zscore) * abs(correlation)
                        },
                        position_size=0.5,  # Half position for each pair
                        reasoning=f"Pairs trade: {symbol2} vs {symbol1}, z-score: {zscore:.2f}, correlation: {correlation:.2f}"
                    )
                    signals.append(signal2_obj)
                    
        except Exception as e:
            logger.error(f"Error creating pair signals: {str(e)}")

    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the pairs trading strategy"""
        results: Dict[str, Any] = {
            'strategy_name': self.name,
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'avg_confidence': 0.0,
            'pairs_analyzed': len(self.known_pairs),
            'backtest_period': f"{data.index[0]} to {data.index[-1]}"
        }
        
        try:
            # Generate signals for backtesting
            data_dict = {'prices': data}
            signals = self.generate_signals(data_dict)
            
            results['total_signals'] = len(signals)
            results['buy_signals'] = len([s for s in signals if s.signal == SignalType.BUY])
            results['sell_signals'] = len([s for s in signals if s.signal == SignalType.SELL])
            
            if signals:
                results['avg_confidence'] = np.mean([s.confidence for s in signals])
            
            # Calculate pair correlations
            correlations = {}
            for symbol1, symbol2 in self.known_pairs:
                corr = self.calculate_correlation(data, symbol1, symbol2)
                correlations[f"{symbol1}_{symbol2}"] = corr
            
            results['pair_correlations'] = correlations
            
            logger.info(f"Pairs trading backtest completed: {results}")
            
        except Exception as e:
            logger.error(f"Error in pairs trading backtest: {str(e)}")
            results['error'] = str(e)
        
        return results

    def calculate_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate indicators for pairs trading strategy"""
        indicators = {}
        try:
            if 'prices' in data:
                df = pd.DataFrame(data['prices'])
                # Calculate correlations for all pairs
                correlations = {}
                for symbol1, symbol2 in self.known_pairs:
                    corr = self.calculate_correlation(df, symbol1, symbol2)
                    correlations[f"{symbol1}_{symbol2}"] = corr
                indicators['pair_correlations'] = correlations
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        return indicators
