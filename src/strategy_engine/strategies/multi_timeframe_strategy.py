"""
Multi-timeframe Strategy Implementation
Combines daily, hourly, and minute-level signals for comprehensive analysis
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .base_strategy import BaseStrategy, TradingSignal, SignalType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """Data structure for different timeframes"""
    timeframe: str  # '1m', '5m', '15m', '1h', '1d'
    data: Dict[str, Any]
    signals: List[TradingSignal]
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-100


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe Strategy that combines signals from different time horizons
    
    Strategy Components:
    1. Daily trend analysis (primary trend)
    2. Hourly momentum signals (intermediate trend)
    3. Minute-level entry/exit signals (short-term timing)
    4. Confluence-based signal generation
    """
    
    def __init__(self,
                 timeframes: List[str] = ['1d', '1h', '15m'],
                 trend_weights: Optional[Dict[str, float]] = None,
                 min_confluence: int = 2,
                 rsi_period: int = 14,
                 ma_fast: int = 10,
                 ma_slow: int = 20,
                 **kwargs: Any):
        """
        Initialize Multi-timeframe Strategy
        
        Args:
            timeframes: List of timeframes to analyze ['1d', '1h', '15m', '5m', '1m']
            trend_weights: Weight for each timeframe in final signal
            min_confluence: Minimum number of timeframes agreeing for signal
            rsi_period: Period for RSI calculation
            ma_fast: Fast moving average period
            ma_slow: Slow moving average period
        """
        super().__init__(**kwargs)
        self.timeframes = timeframes
        self.trend_weights = trend_weights or {
            '1d': 0.4,    # Daily trend most important
            '1h': 0.3,    # Hourly for momentum
            '15m': 0.2,   # 15-min for timing
            '5m': 0.1,    # 5-min for fine-tuning
            '1m': 0.05    # 1-min for precise entry
        }
        self.min_confluence = min_confluence
        self.rsi_period = rsi_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        
        # Timeframe analysis storage
        self.timeframe_analysis: Dict[str, TimeframeData] = {}
        self.last_update = datetime.now()
        
        logger.info(f"Initialized Multi-timeframe Strategy with timeframes: {timeframes}, "
                   f"min_confluence: {min_confluence}")
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate multi-timeframe indicators
        
        Args:
            data: Market data dictionary with OHLCV data
            
        Returns:
            Dictionary with multi-timeframe indicators
        """
        try:
            result = data.copy()
            
            # Get price data
            close_prices = np.array(data.get('close', []))
            volume = np.array(data.get('volume', []))
            
            if len(close_prices) < max(self.ma_slow, self.rsi_period):
                # Not enough data
                result.update({
                    'mtf_trend_daily': 'neutral',
                    'mtf_trend_hourly': 'neutral',
                    'mtf_trend_short': 'neutral',
                    'mtf_confluence_score': 50.0,
                    'mtf_signal_strength': 0.0
                })
                return result
            
            # Calculate basic indicators for primary timeframe
            ma_fast = float(np.mean(close_prices[-self.ma_fast:]) if len(close_prices) >= self.ma_fast else close_prices[-1])
            ma_slow = float(np.mean(close_prices[-self.ma_slow:]) if len(close_prices) >= self.ma_slow else close_prices[-1])
            
            # RSI calculation
            rsi = self._calculate_rsi(close_prices, self.rsi_period)
            
            # Volatility
            returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else np.array([0])
            volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.02
            
            # Volume analysis
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else (volume[-1] if len(volume) > 0 else 0)
            volume_ratio = (volume[-1] / avg_volume) if avg_volume > 0 and len(volume) > 0 else 1.0
            
            # Trend determination
            price_trend = self._determine_trend(close_prices, ma_fast, ma_slow)
            momentum_trend = self._determine_momentum_trend(close_prices, rsi)
            volume_trend = self._determine_volume_trend(volume_ratio)
            
            # Multi-timeframe simulation (using different lookback periods)
            daily_trend = self._simulate_timeframe_trend(close_prices, 20)  # 20-day trend
            hourly_trend = self._simulate_timeframe_trend(close_prices, 5)   # 5-day trend
            short_trend = self._simulate_timeframe_trend(close_prices, 2)    # 2-day trend
            
            # Confluence score
            confluence_score = self._calculate_confluence_score(daily_trend, hourly_trend, short_trend, price_trend)
            
            # Signal strength based on alignment
            signal_strength = self._calculate_signal_strength(daily_trend, hourly_trend, short_trend, rsi, volume_ratio)
            
            # Update result
            result.update({
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'price_trend': price_trend,
                'momentum_trend': momentum_trend,
                'volume_trend': volume_trend,
                'mtf_trend_daily': daily_trend,
                'mtf_trend_hourly': hourly_trend,
                'mtf_trend_short': short_trend,
                'mtf_confluence_score': confluence_score,
                'mtf_signal_strength': signal_strength
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe indicators: {e}")
            result = data.copy()
            result.update({
                'mtf_trend_daily': 'neutral',
                'mtf_trend_hourly': 'neutral',
                'mtf_trend_short': 'neutral',
                'mtf_confluence_score': 50.0,
                'mtf_signal_strength': 0.0
            })
            return result
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _determine_trend(self, prices: np.ndarray, ma_fast: float, ma_slow: float) -> str:
        """Determine price trend based on moving averages"""
        try:
            current_price = prices[-1]
            
            # Multiple criteria for trend
            ma_signal = 'bullish' if ma_fast > ma_slow else 'bearish'
            price_vs_ma = 'bullish' if current_price > ma_slow else 'bearish'
            
            # Price direction over last few periods
            if len(prices) >= 3:
                recent_trend = 'bullish' if prices[-1] > prices[-3] else 'bearish'
            else:
                recent_trend = 'neutral'
            
            # Consensus
            bullish_votes = sum([
                ma_signal == 'bullish',
                price_vs_ma == 'bullish',
                recent_trend == 'bullish'
            ])
            
            if bullish_votes >= 2:
                return 'bullish'
            elif bullish_votes == 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return 'neutral'
    
    def _determine_momentum_trend(self, prices: np.ndarray, rsi: float) -> str:
        """Determine momentum trend"""
        try:
            if rsi > 70:
                return 'overbought'
            elif rsi < 30:
                return 'oversold'
            elif rsi > 50:
                return 'bullish'
            else:
                return 'bearish'
        except Exception:
            return 'neutral'
    
    def _determine_volume_trend(self, volume_ratio: float) -> str:
        """Determine volume trend"""
        try:
            if volume_ratio > 1.5:
                return 'high'
            elif volume_ratio < 0.5:
                return 'low'
            else:
                return 'normal'
        except Exception:
            return 'normal'
    
    def _simulate_timeframe_trend(self, prices: np.ndarray, lookback: int) -> str:
        """Simulate trend for different timeframes using different lookback periods"""
        try:
            if len(prices) < lookback:
                return 'neutral'
            
            start_price = prices[-lookback]
            end_price = prices[-1]
            
            # Calculate trend strength
            price_change = (end_price - start_price) / start_price
            
            if price_change > 0.02:  # > 2% gain
                return 'bullish'
            elif price_change < -0.02:  # > 2% loss
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error simulating timeframe trend: {e}")
            return 'neutral'
    
    def _calculate_confluence_score(self, daily: str, hourly: str, short: str, price: str) -> float:
        """Calculate confluence score based on trend alignment"""
        try:
            trends = [daily, hourly, short, price]
            
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')
            
            # Score based on alignment
            if bullish_count >= 3:
                return 80.0 + (bullish_count * 5)
            elif bearish_count >= 3:
                return 20.0 - (bearish_count * 5)
            elif bullish_count > bearish_count:
                return 60.0 + ((bullish_count - bearish_count) * 5)
            elif bearish_count > bullish_count:
                return 40.0 - ((bearish_count - bullish_count) * 5)
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error calculating confluence score: {e}")
            return 50.0
    
    def _calculate_signal_strength(self, daily: str, hourly: str, short: str, rsi: float, volume_ratio: float) -> float:
        """Calculate overall signal strength"""
        try:
            base_strength = 0.0
            
            # Trend alignment bonus
            if daily == hourly == short:
                base_strength += 40.0
            elif daily == hourly or daily == short or hourly == short:
                base_strength += 20.0
            
            # RSI contribution
            if 30 <= rsi <= 70:  # Healthy range
                base_strength += 20.0
            elif rsi < 30 or rsi > 70:  # Extreme levels
                base_strength += 10.0
            
            # Volume contribution
            if volume_ratio > 1.2:  # Above average volume
                base_strength += 20.0
            elif volume_ratio > 0.8:  # Normal volume
                base_strength += 10.0
            
            # Direction bonus
            if daily == 'bullish' and hourly == 'bullish':
                base_strength += 20.0
            elif daily == 'bearish' and hourly == 'bearish':
                base_strength += 20.0
            
            return min(base_strength, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def generate_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate multi-timeframe trading signals
        
        Args:
            data: Market data dictionary with OHLCV data
            
        Returns:
            List of trading signals
        """
        try:
            signals = []
            
            # Get symbol from data
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Calculate indicators
            enhanced_data = self.calculate_indicators(data)
            
            close_prices = data.get('close', [])
            if not close_prices:
                return signals
                
            current_price = close_prices[-1]
            
            # Extract multi-timeframe data
            daily_trend = enhanced_data.get('mtf_trend_daily', 'neutral')
            hourly_trend = enhanced_data.get('mtf_trend_hourly', 'neutral')
            short_trend = enhanced_data.get('mtf_trend_short', 'neutral')
            confluence_score = enhanced_data.get('mtf_confluence_score', 50.0)
            signal_strength = enhanced_data.get('mtf_signal_strength', 0.0)
            rsi = enhanced_data.get('rsi', 50.0)
            
            # Generate signals based on confluence
            if self._should_generate_buy_signal(daily_trend, hourly_trend, short_trend, confluence_score, rsi):
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.BUY,
                    confidence=min(confluence_score / 100, 0.95),
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name="multi_timeframe_buy",
                    risk_metrics={
                        'confluence_score': confluence_score,
                        'signal_strength': signal_strength,
                        'max_position_pct': 4.0
                    },
                    position_size=0.04,  # 4% position
                    reasoning=f"Multi-timeframe BUY: Daily={daily_trend}, Hourly={hourly_trend}, "
                             f"Short={short_trend}, Confluence={confluence_score:.1f}%, RSI={rsi:.1f}"
                )
                signals.append(signal)
            
            elif self._should_generate_sell_signal(daily_trend, hourly_trend, short_trend, confluence_score, rsi):
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.SELL,
                    confidence=min((100 - confluence_score) / 100, 0.95),
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name="multi_timeframe_sell",
                    risk_metrics={
                        'confluence_score': confluence_score,
                        'signal_strength': signal_strength,
                        'max_position_pct': 4.0
                    },
                    position_size=0.04,  # 4% position
                    reasoning=f"Multi-timeframe SELL: Daily={daily_trend}, Hourly={hourly_trend}, "
                             f"Short={short_trend}, Confluence={confluence_score:.1f}%, RSI={rsi:.1f}"
                )
                signals.append(signal)
            
            # Generate hold signal for confirmation
            elif signal_strength > 40 and 30 < rsi < 70:
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.HOLD,
                    confidence=signal_strength / 100,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name="multi_timeframe_hold",
                    risk_metrics={
                        'confluence_score': confluence_score,
                        'signal_strength': signal_strength,
                        'max_position_pct': 2.0
                    },
                    position_size=0.02,  # 2% position
                    reasoning=f"Multi-timeframe HOLD: Moderate alignment, RSI in healthy range"
                )
                signals.append(signal)
            
            logger.info(f"Generated {len(signals)} multi-timeframe signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating multi-timeframe signals: {e}")
            return []
    
    def _should_generate_buy_signal(self, daily: str, hourly: str, short: str, confluence: float, rsi: float) -> bool:
        """Determine if we should generate a buy signal"""
        try:
            # Count bullish timeframes
            bullish_count = [daily, hourly, short].count('bullish')
            
            # Conditions for buy signal
            return (
                bullish_count >= self.min_confluence and  # Minimum confluence
                confluence > 65.0 and  # High confluence score
                rsi < 70 and  # Not overbought
                daily in ['bullish', 'neutral']  # Daily trend supportive
            )
            
        except Exception as e:
            logger.error(f"Error checking buy signal conditions: {e}")
            return False
    
    def _should_generate_sell_signal(self, daily: str, hourly: str, short: str, confluence: float, rsi: float) -> bool:
        """Determine if we should generate a sell signal"""
        try:
            # Count bearish timeframes
            bearish_count = [daily, hourly, short].count('bearish')
            
            # Conditions for sell signal
            return (
                bearish_count >= self.min_confluence and  # Minimum confluence
                confluence < 35.0 and  # Low confluence score (bearish)
                rsi > 30 and  # Not oversold
                daily in ['bearish', 'neutral']  # Daily trend supportive
            )
            
        except Exception as e:
            logger.error(f"Error checking sell signal conditions: {e}")
            return False
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return {
            'timeframes': self.timeframes,
            'trend_weights': self.trend_weights,
            'min_confluence': self.min_confluence,
            'rsi_period': self.rsi_period,
            'ma_fast': self.ma_fast,
            'ma_slow': self.ma_slow,
            'strategy_type': 'multi_timeframe'
        }
    
    def update_params(self, **kwargs: Any) -> None:
        """Update strategy parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get multi-timeframe specific risk metrics"""
        return {
            'max_position_size': 0.04,  # 4% per position
            'max_portfolio_exposure': 0.25,  # 25% total exposure
            'min_confluence_threshold': self.min_confluence,
            'required_timeframe_agreement': 0.6,  # 60% agreement required
            'volatility_adjustment': True,
            'volume_confirmation': True
        }
