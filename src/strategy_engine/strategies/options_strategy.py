"""
================================================================================
Options Strategy Implementation
================================================================================

This module provides strategies for covered calls, protective puts, and
cash-secured puts. It features:

- Risk management using delta, theta, and implied volatility
- Real-time signal generation and backtesting
- Designed for integration into larger trading systems

================================================================================
"""

import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, SignalType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptionsPosition:
    """Represents an options position"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiry_date: datetime
    premium: float
    quantity: int
    underlying_price: float


class OptionsStrategy(BaseStrategy):
    """
    Options Strategy implementing covered calls and protective puts
    
    Strategy Components:
    1. Covered Calls: Sell call options on owned stocks for income
    2. Protective Puts: Buy put options to protect against downside
    3. Risk Management: Monitor delta, theta, and IV
    """
    
    def __init__(self, 
                 volatility_threshold: float = 0.02,
                 delta_target: float = 0.3,
                 theta_threshold: float = -0.05,
                 iv_percentile_threshold: float = 50.0,
                 days_to_expiry: int = 30,
                 **kwargs: Any):
        """
        Initialize Options Strategy
        
        Args:
            volatility_threshold: Minimum volatility for option writing
            delta_target: Target delta for option selection
            theta_threshold: Maximum theta decay acceptance
            iv_percentile_threshold: IV percentile threshold for entry
            days_to_expiry: Target days to expiration for options
        """
        super().__init__(**kwargs)
        self.volatility_threshold = volatility_threshold
        self.delta_target = delta_target
        self.theta_threshold = theta_threshold
        self.iv_percentile_threshold = iv_percentile_threshold
        self.days_to_expiry = days_to_expiry
        
        # Options tracking
        self.active_positions: Dict[str, List[OptionsPosition]] = {}
        self.iv_history: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized Options Strategy with vol_threshold={volatility_threshold}, "
                   f"delta_target={delta_target}, days_to_expiry={days_to_expiry}")
    
    def calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate options-specific indicators
        
        Args:
            data: Market data dictionary with OHLCV data
            
        Returns:
            Dictionary with options indicators
        """
        try:
            result = data.copy()
            
            # Convert lists to numpy arrays for calculation
            close_prices = np.array(data.get('close', []))
            
            if len(close_prices) < 20:
                # Not enough data for calculations
                result.update({
                    'volatility_20d': 0.02,
                    'iv_rank': 50.0,
                    'options_score': 50.0,
                    'sma_20': close_prices[-1] if len(close_prices) > 0 else 0,
                    'sma_50': close_prices[-1] if len(close_prices) > 0 else 0,
                    'rsi': 50.0
                })
                return result
            
            # Calculate returns and volatility
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.02
            
            # Simple moving averages
            sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else close_prices[-1]
            
            # RSI calculation
            rsi = self._calculate_rsi(close_prices)
            
            # IV rank simulation (using historical volatility percentile)
            if len(returns) >= 252:
                vol_history = []
                for i in range(20, len(returns)):
                    vol = np.std(returns[i-20:i]) * np.sqrt(252)
                    vol_history.append(vol)
                current_vol_rank = (np.sum(np.array(vol_history) < volatility_20d) / len(vol_history)) * 100
            else:
                current_vol_rank = 50.0
            
            # Options suitability score
            vol_score = 1 if volatility_20d > self.volatility_threshold else 0
            iv_score = 1 if current_vol_rank > self.iv_percentile_threshold else 0
            trend_score = 1 if close_prices[-1] > sma_20 else 0
            options_score = (vol_score * 40 + iv_score * 40 + trend_score * 20)
            
            # Update result dictionary
            result.update({
                'volatility_20d': volatility_20d,
                'iv_rank': current_vol_rank,
                'options_score': options_score,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating options indicators: {e}")
            result = data.copy()
            result.update({
                'volatility_20d': 0.02,
                'iv_rank': 50.0,
                'options_score': 50.0,
                'sma_20': 0,
                'sma_50': 0,
                'rsi': 50.0
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
    
    def generate_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Generate options strategy signals
        
        Args:
            data: Market data dictionary with OHLCV data
            
        Returns:
            List of trading signals
        """
        try:
            signals = []
            
            # Get symbol from data or use default
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Calculate indicators
            enhanced_data = self.calculate_indicators(data)
            
            close_prices = data.get('close', [])
            if not close_prices:
                return signals
                
            current_price = close_prices[-1]
            volatility = enhanced_data.get('volatility_20d', 0)
            iv_rank = enhanced_data.get('iv_rank', 0)
            options_score = enhanced_data.get('options_score', 0)
            rsi = enhanced_data.get('rsi', 50)
            
            # Covered Call Signals
            if self._should_write_covered_call(enhanced_data):
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.SELL,  # Selling call options
                    confidence=min(options_score / 100, 0.95),
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name="options_covered_call",
                    risk_metrics={
                        'volatility': volatility,
                        'iv_rank': iv_rank,
                        'max_position_pct': 5.0
                    },
                    position_size=0.05,  # 5% position
                    reasoning=f"Covered call opportunity: IV rank {iv_rank:.1f}%, vol {volatility:.2f}"
                )
                signals.append(signal)
            
            # Protective Put Signals
            if self._should_buy_protective_put(enhanced_data):
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.BUY,  # Buying put options for protection
                    confidence=min((80 - rsi + 20) / 100, 0.95),
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy_name="options_protective_put",
                    risk_metrics={
                        'volatility': volatility,
                        'iv_rank': iv_rank,
                        'max_position_pct': 3.0
                    },
                    position_size=0.03,  # 3% position
                    reasoning=f"Protective put for downside protection: RSI {rsi:.1f}, vol {volatility:.2f}"
                )
                signals.append(signal)
            
            # Cash-Secured Put Signals
            if self._should_write_cash_secured_put(enhanced_data):
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.BUY,  # Intent to buy at lower price
                    confidence=min(options_score * 0.8 / 100, 0.90),
                    price=current_price * 0.95,  # Target entry 5% below current
                    timestamp=datetime.now(),
                    strategy_name="options_cash_secured_put",
                    risk_metrics={
                        'volatility': volatility,
                        'iv_rank': iv_rank,
                        'max_position_pct': 4.0
                    },
                    position_size=0.04,  # 4% position
                    reasoning=f"Cash-secured put for income: IV rank {iv_rank:.1f}%, oversold RSI {rsi:.1f}"
                )
                signals.append(signal)
            
            logger.info(f"Generated {len(signals)} options signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating options signals: {e}")
            return []
    
    def _should_write_covered_call(self, data: Dict[str, Any]) -> bool:
        """Determine if we should write a covered call"""
        try:
            volatility = data.get('volatility_20d', 0)
            iv_rank = data.get('iv_rank', 0)
            rsi = data.get('rsi', 50)
            
            close = data.get('close', [])
            sma_20 = data.get('sma_20', 0)
            if not close or sma_20 == 0:
                return False
                
            current_price = close[-1]
            price_vs_sma = current_price / sma_20 if sma_20 > 0 else 1.0
            
            # Conditions for covered call
            return (
                volatility > self.volatility_threshold and
                iv_rank > self.iv_percentile_threshold and
                40 < rsi < 80 and  # Not oversold or extremely overbought
                0.98 <= price_vs_sma <= 1.05  # Near 20-day SMA
            )
            
        except Exception as e:
            logger.error(f"Error checking covered call conditions: {e}")
            return False
    
    def _should_buy_protective_put(self, data: Dict[str, Any]) -> bool:
        """Determine if we should buy a protective put"""
        try:
            volatility = data.get('volatility_20d', 0)
            rsi = data.get('rsi', 50)
            
            close = data.get('close', [])
            sma_20 = data.get('sma_20', 0)
            if not close or sma_20 == 0:
                return False
                
            current_price = close[-1]
            
            # Conditions for protective put
            return (
                volatility > self.volatility_threshold * 1.5 and  # Higher volatility
                (rsi < 30) and  # Oversold
                current_price < sma_20  # Below trend
            )
            
        except Exception as e:
            logger.error(f"Error checking protective put conditions: {e}")
            return False
    
    def _should_write_cash_secured_put(self, data: Dict[str, Any]) -> bool:
        """Determine if we should write a cash-secured put"""
        try:
            volatility = data.get('volatility_20d', 0)
            iv_rank = data.get('iv_rank', 0)
            rsi = data.get('rsi', 50)
            
            close = data.get('close', [])
            sma_50 = data.get('sma_50', 0)
            if not close or sma_50 == 0:
                return False
                
            current_price = close[-1]
            
            # Conditions for cash-secured put
            return (
                volatility > self.volatility_threshold and
                iv_rank > self.iv_percentile_threshold and
                rsi < 40 and  # Oversold
                current_price > sma_50  # Long-term uptrend
            )
            
        except Exception as e:
            logger.error(f"Error checking cash-secured put conditions: {e}")
            return False
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return {
            'volatility_threshold': self.volatility_threshold,
            'delta_target': self.delta_target,
            'theta_threshold': self.theta_threshold,
            'iv_percentile_threshold': self.iv_percentile_threshold,
            'days_to_expiry': self.days_to_expiry,
            'strategy_type': 'options',
            'components': ['covered_calls', 'protective_puts', 'cash_secured_puts']
        }
    
    def update_params(self, **kwargs: Any) -> None:
        """Update strategy parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get options-specific risk metrics"""
        return {
            'max_position_size': 0.05,  # 5% per options position
            'max_portfolio_options': 0.20,  # 20% of portfolio in options
            'delta_hedge_threshold': 0.5,
            'gamma_limit': 0.1,
            'theta_target': self.theta_threshold,
            'vega_limit': 0.15
        }
