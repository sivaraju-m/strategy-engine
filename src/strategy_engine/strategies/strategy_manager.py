"""
Strategy Manager for coordinating multiple trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List
import logging

from .strategy_registry import create_strategy
from .base_strategy import TradingSignal, SignalType
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.strategy_weights = config.get('strategy_weights', {})
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_signals_per_strategy = config.get('max_signals_per_strategy', 10)
        
        # Initialize strategies
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize all configured strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        for strategy_name, strategy_config in strategy_configs.items():
            try:
                strategy = create_strategy(strategy_name, strategy_config)
                self.strategies[strategy_name] = strategy
                logger.info(f"Initialized strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_name}: {str(e)}")

    def generate_combined_signals(self, data: dict[str, Any]) -> List[TradingSignal]:
        """
        Generate signals from all strategies and combine them
        """
        all_signals = []
        strategy_signals = {}
        
        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                strategy_signals[strategy_name] = signals
                
                # Apply strategy weight
                weight = self.strategy_weights.get(strategy_name, 1.0)
                for signal in signals:
                    # Adjust confidence based on strategy weight
                    signal.confidence = min(0.95, signal.confidence * weight)
                    
                    # Add strategy information to risk metrics
                    signal.risk_metrics['strategy_weight'] = weight
                    signal.risk_metrics['source_strategy'] = strategy_name
                
                all_signals.extend(signals)
                logger.info(f"Strategy {strategy_name} generated {len(signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating signals from {strategy_name}: {str(e)}")
        
        # Filter and rank signals
        filtered_signals = self._filter_and_rank_signals(all_signals)
        
        logger.info(f"Generated {len(filtered_signals)} combined signals from {len(self.strategies)} strategies")
        
        return filtered_signals

    def _filter_and_rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter and rank signals based on confidence and other criteria
        """
        # Filter by minimum confidence
        filtered_signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in filtered_signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # For each symbol, select the best signal
        final_signals = []
        for symbol, sym_signals in symbol_signals.items():
            if len(sym_signals) == 1:
                final_signals.append(sym_signals[0])
            else:
                # Multiple signals for same symbol - combine or select best
                best_signal = self._combine_signals_for_symbol(symbol, sym_signals)
                if best_signal:
                    final_signals.append(best_signal)
        
        # Sort by confidence descending
        final_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_signals

    def _combine_signals_for_symbol(self, symbol: str, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """
        Combine multiple signals for the same symbol
        """
        if not signals:
            return None
        
        # Separate BUY and SELL signals
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        # If we have both BUY and SELL, use the one with higher confidence
        if buy_signals and sell_signals:
            best_buy = max(buy_signals, key=lambda x: x.confidence)
            best_sell = max(sell_signals, key=lambda x: x.confidence)
            return best_buy if best_buy.confidence > best_sell.confidence else best_sell
        
        # If we have only one type, combine them
        relevant_signals = buy_signals if buy_signals else sell_signals
        
        if len(relevant_signals) == 1:
            return relevant_signals[0]
        
        # Combine multiple signals of the same type
        combined_confidence = float(np.mean([s.confidence for s in relevant_signals]))
        combined_price = float(np.mean([s.price for s in relevant_signals]))
        combined_position_size = float(np.mean([s.position_size for s in relevant_signals]))
        
        # Combine risk metrics
        combined_risk_metrics = {}
        for signal in relevant_signals:
            for key, value in signal.risk_metrics.items():
                if key not in combined_risk_metrics:
                    combined_risk_metrics[key] = []
                combined_risk_metrics[key].append(value)
        
        # Average numerical risk metrics
        for key, values in combined_risk_metrics.items():
            if isinstance(values[0], (int, float)):
                combined_risk_metrics[key] = np.mean(values)
            else:
                combined_risk_metrics[key] = values  # Keep as list for non-numerical
        
        # Create combined signal
        combined_signal = TradingSignal(
            symbol=symbol,
            signal=relevant_signals[0].signal,  # Same signal type
            confidence=combined_confidence,
            price=combined_price,
            timestamp=datetime.now(),
            strategy_name="combined",
            risk_metrics=combined_risk_metrics,
            position_size=combined_position_size,
            reasoning=f"Combined signal from {len(relevant_signals)} strategies"
        )
        
        return combined_signal

    def get_strategy_performance(self, data: dict[str, Any]) -> Dict[str, Any]:
        """
        Get performance metrics for all strategies
        """
        performance = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'backtest') and 'prices' in data:
                    df = pd.DataFrame(data['prices'])
                    results = strategy.backtest(df)
                    performance[strategy_name] = results
                else:
                    # Get basic metrics
                    indicators = strategy.calculate_indicators(data)
                    performance[strategy_name] = {
                        'strategy_name': strategy_name,
                        'indicators': indicators,
                        'status': 'active'
                    }
            except Exception as e:
                logger.error(f"Error getting performance for {strategy_name}: {str(e)}")
                performance[strategy_name] = {'error': str(e)}
        
        return performance

    def list_strategies(self) -> Dict[str, str]:
        """
        List all initialized strategies
        """
        return {name: strategy.strategy_name for name, strategy in self.strategies.items()}

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get current strategy weights
        """
        return self.strategy_weights.copy()

    def update_strategy_weights(self, new_weights: Dict[str, float]):
        """
        Update strategy weights
        """
        self.strategy_weights.update(new_weights)
        logger.info(f"Updated strategy weights: {self.strategy_weights}")

    def enable_strategy(self, strategy_name: str):
        """
        Enable a specific strategy
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = True
            logger.info(f"Enabled strategy: {strategy_name}")

    def disable_strategy(self, strategy_name: str):
        """
        Disable a specific strategy
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].is_active = False
            logger.info(f"Disabled strategy: {strategy_name}")
