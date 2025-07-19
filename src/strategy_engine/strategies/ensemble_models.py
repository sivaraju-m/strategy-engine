#!/usr/bin/env python3
"""
================================================================================
Ensemble Models Implementation
================================================================================

This module combines multiple machine learning models for superior signal
generation. It features:

- Base class for ensemble models
- Models include SMA, RSI, Momentum, and Volume Weighted
- Weighted ensemble strategy for combining signals
- Designed for integration into larger trading systems

================================================================================
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnsembleModel:
    """Base class for ensemble models"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.is_trained = False
        self.performance_metrics: Dict[str, Any] = {}
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
    
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        raise NotImplementedError


class SimpleMovingAverageModel(BaseEnsembleModel):
    """Simple moving average crossover model"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, **kwargs: Any):
        super().__init__("SMA_Crossover", **kwargs)
        self.short_window = short_window
        self.long_window = long_window
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """SMA doesn't require training"""
        self.is_trained = True
        logger.info(f"SMA model initialized: {self.short_window}/{self.long_window}")
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Generate buy/sell signals based on SMA crossover"""
        if 'close' not in x_data.columns:
            raise ValueError("Close price required for SMA model")
        
        close_prices = x_data['close']
        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()
        
        # 1 for buy, -1 for sell, 0 for hold
        signals = np.where(short_sma > long_sma, 1, -1)
        return signals
    
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Return confidence scores"""
        signals = self.predict(x_data)
        # Simple confidence based on SMA divergence
        close_prices = x_data['close']
        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()
        
        divergence = abs((short_sma - long_sma) / long_sma)
        confidence = np.clip(divergence * 10, 0.5, 1.0)  # Scale to 0.5-1.0
        
        # Return probabilities for [sell, hold, buy]
        proba = np.zeros((len(signals), 3))
        for i, (signal, conf) in enumerate(zip(signals, confidence)):
            if signal == 1:  # Buy
                proba[i] = [0, 1-conf, conf]
            else:  # Sell
                proba[i] = [conf, 1-conf, 0]
        
        return proba


class RSIModel(BaseEnsembleModel):
    """RSI-based trading model"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30, **kwargs: Any):
        super().__init__("RSI", **kwargs)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def _calculate_rsi(self, prices: pd.Series[Any]) -> pd.Series[Any]:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """RSI doesn't require training"""
        self.is_trained = True
        logger.info(f"RSI model initialized: period={self.period}, OB={self.overbought}, OS={self.oversold}")
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on RSI"""
        if 'close' not in x_data.columns:
            raise ValueError("Close price required for RSI model")
        
        rsi = self._calculate_rsi(x_data['close'])
        signals = np.where(rsi < self.oversold, 1,  # Buy when oversold
                          np.where(rsi > self.overbought, -1, 0))  # Sell when overbought
        return signals
    
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Return confidence based on RSI extremes"""
        rsi = self._calculate_rsi(x_data['close'])
        signals = self.predict(x_data)
        
        # Confidence increases as RSI approaches extremes
        buy_conf = np.clip((self.oversold - rsi) / self.oversold, 0, 1)
        sell_conf = np.clip((rsi - self.overbought) / (100 - self.overbought), 0, 1)
        
        proba = np.zeros((len(signals), 3))
        for i, signal in enumerate(signals):
            if signal == 1:  # Buy
                proba[i] = [0, 1-buy_conf[i], buy_conf[i]]
            elif signal == -1:  # Sell
                proba[i] = [sell_conf[i], 1-sell_conf[i], 0]
            else:  # Hold
                proba[i] = [0.2, 0.6, 0.2]
        
        return proba


class MomentumModel(BaseEnsembleModel):
    """Momentum-based model"""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs: Any):
        super().__init__("Momentum", **kwargs)
        self.lookback = lookback
        self.threshold = threshold
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """Momentum doesn't require training"""
        self.is_trained = True
        logger.info(f"Momentum model initialized: lookback={self.lookback}, threshold={self.threshold}")
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on momentum"""
        if 'close' not in x_data.columns:
            raise ValueError("Close price required for Momentum model")
        
        close_prices = x_data['close']
        momentum = close_prices.pct_change(periods=self.lookback)
        
        signals = np.where(momentum > self.threshold, 1,
                          np.where(momentum < -self.threshold, -1, 0))
        return signals
    
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Return confidence based on momentum strength"""
        close_prices = x_data['close']
        momentum = close_prices.pct_change(periods=self.lookback)
        signals = self.predict(x_data)
        
        # Confidence based on momentum magnitude
        momentum_strength = abs(momentum) / self.threshold
        confidence = np.clip(momentum_strength / 3, 0.5, 1.0)
        
        proba = np.zeros((len(signals), 3))
        for i, (signal, conf) in enumerate(zip(signals, confidence)):
            if signal == 1:  # Buy
                proba[i] = [0, 1-conf, conf]
            elif signal == -1:  # Sell
                proba[i] = [conf, 1-conf, 0]
            else:  # Hold
                proba[i] = [0.25, 0.5, 0.25]
        
        return proba


class VolumeWeightedModel(BaseEnsembleModel):
    """Volume-weighted signal model"""
    
    def __init__(self, volume_threshold: float = 1.5, **kwargs: Any):
        super().__init__("Volume_Weighted", **kwargs)
        self.volume_threshold = volume_threshold
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """Calculate volume statistics"""
        if 'volume' in x_data.columns:
            self.avg_volume = x_data['volume'].rolling(window=20).mean()
        self.is_trained = True
        logger.info(f"Volume model initialized: threshold={self.volume_threshold}")
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """Generate signals based on volume spikes"""
        if 'volume' not in x_data.columns or 'close' not in x_data.columns:
            return np.zeros(len(x_data))
        
        volume_ratio = x_data['volume'] / x_data['volume'].rolling(window=20).mean()
        price_change = x_data['close'].pct_change()
        
        # Buy on volume spike with positive price movement
        # Sell on volume spike with negative price movement
        signals = np.where((volume_ratio > self.volume_threshold) & (price_change > 0), 1,
                          np.where((volume_ratio > self.volume_threshold) & (price_change < 0), -1, 0))
        return signals
    
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        """Confidence based on volume ratio"""
        if 'volume' not in x_data.columns:
            return np.full((len(x_data), 3), [0.33, 0.34, 0.33])
        
        volume_ratio = x_data['volume'] / x_data['volume'].rolling(window=20).mean()
        signals = self.predict(x_data)
        
        confidence = np.clip((volume_ratio - 1) / 2, 0.5, 1.0)
        
        proba = np.zeros((len(signals), 3))
        for i, (signal, conf) in enumerate(zip(signals, confidence)):
            if signal == 1:  # Buy
                proba[i] = [0, 1-conf, conf]
            elif signal == -1:  # Sell
                proba[i] = [conf, 1-conf, 0]
            else:  # Hold
                proba[i] = [0.3, 0.4, 0.3]
        
        return proba


class WeightedEnsemble:
    """Weighted ensemble of multiple models"""
    
    def __init__(self, models: List[BaseEnsembleModel], 
                 voting: str = "soft", confidence_threshold: float = 0.6):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            voting: 'hard' or 'soft' voting
            confidence_threshold: Minimum confidence for signal generation
        """
        self.models = models
        self.voting = voting
        self.confidence_threshold = confidence_threshold
        self.model_weights = None
        self._normalize_weights()
        
        logger.info(f"Ensemble initialized with {len(models)} models")
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1"""
        total_weight = sum(model.weight for model in self.models)
        if total_weight > 0:
            for model in self.models:
                model.weight /= total_weight
    
    def fit(self, x_data: pd.DataFrame, y_data: pd.Series[Any]) -> None:
        """Train all models in the ensemble"""
        logger.info("Training ensemble models...")
        for model in self.models:
            try:
                model.fit(x_data, y_data)
                logger.info(f"✅ {model.name} trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train {model.name}: {e}")
    
    def predict(self, x_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ensemble predictions"""
        if not x_data.empty:
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model in self.models:
                if model.is_trained:
                    try:
                        predictions = model.predict(x_data)
                        probabilities = model.predict_proba(x_data)
                        
                        model_predictions[model.name] = predictions
                        model_probabilities[model.name] = probabilities
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model.name}: {e}")
                        continue
            
            if not model_predictions:
                return {"signal": 0, "confidence": 0.5, "model_votes": {}}
            
            # Combine predictions
            if self.voting == "hard":
                ensemble_signal, confidence = self._hard_voting(model_predictions)
            else:
                ensemble_signal, confidence = self._soft_voting(model_probabilities)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                ensemble_signal = 0  # Hold if confidence too low
            
            return {
                "signal": ensemble_signal,
                "confidence": confidence,
                "model_votes": model_predictions,
                "model_probabilities": model_probabilities
            }
        
        return {"signal": 0, "confidence": 0.5, "model_votes": {}}
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """Hard voting: majority rule"""
        if not predictions:
            return 0, 0.5
        
        # Get the last prediction from each model
        latest_predictions = {}
        for model_name, preds in predictions.items():
            if len(preds) > 0:
                latest_predictions[model_name] = preds[-1]
        
        if not latest_predictions:
            return 0, 0.5
        
        # Count votes
        votes = list(latest_predictions.values())
        weights = [model.weight for model in self.models if model.name in latest_predictions]
        
        # Weighted voting
        buy_weight = sum(w for v, w in zip(votes, weights) if v == 1)
        sell_weight = sum(w for v, w in zip(votes, weights) if v == -1)
        hold_weight = sum(w for v, w in zip(votes, weights) if v == 0)
        
        # Determine winner
        if buy_weight > sell_weight and buy_weight > hold_weight:
            signal = 1
            confidence = buy_weight / (buy_weight + sell_weight + hold_weight)
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            signal = -1
            confidence = sell_weight / (buy_weight + sell_weight + hold_weight)
        else:
            signal = 0
            confidence = hold_weight / (buy_weight + sell_weight + hold_weight)
        
        return signal, confidence
    
    def _soft_voting(self, probabilities: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """Soft voting: weighted average of probabilities"""
        if not probabilities:
            return 0, 0.5
        
        # Get latest probabilities
        latest_probs = {}
        for model_name, probs in probabilities.items():
            if len(probs) > 0:
                latest_probs[model_name] = probs[-1]  # [sell_prob, hold_prob, buy_prob]
        
        if not latest_probs:
            return 0, 0.5
        
        # Weighted average of probabilities
        weighted_probs = np.zeros(3)  # [sell, hold, buy]
        total_weight = 0
        
        for model in self.models:
            if model.name in latest_probs:
                weighted_probs += latest_probs[model.name] * model.weight
                total_weight += model.weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Determine signal
        max_prob_idx = np.argmax(weighted_probs)
        confidence = weighted_probs[max_prob_idx]
        
        if max_prob_idx == 0:  # Sell
            signal = -1
        elif max_prob_idx == 2:  # Buy
            signal = 1
        else:  # Hold
            signal = 0
        
        return signal, confidence
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each model"""
        performance = {}
        for model in self.models:
            performance[model.name] = {
                "weight": model.weight,
                "is_trained": model.is_trained,
                **model.performance_metrics
            }
        return performance
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """Update model weights based on performance"""
        for model in self.models:
            if model.name in performance_metrics:
                # Update weight based on performance (e.g., Sharpe ratio)
                new_weight = max(0.1, performance_metrics[model.name])
                model.weight = new_weight
                logger.info(f"Updated {model.name} weight to {new_weight:.3f}")
        
        self._normalize_weights()


class EnsembleStrategy:
    """Complete ensemble trading strategy"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ensemble strategy"""
        self.config = config or self._get_default_config()
        self.ensemble = self._create_ensemble()
        self.is_trained = False
        
        # Performance tracking
        self.performance_history = []
        self.signal_history = []
        
        logger.info("Ensemble strategy initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {
                "sma": {"short_window": 10, "long_window": 30, "weight": 0.25},
                "rsi": {"period": 14, "overbought": 70, "oversold": 30, "weight": 0.25},
                "momentum": {"lookback": 20, "threshold": 0.02, "weight": 0.25},
                "volume": {"volume_threshold": 1.5, "weight": 0.25}
            },
            "ensemble": {
                "voting": "soft",
                "confidence_threshold": 0.6
            }
        }
    
    def _create_ensemble(self) -> WeightedEnsemble:
        """Create ensemble with configured models"""
        models = []
        
        model_configs = self.config["models"]
        
        # SMA model
        if "sma" in model_configs:
            sma_config = model_configs["sma"]
            models.append(SimpleMovingAverageModel(**sma_config))
        
        # RSI model
        if "rsi" in model_configs:
            rsi_config = model_configs["rsi"]
            models.append(RSIModel(**rsi_config))
        
        # Momentum model
        if "momentum" in model_configs:
            momentum_config = model_configs["momentum"]
            models.append(MomentumModel(**momentum_config))
        
        # Volume model
        if "volume" in model_configs:
            volume_config = model_configs["volume"]
            models.append(VolumeWeightedModel(**volume_config))
        
        ensemble_config = self.config["ensemble"]
        return WeightedEnsemble(models, **ensemble_config)
    
    def fit(self, data: pd.DataFrame, target_column: str = "future_return") -> None:
        """Train the ensemble strategy"""
        logger.info("Training ensemble strategy...")
        
        # Prepare features and target
        features = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Create target if not exists
        if target_column not in data.columns:
            # Create future return target (1-day ahead)
            target = data['close'].pct_change().shift(-1)
            labels = np.where(target > 0.01, 1, np.where(target < -0.01, -1, 0))
            labels = pd.Series(labels, index=data.index)
        else:
            labels = data[target_column]
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        # Train ensemble
        self.ensemble.fit(features, labels)
        self.is_trained = True
        
        logger.info("Ensemble strategy training completed")
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal using ensemble"""
        if not self.is_trained:
            logger.warning("Ensemble not trained. Training with provided data...")
            self.fit(data)
        
        # Use last row for prediction
        latest_data = data.tail(1)[['open', 'high', 'low', 'close', 'volume']]
        
        # Generate ensemble prediction
        result = self.ensemble.predict(latest_data)
        
        # Add timestamp and additional info
        result['timestamp'] = datetime.now()
        result['strategy'] = 'ensemble'
        result['data_points'] = len(data)
        
        # Store signal history
        self.signal_history.append(result)
        
        return result
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """Backtest the ensemble strategy"""
        logger.info(f"Starting ensemble backtest with ${initial_capital:,.0f}")
        
        # Ensure strategy is trained
        if not self.is_trained:
            self.fit(data)
        
        # Initialize backtest variables
        portfolio_value = initial_capital
        position = 0  # 0=cash, 1=long, -1=short
        trades = []
        portfolio_values = []
        signals = []
        
        # Minimum required data for predictions
        min_data_points = 30
        
        for i in range(min_data_points, len(data)):
            # Get historical data up to current point
            hist_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            
            # Generate signal
            signal_result = self.generate_signal(hist_data)
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            
            signals.append({
                'date': data.index[i],
                'signal': signal,
                'confidence': confidence,
                'price': current_price
            })
            
            # Execute trades based on signal
            if signal == 1 and position <= 0 and confidence > self.ensemble.confidence_threshold:
                # Buy signal
                if position == -1:
                    # Close short position
                    portfolio_value += (trades[-1]['price'] - current_price) * abs(trades[-1]['quantity'])
                
                # Open long position
                quantity = portfolio_value / current_price
                position = 1
                trades.append({
                    'date': data.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'portfolio_value': portfolio_value
                })
                
            elif signal == -1 and position >= 0 and confidence > self.ensemble.confidence_threshold:
                # Sell signal
                if position == 1:
                    # Close long position
                    portfolio_value = trades[-1]['quantity'] * current_price
                
                # Open short position (if allowed)
                quantity = portfolio_value / current_price
                position = -1
                trades.append({
                    'date': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'portfolio_value': portfolio_value
                })
            
            # Calculate current portfolio value
            if position == 1 and trades:
                current_value = trades[-1]['quantity'] * current_price
            elif position == -1 and trades:
                current_value = portfolio_value + (trades[-1]['price'] - current_price) * trades[-1]['quantity']
            else:
                current_value = portfolio_value
            
            portfolio_values.append(current_value)
        
        # Calculate performance metrics
        final_value = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_value - initial_capital) / initial_capital
        
        # Convert to pandas series for easier calculation
        pv_series = pd.Series(portfolio_values, index=data.index[min_data_points:])
        returns = pv_series.pct_change().dropna()
        
        # Calculate metrics
        metrics = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': self._calculate_cagr(pv_series),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(pv_series),
            'volatility': returns.std() * np.sqrt(252),
            'num_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'calmar_ratio': total_return / abs(self._calculate_max_drawdown(pv_series)) if self._calculate_max_drawdown(pv_series) != 0 else 0
        };
        
        backtest_results = {
            'performance_metrics': metrics,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'signals': signals,
            'model_performance': self.ensemble.get_model_performance()
        }
        
        logger.info(f"Backtest completed: {total_return:.2%} return, Sharpe: {metrics['sharpe_ratio']:.3f}")
        
        return backtest_results
    
    def _calculate_cagr(self, values: pd.Series[Any]) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(values) < 2:
            return 0.0
        
        years = len(values) / 252  # Assuming daily data
        return (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1
    
    def _calculate_sharpe(self, returns: pd.Series[Any], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, values: pd.Series[Any]) -> float:
        """Calculate maximum drawdown"""
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades"""
        if len(trades) < 2:
            return 0.5
        
        profits = []
        for i in range(1, len(trades)):
            if trades[i-1]['action'] == 'BUY' and trades[i]['action'] == 'SELL':
                profit = trades[i]['price'] - trades[i-1]['price']
                profits.append(profit > 0)
        
        return sum(profits) / len(profits) if profits else 0.5
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor"""
        if len(trades) < 2:
            return 1.0
        
        gross_profit = 0
        gross_loss = 0
        
        for i in range(1, len(trades)):
            if trades[i-1]['action'] == 'BUY' and trades[i]['action'] == 'SELL':
                profit = trades[i]['price'] - trades[i-1]['price']
                if profit > 0:
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)
        
        return gross_profit / gross_loss if gross_loss > 0 else 1.0
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_backtest_results_{timestamp}.json"
        
        filepath = Path("results") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if key == 'portfolio_values':
                serializable_results[key] = value
            elif key == 'trades' or key == 'signals':
                serializable_results[key] = value
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)


def create_sample_data(symbols: List[str] = ['AAPL'], 
                      start_date: str = '2020-01-01', 
                      end_date: str = '2024-12-31',
                      num_points: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')[:num_points]
    
    # Generate realistic price data
    np.random.seed(42)
    initial_price = 100.0
    
    # Generate returns with some correlation structure
    returns = np.random.normal(0.0005, 0.02, len(date_range))  # ~0.12% daily return, 2% volatility
    
    # Add some momentum and mean reversion
    for i in range(1, len(returns)):
        momentum = returns[i-1] * 0.1  # Momentum effect
        returns[i] += momentum
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    data = []
    for i, (date, price) in enumerate(zip(date_range, prices)):
        # Generate realistic OHLC
        daily_vol = np.random.uniform(0.005, 0.03)  # Daily volatility
        high = price * (1 + daily_vol * np.random.uniform(0, 1))
        low = price * (1 - daily_vol * np.random.uniform(0, 1))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        
        # Ensure OHLC constraints
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume
        volume = int(np.random.lognormal(13, 1))  # Log-normal distribution for volume
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df


def main():
    """Main function to demonstrate ensemble models"""
    logger.info("🤖 Starting Ensemble Models Demo")
    
    # Create sample data
    logger.info("Creating sample market data...")
    data = create_sample_data(num_points=500)
    
    # Initialize ensemble strategy
    logger.info("Initializing ensemble strategy...")
    strategy = EnsembleStrategy()
    
    # Run backtest
    logger.info("Running ensemble backtest...")
    results = strategy.backtest(data)
    
    # Save results
    results_file = strategy.save_results(results)
    
    # Print summary
    metrics = results['performance_metrics']
    print("\n" + "="*60)
    print("🤖 ENSEMBLE MODELS BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"\nResults saved to: {results_file}")
    print("="*60)
    
    # Model performance
    print("\n📊 MODEL PERFORMANCE:")
    model_perf = results['model_performance']
    for model_name, perf in model_perf.items():
        print(f"  {model_name}: Weight = {perf['weight']:.3f}, Trained = {perf['is_trained']}")


if __name__ == "__main__":
    main()
