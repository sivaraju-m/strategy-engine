"""
================================================================================
News Sentiment Strategy Implementation
================================================================================

This module analyzes news sentiment for generating trading signals. It
features:

- Sentiment analysis based on news headlines
- Real-time signal generation and backtesting
- Designed for integration into larger trading systems

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict
import logging
import re

from .base_strategy import BaseStrategy, TradingSignal, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentStrategy(BaseStrategy):
    """
    News Sentiment Strategy that analyzes news sentiment for trading decisions
    """

    def __init__(self, config: dict[str, Any]):
        strategy_name = config.get('strategy_name', 'news_sentiment')
        super().__init__(strategy_name)
        self.name = strategy_name
        self.min_confidence = config.get('min_confidence', 0.7)
        self.sentiment_threshold = config.get('sentiment_threshold', 0.6)
        
        # Positive and negative keywords for basic sentiment analysis
        self.positive_keywords = {
            'profit', 'growth', 'increase', 'gain', 'positive', 'strong', 'beat',
            'exceed', 'bullish', 'optimistic', 'upgrade', 'buy', 'outperform',
            'earnings', 'revenue', 'expansion', 'acquisition', 'partnership'
        }
        
        self.negative_keywords = {
            'loss', 'decline', 'decrease', 'negative', 'weak', 'miss', 'below',
            'bearish', 'pessimistic', 'downgrade', 'sell', 'underperform',
            'debt', 'risk', 'concern', 'warning', 'lawsuit', 'investigation'
        }
        
        # Mock news data for demonstration
        self.mock_news_data = {
            'RELIANCE.NS': [
                {'headline': 'Reliance Industries reports strong quarterly earnings growth', 'sentiment': 0.8},
                {'headline': 'RIL announces major expansion in renewable energy', 'sentiment': 0.7}
            ],
            'TCS.NS': [
                {'headline': 'TCS beats earnings estimates with strong digital growth', 'sentiment': 0.9},
                {'headline': 'Tata Consultancy Services wins major client contract', 'sentiment': 0.6}
            ],
            'HDFCBANK.NS': [
                {'headline': 'HDFC Bank faces regulatory concerns over lending practices', 'sentiment': -0.6},
                {'headline': 'Banking sector outlook remains challenging', 'sentiment': -0.4}
            ],
            'ICICIBANK.NS': [
                {'headline': 'ICICI Bank reports healthy loan growth and improved asset quality', 'sentiment': 0.7},
                {'headline': 'Private banks show resilience amid market volatility', 'sentiment': 0.5}
            ],
            'INFY.NS': [
                {'headline': 'Infosys guidance revision raises concerns about IT sector', 'sentiment': -0.5},
                {'headline': 'Tech services demand softening in key markets', 'sentiment': -0.3}
            ]
        }

    def analyze_sentiment(self, text: str) -> float:
        """Basic sentiment analysis using keyword matching"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, sentiment_score * 10))

    def get_news_sentiment(self, symbol: str) -> float:
        """Get aggregated news sentiment for a symbol"""
        if symbol not in self.mock_news_data:
            return 0.0
        
        news_items = self.mock_news_data[symbol]
        
        # Calculate average sentiment from recent news
        sentiments = []
        for news_item in news_items:
            # Use mock sentiment if available, otherwise analyze headline
            if 'sentiment' in news_item:
                sentiment = news_item['sentiment']
            else:
                sentiment = self.analyze_sentiment(news_item['headline'])
            
            sentiments.append(sentiment)
        
        if not sentiments:
            return 0.0
        
        # Weight recent news more heavily
        weights = [0.7, 0.3] if len(sentiments) >= 2 else [1.0]
        weights = weights[:len(sentiments)]
        
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights))
        return weighted_sentiment / sum(weights)

    def generate_signals(self, data: dict[str, Any]) -> list[TradingSignal]:
        """Generate news sentiment-based signals"""
        signals: list[TradingSignal] = []
        
        try:
            # Convert data to DataFrame if needed
            if 'prices' in data:
                df = pd.DataFrame(data['prices'])
            else:
                logger.warning("Expected 'prices' key in data dict")
                return signals
            
            # Analyze sentiment for each symbol
            for symbol in df.columns:
                sentiment = self.get_news_sentiment(symbol)
                
                if abs(sentiment) >= self.sentiment_threshold:
                    current_price = float(df[symbol].iloc[-1])
                    
                    # Calculate confidence based on sentiment strength
                    confidence = min(0.95, 0.5 + (abs(sentiment) * 0.4))
                    
                    if confidence >= self.min_confidence:
                        # Determine signal type
                        if sentiment > 0:
                            signal_type = SignalType.BUY
                            reasoning = f"Positive news sentiment: {sentiment:.2f}"
                        else:
                            signal_type = SignalType.SELL
                            reasoning = f"Negative news sentiment: {sentiment:.2f}"
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal=signal_type,
                            confidence=confidence,
                            price=current_price,
                            timestamp=datetime.now(),
                            strategy_name=self.name,
                            risk_metrics={
                                'sentiment_score': abs(sentiment),
                                'news_impact': abs(sentiment) * confidence
                            },
                            position_size=min(0.02, abs(sentiment) * 0.03),  # Position based on sentiment
                            reasoning=reasoning
                        )
                        signals.append(signal)
            
            logger.info(f"Generated {len(signals)} news sentiment signals")
            
        except Exception as e:
            logger.error(f"Error generating news sentiment signals: {str(e)}")
        
        return signals

    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the news sentiment strategy"""
        results: Dict[str, Any] = {
            'strategy_name': self.name,
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'avg_confidence': 0.0,
            'sentiment_analysis': {},
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
            
            # Analyze sentiment for all symbols
            sentiment_analysis = {}
            for symbol in data.columns:
                sentiment = self.get_news_sentiment(symbol)
                sentiment_analysis[symbol] = sentiment
            
            results['sentiment_analysis'] = sentiment_analysis
            
            logger.info(f"News sentiment backtest completed: {results}")
            
        except Exception as e:
            logger.error(f"Error in news sentiment backtest: {str(e)}")
            results['error'] = str(e)
        
        return results

    def calculate_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate indicators for news sentiment strategy"""
        indicators = {}
        try:
            if 'prices' in data:
                df = pd.DataFrame(data['prices'])
                # Calculate sentiment scores for all symbols
                sentiment_scores = {}
                for symbol in df.columns:
                    sentiment = self.get_news_sentiment(symbol)
                    sentiment_scores[symbol] = sentiment
                indicators['sentiment_scores'] = sentiment_scores
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        return indicators
