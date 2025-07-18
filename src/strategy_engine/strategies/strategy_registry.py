# src/ai_trading_machine/strategies/strategy_registry.py

from typing import Any

try:
    from .momentum import MomentumStrategy
    from .rsi_strategy import RSIStrategy
    from .sector_rotation import SectorRotationStrategy
    from .pairs_trading import PairsTradingStrategy
    from .news_sentiment import NewsSentimentStrategy
    from .lstm_prediction import LSTMPredictionStrategy
    from .options_strategy import OptionsStrategy
    from .multi_timeframe_strategy import MultiTimeframeStrategy
except ImportError:
    # Fallback if strategies are not available
    RSIStrategy = None
    MomentumStrategy = None
    SectorRotationStrategy = None
    PairsTradingStrategy = None
    NewsSentimentStrategy = None
    LSTMPredictionStrategy = None
    OptionsStrategy = None
    MultiTimeframeStrategy = None

STRATEGY_MAP = {
    "rsi": RSIStrategy,
    "momentum": MomentumStrategy,
    "sector_rotation": SectorRotationStrategy,
    "pairs_trading": PairsTradingStrategy,
    "news_sentiment": NewsSentimentStrategy,
    "lstm_prediction": LSTMPredictionStrategy,
    "options": OptionsStrategy,
    "multi_timeframe": MultiTimeframeStrategy,
}


def get_strategy_class(name: str):
    key = name.lower()
    if key not in STRATEGY_MAP:
        raise ValueError(f"Unsupported strategy: {name}")
    return STRATEGY_MAP[key]


def list_available_strategies():
    """List all available strategies"""
    return {name: cls for name, cls in STRATEGY_MAP.items() if cls is not None}


def create_strategy(name: str, config: dict[str, Any]):
    """Create a strategy instance with configuration"""
    strategy_class = get_strategy_class(name)
    if strategy_class is None:
        raise ValueError(f"Strategy {name} is not available")
    return strategy_class(config)
