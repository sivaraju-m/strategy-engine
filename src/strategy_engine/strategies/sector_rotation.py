"""
================================================================================
Sector Rotation Strategy Implementation
================================================================================

This module rotates between sectors based on macroeconomic indicators and
sector performance. It features:

- Relative sector performance and momentum indicators
- Real-time signal generation and backtesting
- Designed for integration into larger trading systems

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple
import logging

from .base_strategy import BaseStrategy, TradingSignal, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SectorRotationStrategy(BaseStrategy):
    """
    Sector Rotation Strategy that rotates between sectors based on:
    1. Relative sector performance
    2. Momentum indicators
    3. Macroeconomic indicators
    4. Market cycle analysis
    """

    def __init__(self, config: dict[str, Any]):
        strategy_name = config.get("strategy_name", "sector_rotation")
        super().__init__(strategy_name)
        self.name = strategy_name
        self.lookback_period = config.get("lookback_period", 60)
        self.momentum_period = config.get("momentum_period", 20)
        self.min_confidence = config.get("min_confidence", 0.65)
        self.max_positions = config.get("max_positions", 3)

        # Sector mappings for NSE stocks
        self.sector_mappings = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Technology",
            "HDFCBANK.NS": "Banking",
            "INFY.NS": "Technology",
            "ICICIBANK.NS": "Banking",
            "HINDUNILVR.NS": "FMCG",
            "HDFC.NS": "Banking",
            "SBIN.NS": "Banking",
            "BHARTIARTL.NS": "Telecom",
            "KOTAKBANK.NS": "Banking",
            "ITC.NS": "FMCG",
            "LT.NS": "Engineering",
            "AXISBANK.NS": "Banking",
            "ASIANPAINT.NS": "Paints",
            "MARUTI.NS": "Auto",
            "HCLTECH.NS": "Technology",
            "BAJFINANCE.NS": "NBFC",
            "WIPRO.NS": "Technology",
            "ULTRACEMCO.NS": "Cement",
            "TITAN.NS": "Consumer Durables",
            "ADANIPORTS.NS": "Infrastructure",
            "POWERGRID.NS": "Utilities",
            "NTPC.NS": "Utilities",
            "NESTLEIND.NS": "FMCG",
            "TECHM.NS": "Technology",
            "JSWSTEEL.NS": "Steel",
            "TATASTEEL.NS": "Steel",
            "COALINDIA.NS": "Mining",
            "ONGC.NS": "Energy",
            "HINDALCO.NS": "Metals",
            "GRASIM.NS": "Cement",
            "CIPLA.NS": "Pharma",
            "DRREDDY.NS": "Pharma",
            "SUNPHARMA.NS": "Pharma",
            "BAJAJFINSV.NS": "NBFC",
            "TATAMOTORS.NS": "Auto",
            "HEROMOTOCO.NS": "Auto",
            "BRITANNIA.NS": "FMCG",
            "DIVISLAB.NS": "Pharma",
            "EICHERMOT.NS": "Auto",
            "APOLLOHOSP.NS": "Healthcare",
            "SHREECEM.NS": "Cement",
            "BPCL.NS": "Energy",
            "INDUSINDBK.NS": "Banking",
            "TATACONSUM.NS": "FMCG",
            "ADANIENT.NS": "Infrastructure",
            "ADANIGREEN.NS": "Green Energy",
            "SBILIFE.NS": "Insurance",
            "BAJAJHLDNG.NS": "NBFC",
            "HDFCLIFE.NS": "Insurance",
        }

        # Sector weights and performance tracking
        self.sector_performance: Dict[str, Any] = {}
        self.sector_momentum: Dict[str, Any] = {}
        self.current_sector_allocation: Dict[str, Any] = {}

    def calculate_sector_performance(
        self, data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate sector performance metrics
        """
        sector_performance: Dict[str, Dict[str, float]] = {}

        for symbol in data.columns:
            if symbol in self.sector_mappings:
                sector = self.sector_mappings[symbol]

                # Calculate returns
                returns = data[symbol].pct_change().dropna()

                if len(returns) > self.lookback_period:
                    # Calculate various performance metrics
                    recent_returns = returns.tail(self.lookback_period)
                    cumulative_return = (1 + recent_returns).prod()
                    # Convert pandas scalar to float more safely
                    if hasattr(cumulative_return, "item"):
                        total_return = cumulative_return.item() - 1.0
                    else:
                        total_return = 0.0

                    volatility = float(recent_returns.std() * np.sqrt(252))
                    sharpe_ratio = (
                        (total_return / volatility) if volatility > 0 else 0.0
                    )

                    # Momentum calculation
                    momentum_val = returns.tail(self.momentum_period).mean()
                    if hasattr(momentum_val, "item"):
                        momentum = momentum_val.item()
                    else:
                        momentum = 0.0

                    if sector not in sector_performance:
                        sector_performance[sector] = {
                            "total_return": 0.0,
                            "volatility": 0.0,
                            "sharpe_ratio": 0.0,
                            "momentum": 0.0,
                            "count": 0.0,
                        }

                    # Aggregate sector metrics
                    sector_performance[sector]["total_return"] += total_return
                    sector_performance[sector]["volatility"] += volatility
                    sector_performance[sector]["sharpe_ratio"] += sharpe_ratio
                    sector_performance[sector]["momentum"] += momentum
                    sector_performance[sector]["count"] += 1

        # Average the metrics by sector
        for sector in sector_performance:
            count = sector_performance[sector]["count"]
            if count > 0:
                sector_performance[sector]["total_return"] /= count
                sector_performance[sector]["volatility"] /= count
                sector_performance[sector]["sharpe_ratio"] /= count
                sector_performance[sector]["momentum"] /= count

        return sector_performance

    def rank_sectors(
        self, sector_performance: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Rank sectors based on combined performance metrics
        """
        sector_scores = []

        for sector, metrics in sector_performance.items():
            # Combined score: 40% momentum, 30% sharpe ratio, 30% total return
            score = (
                0.4 * metrics["momentum"]
                + 0.3 * metrics["sharpe_ratio"]
                + 0.3 * metrics["total_return"]
            )

            sector_scores.append((sector, score))

        # Sort by score descending
        sector_scores.sort(key=lambda x: x[1], reverse=True)

        return sector_scores

    def get_sector_symbols(self, sector: str) -> List[str]:
        """
        Get all symbols belonging to a specific sector
        """
        return [symbol for symbol, sec in self.sector_mappings.items() if sec == sector]

    def generate_signals(self, data: dict[str, Any]) -> list[TradingSignal]:
        """
        Generate sector rotation signals
        """
        signals: list[TradingSignal] = []

        try:
            # Convert data to DataFrame if needed
            if isinstance(data, dict) and "prices" in data:
                df = pd.DataFrame(data["prices"])
            else:
                logger.warning("Expected 'prices' key in data dict")
                return signals

            # Calculate sector performance
            sector_performance = self.calculate_sector_performance(df)

            if not sector_performance:
                logger.warning("No sector performance data available")
                return signals

            # Rank sectors
            sector_rankings = self.rank_sectors(sector_performance)

            # Get top performing sectors
            top_sectors = sector_rankings[: self.max_positions]

            logger.info(f"Top performing sectors: {top_sectors}")

            # Generate signals for top sectors
            for sector, score in top_sectors:
                if score > 0:  # Only positive momentum sectors
                    symbols = self.get_sector_symbols(sector)

                    # For each symbol in top sectors, generate BUY signal
                    for symbol in symbols:
                        if symbol in df.columns:
                            current_price = float(df[symbol].iloc[-1])

                            # Calculate confidence based on sector score
                            confidence = min(0.95, max(0.5, 0.6 + (score * 0.3)))

                            if confidence >= self.min_confidence:
                                signal = TradingSignal(
                                    symbol=symbol,
                                    signal=SignalType.BUY,
                                    confidence=confidence,
                                    price=current_price,
                                    timestamp=datetime.now(),
                                    strategy_name=self.name,
                                    risk_metrics={
                                        "sector": sector,
                                        "sector_score": score,
                                        "sector_rank": sector_rankings.index(
                                            (sector, score)
                                        )
                                        + 1,
                                    },
                                    position_size=1.0
                                    / len(symbols),  # Equal weight within sector
                                    reasoning=f"Sector {sector} showing strong momentum with score {score:.3f}",
                                )
                                signals.append(signal)

            # Generate SELL signals for bottom sectors
            bottom_sectors = sector_rankings[-2:]  # Bottom 2 sectors

            for sector, score in bottom_sectors:
                if score < -0.05:  # Only negative momentum sectors
                    symbols = self.get_sector_symbols(sector)

                    for symbol in symbols:
                        if symbol in df.columns:
                            current_price = float(df[symbol].iloc[-1])

                            # Calculate confidence based on negative sector score
                            confidence = min(0.85, max(0.5, 0.6 + (abs(score) * 0.25)))

                            if confidence >= self.min_confidence:
                                signal = TradingSignal(
                                    symbol=symbol,
                                    signal=SignalType.SELL,
                                    confidence=confidence,
                                    price=current_price,
                                    timestamp=datetime.now(),
                                    strategy_name=self.name,
                                    risk_metrics={
                                        "sector": sector,
                                        "sector_score": score,
                                        "sector_rank": sector_rankings.index(
                                            (sector, score)
                                        )
                                        + 1,
                                    },
                                    position_size=1.0
                                    / len(symbols),  # Equal weight within sector
                                    reasoning=f"Sector {sector} showing weak momentum with score {score:.3f}",
                                )
                                signals.append(signal)

            logger.info(f"Generated {len(signals)} sector rotation signals")

        except Exception as e:
            logger.error(f"Error generating sector rotation signals: {str(e)}")

        return signals

    def calculate_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate indicators for sector rotation strategy"""
        indicators = {}
        try:
            if "prices" in data:
                df = pd.DataFrame(data["prices"])
                sector_performance = self.calculate_sector_performance(df)
                indicators["sector_performance"] = sector_performance
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        return indicators

    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Backtest the sector rotation strategy
        """
        results: Dict[str, Any] = {
            "strategy_name": self.name,
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "avg_confidence": 0.0,
            "sector_performance": {},
            "backtest_period": f"{data.index[0]} to {data.index[-1]}",
        }

        try:
            # Generate signals for backtesting
            data_dict = {"prices": data}
            signals = self.generate_signals(data_dict)

            results["total_signals"] = len(signals)
            results["buy_signals"] = len(
                [s for s in signals if s.signal == SignalType.BUY]
            )
            results["sell_signals"] = len(
                [s for s in signals if s.signal == SignalType.SELL]
            )

            if signals:
                results["avg_confidence"] = np.mean([s.confidence for s in signals])

            # Calculate sector performance
            sector_performance = self.calculate_sector_performance(data)
            results["sector_performance"] = sector_performance

            logger.info(f"Sector rotation backtest completed: {results}")

        except Exception as e:
            logger.error(f"Error in sector rotation backtest: {str(e)}")
            results["error"] = str(e)

        return results
