"""
================================================================================
Base Strategy Class with SEBI Compliance
================================================================================

This module provides an abstract base class for developing trading strategies
with SEBI compliance. It features:

- Unified interface for signal generation, risk management, and position sizing
- SEBI compliance checks for Indian markets
- Logging for traceability and debugging
- Designed for integration into larger trading systems

================================================================================
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Optional

# Import sector mapping utility
try:
    from ..utils.sector_mapper import SectorMapper, get_sector_mapper
except ImportError:
    # Fallback if sector mapper is not available
    get_sector_mapper = None
    SectorMapper = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies"""

    MOMENTUM = auto()  # Strategies based on price momentum
    MEAN_REVERSION = auto()  # Strategies that trade mean reversion
    TREND_FOLLOWING = auto()  # Strategies that follow trends
    BREAKOUT = auto()  # Strategies trading breakouts
    VOLATILITY = auto()  # Volatility-based strategies
    MACHINE_LEARNING = auto()  # ML-based strategies
    TECHNICAL = auto()  # Technical indicator strategies
    FUNDAMENTAL = auto()  # Fundamental analysis strategies
    ARBITRAGE = auto()  # Arbitrage strategies
    MARKET_NEUTRAL = auto()  # Market neutral strategies
    PAIRS_TRADING = auto()  # Pairs trading strategies
    HIGH_FREQUENCY = auto()  # High-frequency trading strategies
    CUSTOM = auto()  # Custom strategies


class SignalType(Enum):
    """Trading signal types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""

    symbol: str
    signal: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy_name: str
    risk_metrics: dict[str, float]
    position_size: float
    reasoning: str


@dataclass
class SEBILimits:
    """SEBI position limits and compliance rules"""

    max_position_pct: float = 2.0  # Max 2% of portfolio per position
    max_sector_exposure: float = 20.0  # Max 20% per sector
    max_single_stock_fo: float = 1.0  # Max 1% for F&O positions
    max_portfolio_leverage: float = 3.0  # Max 3x leverage
    circuit_breaker_pct: float = 20.0  # 20% daily limit
    position_reporting_threshold: float = 1000000  # ₹10L reporting


@dataclass
class StrategyMetadata:
    """Metadata for strategy configuration and categorization"""

    name: str
    strategy_type: StrategyType
    description: str
    version: str = "1.0.0"
    author: str = "AI Trading Machine"
    created_date: datetime = datetime.now(timezone.utc)
    last_modified: Optional[datetime] = None
    risk_level: int = 3  # Scale of 1-5 with 5 being highest risk
    tags: list[str] = None
    market_conditions: list[str] = None
    time_frames: list[str] = None
    instruments: list[str] = None

    def __post_init__(self):
        """Initialize default values for optional fields"""
        if self.tags is None:
            self.tags = []
        if self.market_conditions is None:
            self.market_conditions = ["all"]
        if self.time_frames is None:
            self.time_frames = ["1d"]
        if self.instruments is None:
            self.instruments = ["equity"]


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies with SEBI compliance

    All strategies must inherit from this class and implement the required methods.
    Includes risk management, position sizing, and SEBI compliance validation.
    """

    def __init__(
        self,
        strategy_name: str,
        max_position_pct: float = 2.0,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 10.0,
        sebi_limits: Optional[SEBILimits] = None,
    ):
        self.strategy_name = strategy_name
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.sebi_limits = sebi_limits or SEBILimits()

        # Strategy state
        self.is_active = True
        self.current_positions: dict[str, Any] = {}
        self.performance_metrics: dict[str, Any] = {}
        self.trade_history: list[dict[str, Any]] = []

        logger.info("📊 Initialized {strategy_name} with SEBI compliance")

    @abstractmethod
    def generate_signals(self, data: dict[str, Any]) -> list[TradingSignal]:
        """
        Generate trading signals based on market data

        Args:
            data: Market data dictionary with OHLCV data

        Returns:
            List of trading signals with confidence scores
        """

    @abstractmethod
    def calculate_indicators(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate technical indicators for the strategy

        Args:
            data: Market data dictionary with OHLCV data

        Returns:
            Dictionary with additional indicator data
        """

    def validate_sebi_limits(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        sector: Optional[str] = None,
        is_fo: bool = False,
    ) -> tuple[bool, str]:
        """
        Validate position against SEBI limits

        Args:
            symbol: Stock symbol
            position_value: Value of the proposed position
            portfolio_value: Total portfolio value
            sector: Sector classification (deprecated - auto-detected)
            is_fo: Whether this is a F&O position

        Returns:
            Tuple of (is_valid, reason)
        """
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        position_pct = (position_value / portfolio_value) * 100

        # Check individual position limit
        max_limit = (
            self.sebi_limits.max_single_stock_fo
            if is_fo
            else self.sebi_limits.max_position_pct
        )
        if position_pct > max_limit:
            return False, "Position {position_pct:.2f}% exceeds limit {max_limit}%"

        # Check reporting threshold
        if position_value > self.sebi_limits.position_reporting_threshold:
            logger.warning(
                "⚠️ Position ₹{position_value:,.0f} exceeds reporting threshold"
            )

        # Sector exposure validation
        if get_sector_mapper is not None:
            try:
                sector_mapper = get_sector_mapper()
                is_valid, reason = sector_mapper.validate_sector_exposure(
                    symbol=symbol,
                    new_position_value=position_value,
                    current_positions=getattr(self, "current_positions", {}),
                    portfolio_value=portfolio_value,
                )

                if not is_valid:
                    return False, "Sector exposure limit exceeded: {reason}"

                logger.info("✅ Sector validation passed: {reason}")

            except Exception as e:
                logger.warning("⚠️ Sector validation failed with error: {e}")
                # Don't block the trade if sector validation fails
        else:
            logger.debug("Sector mapper not available - skipping sector validation")

        # Legacy sector check for backward compatibility
        if sector:
            # This parameter is kept for API compatibility but handled above
            pass

        return True, "Position within SEBI limits"

    def calculate_position_size(
        self,
        price: float,
        portfolio_value: float,
        volatility: float,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion with risk management

        Args:
            price: Current stock price
            portfolio_value: Total portfolio value
            volatility: Annualized volatility
            confidence: Signal confidence (0-1)

        Returns:
            Position size in shares
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        # Base position size as percentage of portfolio
        base_position_value = portfolio_value * (self.max_position_pct / 100)

        # Adjust for volatility (reduce size for high volatility)
        volatility_adjustment = min(1.0, 0.2 / max(volatility, 0.1))

        # Adjust for confidence
        confidence_adjustment = max(0.1, min(1.0, confidence))

        # Calculate final position value
        adjusted_position_value = (
            base_position_value * volatility_adjustment * confidence_adjustment
        )

        # Convert to shares
        position_size = int(adjusted_position_value / price)

        logger.debug(
            "💰 Position sizing: base={base_position_value:.0f}, "
            "vol_adj={volatility_adjustment:.2f}, conf_adj={confidence_adjustment:.2f}, "
            "final_shares={position_size}"
        )

        return position_size

    def check_circuit_breakers(
        self, current_price: float, previous_close: float
    ) -> bool:
        """
        Check if stock has hit circuit breaker limits

        Args:
            current_price: Current stock price
            previous_close: Previous day's closing price

        Returns:
            True if circuit breaker triggered
        """
        if previous_close <= 0:
            return False

        price_change_pct = abs((current_price - previous_close) / previous_close) * 100

        if price_change_pct >= self.sebi_limits.circuit_breaker_pct:
            logger.warning(
                "🚨 Circuit breaker triggered: {price_change_pct:.2f}% movement"
            )
            return True

        return False

    def calculate_risk_metrics(
        self, price_data: list[float], signal: SignalType
    ) -> dict[str, float]:
        """
        Calculate risk metrics for a trading signal

        Args:
            price_data: Historical price data
            signal: Trading signal type

        Returns:
            Dictionary of risk metrics
        """
        if len(price_data) < 20:
            return {"error": 0.0, "message": "insufficient_data"}

        # Calculate basic risk metrics without pandas
        returns = []
        for i in range(1, len(price_data)):
            if price_data[i - 1] > 0:
                returns.append((price_data[i] - price_data[i - 1]) / price_data[i - 1])

        if not returns:
            return {"error": 0.0, "message": "no_valid_returns"}

        # Basic statistics
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = (variance**0.5) * (252**0.5)  # Annualized

        # Simple VaR calculation
        sorted_returns = sorted(returns)
        var_95_index = int(len(sorted_returns) * 0.05)
        var_95 = (
            sorted_returns[var_95_index]
            if var_95_index < len(sorted_returns)
            else sorted_returns[0]
        )

        # Max drawdown calculation
        max_drawdown = self._calculate_max_drawdown(price_data)

        # Signal strength
        signal_strength = abs(mean_return) / (variance**0.5) if variance > 0 else 0

        return {
            "volatility": volatility,
            "var_95": var_95,
            "max_drawdown": max_drawdown,
            "signal_strength": signal_strength,
            "data_points": len(price_data),
        }

    def _calculate_max_drawdown(self, prices: list[float]) -> float:
        """Calculate maximum drawdown from price list"""
        if len(prices) < 2:
            return 0.0

        cumulative = [1.0]
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                return_pct = prices[i] / prices[i - 1]
                cumulative.append(cumulative[-1] * return_pct)

        max_drawdown = 0.0
        peak = cumulative[0]

        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def log_trading_decision(
        self, signal: TradingSignal, execution_details: dict[str, Any]
    ):
        """
        Log trading decision for audit trail (SEBI compliance)

        Args:
            signal: Trading signal generated
            execution_details: Details of order execution
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "symbol": signal.symbol,
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "price": signal.price,
            "position_size": signal.position_size,
            "reasoning": signal.reasoning,
            "risk_metrics": signal.risk_metrics,
            "execution": execution_details,
        }

        # In production, this would go to structured logging/audit database
        logger.info("📝 Trading Decision: {log_entry}")

        # Store in trade history
        self.trade_history.append(log_entry)

    def update_performance_metrics(
        self, portfolio_value: float, benchmark_return: float = 0.0
    ):
        """
        Update strategy performance metrics

        Args:
            portfolio_value: Current portfolio value
            benchmark_return: Benchmark return for comparison
        """
        self.performance_metrics.update(
            {
                "current_value": portfolio_value,
                "last_updated": datetime.now(timezone.utc),
                "benchmark_return": benchmark_return,
            }
        )

    def validate_market_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Validate NSE/BSE trading hours (9:15 AM - 3:30 PM IST)

        Args:
            timestamp: Time to validate (defaults to current time)

        Returns:
            True if within trading hours
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Convert to IST (UTC+5:30)
        ist_time = timestamp.replace(tzinfo=timezone.utc).astimezone(
            timezone(timedelta(hours=5, minutes=30))
        )

        # Check if weekday
        if ist_time.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check trading hours (9:15 AM - 3:30 PM)
        market_open = ist_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = ist_time.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= ist_time <= market_close

    def emergency_stop(self, reason: str):
        """
        Emergency stop for the strategy

        Args:
            reason: Reason for emergency stop
        """
        self.is_active = False
        logger.critical("🛑 EMERGENCY STOP: {self.strategy_name} - {reason}")

        # In production, this would trigger alerts and position closure

    def get_status(self) -> dict[str, Any]:
        """Get current strategy status"""
        return {
            "strategy_name": self.strategy_name,
            "is_active": self.is_active,
            "current_positions": len(self.current_positions),
            "performance_metrics": self.performance_metrics,
            "sebi_limits": self.sebi_limits.__dict__,
            "trade_count": len(self.trade_history),
        }

    def get_sector_exposure(self, portfolio_value: float) -> dict[str, Any]:
        """
        Get current sector exposure analysis

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with sector exposure information
        """
        if get_sector_mapper is None:
            return {
                "status": "unavailable",
                "message": "Sector mapper not available",
                "exposures": {},
            }

        try:
            sector_mapper = get_sector_mapper()
            exposures = sector_mapper.calculate_portfolio_sector_exposure(
                self.current_positions, portfolio_value
            )

            recommendations = sector_mapper.get_sector_recommendations(
                self.current_positions, portfolio_value
            )

            return {
                "status": "success",
                "portfolio_value": portfolio_value,
                "sector_exposures": {
                    code: {
                        "name": exp.sector_name,
                        "current_pct": exp.current_exposure_pct,
                        "max_pct": exp.max_allowed_pct,
                        "position_count": exp.position_count,
                        "total_value": exp.total_value,
                        "utilization": (
                            exp.current_exposure_pct / exp.max_allowed_pct * 100
                            if exp.max_allowed_pct > 0
                            else 0
                        ),
                    }
                    for code, exp in exposures.items()
                    if exp.current_exposure_pct > 0  # Only show sectors with positions
                },
                "recommendations": recommendations,
                "total_sectors": len(
                    [exp for exp in exposures.values() if exp.current_exposure_pct > 0]
                ),
            }

        except Exception as e:
            logger.error("Error calculating sector exposure: {e}")
            return {"status": "error", "message": str(e), "exposures": {}}

    def validate_portfolio_diversification(
        self, portfolio_value: float
    ) -> dict[str, Any]:
        """
        Validate portfolio diversification against sector limits

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Diversification validation results
        """
        sector_exposure = self.get_sector_exposure(portfolio_value)

        if sector_exposure["status"] != "success":
            return sector_exposure

        violations = []
        warnings = []

        for sector_code, exposure in sector_exposure["sector_exposures"].items():
            current_pct = exposure["current_pct"]
            max_pct = exposure["max_pct"]

            if current_pct > max_pct:
                violations.append(
                    {
                        "sector": exposure["name"],
                        "current_pct": current_pct,
                        "max_pct": max_pct,
                        "excess_pct": current_pct - max_pct,
                        "severity": "HIGH",
                    }
                )
            elif current_pct > max_pct * 0.9:  # Within 90% of limit
                warnings.append(
                    {
                        "sector": exposure["name"],
                        "current_pct": current_pct,
                        "max_pct": max_pct,
                        "utilization_pct": (current_pct / max_pct) * 100,
                        "severity": "MEDIUM",
                    }
                )

        is_compliant = len(violations) == 0

        return {
            "status": "success",
            "is_compliant": is_compliant,
            "violations": violations,
            "warnings": warnings,
            "total_sectors": sector_exposure["total_sectors"],
            "recommendations": sector_exposure["recommendations"],
        }
