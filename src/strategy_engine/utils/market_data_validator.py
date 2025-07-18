"""
# Market Data Validation for NSE Feed
#
# SJ-VERIFY
# - Path: /ai-trading-machine/src/ai_trading_machine/utils
# - Type: validator
# - Checks: types,sebi,data_quality,nse
#
# Purpose: Real-time market data validation with missing data, price anomalies, and circuit breaker checks
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    """Result of data validation"""

    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected_range: Optional[tuple[float, float]] = None


class MarketDataValidator:
    """
    Comprehensive market data validator for NSE/BSE feeds

    Validates:
    - Missing data detection
    - Price anomalies and outliers
    - Circuit breaker triggers
    - OHLC relationship consistency
    - Volume anomalies
    - Market hours compliance
    """

    def __init__(
        self,
        circuit_breaker_limits: dict[str, float] = None,
        volume_spike_threshold: float = 5.0,
        price_gap_threshold: float = 0.10,
        enable_strict_validation: bool = True,
    ):
        """
        Initialize market data validator

        Args:
            circuit_breaker_limits: Custom circuit breaker limits by category
            volume_spike_threshold: Volume spike detection multiplier
            price_gap_threshold: Price gap threshold (10% default)
            enable_strict_validation: Enable strict validation rules
        """
        self.circuit_breaker_limits = circuit_breaker_limits or {
            "normal": 0.20,  # 20% for normal stocks
            "et": 0.10,  # 10% for ETFs
            "index": 0.05,  # 5% for index stocks
            "illiquid": 0.05,  # 5% for illiquid stocks
        }
        self.volume_spike_threshold = volume_spike_threshold
        self.price_gap_threshold = price_gap_threshold
        self.enable_strict_validation = enable_strict_validation

        # NSE market hours (IST)
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        self.pre_market_open = time(9, 0)
        self.pre_market_close = time(9, 15)
        self.post_market_open = time(15, 40)
        self.post_market_close = time(16, 0)

        logger.info("📊 Market Data Validator initialized with SEBI compliance")

    def validate_ticker_data(
        self, ticker_data: dict[str, Any]
    ) -> list[ValidationResult]:
        """
        Comprehensive validation of ticker data

        Args:
            ticker_data: Dictionary containing market data for a ticker

        Returns:
            List of validation results
        """
        results = []

        # Required fields validation
        results.extend(self._validate_required_fields(ticker_data))

        # Basic data type validation
        results.extend(self._validate_data_types(ticker_data))

        # OHLCV validation
        if self._has_ohlcv_data(ticker_data):
            results.extend(self._validate_ohlc_relationships(ticker_data))
            results.extend(self._validate_price_ranges(ticker_data))
            results.extend(self._validate_volume_data(ticker_data))

        # Circuit breaker validation
        results.extend(self._validate_circuit_breakers(ticker_data))

        # Time-based validation
        results.extend(self._validate_timestamp(ticker_data))

        # Market hours validation
        results.extend(self._validate_market_hours(ticker_data))

        # Historical consistency validation
        results.extend(self._validate_historical_consistency(ticker_data))

        return results

    def _validate_required_fields(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate presence of required fields"""
        results = []
        required_fields = ["symbol", "timestamp"]

        for field in required_fields:
            if field not in data or data[field] is None:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Missing required field: {field}",
                        field=field,
                    )
                )

        return results

    def _validate_data_types(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate data types for all fields"""
        results = []

        type_validations = {
            "symbol": str,
            "open": (int, float),
            "high": (int, float),
            "low": (int, float),
            "close": (int, float),
            "volume": (int, float),
            "timestamp": (str, datetime),
        }

        for field, expected_type in type_validations.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.ERROR,
                            message="Invalid data type for {field}: expected {expected_type}, got {type(data[field])}",
                            field=field,
                            value=data[field],
                        )
                    )

        return results

    def _has_ohlcv_data(self, data: dict[str, Any]) -> bool:
        """Check if data contains OHLCV fields"""
        ohlcv_fields = ["open", "high", "low", "close", "volume"]
        return all(field in data and data[field] is not None for field in ohlcv_fields)

    def _validate_ohlc_relationships(
        self, data: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate OHLC price relationships"""
        results = []

        try:
            open_price = float(data["open"])
            high_price = float(data["high"])
            low_price = float(data["low"])
            close_price = float(data["close"])

            # High should be >= max(open, close)
            max_oc = max(open_price, close_price)
            if high_price < max_oc:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="High price ({high_price}) < max(open, close) ({max_oc})",
                        field="high",
                        value=high_price,
                        expected_range=(max_oc, float("in")),
                    )
                )

            # Low should be <= min(open, close)
            min_oc = min(open_price, close_price)
            if low_price > min_oc:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message=f"Low price ({low_price}) > min(open, close) ({min_oc})",
                        field="low",
                        value=low_price,
                        expected_range=(0, min_oc),
                    )
                )

            # All prices should be positive
            for field, price in [
                ("open", open_price),
                ("high", high_price),
                ("low", low_price),
                ("close", close_price),
            ]:
                if price <= 0:
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.ERROR,
                            message="Non-positive price for {field}: {price}",
                            field=field,
                            value=price,
                            expected_range=(0.01, float("in")),
                        )
                    )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Error validating OHLC relationships: {e}",
                    field="ohlc",
                )
            )

        return results

    def _validate_price_ranges(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate price ranges for anomaly detection"""
        results = []

        try:
            prices = [
                float(data[field])
                for field in ["open", "high", "low", "close"]
                if field in data and data[field] is not None
            ]

            if not prices:
                return results

            # Check for extreme price values
            max_price = max(prices)
            min_price = min(prices)

            # Detect potential data errors (prices too high or too low)
            if max_price > 100000:  # ₹1L per share seems unrealistic for most stocks
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message="Extremely high price detected: ₹{max_price}",
                        field="price_range",
                        value=max_price,
                    )
                )

            if min_price < 0.01:  # Less than 1 paisa
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Extremely low price detected: ₹{min_price}",
                        field="price_range",
                        value=min_price,
                    )
                )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Error validating price ranges: {e}",
                    field="price_range",
                )
            )

        return results

    def _validate_volume_data(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate volume data"""
        results = []

        if "volume" not in data or data["volume"] is None:
            return results

        try:
            volume = float(data["volume"])

            # Volume should be non-negative
            if volume < 0:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Negative volume: {volume}",
                        field="volume",
                        value=volume,
                        expected_range=(0, float("in")),
                    )
                )

            # Check for unusual volume spikes (if historical data available)
            if "avg_volume" in data and data["avg_volume"] is not None:
                avg_volume = float(data["avg_volume"])
                if avg_volume > 0 and volume > avg_volume * self.volume_spike_threshold:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            level=ValidationLevel.WARNING,
                            message="Volume spike detected: {volume:.0f} vs avg {avg_volume:.0f} "
                            "({volume/avg_volume:.1f}x normal)",
                            field="volume",
                            value=volume,
                        )
                    )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Error validating volume data: {e}",
                    field="volume",
                )
            )

        return results

    def _validate_circuit_breakers(
        self, data: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate circuit breaker triggers"""
        results = []

        if not all(field in data for field in ["close", "previous_close"]):
            return results

        try:
            current_price = float(data["close"])
            previous_close = float(data["previous_close"])

            if previous_close <= 0:
                return results

            price_change_pct = abs((current_price - previous_close) / previous_close)

            # Determine stock category for circuit breaker limits
            stock_category = data.get("category", "normal")
            limit = self.circuit_breaker_limits.get(stock_category, 0.20)

            if price_change_pct >= limit:
                direction = "up" if current_price > previous_close else "down"
                results.append(
                    ValidationResult(
                        is_valid=True,
                        level=ValidationLevel.CRITICAL,
                        message="Circuit breaker triggered: {price_change_pct*100:.2f}% move {direction} "
                        "(limit: {limit*100:.0f}%)",
                        field="circuit_breaker",
                        value=price_change_pct,
                    )
                )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Error validating circuit breakers: {e}",
                    field="circuit_breaker",
                )
            )

        return results

    def _validate_timestamp(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate timestamp data"""
        results = []

        if "timestamp" not in data:
            return results

        timestamp = data["timestamp"]

        try:
            if isinstance(timestamp, str):
                # Try to parse timestamp string
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.ERROR,
                        message="Invalid timestamp type: {type(timestamp)}",
                        field="timestamp",
                        value=timestamp,
                    )
                )
                return results

            # Check if timestamp is too far in the future
            now = datetime.now()
            if dt > now:
                time_diff = (dt - now).total_seconds()
                if time_diff > 300:  # More than 5 minutes in future
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            level=ValidationLevel.WARNING,
                            message="Timestamp is {time_diff:.0f} seconds in the future",
                            field="timestamp",
                            value=timestamp,
                        )
                    )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Error parsing timestamp: {e}",
                    field="timestamp",
                    value=timestamp,
                )
            )

        return results

    def _validate_market_hours(self, data: dict[str, Any]) -> list[ValidationResult]:
        """Validate market hours compliance"""
        results = []

        if "timestamp" not in data:
            return results

        try:
            if isinstance(data["timestamp"], str):
                dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            else:
                dt = data["timestamp"]

            # Convert to IST (assuming UTC input)
            ist_dt = dt.replace(tzinfo=None)  # Simplified for demo
            current_time = ist_dt.time()
            weekday = ist_dt.weekday()

            # Check if it's a weekday
            if weekday >= 5:  # Saturday=5, Sunday=6
                results.append(
                    ValidationResult(
                        is_valid=True,
                        level=ValidationLevel.INFO,
                        message="Data received on weekend ({weekday})",
                        field="market_hours",
                    )
                )
                return results

            # Check market session
            if self.market_open <= current_time <= self.market_close:
                session = "regular"
            elif self.pre_market_open <= current_time < self.pre_market_close:
                session = "pre_market"
            elif self.post_market_open <= current_time <= self.post_market_close:
                session = "post_market"
            else:
                session = "closed"
                results.append(
                    ValidationResult(
                        is_valid=True,
                        level=ValidationLevel.WARNING,
                        message="Data received outside market hours: {current_time}",
                        field="market_hours",
                        value=current_time,
                    )
                )

        except (ValueError, TypeError) as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Error validating market hours: {e}",
                    field="market_hours",
                )
            )

        return results

    def _validate_historical_consistency(
        self, data: dict[str, Any]
    ) -> list[ValidationResult]:
        """Validate consistency with historical data"""
        results = []

        # This would check against historical patterns
        # For now, just basic gap detection
        if all(field in data for field in ["open", "previous_close"]):
            try:
                open_price = float(data["open"])
                prev_close = float(data["previous_close"])

                if prev_close > 0:
                    gap_pct = abs((open_price - prev_close) / prev_close)

                    if gap_pct > self.price_gap_threshold:
                        direction = "up" if open_price > prev_close else "down"
                        results.append(
                            ValidationResult(
                                is_valid=True,
                                level=ValidationLevel.WARNING,
                                message="Large price gap {direction}: {gap_pct*100:.2f}% "
                                "(open: ₹{open_price}, prev_close: ₹{prev_close})",
                                field="price_gap",
                                value=gap_pct,
                            )
                        )

            except (ValueError, TypeError):
                pass

        return results

    def get_validation_summary(self, results: list[ValidationResult]) -> dict[str, Any]:
        """Get summary of validation results"""
        summary = {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.is_valid),
            "failed": sum(1 for r in results if not r.is_valid),
            "by_level": {
                level.value: sum(1 for r in results if r.level == level)
                for level in ValidationLevel
            },
            "critical_issues": [
                r.message for r in results if r.level == ValidationLevel.CRITICAL
            ],
            "errors": [
                r.message
                for r in results
                if r.level == ValidationLevel.ERROR and not r.is_valid
            ],
        }

        summary["success_rate"] = (
            (summary["passed"] / summary["total_checks"]) * 100
            if summary["total_checks"] > 0
            else 0
        )

        return summary


def validate_market_data(
    ticker_data: dict[str, Any], strict: bool = True
) -> tuple[bool, dict[str, Any]]:
    """
    Convenience function for market data validation

    Args:
        ticker_data: Market data dictionary
        strict: Enable strict validation mode

    Returns:
        Tuple of (is_valid, validation_summary)
    """
    validator = MarketDataValidator(enable_strict_validation=strict)
    results = validator.validate_ticker_data(ticker_data)
    summary = validator.get_validation_summary(results)

    # Data is considered valid if no critical errors
    is_valid = summary["failed"] == 0 and len(summary["critical_issues"]) == 0

    return is_valid, summary
