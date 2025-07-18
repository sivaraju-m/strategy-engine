"""
# Portfolio Rebalancing with Transaction Cost Optimization
#
# SJ-VERIFY
# - Path: /ai-trading-machine/src/ai_trading_machine/strategies
# - Type: portfolio
# - Checks: types,sebi,costs,optimization
#
# Purpose: Portfolio rebalancing logic with transaction cost optimization, minimize brokerage and impact costs
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Reasons for portfolio rebalancing"""

    DRIFT = "DRIFT"  # Target allocation drift
    SIGNAL = "SIGNAL"  # New trading signals
    RISK = "RISK"  # Risk management
    LIQUIDITY = "LIQUIDITY"  # Liquidity requirements
    SCHEDULED = "SCHEDULED"  # Scheduled rebalancing


@dataclass
class Position:
    """Current portfolio position"""

    symbol: str
    quantity: int
    current_price: float
    market_value: float
    target_weight: float
    current_weight: float
    sector: Optional[str] = None
    liquidity_score: float = 1.0


@dataclass
class TransactionCosts:
    """Transaction cost structure"""

    brokerage_pct: float = 0.0005  # 0.05% brokerage
    stt_pct: float = 0.001  # 0.1% STT (Securities Transaction Tax)
    exchange_fee_pct: float = 0.0001  # 0.01% exchange fee
    gst_pct: float = 0.18  # 18% GST on brokerage
    impact_cost_pct: float = 0.002  # 0.2% market impact cost

    def calculate_total_cost(self, trade_value: float) -> float:
        """Calculate total transaction cost"""
        brokerage = trade_value * self.brokerage_pct
        stt = trade_value * self.stt_pct
        exchange_fee = trade_value * self.exchange_fee_pct
        gst = brokerage * self.gst_pct
        impact_cost = trade_value * self.impact_cost_pct

        return brokerage + stt + exchange_fee + gst + impact_cost


@dataclass
class TradeRecommendation:
    """Recommended trade for rebalancing"""

    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    target_value: float
    expected_cost: float
    priority: int
    reasoning: str


class PortfolioRebalancer:
    """
    Advanced portfolio rebalancing with transaction cost optimization

    Features:
    - Minimize total transaction costs
    - Maintain target allocations
    - Respect SEBI position limits
    - Optimize trade execution order
    - Consider market impact and liquidity
    """

    def __init__(
        self,
        transaction_costs: Optional[TransactionCosts] = None,
        min_trade_value: float = 1000,  # Minimum trade value ₹1000
        max_position_weight: float = 0.10,  # Max 10% per position
        rebalance_threshold: float = 0.02,  # 2% drift threshold
        max_turnover_pct: float = 0.20,  # Max 20% portfolio turnover
    ):
        """
        Initialize portfolio rebalancer

        Args:
            transaction_costs: Transaction cost structure
            min_trade_value: Minimum trade value to execute
            max_position_weight: Maximum weight per position
            rebalance_threshold: Minimum drift before rebalancing
            max_turnover_pct: Maximum portfolio turnover per rebalancing
        """
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.min_trade_value = min_trade_value
        self.max_position_weight = max_position_weight
        self.rebalance_threshold = rebalance_threshold
        self.max_turnover_pct = max_turnover_pct

        logger.info("💼 Portfolio Rebalancer initialized with cost optimization")

    def rebalance_portfolio(
        self,
        current_positions: dict[str, Position],
        target_weights: dict[str, float],
        available_cash: float,
        portfolio_value: float,
        reason: RebalanceReason = RebalanceReason.DRIFT,
    ) -> list[TradeRecommendation]:
        """
        Generate optimal rebalancing trades

        Args:
            current_positions: Current portfolio positions
            target_weights: Target allocation weights
            available_cash: Available cash for investment
            portfolio_value: Total portfolio value
            reason: Reason for rebalancing

        Returns:
            List of recommended trades sorted by priority
        """
        logger.info("🔄 Starting portfolio rebalancing: {reason.value}")

        # Validate inputs
        if not self._validate_inputs(
            current_positions, target_weights, portfolio_value
        ):
            return []

        # Calculate current allocations
        current_allocations = self._calculate_current_allocations(
            current_positions, portfolio_value
        )

        # Identify positions that need rebalancing
        drift_analysis = self._analyze_allocation_drift(
            current_allocations, target_weights
        )

        # Generate trade recommendations
        trade_recommendations = self._generate_trade_recommendations(
            current_positions,
            target_weights,
            portfolio_value,
            available_cash,
            drift_analysis,
        )

        # Optimize trade execution order
        optimized_trades = self._optimize_trade_execution(trade_recommendations)

        # Apply portfolio constraints
        final_trades = self._apply_portfolio_constraints(
            optimized_trades, portfolio_value
        )

        logger.info("✅ Generated {len(final_trades)} rebalancing trades")
        return final_trades

    def _validate_inputs(
        self,
        positions: dict[str, Position],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> bool:
        """Validate input parameters"""

        # Check portfolio value
        if portfolio_value <= 0:
            logger.error("❌ Invalid portfolio value")
            return False

        # Check target weights sum to ~1.0
        total_weight = sum(target_weights.values())
        if abs(total_weight - 1.0) > 0.05:  # 5% tolerance
            logger.warning("⚠️ Target weights sum to {total_weight:.3f}, not 1.0")

        # Check individual position limits
        for symbol, weight in target_weights.items():
            if weight > self.max_position_weight:
                logger.error(
                    "❌ Target weight for {symbol} ({weight:.2%}) exceeds limit ({self.max_position_weight:.2%})"
                )
                return False

        return True

    def _calculate_current_allocations(
        self, positions: dict[str, Position], portfolio_value: float
    ) -> dict[str, float]:
        """Calculate current position allocations"""
        allocations = {}

        for symbol, position in positions.items():
            allocations[symbol] = position.market_value / portfolio_value

        return allocations

    def _analyze_allocation_drift(
        self, current_allocations: dict[str, float], target_weights: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Analyze allocation drift from targets"""
        drift_analysis = {}

        # Get all symbols (current + target)
        all_symbols = set(current_allocations.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_weight = current_allocations.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            drift = current_weight - target_weight

            drift_analysis[symbol] = {
                "current_weight": current_weight,
                "target_weight": target_weight,
                "drift": drift,
                "drift_pct": abs(drift) / max(target_weight, 0.01),
                "needs_rebalance": abs(drift) > self.rebalance_threshold,
            }

        return drift_analysis

    def _generate_trade_recommendations(
        self,
        current_positions: dict[str, Position],
        target_weights: dict[str, float],
        portfolio_value: float,
        available_cash: float,
        drift_analysis: dict[str, dict[str, float]],
    ) -> list[TradeRecommendation]:
        """Generate trade recommendations based on drift analysis"""
        recommendations = []

        for symbol, analysis in drift_analysis.items():
            if not analysis["needs_rebalance"]:
                continue

            current_position = current_positions.get(symbol)
            target_value = analysis["target_weight"] * portfolio_value

            if current_position:
                current_value = current_position.market_value
                trade_value = target_value - current_value

                if abs(trade_value) < self.min_trade_value:
                    continue

                if trade_value > 0:
                    # Need to buy more
                    action = "BUY"
                    quantity = int(trade_value / current_position.current_price)
                else:
                    # Need to sell
                    action = "SELL"
                    quantity = int(abs(trade_value) / current_position.current_price)
                    quantity = min(
                        quantity, current_position.quantity
                    )  # Can't sell more than we have

                expected_cost = self.transaction_costs.calculate_total_cost(
                    abs(trade_value)
                )

            else:
                # New position
                if target_value < self.min_trade_value:
                    continue

                action = "BUY"
                # Estimate price (would need market data in real implementation)
                estimated_price = 100.0  # Placeholder
                quantity = int(target_value / estimated_price)
                expected_cost = self.transaction_costs.calculate_total_cost(
                    target_value
                )

            # Calculate priority (higher drift = higher priority)
            priority = int(analysis["drift_pct"] * 100)

            recommendations.append(
                TradeRecommendation(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    target_value=target_value,
                    expected_cost=expected_cost,
                    priority=priority,
                    reasoning="Drift: {analysis['drift']:.3f} ({analysis['drift_pct']:.1%})",
                )
            )

        return recommendations

    def _optimize_trade_execution(
        self, trade_recommendations: list[TradeRecommendation]
    ) -> list[TradeRecommendation]:
        """Optimize trade execution order to minimize costs"""

        # Sort by priority (highest first) and cost efficiency
        optimized_trades = sorted(
            trade_recommendations,
            key=lambda x: (
                -x.priority,  # Higher priority first
                x.expected_cost / max(x.target_value, 1),  # Lower cost ratio first
                -x.target_value,  # Larger trades first
            ),
        )

        # Group sells before buys to generate cash
        sells = [t for t in optimized_trades if t.action == "SELL"]
        buys = [t for t in optimized_trades if t.action == "BUY"]

        return sells + buys

    def _apply_portfolio_constraints(
        self, trade_recommendations: list[TradeRecommendation], portfolio_value: float
    ) -> list[TradeRecommendation]:
        """Apply portfolio-level constraints"""

        # Calculate total turnover
        total_trade_value = sum(t.target_value for t in trade_recommendations)
        turnover_pct = total_trade_value / portfolio_value

        # If turnover is too high, prioritize most important trades
        if turnover_pct > self.max_turnover_pct:
            logger.warning("⚠️ High turnover {turnover_pct:.1%}, reducing trades")

            # Keep only highest priority trades within turnover limit
            cumulative_value = 0
            max_value = self.max_turnover_pct * portfolio_value

            constrained_trades = []
            for trade in trade_recommendations:
                if cumulative_value + trade.target_value <= max_value:
                    constrained_trades.append(trade)
                    cumulative_value += trade.target_value
                else:
                    break

            return constrained_trades

        return trade_recommendations

    def calculate_rebalancing_impact(
        self, trade_recommendations: list[TradeRecommendation], portfolio_value: float
    ) -> dict[str, float]:
        """Calculate the impact of rebalancing on portfolio"""

        total_trade_value = sum(t.target_value for t in trade_recommendations)
        total_costs = sum(t.expected_cost for t in trade_recommendations)

        impact = {
            "total_trades": len(trade_recommendations),
            "total_trade_value": total_trade_value,
            "total_costs": total_costs,
            "cost_pct": total_costs / portfolio_value,
            "turnover_pct": total_trade_value / portfolio_value,
            "buy_trades": len([t for t in trade_recommendations if t.action == "BUY"]),
            "sell_trades": len(
                [t for t in trade_recommendations if t.action == "SELL"]
            ),
            "avg_trade_size": (
                total_trade_value / len(trade_recommendations)
                if trade_recommendations
                else 0
            ),
        }

        return impact

    def simulate_rebalancing_outcomes(
        self,
        current_positions: dict[str, Position],
        target_weights: dict[str, float],
        portfolio_value: float,
        scenarios: list[dict[str, float]],  # Different price scenarios
    ) -> dict[str, Any]:
        """Simulate rebalancing outcomes under different scenarios"""

        results = {
            "scenarios": [],
            "avg_cost": 0,
            "max_cost": 0,
            "min_cost": float("in"),
            "risk_metrics": {},
        }

        for i, scenario in enumerate(scenarios):
            # Update position values with scenario prices
            scenario_positions = {}
            for symbol, position in current_positions.items():
                new_price = scenario.get(symbol, position.current_price)
                scenario_positions[symbol] = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    current_price=new_price,
                    market_value=position.quantity * new_price,
                    target_weight=position.target_weight,
                    current_weight=position.current_weight,
                    sector=position.sector,
                    liquidity_score=position.liquidity_score,
                )

            # Generate trades for this scenario
            trades = self.rebalance_portfolio(
                scenario_positions, target_weights, 0, portfolio_value
            )

            impact = self.calculate_rebalancing_impact(trades, portfolio_value)

            results["scenarios"].append(
                {
                    "scenario_id": i,
                    "trades": len(trades),
                    "cost": impact["total_costs"],
                    "turnover": impact["turnover_pct"],
                }
            )

        # Calculate summary statistics
        costs = [s["cost"] for s in results["scenarios"]]
        if costs:
            results["avg_cost"] = sum(costs) / len(costs)
            results["max_cost"] = max(costs)
            results["min_cost"] = min(costs)

        return results


def optimize_rebalancing(
    current_positions: dict[str, Any],
    target_weights: dict[str, float],
    portfolio_value: float,
    available_cash: float = 0,
    min_trade_value: float = 1000,
) -> list[dict[str, Any]]:
    """
    Convenience function for portfolio rebalancing optimization

    Args:
        current_positions: Dictionary of current positions
        target_weights: Target allocation weights
        portfolio_value: Total portfolio value
        available_cash: Available cash for trades
        min_trade_value: Minimum trade value

    Returns:
        List of optimized trade recommendations
    """
    # Convert to Position objects
    positions = {}
    for symbol, pos_data in current_positions.items():
        positions[symbol] = Position(
            symbol=symbol,
            quantity=pos_data.get("quantity", 0),
            current_price=pos_data.get("price", 0),
            market_value=pos_data.get("value", 0),
            target_weight=target_weights.get(symbol, 0),
            current_weight=pos_data.get("value", 0) / portfolio_value,
            sector=pos_data.get("sector"),
            liquidity_score=pos_data.get("liquidity_score", 1.0),
        )

    rebalancer = PortfolioRebalancer(min_trade_value=min_trade_value)

    trades = rebalancer.rebalance_portfolio(
        positions, target_weights, available_cash, portfolio_value
    )

    # Convert back to dictionaries
    return [
        {
            "symbol": trade.symbol,
            "action": trade.action,
            "quantity": trade.quantity,
            "target_value": trade.target_value,
            "expected_cost": trade.expected_cost,
            "priority": trade.priority,
            "reasoning": trade.reasoning,
        }
        for trade in trades
    ]
