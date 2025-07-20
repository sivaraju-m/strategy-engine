"""
Execution cost and slippage modeling for realistic backtesting.
Models transaction costs, market impact, slippage, and tax implications.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, time
from enum import Enum


class BrokerType(Enum):
    """Broker type enumeration for fee calculation."""

    ZERODHA = "zerodha"
    UPSTOX = "upstox"
    ICICI = "icici_direct"
    HDFC = "hdfc_sec"
    CUSTOM = "custom"


class ExecutionType(Enum):
    """Execution type enumeration for slippage calculation."""

    MARKET = "market"
    LIMIT = "limit"
    SL_MARKET = "sl_market"
    SL_LIMIT = "sl_limit"


class ExecutionTimeOfDay(Enum):
    """Time of day enumeration for slippage calculation."""

    OPEN = "open"
    CLOSE = "close"
    INTRADAY = "intraday"


class TransactionCostModel:
    """
    Models transaction costs including brokerage, exchange fees, and taxes.

    Attributes:
        broker_type: Type of broker for fee calculation
        custom_fees: Custom fee structure if broker_type is CUSTOM
        include_stt: Whether to include Securities Transaction Tax
        include_gst: Whether to include GST
    """

    def __init__(
        self,
        broker_type: BrokerType = BrokerType.ZERODHA,
        custom_fees: Optional[Dict[str, float]] = None,
        include_stt: bool = True,
        include_gst: bool = True,
    ):
        """
        Initialize the transaction cost model.

        Args:
            broker_type: Type of broker for fee calculation
            custom_fees: Custom fee structure if broker_type is CUSTOM
            include_stt: Whether to include Securities Transaction Tax
            include_gst: Whether to include GST
        """
        self.broker_type = broker_type
        self.custom_fees = custom_fees or {}
        self.include_stt = include_stt
        self.include_gst = include_gst

        # Standard fee structures
        self.fee_structures = {
            BrokerType.ZERODHA: {
                "brokerage_equity_delivery": 0.0,  # Zero for delivery
                "brokerage_equity_intraday": 0.03,  # 0.03% or Rs 20 per executed order, whichever is lower
                "brokerage_equity_intraday_cap": 20,  # Cap at Rs 20
                "exchange_transaction_charge": 0.00035,  # 0.035% NSE/BSE
                "clearing_charges": 0.0,
                "sebi_charges": 0.000001,  # Rs 10 per crore
                "stamp_duty": 0.00015,  # Varies by state, using Maharashtra
            },
            BrokerType.UPSTOX: {
                "brokerage_equity_delivery": 0.0,
                "brokerage_equity_intraday": 0.05,
                "brokerage_equity_intraday_cap": 20,
                "exchange_transaction_charge": 0.00035,
                "clearing_charges": 0.0,
                "sebi_charges": 0.000001,
                "stamp_duty": 0.00015,
            },
            BrokerType.ICICI: {
                "brokerage_equity_delivery": 0.0,
                "brokerage_equity_intraday": 0.0275,
                "brokerage_equity_intraday_cap": 20,
                "exchange_transaction_charge": 0.00035,
                "clearing_charges": 0.0,
                "sebi_charges": 0.000001,
                "stamp_duty": 0.00015,
            },
            BrokerType.HDFC: {
                "brokerage_equity_delivery": 0.0,
                "brokerage_equity_intraday": 0.05,
                "brokerage_equity_intraday_cap": 20,
                "exchange_transaction_charge": 0.00035,
                "clearing_charges": 0.0,
                "sebi_charges": 0.000001,
                "stamp_duty": 0.00015,
            },
        }

        # Tax rates
        self.stt_buy_delivery = 0.001  # 0.1%
        self.stt_sell_delivery = 0.001  # 0.1%
        self.stt_intraday = 0.00025  # 0.025%
        self.gst_rate = 0.18  # 18%

    def calculate_cost(
        self, price: float, quantity: int, is_buy: bool = True, is_delivery: bool = True
    ) -> Dict[str, float]:
        """
        Calculate transaction costs for a trade.

        Args:
            price: Execution price
            quantity: Number of shares
            is_buy: Whether it's a buy (True) or sell (False)
            is_delivery: Whether it's delivery (True) or intraday (False)

        Returns:
            Dictionary of cost components and total cost

        Raises:
            ValueError: If inputs are invalid
        """
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive")

        # Get fee structure based on broker type
        if self.broker_type == BrokerType.CUSTOM:
            fees = self.custom_fees
        else:
            fees = self.fee_structures.get(
                self.broker_type, self.fee_structures[BrokerType.ZERODHA]
            )

        # Calculate trade value
        trade_value = price * quantity

        # Calculate brokerage
        if is_delivery:
            brokerage = trade_value * fees.get("brokerage_equity_delivery", 0.0)
        else:
            brokerage = min(
                trade_value * fees.get("brokerage_equity_intraday", 0.03),
                fees.get("brokerage_equity_intraday_cap", 20),
            )

        # Calculate exchange transaction charge
        exchange_charge = trade_value * fees.get("exchange_transaction_charge", 0.00035)

        # Calculate SEBI charges
        sebi_charge = trade_value * fees.get("sebi_charges", 0.000001)

        # Calculate stamp duty (only for buy orders)
        stamp_duty = trade_value * fees.get("stamp_duty", 0.00015) if is_buy else 0.0

        # Calculate STT (Securities Transaction Tax)
        stt = 0.0
        if self.include_stt:
            if is_delivery:
                stt = trade_value * (
                    self.stt_buy_delivery if is_buy else self.stt_sell_delivery
                )
            else:
                # STT is charged only on sells for intraday
                stt = trade_value * (0.0 if is_buy else self.stt_intraday)

        # Calculate GST (on brokerage and exchange charges)
        gst = 0.0
        if self.include_gst:
            gst = (brokerage + exchange_charge) * self.gst_rate

        # Calculate total cost
        total_cost = brokerage + exchange_charge + sebi_charge + stamp_duty + stt + gst

        # Return cost breakdown
        return {
            "trade_value": trade_value,
            "brokerage": brokerage,
            "exchange_charge": exchange_charge,
            "sebi_charge": sebi_charge,
            "stamp_duty": stamp_duty,
            "stt": stt,
            "gst": gst,
            "total_cost": total_cost,
            "cost_percentage": (
                (total_cost / trade_value) * 100 if trade_value > 0 else 0
            ),
        }


class SlippageModel:
    """
    Models slippage based on order size, liquidity, volatility, and time of day.

    Attributes:
        base_slippage: Base slippage rate
        volatility_factor: Impact of volatility on slippage
        volume_factor: Impact of volume on slippage
        time_factors: Time of day impact on slippage
        execution_type_factors: Execution type impact on slippage
    """

    def __init__(
        self,
        base_slippage: float = 0.0005,  # 5 basis points
        volatility_factor: float = 0.5,
        volume_factor: float = 0.5,
        time_factors: Optional[Dict[ExecutionTimeOfDay, float]] = None,
        execution_type_factors: Optional[Dict[ExecutionType, float]] = None,
    ):
        """
        Initialize the slippage model.

        Args:
            base_slippage: Base slippage rate as decimal
            volatility_factor: Impact of volatility on slippage
            volume_factor: Impact of volume on slippage
            time_factors: Time of day impact factors
            execution_type_factors: Execution type impact factors
        """
        self.base_slippage = base_slippage
        self.volatility_factor = volatility_factor
        self.volume_factor = volume_factor

        # Default time factors
        self.time_factors = time_factors or {
            ExecutionTimeOfDay.OPEN: 1.5,  # 50% more at open
            ExecutionTimeOfDay.CLOSE: 1.3,  # 30% more at close
            ExecutionTimeOfDay.INTRADAY: 1.0,  # Base during day
        }

        # Default execution type factors
        self.execution_type_factors = execution_type_factors or {
            ExecutionType.MARKET: 1.0,  # Base for market orders
            ExecutionType.LIMIT: 0.5,  # 50% less for limit orders
            ExecutionType.SL_MARKET: 1.2,  # 20% more for stop-loss market
            ExecutionType.SL_LIMIT: 0.7,  # 30% less for stop-loss limit
        }

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        average_volume: int,
        volatility: float,
        execution_type: ExecutionType = ExecutionType.MARKET,
        time_of_day: ExecutionTimeOfDay = ExecutionTimeOfDay.INTRADAY,
        is_buy: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate expected slippage for a trade.

        Args:
            price: Execution price
            quantity: Number of shares
            average_volume: Average daily volume
            volatility: Historical volatility (std of daily returns)
            execution_type: Type of order execution
            time_of_day: Time of day for execution
            is_buy: Whether it's a buy (True) or sell (False)

        Returns:
            Dictionary with slippage information

        Raises:
            ValueError: If inputs are invalid
        """
        if price <= 0 or quantity <= 0 or average_volume <= 0:
            raise ValueError("Price, quantity, and volume must be positive")

        # Calculate order's percentage of average volume
        volume_impact = min(quantity / average_volume, 1.0)

        # Calculate volatility impact (normalized to 0-1 range)
        volatility_impact = min(volatility / 0.05, 1.0)  # Assume 5% vol is high

        # Get time factor
        time_factor = self.time_factors.get(time_of_day, 1.0)

        # Get execution type factor
        execution_factor = self.execution_type_factors.get(execution_type, 1.0)

        # Calculate total slippage
        slippage_rate = (
            self.base_slippage
            * (1 + self.volume_factor * volume_impact)
            * (1 + self.volatility_factor * volatility_impact)
            * time_factor
            * execution_factor
        )

        # Adjust direction for buys/sells
        if not is_buy:
            slippage_rate = -slippage_rate

        # Calculate slippage amount
        slippage_amount = price * slippage_rate
        slipped_price = price + slippage_amount

        return {
            "original_price": price,
            "slippage_rate": slippage_rate,
            "slippage_amount": slippage_amount,
            "slippage_percentage": slippage_rate * 100,
            "final_price": slipped_price,
            "volume_impact": volume_impact,
            "volatility_impact": volatility_impact,
            "time_factor": time_factor,
            "execution_factor": execution_factor,
        }


class RealisticExecutionModel:
    """
    Combined model for realistic execution with costs and slippage.

    Attributes:
        cost_model: Transaction cost model
        slippage_model: Slippage model
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        slippage_model: Optional[SlippageModel] = None,
    ):
        """
        Initialize the realistic execution model.

        Args:
            cost_model: Transaction cost model
            slippage_model: Slippage model
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.slippage_model = slippage_model or SlippageModel()

    def calculate_realistic_execution(
        self,
        price: float,
        quantity: int,
        average_volume: int,
        volatility: float,
        is_buy: bool = True,
        is_delivery: bool = True,
        execution_type: ExecutionType = ExecutionType.MARKET,
        time_of_day: ExecutionTimeOfDay = ExecutionTimeOfDay.INTRADAY,
    ) -> Dict[str, Any]:
        """
        Calculate realistic execution with costs and slippage.

        Args:
            price: Execution price
            quantity: Number of shares
            average_volume: Average daily volume
            volatility: Historical volatility
            is_buy: Whether it's a buy (True) or sell (False)
            is_delivery: Whether it's delivery (True) or intraday (False)
            execution_type: Type of order execution
            time_of_day: Time of day for execution

        Returns:
            Dictionary with execution details

        Raises:
            ValueError: If inputs are invalid
        """
        # Calculate slippage
        slippage_result = self.slippage_model.calculate_slippage(
            price=price,
            quantity=quantity,
            average_volume=average_volume,
            volatility=volatility,
            execution_type=execution_type,
            time_of_day=time_of_day,
            is_buy=is_buy,
        )

        # Get execution price with slippage
        execution_price = slippage_result["final_price"]

        # Calculate transaction costs
        cost_result = self.cost_model.calculate_cost(
            price=execution_price,
            quantity=quantity,
            is_buy=is_buy,
            is_delivery=is_delivery,
        )

        # Calculate total impact
        initial_value = price * quantity
        final_value = (execution_price * quantity) - cost_result["total_cost"]
        total_impact = (
            (final_value - initial_value) / initial_value if initial_value > 0 else 0
        )

        # Combine results
        return {
            "original_price": price,
            "execution_price": execution_price,
            "quantity": quantity,
            "is_buy": is_buy,
            "is_delivery": is_delivery,
            "execution_type": execution_type.value,
            "time_of_day": time_of_day.value,
            "slippage": slippage_result,
            "costs": cost_result,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_impact_percentage": total_impact * 100,
        }

    def apply_execution_costs_to_backtest(
        self,
        df: pd.DataFrame,
        signals: List[int],
        position_sizes: List[int],
        average_volume: int,
        volatility: float,
        is_delivery: bool = True,
        execution_type: ExecutionType = ExecutionType.MARKET,
        time_of_day: ExecutionTimeOfDay = ExecutionTimeOfDay.INTRADAY,
    ) -> Dict[str, Any]:
        """
        Apply realistic execution costs to a backtest.

        Args:
            df: OHLCV DataFrame
            signals: List of signals (-1, 0, 1)
            position_sizes: List of position sizes (number of shares)
            average_volume: Average daily volume
            volatility: Historical volatility
            is_delivery: Whether it's delivery (True) or intraday (False)
            execution_type: Type of order execution
            time_of_day: Time of day for execution

        Returns:
            Dictionary with modified backtest results

        Raises:
            ValueError: If inputs are invalid
        """
        if len(signals) != len(position_sizes):
            raise ValueError("Signals and position sizes must have same length")

        if len(signals) > len(df) - 1:
            raise ValueError("Too many signals for DataFrame length")

        prices = df["close"].values[: len(signals)]

        # Initialize results
        original_returns = []
        adjusted_returns = []
        slippage_impacts = []
        cost_impacts = []
        total_impacts = []

        # Track positions
        current_position = 0

        for i in range(len(signals)):
            signal = signals[i]

            # Skip if no change in position
            if signal == 0 and current_position == 0:
                original_returns.append(0)
                adjusted_returns.append(0)
                slippage_impacts.append(0)
                cost_impacts.append(0)
                total_impacts.append(0)
                continue

            # Calculate position change
            position_size = position_sizes[i]
            price = prices[i]
            next_price = prices[i + 1] if i < len(prices) - 1 else price

            # If signal indicates position change
            if signal != 0 or current_position != 0:
                # Determine if buying or selling
                is_buy = signal > 0

                if signal == 0 and current_position != 0:
                    # Closing position
                    is_buy = current_position < 0
                    position_size = abs(current_position)

                # Calculate realistic execution
                execution_result = self.calculate_realistic_execution(
                    price=price,
                    quantity=position_size,
                    average_volume=average_volume,
                    volatility=volatility,
                    is_buy=is_buy,
                    is_delivery=is_delivery,
                    execution_type=execution_type,
                    time_of_day=time_of_day,
                )

                # Calculate returns
                original_return = (next_price - price) / price * signal

                # Adjust for execution costs
                slippage_impact = (
                    execution_result["slippage"]["slippage_percentage"] / 100
                )
                cost_impact = execution_result["costs"]["cost_percentage"] / 100
                total_impact = execution_result["total_impact_percentage"] / 100

                # Adjusted return accounts for execution impacts
                adjusted_return = original_return - (total_impact * abs(signal))

                # Save results
                original_returns.append(original_return)
                adjusted_returns.append(adjusted_return)
                slippage_impacts.append(slippage_impact)
                cost_impacts.append(cost_impact)
                total_impacts.append(total_impact)

                # Update position
                current_position = signal * position_size
            else:
                # No trade
                original_returns.append(0)
                adjusted_returns.append(0)
                slippage_impacts.append(0)
                cost_impacts.append(0)
                total_impacts.append(0)

        # Calculate performance metrics
        original_cumulative = np.cumprod(1 + np.array(original_returns))
        adjusted_cumulative = np.cumprod(1 + np.array(adjusted_returns))

        # Sharpe ratio calculation
        original_sharpe = (
            np.mean(original_returns) / np.std(original_returns) * np.sqrt(252)
            if np.std(original_returns) > 0
            else 0
        )
        adjusted_sharpe = (
            np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(252)
            if np.std(adjusted_returns) > 0
            else 0
        )

        # Calculate drawdowns
        original_max_drawdown = self._calculate_max_drawdown(original_returns)
        adjusted_max_drawdown = self._calculate_max_drawdown(adjusted_returns)

        return {
            "original_returns": original_returns,
            "adjusted_returns": adjusted_returns,
            "original_cumulative": original_cumulative.tolist(),
            "adjusted_cumulative": adjusted_cumulative.tolist(),
            "slippage_impacts": slippage_impacts,
            "cost_impacts": cost_impacts,
            "total_impacts": total_impacts,
            "original_sharpe": original_sharpe,
            "adjusted_sharpe": adjusted_sharpe,
            "original_total_return": (
                original_cumulative[-1] - 1 if len(original_cumulative) > 0 else 0
            ),
            "adjusted_total_return": (
                adjusted_cumulative[-1] - 1 if len(adjusted_cumulative) > 0 else 0
            ),
            "original_max_drawdown": original_max_drawdown,
            "adjusted_max_drawdown": adjusted_max_drawdown,
            "average_slippage_impact": np.mean([i for i in slippage_impacts if i != 0]),
            "average_cost_impact": np.mean([i for i in cost_impacts if i != 0]),
            "average_total_impact": np.mean([i for i in total_impacts if i != 0]),
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from returns.

        Args:
            returns: List of return values

        Returns:
            Maximum drawdown as a positive value
        """
        # Convert returns to cumulative equity curve
        cumulative = np.cumprod(1 + np.array(returns))

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Return maximum drawdown (as a positive value)
        return abs(min(drawdown)) if len(drawdown) > 0 else 0
