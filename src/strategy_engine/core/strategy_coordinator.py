"""
Strategy Coordinator for Strategy Engine
========================================

This module coordinates strategy execution across multiple strategies and symbols.
It manages strategy lifecycle, execution scheduling, and resource allocation.

It provides a unified interface for registering, executing, and monitoring strategies,
including error handling, recovery, and performance tracking.

"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class StrategyCoordinator:
    """Coordinates strategy execution and resource management."""

    def __init__(self):
        """Initialize the strategy coordinator."""
        self.initialized = False
        self.registered_strategies = {}
        self.active_executions = {}
        self.execution_history = []

    async def initialize(self) -> bool:
        """Initialize the strategy coordinator."""
        try:
            logger.info("🔧 Initializing Strategy Coordinator...")

            # Initialize coordination components
            self.initialized = True

            logger.info("✅ Strategy Coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Strategy Coordinator: {e}")
            return False

    async def register_strategy(
        self, strategy_name: str, config: Dict[str, Any]
    ) -> bool:
        """Register a strategy with the coordinator."""
        try:
            self.registered_strategies[strategy_name] = {
                "config": config,
                "registered_at": datetime.now().isoformat(),
                "execution_count": 0,
                "last_execution": None,
            }

            logger.info(f"✅ Strategy registered: {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to register strategy {strategy_name}: {e}")
            return False

    async def execute_strategy(
        self, strategy_name: str, symbol: str, signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a strategy with given signals.

        Args:
            strategy_name: Name of the strategy to execute
            symbol: Trading symbol
            signals: List of trading signals

        Returns:
            Execution result dictionary
        """
        execution_id = f"{strategy_name}_{symbol}_{datetime.now().strftime('%H%M%S')}"

        try:
            logger.debug(f"Executing strategy: {strategy_name} for {symbol}")

            # Track execution
            execution_start = datetime.now()
            self.active_executions[execution_id] = {
                "strategy": strategy_name,
                "symbol": symbol,
                "signals": signals,
                "start_time": execution_start.isoformat(),
                "status": "RUNNING",
            }

            # Simulate strategy execution (replace with actual logic)
            await asyncio.sleep(0.1)  # Simulate processing time

            # Execution result
            result = {
                "success": True,
                "execution_id": execution_id,
                "strategy": strategy_name,
                "symbol": symbol,
                "signals_processed": len(signals),
                "execution_time": (datetime.now() - execution_start).total_seconds(),
                "signals_generated": len(signals),
                "performance_metrics": {
                    "signal_quality": 0.85,
                    "execution_speed": "fast",
                    "risk_score": 0.3,
                },
            }

            # Update tracking
            self.active_executions[execution_id]["status"] = "COMPLETED"
            self.active_executions[execution_id]["result"] = result

            # Update strategy stats
            if strategy_name in self.registered_strategies:
                self.registered_strategies[strategy_name]["execution_count"] += 1
                self.registered_strategies[strategy_name][
                    "last_execution"
                ] = datetime.now().isoformat()

            # Archive execution
            self.execution_history.append(self.active_executions[execution_id])
            del self.active_executions[execution_id]

            logger.debug(f"Strategy execution completed: {execution_id}")
            return result

        except Exception as e:
            logger.error(f"Strategy execution failed for {execution_id}: {e}")

            # Update execution status
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "FAILED"
                self.active_executions[execution_id]["error"] = str(e)

            return {
                "success": False,
                "execution_id": execution_id,
                "error": str(e),
                "strategy": strategy_name,
                "symbol": symbol,
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the coordinator."""
        return {
            "healthy": True,
            "initialized": self.initialized,
            "registered_strategies": len(self.registered_strategies),
            "active_executions": len(self.active_executions),
            "total_executions": len(self.execution_history),
            "timestamp": datetime.now().isoformat(),
        }

    async def recover(self) -> bool:
        """Attempt to recover from errors."""
        try:
            logger.info("🔄 Attempting coordinator recovery...")

            # Clear failed executions
            failed_executions = [
                eid
                for eid, execution in self.active_executions.items()
                if execution.get("status") == "FAILED"
            ]

            for execution_id in failed_executions:
                del self.active_executions[execution_id]

            logger.info(
                f"✅ Recovery completed, cleared {len(failed_executions)} failed executions"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")
            return False

    async def clear_cache(self) -> None:
        """Clear coordinator cache and temporary data."""
        try:
            # Archive any remaining active executions
            for execution_id, execution in self.active_executions.items():
                self.execution_history.append(execution)

            self.active_executions.clear()

            logger.debug("🧹 Coordinator cache cleared")

        except Exception as e:
            logger.error(f"❌ Cache clearing failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the coordinator gracefully."""
        try:
            logger.info("🛑 Shutting down Strategy Coordinator...")

            # Wait for active executions to complete
            if self.active_executions:
                logger.info(
                    f"Waiting for {len(self.active_executions)} active executions..."
                )
                await asyncio.sleep(1)  # Give some time for completion

            # Clear remaining data
            await self.clear_cache()

            self.initialized = False

            logger.info("✅ Strategy Coordinator shutdown completed")

        except Exception as e:
            logger.error(f"❌ Coordinator shutdown error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "registered_strategies": list(self.registered_strategies.keys()),
            "strategy_count": len(self.registered_strategies),
            "active_executions": len(self.active_executions),
            "total_executions_completed": len(self.execution_history),
            "execution_history": self.execution_history[-10:],  # Last 10 executions
            "uptime": "active" if self.initialized else "inactive",
        }
