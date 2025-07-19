"""
Performance Tracker for Strategy Engine
=======================================

This module tracks strategy performance metrics, execution statistics,
and provides analytics for strategy optimization.

It provides a unified interface for recording, analyzing, and reporting
performance metrics across different strategies and trading symbols.
It supports real-time performance tracking, daily summaries, and historical
data archiving for comprehensive performance analysis.
It also includes utilities for resetting daily metrics, archiving data,
and calculating derived metrics such as success rates and average execution times.  
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks and analyzes strategy performance metrics."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.initialized = False
        self.daily_metrics = {}
        self.execution_metrics = []
        self.strategy_performance = {}
        
    async def initialize(self) -> None:
        """Initialize the performance tracker."""
        try:
            logger.info("🔧 Initializing Performance Tracker...")
            
            # Load previous metrics if available
            await self._load_previous_metrics()
            
            self.initialized = True
            
            logger.info("✅ Performance Tracker initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Performance Tracker: {e}")
    
    async def _load_previous_metrics(self) -> None:
        """Load previous performance metrics from storage."""
        try:
            # Check if previous day's metrics exist
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            metrics_file = f"data/performance/daily_metrics_{yesterday}.json"
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    previous_metrics = json.load(f)
                
                # Use previous metrics for baseline comparison
                self.baseline_metrics = previous_metrics
                logger.info(f"📊 Loaded baseline metrics from {metrics_file}")
            else:
                self.baseline_metrics = {}
                
        except Exception as e:
            logger.debug(f"No previous metrics loaded: {e}")
            self.baseline_metrics = {}
    
    async def record_execution(self, strategy_name: str, symbol: str, 
                             execution_result: Dict[str, Any]) -> None:
        """Record strategy execution metrics."""
        try:
            timestamp = datetime.now()
            
            # Create execution record
            execution_record = {
                'timestamp': timestamp.isoformat(),
                'strategy': strategy_name,
                'symbol': symbol,
                'execution_time': execution_result.get('execution_time', 0),
                'signals_generated': execution_result.get('signals_generated', 0),
                'success': execution_result.get('success', False),
                'performance_metrics': execution_result.get('performance_metrics', {})
            }
            
            # Add to execution metrics
            self.execution_metrics.append(execution_record)
            
            # Update strategy-specific performance
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'total_signals': 0,
                    'total_execution_time': 0.0,
                    'symbols_traded': set(),
                    'first_execution': timestamp.isoformat(),
                    'last_execution': timestamp.isoformat()
                }
            
            strategy_stats = self.strategy_performance[strategy_name]
            strategy_stats['total_executions'] += 1
            if execution_result.get('success', False):
                strategy_stats['successful_executions'] += 1
            
            strategy_stats['total_signals'] += execution_result.get('signals_generated', 0)
            strategy_stats['total_execution_time'] += execution_result.get('execution_time', 0)
            strategy_stats['symbols_traded'].add(symbol)
            strategy_stats['last_execution'] = timestamp.isoformat()
            
            # Update daily metrics
            date_key = timestamp.strftime("%Y-%m-%d")
            if date_key not in self.daily_metrics:
                self.daily_metrics[date_key] = {
                    'executions': 0,
                    'successful_executions': 0,
                    'total_signals': 0,
                    'strategies_used': set(),
                    'symbols_traded': set(),
                    'total_execution_time': 0.0
                }
            
            daily_stats = self.daily_metrics[date_key]
            daily_stats['executions'] += 1
            if execution_result.get('success', False):
                daily_stats['successful_executions'] += 1
            
            daily_stats['total_signals'] += execution_result.get('signals_generated', 0)
            daily_stats['strategies_used'].add(strategy_name)
            daily_stats['symbols_traded'].add(symbol)
            daily_stats['total_execution_time'] += execution_result.get('execution_time', 0)
            
            logger.debug(f"📊 Recorded execution metrics for {strategy_name}:{symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution metrics: {e}")
    
    async def calculate_daily_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive daily performance metrics."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            if today not in self.daily_metrics:
                return {
                    'date': today,
                    'no_executions': True,
                    'message': 'No executions recorded for today'
                }
            
            daily_stats = self.daily_metrics[today]
            
            # Convert sets to lists for JSON serialization
            daily_stats_clean = {
                'executions': daily_stats['executions'],
                'successful_executions': daily_stats['successful_executions'],
                'total_signals': daily_stats['total_signals'],
                'strategies_used': list(daily_stats['strategies_used']),
                'symbols_traded': list(daily_stats['symbols_traded']),
                'total_execution_time': daily_stats['total_execution_time']
            }
            
            # Calculate derived metrics
            success_rate = (
                (daily_stats['successful_executions'] / daily_stats['executions'] * 100)
                if daily_stats['executions'] > 0 else 0
            )
            
            avg_execution_time = (
                daily_stats['total_execution_time'] / daily_stats['executions']
                if daily_stats['executions'] > 0 else 0
            )
            
            avg_signals_per_execution = (
                daily_stats['total_signals'] / daily_stats['executions']
                if daily_stats['executions'] > 0 else 0
            )
            
            # Find top performing strategy
            top_strategy = self._get_top_performing_strategy()
            
            performance = {
                'date': today,
                'daily_stats': daily_stats_clean,
                'derived_metrics': {
                    'success_rate': round(success_rate, 2),
                    'avg_execution_time': round(avg_execution_time, 4),
                    'avg_signals_per_execution': round(avg_signals_per_execution, 2)
                },
                'top_strategy': top_strategy,
                'strategy_breakdown': self._get_strategy_breakdown(),
                'timestamp': datetime.now().isoformat()
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate daily performance: {e}")
            return {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_top_performing_strategy(self) -> str:
        """Get the top performing strategy based on success rate."""
        try:
            if not self.strategy_performance:
                return "N/A"
            
            best_strategy = ""
            best_rate = 0
            
            for strategy, stats in self.strategy_performance.items():
                if stats['total_executions'] > 0:
                    success_rate = stats['successful_executions'] / stats['total_executions']
                    if success_rate > best_rate:
                        best_rate = success_rate
                        best_strategy = strategy
            
            return best_strategy or "N/A"
            
        except Exception:
            return "N/A"
    
    def _get_strategy_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of performance by strategy."""
        try:
            breakdown = {}
            
            for strategy, stats in self.strategy_performance.items():
                # Convert set to list for JSON serialization
                symbols_list = list(stats['symbols_traded']) if isinstance(stats['symbols_traded'], set) else stats['symbols_traded']
                
                breakdown[strategy] = {
                    'executions': stats['total_executions'],
                    'success_rate': (
                        stats['successful_executions'] / stats['total_executions'] * 100
                        if stats['total_executions'] > 0 else 0
                    ),
                    'total_signals': stats['total_signals'],
                    'avg_execution_time': (
                        stats['total_execution_time'] / stats['total_executions']
                        if stats['total_executions'] > 0 else 0
                    ),
                    'symbols_count': len(symbols_list),
                    'last_execution': stats['last_execution']
                }
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Strategy breakdown calculation failed: {e}")
            return {}
    
    async def reset_daily_metrics(self) -> None:
        """Reset daily metrics for a new trading day."""
        try:
            # Archive current day's metrics before reset
            await self.archive_daily_data()
            
            # Reset for new day
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_metrics = {
                today: {
                    'executions': 0,
                    'successful_executions': 0,
                    'total_signals': 0,
                    'strategies_used': set(),
                    'symbols_traded': set(),
                    'total_execution_time': 0.0
                }
            }
            
            logger.info("🔄 Daily metrics reset for new trading day")
            
        except Exception as e:
            logger.error(f"❌ Failed to reset daily metrics: {e}")
    
    async def archive_daily_data(self) -> None:
        """Archive daily performance data."""
        try:
            today = datetime.now().strftime("%Y%m%d")
            
            # Create performance data directory
            os.makedirs("data/performance", exist_ok=True)
            
            # Prepare data for archiving (convert sets to lists)
            archive_data = {
                'daily_metrics': {},
                'strategy_performance': {},
                'execution_count': len(self.execution_metrics),
                'archived_at': datetime.now().isoformat()
            }
            
            # Convert daily metrics
            for date, metrics in self.daily_metrics.items():
                archive_data['daily_metrics'][date] = {
                    'executions': metrics['executions'],
                    'successful_executions': metrics['successful_executions'],
                    'total_signals': metrics['total_signals'],
                    'strategies_used': list(metrics['strategies_used']),
                    'symbols_traded': list(metrics['symbols_traded']),
                    'total_execution_time': metrics['total_execution_time']
                }
            
            # Convert strategy performance
            for strategy, stats in self.strategy_performance.items():
                archive_data['strategy_performance'][strategy] = {
                    'total_executions': stats['total_executions'],
                    'successful_executions': stats['successful_executions'],
                    'total_signals': stats['total_signals'],
                    'total_execution_time': stats['total_execution_time'],
                    'symbols_traded': list(stats['symbols_traded']),
                    'first_execution': stats['first_execution'],
                    'last_execution': stats['last_execution']
                }
            
            # Save to file
            filename = f"data/performance/daily_metrics_{today}.json"
            with open(filename, 'w') as f:
                json.dump(archive_data, f, indent=2)
            
            logger.info(f"📦 Daily data archived: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to archive daily data: {e}")
    
    async def finalize(self) -> None:
        """Finalize performance tracking and save data."""
        try:
            logger.info("🔚 Finalizing Performance Tracker...")
            
            # Archive current session data
            await self.archive_daily_data()
            
            # Save execution metrics
            if self.execution_metrics:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("data/executions", exist_ok=True)
                
                filename = f"data/executions/execution_metrics_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(self.execution_metrics, f, indent=2)
                
                logger.info(f"💾 Execution metrics saved: {filename}")
            
            logger.info("✅ Performance Tracker finalized")
            
        except Exception as e:
            logger.error(f"❌ Performance Tracker finalization failed: {e}")
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time performance statistics."""
        try:
            total_executions = len(self.execution_metrics)
            successful_executions = sum(1 for exec in self.execution_metrics if exec.get('success', False))
            
            return {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                'strategies_active': len(self.strategy_performance),
                'last_execution': self.execution_metrics[-1]['timestamp'] if self.execution_metrics else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get realtime stats: {e}")
            return {'error': str(e)}
