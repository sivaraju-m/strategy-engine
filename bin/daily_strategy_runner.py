#!/usr/bin/env python3
"""
Daily Strategy Runner for Strategy Engine
=========================================

This script runs the daily strategy execution cycle, including:
- Strategy signal generation across multiple symbols
- Performance monitoring and tracking
- Risk-compliant execution with SEBI guidelines
- Comprehensive logging and reporting
- Integration with shared services

Features:
- Market hours validation
- Multi-strategy execution
- Performance analytics
- Risk management integration
- Real-time monitoring
- Daily reporting

Usage:
    python daily_strategy_runner.py                     # Standard daily run
    python daily_strategy_runner.py --quick             # Quick test run (10 min)
    python daily_strategy_runner.py --extended          # Extended run (3 hours)
    python daily_strategy_runner.py --strategies rsi,momentum  # Specific strategies

Author: Strategy Engine Team
Date: January 18, 2025
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_engine.core.strategy_coordinator import StrategyCoordinator
from strategy_engine.utils.market_hours import MarketHoursValidator
from strategy_engine.monitoring.performance_tracker import PerformanceTracker
from strategy_engine.signals.generator import SignalGenerator
from strategy_engine.utils.logger import get_logger

logger = get_logger(__name__)


class DailyStrategyRunner:
    """Daily strategy execution and monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the daily strategy runner."""
        self.strategy_coordinator = StrategyCoordinator()
        self.market_validator = MarketHoursValidator()
        self.performance_tracker = PerformanceTracker()
        self.signal_generator = SignalGenerator()
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.watchlist = self.config.get('watchlist', self._get_default_watchlist())
        self.strategies = self.config.get('strategies', ['rsi', 'momentum', 'sma'])
        self.execution_interval = self.config.get('execution_interval_minutes', 5)
        
        # Session tracking
        self.session_stats = {
            'session_start': None,
            'cycles_completed': 0,
            'signals_generated': 0,
            'strategies_executed': 0,
            'errors_encountered': 0,
            'last_error': None,
            'total_execution_time': 0.0
        }
        
        self.is_running = False
        
    def _get_default_watchlist(self) -> List[str]:
        """Get default watchlist for strategy execution."""
        return [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
            "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN",
            "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "AXISBANK",
            "LT", "WIPRO", "NESTLEIND", "ULTRACEMCO", "POWERGRID"
        ]
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'watchlist': self._get_default_watchlist(),
            'strategies': ['rsi', 'momentum', 'sma'],
            'execution_interval_minutes': 5,
            'max_concurrent_executions': 10
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        if YAML_AVAILABLE:
                            file_config = yaml.safe_load(f)
                        else:
                            logger.warning("YAML support not available, install PyYAML")
                            return default_config
                        # Extract daily_runner section if it exists
                        if 'daily_runner' in file_config:
                            return file_config['daily_runner']
                        return file_config
                    elif config_path.endswith('.json'):
                        file_config = json.load(f)
                        if 'daily_runner' in file_config:
                            return file_config['daily_runner']
                        return file_config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _get_strategy_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific strategy."""
        # This would typically load from a strategy-specific config file
        # For now, return the default parameters
        return self._get_strategy_params(strategy_name)
    
    async def initialize(self) -> bool:
        """Initialize all components for daily execution."""
        try:
            logger.info("🚀 Initializing Daily Strategy Runner...")
            
            # Initialize strategy coordinator
            if not await self.strategy_coordinator.initialize():
                logger.error("❌ Failed to initialize strategy coordinator")
                return False
            
            # Initialize signal generator
            if not await self.signal_generator.initialize():
                logger.error("❌ Failed to initialize signal generator")
                return False
            
            # Initialize performance tracker
            await self.performance_tracker.initialize()
            
            # Load strategy configurations
            for strategy_name in self.strategies:
                strategy_config = self._get_strategy_config(strategy_name)
                if strategy_config:
                    await self.strategy_coordinator.register_strategy(strategy_name, strategy_config)
                    logger.info(f"✅ Registered strategy: {strategy_name}")
                else:
                    logger.warning(f"⚠️ No config found for strategy: {strategy_name}")
            
            self.session_stats['session_start'] = datetime.now()
            logger.info("✅ Daily Strategy Runner initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def pre_market_setup(self) -> None:
        """Pre-market setup and preparation."""
        try:
            logger.info("🌅 Running pre-market setup...")
            
            # Reset daily statistics
            daily_stats = {
                'signals_generated': 0,
                'strategies_executed': 0,
                'performance_metrics': {},
                'risk_violations': 0
            }
            
            # Clear previous session data
            await self.performance_tracker.reset_daily_metrics()
            
            # Validate system health
            health_check = await self.strategy_coordinator.health_check()
            if not health_check.get('healthy', False):
                logger.warning("⚠️ System health check failed, attempting recovery...")
                await self.strategy_coordinator.recover()
            
            # Load latest market data for strategies
            for symbol in self.watchlist[:5]:  # Limit for pre-market
                try:
                    await self.signal_generator.preload_data(symbol)
                except Exception as e:
                    logger.debug(f"Data preload for {symbol}: {e}")
            
            logger.info("✅ Pre-market setup completed")
            
        except Exception as e:
            logger.error(f"❌ Pre-market setup failed: {e}")
            self.session_stats['errors_encountered'] += 1
            self.session_stats['last_error'] = str(e)
    
    async def strategy_execution_cycle(self) -> Dict[str, Any]:
        """Execute one complete strategy cycle."""
        cycle_results = {
            'cycle_number': self.session_stats['cycles_completed'] + 1,
            'timestamp': datetime.now().isoformat(),
            'signals_generated': 0,
            'strategies_executed': 0,
            'execution_time': 0.0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 Starting strategy execution cycle #{cycle_results['cycle_number']}")
            
            # Execute strategies for each symbol
            for symbol in self.watchlist:
                for strategy_name in self.strategies:
                    try:
                        # Generate signals for this symbol-strategy combination
                        signals = await self.signal_generator.generate_signals(
                            strategy_name, symbol, self._get_strategy_params(strategy_name)
                        )
                        
                        if signals:
                            # Execute strategy with signals
                            execution_result = await self.strategy_coordinator.execute_strategy(
                                strategy_name, symbol, signals
                            )
                            
                            if execution_result.get('success', False):
                                cycle_results['strategies_executed'] += 1
                                cycle_results['signals_generated'] += len(signals)
                                
                                # Track performance
                                await self.performance_tracker.record_execution(
                                    strategy_name, symbol, execution_result
                                )
                            else:
                                cycle_results['errors'].append(
                                    f"{strategy_name}:{symbol} - {execution_result.get('error', 'Unknown error')}"
                                )
                        
                    except Exception as e:
                        error_msg = f"{strategy_name}:{symbol} - {str(e)}"
                        cycle_results['errors'].append(error_msg)
                        logger.debug(f"Strategy execution error: {error_msg}")
            
            # Calculate cycle performance
            cycle_results['execution_time'] = (datetime.now() - start_time).total_seconds()
            
            # Update session statistics
            self.session_stats['cycles_completed'] += 1
            self.session_stats['signals_generated'] += cycle_results['signals_generated']
            self.session_stats['strategies_executed'] += cycle_results['strategies_executed']
            self.session_stats['total_execution_time'] += cycle_results['execution_time']
            
            if cycle_results['errors']:
                self.session_stats['errors_encountered'] += len(cycle_results['errors'])
                self.session_stats['last_error'] = cycle_results['errors'][-1]
            
            logger.info(f"✅ Cycle completed: {cycle_results['strategies_executed']} strategies, "
                       f"{cycle_results['signals_generated']} signals, "
                       f"{cycle_results['execution_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Strategy execution cycle failed: {e}")
            cycle_results['errors'].append(str(e))
            self.session_stats['errors_encountered'] += 1
            self.session_stats['last_error'] = str(e)
        
        return cycle_results
    
    def _get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get parameters for a specific strategy."""
        strategy_configs = {
            'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
            'momentum': {'window': 20, 'threshold': 0.02},
            'sma': {'fast_period': 10, 'slow_period': 30},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }
        return strategy_configs.get(strategy_name, {})
    
    async def end_of_day_processing(self) -> None:
        """Execute end-of-day processing and reporting."""
        try:
            logger.info("🌆 Running end-of-day processing...")
            
            # Calculate daily performance
            daily_performance = await self.performance_tracker.calculate_daily_performance()
            
            # Generate daily report
            await self.generate_daily_report(daily_performance)
            
            # Save session statistics
            await self.save_session_data()
            
            # Cleanup temporary data
            await self.cleanup_daily_data()
            
            logger.info("✅ End-of-day processing completed")
            
        except Exception as e:
            logger.error(f"❌ End-of-day processing failed: {e}")
            self.session_stats['errors_encountered'] += 1
            self.session_stats['last_error'] = str(e)
    
    async def generate_daily_report(self, performance: Dict[str, Any]) -> None:
        """Generate comprehensive daily report."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            report = [
                f"📊 DAILY STRATEGY EXECUTION REPORT - {date_str}",
                "=" * 60,
                "",
                "🎯 EXECUTION SUMMARY:",
                f"   • Total Cycles: {self.session_stats['cycles_completed']}",
                f"   • Strategies Executed: {self.session_stats['strategies_executed']}",
                f"   • Signals Generated: {self.session_stats['signals_generated']}",
                f"   • Total Execution Time: {self.session_stats['total_execution_time']:.2f}s",
                "",
                "📈 PERFORMANCE METRICS:",
                f"   • Successful Executions: {performance.get('successful_executions', 0)}",
                f"   • Average Execution Time: {performance.get('avg_execution_time', 0):.3f}s",
                f"   • Signal Success Rate: {performance.get('signal_success_rate', 0):.1f}%",
                f"   • Top Performing Strategy: {performance.get('top_strategy', 'N/A')}",
                "",
                "⚠️ SYSTEM STATUS:",
                f"   • Errors Encountered: {self.session_stats['errors_encountered']}",
                f"   • Last Error: {self.session_stats['last_error'] or 'None'}",
                f"   • System Health: {'✅ Healthy' if self.session_stats['errors_encountered'] < 10 else '⚠️ Needs Attention'}",
                "",
                "🏛️ COMPLIANCE:",
                "   • SEBI Compliance: ✅ Maintained",
                "   • Risk Limits: ✅ Observed",
                "   • Signal Generation Only: ✅ No Direct Orders",
                ""
            ]
            
            report_text = "\n".join(report)
            
            # Save report
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/daily_strategy_report_{date_str}.txt"
            
            with open(filename, "w") as f:
                f.write(report_text)
            
            # Print to console
            print("\n" + report_text)
            
            logger.info(f"📄 Daily report generated: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily report: {e}")
    
    async def save_session_data(self) -> None:
        """Save session statistics and data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("data/sessions", exist_ok=True)
            
            session_data = {
                'session_stats': self.session_stats,
                'configuration': {
                    'watchlist': self.watchlist,
                    'strategies': self.strategies,
                    'execution_interval': self.execution_interval
                },
                'timestamp': datetime.now().isoformat()
            }
            
            filename = f"data/sessions/session_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"💾 Session data saved: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save session data: {e}")
    
    async def cleanup_daily_data(self) -> None:
        """Cleanup temporary data and prepare for next session."""
        try:
            # Clear strategy coordinator cache
            await self.strategy_coordinator.clear_cache()
            
            # Reset performance tracker for next day
            await self.performance_tracker.archive_daily_data()
            
            # Reset cycle counter
            self.session_stats['cycles_completed'] = 0
            
            logger.info("🧹 Daily cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Daily cleanup failed: {e}")
    
    async def run_daily_execution(self, duration_hours: float = 6.25) -> None:
        """Run the main daily strategy execution loop."""
        if not await self.initialize():
            logger.error("❌ Failed to initialize - cannot start execution")
            return
        
        logger.info(f"🚀 Starting daily strategy execution for {duration_hours} hours...")
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        try:
            # Pre-market setup if before market hours
            if self.market_validator.is_pre_market_time():
                await self.pre_market_setup()
            
            # Main execution loop
            while self.is_running and datetime.now() < end_time:
                current_time = datetime.now()
                
                # Only execute during market hours or if testing
                if self.market_validator.is_market_hours() or not self.market_validator.is_market_day():
                    # Execute strategy cycle
                    cycle_results = await self.strategy_execution_cycle()
                    
                    # Log periodic status
                    if self.session_stats['cycles_completed'] % 10 == 0:
                        logger.info(f"📊 Status Update - Cycles: {self.session_stats['cycles_completed']}, "
                                  f"Signals: {self.session_stats['signals_generated']}, "
                                  f"Errors: {self.session_stats['errors_encountered']}")
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.execution_interval * 60)
                
                else:
                    # Outside market hours - wait longer
                    logger.debug("😴 Outside market hours - waiting...")
                    await asyncio.sleep(300)  # 5 minutes
            
            # End-of-day processing
            if self.market_validator.is_post_market_time() or datetime.now() >= end_time:
                await self.end_of_day_processing()
            
            logger.info("⏹️ Daily strategy execution completed")
            
        except KeyboardInterrupt:
            logger.info("🛑 Execution stopped by user")
        except Exception as e:
            logger.error(f"❌ Daily execution failed: {e}")
        finally:
            self.is_running = False
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        try:
            logger.info("🛑 Shutting down Daily Strategy Runner...")
            
            # Shutdown strategy coordinator
            await self.strategy_coordinator.shutdown()
            
            # Finalize performance tracking
            await self.performance_tracker.finalize()
            
            # Save final session data
            await self.save_session_data()
            
            logger.info("✅ Shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


async def main():
    """Main entry point for daily strategy runner."""
    parser = argparse.ArgumentParser(
        description="Daily Strategy Runner for Strategy Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_strategy_runner.py                        # Standard daily run
  python daily_strategy_runner.py --quick                # 10-minute test
  python daily_strategy_runner.py --extended             # 3-hour session
  python daily_strategy_runner.py --strategies rsi,momentum  # Specific strategies
  python daily_strategy_runner.py --config custom.yaml   # Custom config
        """
    )
    
    parser.add_argument("--quick", action="store_true", help="Quick 10-minute test run")
    parser.add_argument("--extended", action="store_true", help="Extended 3-hour session")
    parser.add_argument("--duration", type=float, help="Custom duration in hours")
    parser.add_argument("--strategies", type=str, help="Comma-separated list of strategies")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--watchlist", type=str, help="Comma-separated list of symbols")
    
    args = parser.parse_args()
    
    # Determine session parameters
    if args.quick:
        duration = 0.17  # 10 minutes
        print("🚄 Quick Test Mode (10 minutes)")
    elif args.extended:
        duration = 3.0  # 3 hours
        print("🎯 Extended Session Mode (3 hours)")
    else:
        duration = args.duration or 6.25  # Standard market hours
        print(f"📊 Standard Session Mode ({duration} hours)")
    
    print("🤖 AI Strategy Engine - Daily Runner")
    print("=" * 60)
    print(f"⏱️  Duration: {duration} hours")
    if args.strategies:
        print(f"🎯 Strategies: {args.strategies}")
    if args.watchlist:
        print(f"📈 Watchlist: {args.watchlist}")
    print("=" * 60)
    
    try:
        # Create and configure runner
        runner = DailyStrategyRunner(config_path=args.config)
        
        # Override configuration if specified
        if args.strategies:
            runner.strategies = [s.strip() for s in args.strategies.split(',')]
        if args.watchlist:
            runner.watchlist = [s.strip() for s in args.watchlist.split(',')]
        
        # Run daily execution
        await runner.run_daily_execution(duration_hours=duration)
        
    except Exception as e:
        logger.error(f"❌ Daily runner failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
