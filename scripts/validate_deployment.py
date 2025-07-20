#!/usr/bin/env python3
"""
Strategy Engine Deployment Validation Script

This script validates that the strategy-engine deployment is working correctly
by testing key components and functionality.
"""

import sys
import os
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing module imports...")

    required_modules = [
        "strategy_engine",
        "pandas",
        "numpy",
        "google.cloud.bigquery",
        "google.cloud.storage",
        "yaml",
        "requests",
        "sqlalchemy",
    ]

    failed_imports = []

    for module in required_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✅ Successfully imported {module}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            failed_imports.append(module)

    return len(failed_imports) == 0, failed_imports


def test_configuration():
    """Test that configuration files are accessible"""
    logger.info("Testing configuration accessibility...")

    config_files = [
        "config/strategies/momentum.yaml",
        "config/strategies/mean_reversion.yaml",
        "config/daily_runner/config.yaml",
    ]

    missing_configs = []

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            logger.info(f"✅ Configuration file found: {config_file}")
        else:
            logger.error(f"❌ Configuration file missing: {config_file}")
            missing_configs.append(config_file)

    return len(missing_configs) == 0, missing_configs


def test_entry_points():
    """Test that entry points are accessible"""
    logger.info("Testing entry points...")

    entry_points = [
        "bin/daily_strategy_runner.py",
        "bin/backtest_runner.py",
        "bin/signal_generator.py",
        "bin/strategy_optimizer.py",
    ]

    missing_entry_points = []

    for entry_point in entry_points:
        entry_path = Path(entry_point)
        if entry_path.exists():
            logger.info(f"✅ Entry point found: {entry_point}")
        else:
            logger.error(f"❌ Entry point missing: {entry_point}")
            missing_entry_points.append(entry_point)

    return len(missing_entry_points) == 0, missing_entry_points


def test_package_installation():
    """Test that the package is properly installed"""
    logger.info("Testing package installation...")

    try:
        import strategy_engine

        version = getattr(strategy_engine, "__version__", "unknown")
        logger.info(f"✅ Strategy Engine package installed, version: {version}")
        return True
    except Exception as e:
        logger.error(f"❌ Strategy Engine package not properly installed: {e}")
        return False


def test_docker_functionality():
    """Test basic Docker functionality if running in container"""
    logger.info("Testing Docker environment...")

    # Check if running in Docker
    if os.path.exists("/.dockerenv"):
        logger.info("✅ Running in Docker container")

        # Check user
        import pwd

        current_user = pwd.getpwuid(os.getuid()).pw_name
        logger.info(f"✅ Running as user: {current_user}")

        # Check working directory
        cwd = os.getcwd()
        logger.info(f"✅ Working directory: {cwd}")

        return True
    else:
        logger.info("ℹ️  Not running in Docker container")
        return True


def main():
    """Main validation function"""
    logger.info("🚀 Starting Strategy Engine Deployment Validation")
    logger.info("=" * 60)

    test_results = {}

    # Run all tests
    test_results["imports"], failed_imports = test_imports()
    test_results["config"], missing_configs = test_configuration()
    test_results["entry_points"], missing_entry_points = test_entry_points()
    test_results["package"] = test_package_installation()
    test_results["docker"] = test_docker_functionality()

    # Summary
    logger.info("=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")

    logger.info("-" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Strategy Engine is ready for deployment.")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED! Please review the issues above.")

        # Print specific failures
        if not test_results["imports"]:
            logger.error(f"Failed imports: {failed_imports}")
        if not test_results["config"]:
            logger.error(f"Missing configurations: {missing_configs}")
        if not test_results["entry_points"]:
            logger.error(f"Missing entry points: {missing_entry_points}")

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
