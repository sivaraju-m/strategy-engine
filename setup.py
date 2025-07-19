from setuptools import setup, find_packages

setup(
    name="strategy-engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.0",
        "pandas-ta>=0.3.14b0",
        "shared-services>=0.1.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "backtest=strategy_engine.bin.backtest_runner:main",
            "generate-signals=strategy_engine.bin.signal_generator:main",
            "optimize-strategy=strategy_engine.bin.strategy_optimizer:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Engine for defining, backtesting, and executing trading strategies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/strategy-engine",
)
