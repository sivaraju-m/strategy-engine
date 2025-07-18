"""
Feature generation module for AI Trading Machine.

This module contains classes and functions for generating features from market data.
"""

import logging

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class BaseFeatureGenerator:
    """Base class for all feature generators."""

    def __init__(self, name: str):
        """
        Initialize the feature generator.

        Args:
            name: Name of the feature generator
        """
        self.name = name

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from input data.

        Args:
            data: Input market data

        Returns:
            DataFrame with additional feature columns
        """
        raise NotImplementedError("Subclasses must implement generate method")


class TechnicalIndicatorFeatures(BaseFeatureGenerator):
    """
    Generate features based on technical indicators.
    """

    def __init__(self, window_sizes: list[int] = None):
        """
        Initialize the technical indicator feature generator.

        Args:
            window_sizes: List of window sizes for moving averages, RSI, etc.
        """
        super().__init__(name="technical_indicators")
        self.window_sizes = window_sizes or [5, 10, 20, 50, 200]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicator features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical indicator features
        """
        logger.info(
            "Generating technical indicator features with windows: {self.window_sizes}"
        )

        # Make a copy of the input data
        result = data.copy()

        # Check if required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in result.columns:
                raise ValueError("Required column '{col}' not found in input data")

        # Generate moving averages
        for window in self.window_sizes:
            result["ma_{window}"] = result["close"].rolling(window=window).mean()

            # Moving average crossovers
            if window < 50:
                result["ma_cross_{window}_50"] = (
                    result["ma_{window}"] > result["ma_50"]
                ).astype(int)

            # Price position relative to moving average
            result["close_over_ma_{window}"] = (
                result["close"] > result["ma_{window}"]
            ).astype(int)

            # Volume features
            result["volume_ma_{window}"] = (
                result["volume"].rolling(window=window).mean()
            )
            result["volume_ratio_{window}"] = (
                result["volume"] / result["volume_ma_{window}"]
            )

        # RSI calculation
        for window in self.window_sizes:
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()

            rs = avg_gain / avg_loss
            result["rsi_{window}"] = 100 - (100 / (1 + rs))

        # MACD
        result["macd"] = (
            result["close"].ewm(span=12, adjust=False).mean()
            - result["close"].ewm(span=26, adjust=False).mean()
        )
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # Bollinger Bands
        for window in [20]:  # Typically 20-day is standard
            result["bb_middle_{window}"] = result["close"].rolling(window=window).mean()
            result["bb_std_{window}"] = result["close"].rolling(window=window).std()

            result["bb_upper_{window}"] = (
                result["bb_middle_{window}"] + 2 * result["bb_std_{window}"]
            )
            result["bb_lower_{window}"] = (
                result["bb_middle_{window}"] - 2 * result["bb_std_{window}"]
            )

            # BB Width
            result["bb_width_{window}"] = (
                result["bb_upper_{window}"] - result["bb_lower_{window}"]
            ) / result["bb_middle_{window}"]

            # BB Position
            result["bb_pos_{window}"] = (
                result["close"] - result["bb_lower_{window}"]
            ) / (result["bb_upper_{window}"] - result["bb_lower_{window}"])

        # Drop NaN values from calculations
        result = result.dropna()

        logger.info("Generated {len(result.columns) - len(data.columns)} new features")

        return result


class MarketRegimeFeatures(BaseFeatureGenerator):
    """
    Generate features related to market regimes (trending, ranging, etc.).
    """

    def __init__(self, window_sizes: list[int] = None):
        """
        Initialize the market regime feature generator.

        Args:
            window_sizes: List of window sizes for market regime calculations
        """
        super().__init__(name="market_regime")
        self.window_sizes = window_sizes or [20, 50, 100]

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market regime features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional market regime features
        """
        logger.info("Generating market regime features")

        # Make a copy of the input data
        result = data.copy()

        # Check if required columns exist
        if "close" not in result.columns:
            raise ValueError("Required column 'close' not found in input data")

        # Calculate volatility for different windows
        for window in self.window_sizes:
            # Calculate returns
            result["returns_{window}"] = result["close"].pct_change(periods=1)

            # Volatility (standard deviation of returns)
            result["volatility_{window}"] = (
                result["returns_{window}"].rolling(window=window).std()
            )

            # Trend strength (absolute value of the correlation between price and time)
            result["trend_strength_{window}"] = (
                result["close"]
                .rolling(window=window)
                .apply(lambda x: abs(np.corrcoef(x, range(len(x)))[0, 1]), raw=True)
            )

            # Directional Movement Index (DMI)
            high_diff = result["high"].diff()
            low_diff = result["low"].diff().abs() * -1

            plus_dm = (high_diff > low_diff) & (high_diff > 0)
            minus_dm = (low_diff > high_diff) & (low_diff > 0)

            result["plus_dm_{window}"] = plus_dm.astype(int) * high_diff
            result["minus_dm_{window}"] = minus_dm.astype(int) * low_diff.abs()

            # Average Directional Index (ADX)
            true_range = pd.DataFrame(
                {
                    "tr1": result["high"] - result["low"],
                    "tr2": (result["high"] - result["close"].shift(1)).abs(),
                    "tr3": (result["low"] - result["close"].shift(1)).abs(),
                }
            ).max(axis=1)

            atr = true_range.rolling(window=window).mean()

            plus_di = 100 * (
                result["plus_dm_{window}"].rolling(window=window).mean() / atr
            )
            minus_di = 100 * (
                result["minus_dm_{window}"].rolling(window=window).mean() / atr
            )

            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            result["adx_{window}"] = dx.rolling(window=window).mean()

            # Define market regime based on ADX
            # ADX > 25: Trending market
            # ADX < 25: Ranging market
            result["is_trending_{window}"] = (result["adx_{window}"] > 25).astype(int)

        # Drop NaN values from calculations
        result = result.dropna()

        logger.info("Generated {len(result.columns) - len(data.columns)} new features")

        return result


class FeaturePipeline:
    """
    Pipeline for generating and combining features from multiple generators.
    """

    def __init__(self, generators: list[BaseFeatureGenerator] = None):
        """
        Initialize the feature pipeline.

        Args:
            generators: List of feature generators to use
        """
        self.generators = generators or []

    def add_generator(self, generator: BaseFeatureGenerator) -> None:
        """
        Add a feature generator to the pipeline.

        Args:
            generator: Feature generator to add
        """
        self.generators.append(generator)

    def generate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from all generators.

        Args:
            data: Input market data

        Returns:
            DataFrame with all generated features
        """
        logger.info("Running feature pipeline with {len(self.generators)} generators")

        result = data.copy()

        for generator in self.generators:
            logger.info("Running generator: {generator.name}")
            result = generator.generate(result)

        logger.info(
            "Feature pipeline complete. Generated {len(result.columns) - len(data.columns)} features"
        )

        return result


def create_default_pipeline() -> FeaturePipeline:
    """
    Create a default feature pipeline with common generators.

    Returns:
        A configured feature pipeline
    """
    pipeline = FeaturePipeline()

    # Add technical indicators
    pipeline.add_generator(TechnicalIndicatorFeatures())

    # Add market regime features
    pipeline.add_generator(MarketRegimeFeatures())

    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate features for AI Trading Machine"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input OHLCV data CSV"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the generated features CSV"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load input data
    logger.info("Loading data from {args.input}")
    data = pd.read_csv(args.input)

    # Create and run the feature pipeline
    pipeline = create_default_pipeline()
    features = pipeline.generate_all(data)

    # Save the features
    logger.info(
        "Saving {len(features)} rows with {len(features.columns)} features to {args.output}"
    )
    features.to_csv(args.output, index=False)
