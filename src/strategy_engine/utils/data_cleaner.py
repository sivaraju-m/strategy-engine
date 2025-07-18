# src/ai_trading_machine/utils/data_cleaner.py

import pandas as pd


def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and fill missing OHLCV data using simple interpolation.
    """
    df = df.copy()
    df = df.sort_values(by="date")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Drop rows still with missing values
    df.dropna(inplace=True)

    return df


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias for clean_and_impute_data for compatibility with backtest_runner.
    """
    return clean_and_impute_data(df)
