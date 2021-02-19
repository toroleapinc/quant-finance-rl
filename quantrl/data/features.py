"""Technical indicator features."""
import pandas as pd
import numpy as np
import ta

def add_indicators(df):
    """Add technical indicators to OHLCV dataframe."""
    df = df.copy()
    # trend
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['macd'] = ta.trend.macd_diff(df['close'])
    # momentum
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    # volatility
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    # volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    df = df.dropna().reset_index(drop=True)
    # normalize
    for col in df.columns:
        if col not in ['date', 'timestamp']:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    return df

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    return add_indicators(df)
