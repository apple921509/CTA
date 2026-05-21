"""Technical factor calculations for cross-sectional CTA research."""

import numpy as np
import pandas as pd

from cta_research.data import MarketData


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator.divide(denominator.replace(0, np.nan))


def momentum(close: pd.DataFrame, lookback: int = 20, skip: int = 1) -> pd.DataFrame:
    """Return skipped momentum so the latest bar can be excluded from signals."""
    return close.shift(skip) / close.shift(lookback + skip) - 1.0


def moving_average_slope(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    average = close.rolling(window).mean()
    return average / average.shift(window) - 1.0


def donchian_position(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    channel_width = rolling_high - rolling_low
    return _safe_divide(close - rolling_low, channel_width) * 2.0 - 1.0


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = _safe_divide(gain, loss)
    result = 100.0 - (100.0 / (1.0 + relative_strength))

    result = result.mask((loss == 0) & (gain > 0), 100.0)
    result = result.mask((gain == 0) & (loss > 0), 0.0)
    result = result.mask((gain == 0) & (loss == 0), 50.0)
    return result.clip(lower=0.0, upper=100.0)


def bollinger_zscore(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    average = close.rolling(window).mean()
    standard_deviation = close.rolling(window).std()
    return _safe_divide(close - average, standard_deviation)


def atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).stack(),
            (high - previous_close).abs().stack(),
            (low - previous_close).abs().stack(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1).unstack()
    true_range = true_range.reindex(index=close.index, columns=close.columns)
    return true_range.rolling(window).mean()


def historical_volatility(
    close: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    returns = close.pct_change()
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def volume_anomaly(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    average_volume = volume.rolling(window).mean()
    return _safe_divide(volume, average_volume) - 1.0


def vwap_deviation(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    rolling_vwap = _safe_divide(
        (close * volume).rolling(window).sum(),
        volume.rolling(window).sum(),
    )
    return _safe_divide(close, rolling_vwap) - 1.0


def price_volume_divergence(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    price_direction = np.sign(close.pct_change(window))
    volume_direction = np.sign(volume.pct_change(window))
    return price_direction * volume_direction


def rolling_zscore(frame: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    minimum_periods = max(3, window // 3)
    average = frame.rolling(window, min_periods=minimum_periods).mean()
    standard_deviation = frame.rolling(window, min_periods=minimum_periods).std()
    return _safe_divide(frame - average, standard_deviation)


def calculate_factor_set(data: MarketData) -> dict[str, pd.DataFrame]:
    return {
        "momentum": momentum(data.close, lookback=5, skip=1),
        "ma_slope": moving_average_slope(data.close, window=5),
        "donchian": donchian_position(data.high, data.low, data.close, window=5),
        "rsi": rsi(data.close, window=3),
        "bollinger_zscore": bollinger_zscore(data.close, window=5),
        "atr": atr(data.high, data.low, data.close, window=3),
        "historical_volatility": historical_volatility(data.close, window=5),
        "volume_anomaly": volume_anomaly(data.volume, window=5),
        "vwap_deviation": vwap_deviation(data.close, data.volume, window=5),
        "price_volume_divergence": price_volume_divergence(
            data.close,
            data.volume,
            window=5,
        ),
    }
