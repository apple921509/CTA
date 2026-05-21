from pathlib import Path

import numpy as np
import pandas as pd

from cta_research.data import load_ohlcv_directory
from cta_research.factors import (
    bollinger_zscore,
    calculate_factor_set,
    donchian_position,
    momentum,
    price_volume_divergence,
    rolling_zscore,
    rsi,
)


def test_momentum_with_skip_avoids_latest_bar() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    result = momentum(data.close, lookback=3, skip=1)

    expected = data.close.shift(1) / data.close.shift(4) - 1
    pd.testing.assert_frame_equal(result, expected)


def test_rsi_is_bounded_between_zero_and_one_hundred() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    result = rsi(data.close, window=3).dropna()

    assert (result >= 0).all().all()
    assert (result <= 100).all().all()


def test_donchian_position_is_finite_after_window() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    result = donchian_position(data.high, data.low, data.close, window=3).dropna()

    assert np.isfinite(result.to_numpy()).all()


def test_factor_set_contains_expected_names() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    factors = calculate_factor_set(data)

    assert {"momentum", "rsi", "bollinger_zscore", "atr", "volume_anomaly"}.issubset(factors)
    assert factors["momentum"].shape == data.close.shape
    assert bollinger_zscore(data.close, window=3).shape == data.close.shape


def test_price_volume_divergence_uses_windowed_sign_product() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    result = price_volume_divergence(data.close, data.volume, window=3)

    expected = np.sign(data.close.pct_change(3)) * np.sign(data.volume.pct_change(3))
    pd.testing.assert_frame_equal(result, expected)


def test_rolling_zscore_uses_partial_window_after_minimum_periods() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    result = rolling_zscore(data.close, window=6)

    assert result.iloc[:2].isna().all().all()
    assert result.iloc[2:].notna().all().all()
