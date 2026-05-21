from pathlib import Path

import numpy as np
import pandas as pd

from cta_research.data import load_ohlcv_directory
from cta_research.factors import calculate_factor_set
from cta_research.strategies import (
    alternative_signal,
    mean_reversion_signal,
    swing_signal,
    trend_signal,
)


def test_strategy_signals_are_bounded() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    factors = calculate_factor_set(data)

    for signal in [
        trend_signal(factors),
        mean_reversion_signal(factors),
        swing_signal(factors),
        alternative_signal(factors),
    ]:
        bounded = signal.dropna(how="all").fillna(0)
        assert (bounded <= 1).all().all()
        assert (bounded >= -1).all().all()


def test_alternative_signal_is_zero_without_alternative_data() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    factors = calculate_factor_set(data)

    signal = alternative_signal(factors)

    assert signal.fillna(0).abs().sum().sum() == 0


def test_trend_signal_can_short_single_symbol_negative_trends() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    factors = {
        "momentum": pd.DataFrame({"BTCUSDT": [-0.2, -0.4]}, index=index),
        "ma_slope": pd.DataFrame({"BTCUSDT": [-0.1, -0.3]}, index=index),
        "donchian": pd.DataFrame({"BTCUSDT": [0.0, 0.0]}, index=index),
    }

    signal = trend_signal(factors)

    assert (signal["BTCUSDT"] < 0).all()


def test_trend_signal_ranks_multi_symbol_factors_cross_sectionally() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    factors = {
        "momentum": pd.DataFrame({"BTCUSDT": [0.1, 0.2], "ETHUSDT": [0.3, 0.4]}, index=index),
        "ma_slope": pd.DataFrame({"BTCUSDT": [0.01, 0.02], "ETHUSDT": [0.03, 0.04]}, index=index),
        "donchian": pd.DataFrame({"BTCUSDT": [0.0, 0.0], "ETHUSDT": [0.0, 0.0]}, index=index),
    }

    signal = trend_signal(factors)

    assert (signal["BTCUSDT"] < 0).all()
    assert (signal["ETHUSDT"] > 0).all()
    assert (signal["ETHUSDT"] > signal["BTCUSDT"]).all()


def test_alternative_signal_fallback_replaces_template_nans_with_zero() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    factors = {"momentum": pd.DataFrame({"BTCUSDT": [np.nan, 1.0]}, index=index)}

    signal = alternative_signal(factors)

    assert not signal.isna().any().any()
    assert signal.sum().sum() == 0.0
