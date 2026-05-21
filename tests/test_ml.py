from pathlib import Path

import pandas as pd

from cta_research.data import MarketData, load_ohlcv_directory
from cta_research.factors import calculate_factor_set
from cta_research.ml import (
    build_supervised_dataset,
    expanding_window_predictions,
    make_model,
    predictions_to_signal,
    run_ml_signal_backtest,
)


def _toy_market_data(periods: int = 16) -> MarketData:
    index = pd.date_range("2024-01-01", periods=periods)
    close = pd.DataFrame(
        {
            "BTCUSDT": [100 + i for i in range(periods)],
            "ETHUSDT": [100 - i * 0.5 for i in range(periods)],
            "SOLUSDT": [50 + (i % 4) for i in range(periods)],
        },
        index=index,
    )
    return MarketData(open=close, high=close, low=close, close=close, volume=close * 10)


def test_build_supervised_dataset_stacks_factor_panel() -> None:
    data = _toy_market_data()
    factors = {
        "return_1": data.close.pct_change(),
        "return_2": data.close.pct_change(2),
    }

    dataset = build_supervised_dataset(factors, data.close, horizon=1)

    assert list(dataset.features.columns) == ["return_1", "return_2"]
    assert dataset.features.index.names == ["timestamp", "symbol"]
    assert len(dataset.features) == len(dataset.target)


def test_make_model_supports_baselines() -> None:
    assert make_model("linear") is not None
    assert make_model("ridge", alpha=0.5) is not None
    assert make_model("random_forest", n_estimators=5) is not None


def test_expanding_window_predictions_returns_metrics() -> None:
    data = _toy_market_data()
    factors = {
        "return_1": data.close.pct_change(),
        "return_2": data.close.pct_change(2),
    }
    dataset = build_supervised_dataset(factors, data.close, horizon=1)

    predictions, metrics = expanding_window_predictions(
        dataset,
        model_name="ridge",
        train_size=6,
        test_size=3,
    )

    assert not predictions.empty
    assert {"rmse", "directional_accuracy", "rank_ic"}.issubset(metrics.columns)


def test_predictions_to_signal_controls_gross_exposure() -> None:
    predictions = pd.Series(
        [0.1, -0.2, 0.3],
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-01"), "BTCUSDT"), (pd.Timestamp("2024-01-01"), "ETHUSDT"), (pd.Timestamp("2024-01-01"), "SOLUSDT")],
            names=["timestamp", "symbol"],
        ),
    )

    signal = predictions_to_signal(
        predictions,
        index=pd.DatetimeIndex(["2024-01-01"]),
        columns=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        gross_exposure=0.8,
    )

    assert round(signal.abs().sum(axis=1).iloc[0], 12) == 0.8


def test_run_ml_signal_backtest_returns_predictions_positions_and_backtest() -> None:
    data = _toy_market_data(20)
    factors = {
        "return_1": data.close.pct_change(),
        "return_2": data.close.pct_change(2),
    }

    result = run_ml_signal_backtest(
        data,
        factors,
        model_name="random_forest",
        train_size=8,
        test_size=4,
        model_params={"n_estimators": 5, "max_depth": 2},
        fee_bps=0,
        slippage_bps=0,
    )

    assert result.predictions.shape == data.close.shape
    assert result.positions.shape == data.close.shape
    assert result.backtest.equity.iloc[0] == 100000
    assert not result.metrics.empty


def test_ml_backtest_with_fixture_factor_set() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    factors = calculate_factor_set(data)

    result = run_ml_signal_backtest(
        data,
        factors,
        model_name="ridge",
        feature_names=["momentum", "ma_slope", "donchian"],
        train_size=4,
        test_size=2,
        fee_bps=0,
        slippage_bps=0,
    )

    assert result.backtest.positions.shape == data.close.shape
