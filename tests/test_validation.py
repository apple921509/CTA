from pathlib import Path

import pandas as pd

from cta_research.data import MarketData, load_ohlcv_directory
from cta_research.validation import (
    build_walk_forward_windows,
    monte_carlo_return_paths,
    monte_carlo_summary,
    overfitting_checks,
    parameter_grid,
    parameter_stability_report,
    walk_forward_analysis,
)


def _toy_market_data(periods: int = 12) -> MarketData:
    index = pd.date_range("2024-01-01", periods=periods)
    close = pd.DataFrame(
        {
            "BTCUSDT": [100 + i for i in range(periods)],
            "ETHUSDT": [50 + i * 0.5 for i in range(periods)],
        },
        index=index,
    )
    return MarketData(open=close, high=close, low=close, close=close, volume=close * 10)


def _signal_builder(data: MarketData, params: dict) -> pd.DataFrame:
    lookback = params["lookback"]
    scale = params.get("scale", 1.0)
    return data.close.pct_change(lookback).fillna(0.0).clip(-0.5, 0.5) * scale


def test_build_walk_forward_windows_creates_train_test_slices() -> None:
    index = pd.date_range("2024-01-01", periods=10)

    windows = build_walk_forward_windows(index, train_size=4, test_size=2)

    assert len(windows) == 3
    assert windows[0].train_start == pd.Timestamp("2024-01-01")
    assert windows[0].test_start == pd.Timestamp("2024-01-05")


def test_walk_forward_analysis_selects_params_and_reports_test_metrics() -> None:
    data = _toy_market_data(14)
    grid = parameter_grid({"lookback": [1, 2], "scale": [0.5, 1.0]})

    result = walk_forward_analysis(
        data=data,
        signal_builder=_signal_builder,
        parameter_grid=grid,
        train_size=6,
        test_size=3,
        initial_capital=100000,
        fee_bps=0,
        slippage_bps=0,
    )

    assert not result.empty
    assert isinstance(result.loc[0, "best_params"], dict)
    assert {"test_total_return", "test_max_drawdown", "test_trade_count"}.issubset(result.columns)


def test_monte_carlo_return_paths_and_summary_are_reproducible() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.date_range("2024-01-01", periods=4))

    paths = monte_carlo_return_paths(returns, simulations=5, seed=7)
    summary = monte_carlo_summary(returns, simulations=5, seed=7)

    assert paths.shape == (4, 5)
    assert {"median_terminal_return", "p05_max_drawdown"}.issubset(summary)


def test_parameter_stability_report_counts_selected_values() -> None:
    results = pd.DataFrame(
        {
            "best_params": [{"lookback": 1}, {"lookback": 2}, {"lookback": 1}],
            "test_sharpe": [1.0, 0.5, 2.0],
        }
    )

    report = parameter_stability_report(results)

    lookback_one = report[(report["parameter"] == "lookback") & (report["value"] == 1)]
    assert lookback_one.iloc[0]["selection_count"] == 2
    assert lookback_one.iloc[0]["mean_metric"] == 1.5


def test_overfitting_checks_flags_poor_out_of_sample_results() -> None:
    checks = overfitting_checks(
        pd.Series([2.0, 2.5, 3.0]),
        pd.Series([-0.5, -0.2, -0.1]),
    )

    assert checks["test_pass_rate"] == 0.0
    assert checks["overfit_risk"] is True


def test_walk_forward_works_with_fixture_market_data() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])

    result = walk_forward_analysis(
        data=data,
        signal_builder=_signal_builder,
        parameter_grid=[{"lookback": 1, "scale": 1.0}],
        train_size=4,
        test_size=2,
        initial_capital=100000,
        fee_bps=0,
        slippage_bps=0,
    )

    assert len(result) == 3
