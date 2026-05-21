from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from cta_research.analysis import calculate_metrics, max_drawdown
from cta_research.backtest import run_backtest
from cta_research.data import MarketData


SignalBuilder = Callable[[MarketData, dict[str, Any]], pd.DataFrame]


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardWindow]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if step_size is None:
        step_size = test_size
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    timestamps = pd.DatetimeIndex(index)
    windows = []
    start = 0
    while start + train_size + test_size <= len(timestamps):
        train = timestamps[start : start + train_size]
        test = timestamps[start + train_size : start + train_size + test_size]
        windows.append(
            WalkForwardWindow(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )
        )
        start += step_size
    return windows


def _slice_market_data(data: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> MarketData:
    return MarketData(
        open=data.open.loc[start:end],
        high=data.high.loc[start:end],
        low=data.low.loc[start:end],
        close=data.close.loc[start:end],
        volume=data.volume.loc[start:end],
    )


def walk_forward_analysis(
    data: MarketData,
    signal_builder: SignalBuilder,
    parameter_grid: list[dict[str, Any]],
    train_size: int,
    test_size: int,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    step_size: int | None = None,
    metric: str = "sharpe",
) -> pd.DataFrame:
    windows = build_walk_forward_windows(data.close.index, train_size, test_size, step_size)
    records = []
    for window_id, window in enumerate(windows, start=1):
        train_data = _slice_market_data(data, window.train_start, window.train_end)
        test_data = _slice_market_data(data, window.test_start, window.test_end)

        scored_params = []
        for params in parameter_grid:
            train_positions = signal_builder(train_data, params)
            train_result = run_backtest(train_data, train_positions, initial_capital, fee_bps, slippage_bps)
            train_metrics = calculate_metrics(train_result.equity, train_result.returns, train_result.trades)
            scored_params.append((train_metrics.get(metric, 0.0), params, train_metrics))

        scored_params.sort(key=lambda item: item[0], reverse=True)
        _, best_params, train_metrics = scored_params[0]
        test_positions = signal_builder(test_data, best_params)
        test_result = run_backtest(test_data, test_positions, initial_capital, fee_bps, slippage_bps)
        test_metrics = calculate_metrics(test_result.equity, test_result.returns, test_result.trades)

        records.append(
            {
                "window": window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "best_params": best_params,
                f"train_{metric}": train_metrics.get(metric, 0.0),
                f"test_{metric}": test_metrics.get(metric, 0.0),
                "test_total_return": test_metrics["total_return"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "test_trade_count": test_metrics["trade_count"],
            }
        )
    return pd.DataFrame.from_records(records)


def monte_carlo_return_paths(
    returns: pd.Series,
    simulations: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    clean_returns = returns.dropna().to_numpy()
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if len(clean_returns) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean_returns, size=(len(clean_returns), simulations), replace=True)
    paths = pd.DataFrame((1.0 + sampled).cumprod(axis=0), index=returns.dropna().index)
    paths.columns = [f"simulation_{i + 1}" for i in range(simulations)]
    return paths


def monte_carlo_summary(
    returns: pd.Series,
    simulations: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    paths = monte_carlo_return_paths(returns, simulations=simulations, seed=seed)
    if paths.empty:
        return {
            "median_terminal_return": 0.0,
            "p05_terminal_return": 0.0,
            "p95_terminal_return": 0.0,
            "p05_max_drawdown": 0.0,
            "p95_max_drawdown": 0.0,
        }

    terminal_returns = paths.iloc[-1] - 1.0
    drawdowns = paths.apply(max_drawdown)
    return {
        "median_terminal_return": float(terminal_returns.median()),
        "p05_terminal_return": float(terminal_returns.quantile(0.05)),
        "p95_terminal_return": float(terminal_returns.quantile(0.95)),
        "p05_max_drawdown": float(drawdowns.quantile(0.05)),
        "p95_max_drawdown": float(drawdowns.quantile(0.95)),
    }


def parameter_grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(params)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*(params[name] for name in names))
    ]


def parameter_stability_report(
    results: pd.DataFrame,
    param_column: str = "best_params",
    metric_column: str = "test_sharpe",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["parameter", "value", "selection_count", "mean_metric"])

    records = []
    for _, row in results.iterrows():
        params = row[param_column]
        if not isinstance(params, dict):
            continue
        for name, value in params.items():
            records.append({"parameter": name, "value": value, "metric": row[metric_column]})
    if not records:
        return pd.DataFrame(columns=["parameter", "value", "selection_count", "mean_metric"])

    frame = pd.DataFrame.from_records(records)
    return (
        frame.groupby(["parameter", "value"])["metric"]
        .agg(selection_count="count", mean_metric="mean")
        .reset_index()
        .sort_values(["parameter", "selection_count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def overfitting_checks(
    train_metric: pd.Series,
    test_metric: pd.Series,
    min_test_pass_rate: float = 0.5,
) -> dict[str, float | bool]:
    aligned_train, aligned_test = train_metric.align(test_metric, join="inner")
    if len(aligned_train) == 0:
        return {
            "train_test_correlation": 0.0,
            "degradation_ratio": 0.0,
            "test_pass_rate": 0.0,
            "overfit_risk": True,
        }

    train_mean = aligned_train.mean()
    test_mean = aligned_test.mean()
    correlation = aligned_train.corr(aligned_test)
    degradation = 0.0 if train_mean == 0 else float((train_mean - test_mean) / abs(train_mean))
    pass_rate = float((aligned_test > 0).mean())
    return {
        "train_test_correlation": 0.0 if pd.isna(correlation) else float(correlation),
        "degradation_ratio": degradation,
        "test_pass_rate": pass_rate,
        "overfit_risk": bool(pass_rate < min_test_pass_rate or degradation > 1.0),
    }
