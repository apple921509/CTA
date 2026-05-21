from pathlib import Path

import pandas as pd

from cta_research.analysis import calculate_metrics, factor_ic
from cta_research.backtest import run_backtest
from cta_research.data import load_ohlcv_directory


def test_run_backtest_outputs_equity_positions_and_trades() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    positions = data.close.pct_change(3).fillna(0).clip(-0.5, 0.5)

    result = run_backtest(
        data=data,
        target_positions=positions,
        initial_capital=100000,
        fee_bps=6,
        slippage_bps=2,
    )

    assert result.equity.iloc[0] == 100000
    assert result.positions.shape == data.close.shape
    assert {"timestamp", "symbol", "target_weight", "turnover"}.issubset(
        result.trades.columns
    )


def test_run_backtest_first_bar_is_initial_state_without_costs_or_returns() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    positions = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
    positions.iloc[0] = [0.5, -0.25]

    result = run_backtest(data, positions, 100000, 6, 2)
    equity_returns = result.equity.pct_change().fillna(0.0)

    assert result.returns.iloc[0] == 0.0
    assert result.gross_returns.iloc[0] == 0.0
    assert result.costs.iloc[0] == 0.0
    assert result.equity.iloc[0] == 100000
    assert not (result.trades["timestamp"] == data.close.index[0]).any()
    pd.testing.assert_series_equal(
        equity_returns,
        result.returns,
        check_names=False,
    )


def test_calculate_metrics_contains_required_fields() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    positions = data.close.pct_change(3).fillna(0).clip(-0.5, 0.5)
    result = run_backtest(data, positions, 100000, 6, 2)

    metrics = calculate_metrics(result.equity, result.returns, result.trades)

    assert {"total_return", "sharpe", "max_drawdown", "trade_count"}.issubset(metrics)


def test_factor_ic_returns_decay_horizons() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    factor = data.close.pct_change(2)

    ic = factor_ic(factor, data.close, horizons=[1, 3])

    assert set(ic["horizon"]) == {1, 3}
