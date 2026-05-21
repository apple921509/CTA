import pandas as pd

from cta_research.data import MarketData
from cta_research.portfolio import (
    blend_strategy_signals,
    size_by_volatility,
    strategy_return_attribution,
)
from cta_research.risk import apply_drawdown_derisking, cap_exposure


def test_blend_strategy_signals_renormalizes_active_weights() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    columns = ["BTCUSDT", "ETHUSDT"]
    signals = {
        "trend": pd.DataFrame(1.0, index=index, columns=columns),
        "mean_reversion": pd.DataFrame(-0.5, index=index, columns=columns),
        "alternative": pd.DataFrame(0.0, index=index, columns=columns),
    }

    blended = blend_strategy_signals(
        signals, {"trend": 0.5, "mean_reversion": 0.25, "alternative": 0.25}
    )

    assert blended.iloc[0, 0] == 0.5


def test_blend_strategy_signals_preserves_disjoint_active_dates() -> None:
    signals = {
        "trend": pd.DataFrame({"BTCUSDT": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])),
        "mean_reversion": pd.DataFrame({"BTCUSDT": [-0.5]}, index=pd.DatetimeIndex(["2024-01-02"])),
    }

    blended = blend_strategy_signals(signals, {"trend": 0.5, "mean_reversion": 0.5})

    assert blended.loc["2024-01-01", "BTCUSDT"] == 0.5
    assert blended.loc["2024-01-02", "BTCUSDT"] == -0.25


def test_blend_strategy_signals_preserves_union_columns() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    signals = {
        "trend": pd.DataFrame({"BTCUSDT": [1.0, 1.0]}, index=index),
        "mean_reversion": pd.DataFrame({"ETHUSDT": [-0.5, -0.5]}, index=index),
    }

    blended = blend_strategy_signals(signals, {"trend": 0.5, "mean_reversion": 0.5})

    assert set(blended.columns) == {"BTCUSDT", "ETHUSDT"}
    assert blended.loc[index[0], "BTCUSDT"] == 0.5
    assert blended.loc[index[0], "ETHUSDT"] == -0.25


def test_size_and_cap_exposure_limits_gross() -> None:
    index = pd.date_range("2024-01-01", periods=3)
    signal = pd.DataFrame([[1, 1], [1, -1], [0.5, -0.5]], index=index, columns=["BTCUSDT", "ETHUSDT"])
    volatility = pd.DataFrame(0.5, index=index, columns=signal.columns)

    sized = size_by_volatility(signal, volatility, volatility_target=0.25)
    capped = cap_exposure(sized, max_symbol_weight=0.3, max_gross_exposure=0.5)

    assert capped.abs().max().max() <= 0.3
    assert capped.abs().sum(axis=1).max() <= 0.5


def test_strategy_return_attribution_sums_to_gross_returns() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    close = pd.DataFrame({"BTCUSDT": [100.0, 110.0, 121.0, 133.1]}, index=index)
    data = MarketData(
        open=close,
        high=close,
        low=close,
        close=close,
        volume=pd.DataFrame(1.0, index=index, columns=close.columns),
    )
    signals = {
        "trend": pd.DataFrame(1.0, index=index, columns=close.columns),
        "mean_reversion": pd.DataFrame(-0.5, index=index, columns=close.columns),
    }
    volatility = pd.DataFrame(0.25, index=index, columns=close.columns)
    final_positions = blend_strategy_signals(
        signals,
        {"trend": 0.5, "mean_reversion": 0.5},
    )

    strategy_returns = strategy_return_attribution(
        data=data,
        strategy_signals=signals,
        weights={"trend": 0.5, "mean_reversion": 0.5},
        volatility=volatility,
        volatility_target=0.25,
        final_positions=final_positions,
    )

    gross_returns = (final_positions.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)).sum(axis=1)
    pd.testing.assert_series_equal(
        strategy_returns.sum(axis=1),
        gross_returns,
        check_names=False,
    )


def test_apply_drawdown_derisking_reduces_exposure() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    positions = pd.DataFrame(1.0, index=index, columns=["BTCUSDT"])
    equity = pd.Series([100, 90, 80, 70], index=index)

    derisked = apply_drawdown_derisking(
        positions,
        equity,
        drawdown_warning=0.10,
        drawdown_hard=0.30,
        min_exposure_multiplier=0.25,
    )

    assert derisked.iloc[0, 0] == 1.0
    assert derisked.iloc[-1, 0] == 0.25
