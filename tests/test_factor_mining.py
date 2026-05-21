import pandas as pd

from cta_research.data import MarketData
from cta_research.factor_mining import (
    FactorAdmissionCriteria,
    accepted_factors,
    factor_quantile_report,
    factor_rank_signal,
    mine_factors,
    single_factor_backtest_report,
)


def _market_data() -> MarketData:
    index = pd.date_range("2024-01-01", periods=8)
    close = pd.DataFrame(
        {
            "A": [100, 101, 103, 106, 110, 115, 121, 128],
            "B": [100, 99, 97, 94, 90, 85, 79, 72],
            "C": [100, 100, 101, 100, 101, 100, 101, 100],
        },
        index=index,
        dtype=float,
    )
    return MarketData(open=close, high=close, low=close, close=close, volume=close * 10)


def test_factor_rank_signal_controls_gross_exposure() -> None:
    data = _market_data()
    factor = data.close.pct_change().fillna(0.0)

    signal = factor_rank_signal(factor, gross_exposure=0.75)

    assert signal.abs().sum(axis=1).max() <= 0.75


def test_mine_factors_scores_and_classifies_factors() -> None:
    data = _market_data()
    good = data.close.pct_change().shift(1)
    bad = -good
    flat = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
    factors = {"good": good, "bad": bad, "flat": flat}

    scorecard = mine_factors(
        factors,
        data,
        horizon=1,
        quantiles=3,
        criteria=FactorAdmissionCriteria(
            min_abs_ic_mean=0.01,
            min_abs_information_ratio=0.01,
            min_positive_rate=0.50,
            min_backtest_sharpe=-10.0,
        ),
        fee_bps=0,
        slippage_bps=0,
    )

    assert {"factor", "ic_mean", "information_ratio", "decision"}.issubset(scorecard.columns)
    assert set(scorecard["factor"]) == {"good", "bad", "flat"}
    assert isinstance(accepted_factors(scorecard), list)


def test_factor_quantile_and_single_backtest_reports() -> None:
    data = _market_data()
    factors = {"momentum": data.close.pct_change().shift(1)}

    quantiles = factor_quantile_report(factors, data.close, horizon=1, quantiles=3)
    backtests = single_factor_backtest_report(factors, data, fee_bps=0, slippage_bps=0)

    assert "factor" in quantiles.columns
    assert "mean_forward_return" in quantiles.columns
    assert {"factor", "sharpe", "total_turnover"}.issubset(backtests.columns)
