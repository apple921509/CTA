from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from cta_research.analysis import calculate_metrics
from cta_research.backtest import run_backtest
from cta_research.data import MarketData
from cta_research.factor_research import (
    factor_ic_decay,
    quantile_forward_returns,
    rank_cross_section,
)


Decision = Literal["pass", "weak", "inverse_candidate", "reject"]


@dataclass(frozen=True)
class FactorAdmissionCriteria:
    min_abs_ic_mean: float = 0.03
    min_abs_information_ratio: float = 0.50
    min_positive_rate: float = 0.55
    min_long_short_spread: float = 0.0
    min_backtest_sharpe: float = 0.0
    max_turnover: float | None = None


def factor_rank_signal(
    factor: pd.DataFrame,
    gross_exposure: float = 1.0,
    inverse: bool = False,
) -> pd.DataFrame:
    signal = rank_cross_section(factor).fillna(0.0)
    if inverse:
        signal = -signal
    gross = signal.abs().sum(axis=1).replace(0, np.nan)
    return signal.divide(gross, axis=0).mul(gross_exposure).fillna(0.0)


def single_factor_backtest_metrics(
    data: MarketData,
    factor: pd.DataFrame,
    initial_capital: float = 100000,
    fee_bps: float = 6,
    slippage_bps: float = 2,
    gross_exposure: float = 1.0,
    inverse: bool = False,
) -> dict[str, float | int]:
    positions = factor_rank_signal(factor, gross_exposure=gross_exposure, inverse=inverse)
    result = run_backtest(data, positions, initial_capital, fee_bps, slippage_bps)
    metrics = calculate_metrics(result.equity, result.returns, result.trades)
    return metrics


def _long_short_spread(
    quantile_returns: pd.DataFrame,
) -> float:
    if quantile_returns.empty:
        return 0.0
    grouped = quantile_returns.groupby("quantile")["mean_forward_return"].mean()
    if len(grouped) < 2:
        return 0.0
    return float(grouped.loc[grouped.index.max()] - grouped.loc[grouped.index.min()])


def _admission_decision(
    ic_mean: float,
    information_ratio: float,
    positive_rate: float,
    long_short_spread: float,
    backtest_sharpe: float,
    turnover: float,
    criteria: FactorAdmissionCriteria,
) -> Decision:
    turnover_ok = criteria.max_turnover is None or turnover <= criteria.max_turnover
    passes = (
        abs(ic_mean) >= criteria.min_abs_ic_mean
        and abs(information_ratio) >= criteria.min_abs_information_ratio
        and positive_rate >= criteria.min_positive_rate
        and long_short_spread >= criteria.min_long_short_spread
        and backtest_sharpe >= criteria.min_backtest_sharpe
        and turnover_ok
    )
    if passes:
        return "pass"

    inverse_candidate = (
        ic_mean <= -criteria.min_abs_ic_mean
        and information_ratio <= -criteria.min_abs_information_ratio
        and positive_rate <= (1.0 - criteria.min_positive_rate)
        and long_short_spread < -criteria.min_long_short_spread
    )
    if inverse_candidate:
        return "inverse_candidate"

    weak = (
        abs(ic_mean) >= criteria.min_abs_ic_mean * 0.5
        or abs(information_ratio) >= criteria.min_abs_information_ratio * 0.5
        or abs(long_short_spread) > 0
    )
    return "weak" if weak else "reject"


def evaluate_factor(
    name: str,
    factor: pd.DataFrame,
    data: MarketData,
    horizon: int = 1,
    quantiles: int = 5,
    criteria: FactorAdmissionCriteria = FactorAdmissionCriteria(),
    initial_capital: float = 100000,
    fee_bps: float = 6,
    slippage_bps: float = 2,
    gross_exposure: float = 1.0,
) -> dict[str, float | int | str]:
    ic = factor_ic_decay(factor, data.close, horizons=[horizon])
    if ic.empty:
        ic_mean = 0.0
        information_ratio = 0.0
        positive_rate = 0.0
        observation_count = 0
    else:
        row = ic.iloc[0]
        ic_mean = 0.0 if pd.isna(row["ic_mean"]) else float(row["ic_mean"])
        information_ratio = (
            0.0 if pd.isna(row["information_ratio"]) else float(row["information_ratio"])
        )
        positive_rate = 0.0 if pd.isna(row["positive_rate"]) else float(row["positive_rate"])
        observation_count = int(row["observation_count"])

    q_returns = quantile_forward_returns(
        factor,
        data.close,
        horizon=horizon,
        quantiles=min(quantiles, len(data.symbols)),
    )
    spread = _long_short_spread(q_returns)

    backtest_metrics = single_factor_backtest_metrics(
        data,
        factor,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        gross_exposure=gross_exposure,
    )
    decision = _admission_decision(
        ic_mean=ic_mean,
        information_ratio=information_ratio,
        positive_rate=positive_rate,
        long_short_spread=spread,
        backtest_sharpe=float(backtest_metrics["sharpe"]),
        turnover=float(backtest_metrics["total_turnover"]),
        criteria=criteria,
    )

    return {
        "factor": name,
        "horizon": horizon,
        "ic_mean": ic_mean,
        "information_ratio": information_ratio,
        "positive_rate": positive_rate,
        "observation_count": observation_count,
        "long_short_spread": spread,
        "single_total_return": float(backtest_metrics["total_return"]),
        "single_sharpe": float(backtest_metrics["sharpe"]),
        "single_max_drawdown": float(backtest_metrics["max_drawdown"]),
        "single_turnover": float(backtest_metrics["total_turnover"]),
        "decision": decision,
    }


def mine_factors(
    factors: dict[str, pd.DataFrame],
    data: MarketData,
    horizon: int = 1,
    quantiles: int = 5,
    criteria: FactorAdmissionCriteria = FactorAdmissionCriteria(),
    initial_capital: float = 100000,
    fee_bps: float = 6,
    slippage_bps: float = 2,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    records = [
        evaluate_factor(
            name=name,
            factor=factor,
            data=data,
            horizon=horizon,
            quantiles=quantiles,
            criteria=criteria,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            gross_exposure=gross_exposure,
        )
        for name, factor in factors.items()
    ]
    return pd.DataFrame.from_records(records).sort_values(
        ["decision", "information_ratio"],
        ascending=[True, False],
    )


def factor_quantile_report(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int = 1,
    quantiles: int = 5,
) -> pd.DataFrame:
    records = []
    for name, factor in factors.items():
        q = quantile_forward_returns(
            factor,
            close,
            horizon=horizon,
            quantiles=min(quantiles, len(close.columns)),
        )
        if q.empty:
            continue
        q = q.copy()
        q["factor"] = name
        records.append(q)
    if not records:
        return pd.DataFrame(
            columns=["timestamp", "horizon", "quantile", "mean_forward_return", "count", "factor"]
        )
    return pd.concat(records, ignore_index=True)


def single_factor_backtest_report(
    factors: dict[str, pd.DataFrame],
    data: MarketData,
    initial_capital: float = 100000,
    fee_bps: float = 6,
    slippage_bps: float = 2,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    records = []
    for name, factor in factors.items():
        metrics = single_factor_backtest_metrics(
            data,
            factor,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            gross_exposure=gross_exposure,
        )
        records.append({"factor": name, **metrics})
    return pd.DataFrame.from_records(records)


def accepted_factors(scorecard: pd.DataFrame) -> list[str]:
    if scorecard.empty or "decision" not in scorecard:
        return []
    return scorecard.loc[scorecard["decision"] == "pass", "factor"].tolist()
