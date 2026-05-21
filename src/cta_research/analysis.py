from __future__ import annotations

import numpy as np
import pandas as pd

from cta_research.factor_research import factor_ic_table


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0

    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _annualized_return(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or total_return <= -1.0:
        return 0.0

    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def calculate_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int = 365,
) -> dict[str, float | int]:
    total_return = 0.0 if equity.empty or equity.iloc[0] == 0 else equity.iloc[-1] / equity.iloc[0] - 1.0
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = 0.0
    if returns.std(ddof=0) != 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = 0.0
    if not pd.isna(downside_std) and downside_std != 0:
        sortino = float(returns.mean() / downside_std * np.sqrt(periods_per_year))

    drawdown = max_drawdown(equity)
    cagr = _annualized_return(equity, periods_per_year)
    calmar = 0.0 if drawdown == 0 else float(cagr / abs(drawdown))
    total_turnover = 0.0
    if "turnover" in trades:
        total_turnover = float(trades["turnover"].sum())

    return {
        "total_return": float(total_return),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trade_count": int(len(trades)),
        "total_turnover": total_turnover,
    }


def factor_ic(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    return factor_ic_table(factor, close, horizons, method="spearman").rename(
        columns={"ic": "rank_ic"}
    )
