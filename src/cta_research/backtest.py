from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cta_research.data import MarketData


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series


def _trade_records(positions: pd.DataFrame, turnover: pd.DataFrame) -> list[dict]:
    records = []
    active_turnover = turnover.fillna(0.0)

    for timestamp, row in active_turnover.iterrows():
        traded = row[row != 0.0]
        for symbol, traded_turnover in traded.items():
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "target_weight": positions.loc[timestamp, symbol],
                    "turnover": traded_turnover,
                }
            )

    return records


def run_backtest(
    data: MarketData,
    target_positions: pd.DataFrame,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
) -> BacktestResult:
    close = data.close
    positions = target_positions.reindex(index=close.index, columns=close.columns).fillna(0.0)
    executed_positions = positions.shift(1).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)

    gross_returns = (executed_positions * asset_returns).sum(axis=1)
    turnover = positions.diff().abs().fillna(positions.abs())
    if not turnover.empty:
        turnover.iloc[0] = 0.0
        gross_returns.iloc[0] = 0.0

    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = turnover.sum(axis=1) * cost_rate
    net_returns = gross_returns - costs
    if not net_returns.empty:
        costs.iloc[0] = 0.0
        net_returns.iloc[0] = 0.0

    equity = (1.0 + net_returns).cumprod() * initial_capital

    trades = pd.DataFrame.from_records(
        _trade_records(positions, turnover),
        columns=["timestamp", "symbol", "target_weight", "turnover"],
    )

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        positions=positions,
        trades=trades,
        gross_returns=gross_returns,
        costs=costs,
    )
